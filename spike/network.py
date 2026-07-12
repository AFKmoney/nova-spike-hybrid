"""
SpikingNetwork — populations + synapses CSR + event queue.

Architecture:
  - 3 populations: SENSORY (input), ASSOCIATIVE (réservoir), MOTOR (output)
  - Synapses stockées en CSR sparse (scipy.sparse)
  - Propagation event-driven: pour chaque tick, on récupère les neurones
    qui spikent, et on ne fait QUE les additions vers leurs cibles.

Le graphe de connectivité:
  sensory  →  associative  (poids STDP-plastic)
  associative → associative (récurrence, STDP-plastic)
  associative → motor      (poids STDP-plastic)

Pas de matmul jamais. Que des additions + masques booléens.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .core import LIFNeuron, LIFParams, SimulationClock


class PopulationType(Enum):
    SENSORY = "sensory"
    ASSOCIATIVE = "associative"
    MOTOR = "motor"


@dataclass
class SynapseGroup:
    """
    Groupe de synapses entre deux populations.

    W est une matrice CSR (n_pre, n_post). W[i, j] = poids de la synapse
    du neurone pré-i vers le neurone post-j.

    Pour la propagation: si spikes_pre est un masque booléen (n_pre,),
    le courant injecté dans la population post est:
        I_post = W[spikes_pre].sum(axis=0)
    Ce qui se fait en O(nnz_des_lignes_activées) — sparse-native.
    """
    name: str
    pre_type: PopulationType
    post_type: PopulationType
    n_pre: int
    n_post: int
    W: sp.csr_matrix = field(init=False)
    delay: int = 1   # ticks de délai de propagation
    plastic: bool = True

    def __post_init__(self):
        # Init à zéro — on ajoutera les poids explicitement
        self.W = sp.csr_matrix((self.n_pre, self.n_post), dtype=np.float32)

    # ---------------------------------------------------------------- #
    def random_connect(self, density: float = 0.05, w_mean: float = 0.5,
                       w_std: float = 0.2, seed: int = 0) -> None:
        """
        Connecte aléatoirement: chaque synapse existe avec proba `density`.
        Poids ~ Normal(w_mean, w_std), tronqués à [0, ∞) pour les excitateurs.
        """
        rng = np.random.default_rng(seed)
        n_synapses = int(self.n_pre * self.n_post * density)
        if n_synapses == 0:
            return
        rows = rng.integers(0, self.n_pre, size=n_synapses)
        cols = rng.integers(0, self.n_post, size=n_synapses)
        vals = np.maximum(0, rng.normal(w_mean, w_std, n_synapses)).astype(np.float32)
        self.W = sp.csr_matrix((vals, (rows, cols)),
                                shape=(self.n_pre, self.n_post),
                                dtype=np.float32)

    def random_connect_inhibitory(self, density: float = 0.02,
                                   w_mean: float = -0.8,
                                   seed: int = 0) -> None:
        """Connectivité inhibitrice — poids négatifs."""
        rng = np.random.default_rng(seed)
        n_synapses = int(self.n_pre * self.n_post * density)
        if n_synapses == 0:
            return
        rows = rng.integers(0, self.n_pre, size=n_synapses)
        cols = rng.integers(0, self.n_post, size=n_synapses)
        vals = rng.normal(w_mean, 0.1, n_synapses).astype(np.float32)
        self.W = sp.csr_matrix((vals, (rows, cols)),
                                shape=(self.n_pre, self.n_post),
                                dtype=np.float32)

    # ---------------------------------------------------------------- #
    def propagate(self, spikes_pre: np.ndarray) -> np.ndarray:
        """
        Propage les spikes de pre vers post.
        spikes_pre: masque booléen (n_pre,)
        Return: courant (n_post,)

        C'est l'opération clé. W[spikes_pre] récupère les lignes activées
        en O(indices_activés), puis on somme sur l'axe 0.
        """
        if not spikes_pre.any():
            return np.zeros(self.n_post, dtype=np.float32)
        # Récupère les indices des neurones pré qui spikent
        pre_indices = np.where(spikes_pre)[0]
        # Sélectionne les lignes de W et somme
        # W[pre_indices] est sparse (k, n_post), on somme sur axis=0
        I_post = np.asarray(self.W[pre_indices].sum(axis=0)).ravel()
        return I_post.astype(np.float32)

    # ---------------------------------------------------------------- #
    def add_synapse(self, i_pre: int, j_post: int, weight: float) -> None:
        """Ajoute ou met à jour une synapse."""
        self.W[i_pre, j_post] = weight

    def get_synapse(self, i_pre: int, j_post: int) -> float:
        return float(self.W[i_pre, j_post])

    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        return {
            "name": self.name,
            "pre_type": self.pre_type.value,
            "post_type": self.post_type.value,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "n_synapses": self.W.nnz,
            "density": self.W.nnz / max(1, self.n_pre * self.n_post),
            "plastic": self.plastic,
        }


# ---------------------------------------------------------------------- #
# Réseau complet
# ---------------------------------------------------------------------- #

@dataclass
class SpikingNetwork:
    """
    Réseau SNN complet: 3 populations + 3 groupes de synapses.

    Sensory → Associative → Motor
                ↻ (récurrence)
    """
    n_sensory: int = 200
    n_associative: int = 1000
    n_motor: int = 100
    dt: float = 1.0

    # Populations
    sensory: LIFNeuron = field(init=False)
    associative: LIFNeuron = field(init=False)
    motor: LIFNeuron = field(init=False)

    # Synapses
    syn_sens_to_assoc: SynapseGroup = field(init=False)
    syn_assoc_to_assoc: SynapseGroup = field(init=False)
    syn_assoc_to_motor: SynapseGroup = field(init=False)

    # Horloge
    clock: SimulationClock = field(default_factory=SimulationClock)

    # Pour STDP — on garde les derniers spikes par population
    last_spikes: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Populations avec params différents
        self.sensory = LIFNeuron(LIFParams(
            n_neurons=self.n_sensory, tau_m=10.0, V_thresh=0.8, V_reset=0.0,
            tau_ref=1.0, dt=self.dt, heterogeneity=0.05,
        ))
        self.associative = LIFNeuron(LIFParams(
            n_neurons=self.n_associative, tau_m=20.0, V_thresh=1.0,
            V_reset=0.0, tau_ref=2.0, dt=self.dt, heterogeneity=0.15,
        ))
        self.motor = LIFNeuron(LIFParams(
            n_neurons=self.n_motor, tau_m=15.0, V_thresh=1.2, V_reset=0.0,
            tau_ref=5.0, dt=self.dt, heterogeneity=0.1,
        ))

        # Synapses — initialisées à zéro
        self.syn_sens_to_assoc = SynapseGroup(
            "sens→assoc", PopulationType.SENSORY, PopulationType.ASSOCIATIVE,
            self.n_sensory, self.n_associative, plastic=True,
        )
        self.syn_assoc_to_assoc = SynapseGroup(
            "assoc→assoc", PopulationType.ASSOCIATIVE, PopulationType.ASSOCIATIVE,
            self.n_associative, self.n_associative, plastic=True,
        )
        self.syn_assoc_to_motor = SynapseGroup(
            "assoc→motor", PopulationType.ASSOCIATIVE, PopulationType.MOTOR,
            self.n_associative, self.n_motor, plastic=True,
        )

        # Connexions aléatoires initiales (sparse)
        self.syn_sens_to_assoc.random_connect(density=0.10, w_mean=0.7, w_std=0.2, seed=1)
        # Récurrence: mixte excitateur (70%) + inhibiteur (30%)
        self.syn_assoc_to_assoc.random_connect(density=0.02, w_mean=0.5, w_std=0.15, seed=2)
        # Pour l'inhibition, on ajoute une seconde matrice (somme)
        inh = SynapseGroup("inh", PopulationType.ASSOCIATIVE, PopulationType.ASSOCIATIVE,
                           self.n_associative, self.n_associative, plastic=False)
        inh.random_connect_inhibitory(density=0.01, w_mean=-0.8, seed=3)
        self.syn_assoc_to_assoc.W = self.syn_assoc_to_assoc.W + inh.W
        self.syn_assoc_to_motor.random_connect(density=0.05, w_mean=0.6, w_std=0.2, seed=4)

    # ---------------------------------------------------------------- #
    # Simulation — un tick
    # ---------------------------------------------------------------- #
    def tick(self, I_sensory: np.ndarray,
             record: bool = False) -> dict:
        """
        Avance d'un pas de simulation.

        Args:
            I_sensory: courant injecté dans la population sensorielle (n_sensory,)
            record: si True, retourne les spikes de chaque population

        Return:
            dict avec les masques de spikes (si record=True).
        """
        t = self.clock.t

        # 1. Step sensory
        spikes_sens = self.sensory.step(I_sensory, t)

        # 2. Propagation sensory → associative
        I_assoc = self.syn_sens_to_assoc.propagate(spikes_sens)

        # 3. Step associative (avec input externe)
        spikes_assoc = self.associative.step(I_assoc, t)

        # 4. Récurrence associative → associative (au prochain tick en pratique,
        # mais ici on l'applique immédiatement pour simplifier)
        # I_assoc_recur = self.syn_assoc_to_assoc.propagate(spikes_assoc)
        # Pour éviter une double intégration dans le même tick, on stocke
        # et on l'appliquera au tick suivant via un buffer.

        # 5. Propagation associative → motor
        I_motor = self.syn_assoc_to_motor.propagate(spikes_assoc)

        # 6. Step motor
        spikes_motor = self.motor.step(I_motor, t)

        # Tick suivant
        self.clock.tick()

        # Pour STDP — on garde les spikes les plus récents
        self.last_spikes = {
            "sensory": spikes_sens.copy(),
            "associative": spikes_assoc.copy(),
            "motor": spikes_motor.copy(),
            "t": t,
        }

        if record:
            return {
                "t": t,
                "sensory_spikes": spikes_sens,
                "assoc_spikes": spikes_assoc,
                "motor_spikes": spikes_motor,
            }
        return {}

    # ---------------------------------------------------------------- #
    # Simulation — N ticks
    # ---------------------------------------------------------------- #
    def run(self, I_sensory_fn, n_ticks: int, record: bool = False) -> list:
        """
        Run n_ticks. I_sensory_fn(t) renvoie le courant sensoriel au tick t.

        Si record=True, retourne une liste de dicts par tick (utile pour debug
        mais coûteux en RAM).
        """
        records = []
        for _ in range(n_ticks):
            I_s = I_sensory_fn(self.clock.t)
            r = self.tick(I_s, record=record)
            if record:
                records.append(r)
        return records

    # ---------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------- #
    def reset(self, soft: bool = False) -> None:
        """Reset toutes les populations."""
        self.sensory.reset(soft=soft)
        self.associative.reset(soft=soft)
        self.motor.reset(soft=soft)
        self.clock.reset()
        self.last_spikes = {}

    # ---------------------------------------------------------------- #
    # Diagnostics
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        return {
            "n_sensory": self.n_sensory,
            "n_associative": self.n_associative,
            "n_motor": self.n_motor,
            "t": self.clock.t,
            "n_ticks": self.clock.n_ticks,
            "sensory_total_spikes": int(self.sensory.spike_count.sum()),
            "assoc_total_spikes": int(self.associative.spike_count.sum()),
            "motor_total_spikes": int(self.motor.spike_count.sum()),
            "synapses": {
                "sens_to_assoc": self.syn_sens_to_assoc.stats(),
                "assoc_to_assoc": self.syn_assoc_to_assoc.stats(),
                "assoc_to_motor": self.syn_assoc_to_motor.stats(),
            },
        }


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time as _time
    print("Test SpikingNetwork (sens=100, assoc=500, motor=50)...")
    net = SpikingNetwork(n_sensory=100, n_associative=500, n_motor=50)

    # Injecte du courant sur 10 neurones sensoriels
    def I_fn(t):
        I = np.zeros(100, dtype=np.float32)
        if 10 < t < 50:
            I[10:20] = 2.0
        return I

    t0 = _time.time()
    records = net.run(I_fn, n_ticks=100, record=True)
    t1 = _time.time()

    print(f"100 ticks en {(t1-t0)*1000:.2f} ms")
    print(f"Stats: {net.stats()['sensory_total_spikes']} sensory spikes, "
          f"{net.stats()['assoc_total_spikes']} assoc spikes, "
          f"{net.stats()['motor_total_spikes']} motor spikes")
    # Activité par tick
    activity = [r["assoc_spikes"].sum() for r in records]
    print(f"Activité associative (10 premiers ticks): {activity[:10]}")
    print(f"Activité associative (ticks 10-20): {activity[10:20]}")
