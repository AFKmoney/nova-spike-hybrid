"""
Lazy spikes — buffer de propagation asynchrone.

Problème: dans un SNN synchrone, à chaque tick on doit calculer la
propagation pour TOUS les neurones, même si 99% ne spikent pas.
C'est OK pour N=1000, mais pour N=100000+ c'est prohibitif.

Solution: buffer de spikes différés. Au lieu de propager immédiatement,
on empile les spikes dans une queue avec leur délai. À chaque tick, on
ne dépile que les spikes dont le délai est écoulé.

Avantages:
  - O(spikes_actifs) au lieu de O(N) par tick
  - Permet des délais hétérogènes par synapse (biologiquement réaliste)
  - Taille du réseau illimitée (mémoire = nb de spikes en attente)

Inconvénients:
  - Légère latence (délai min = 1 tick par défaut)
  - Plus complexe à debug

Usage:
    buffer = LazySpikeBuffer()
    buffer.add_spike(neuron_id=42, delay=3, weight=1.5)
    buffer.add_spikes([0, 5, 12], delay=1, weights=[0.5, 0.7, 0.3])
    spikes_now = buffer.tick()  # dépile les spikes dus au tick courant
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LazySpikeBuffer:
    """
    Buffer de spikes différés.

    Structure interne:
      - delayed_spikes: dict { arrival_tick -> list of (target_neuron, weight) }
      - current_tick: temps courant

    À chaque tick:
      1. On extrait les spikes dus à current_tick
      2. On agrège par neurone cible (somme des poids)
      3. On incrémente current_tick
    """
    current_tick: int = 0
    # delayed_spikes[t] = list of (target_neuron, weight)
    delayed_spikes: dict = field(init=False, default_factory=lambda: defaultdict(list))
    n_buffered: int = 0  # stats

    def add_spike(self, target_neuron: int, weight: float, delay: int = 1) -> None:
        """Ajoute un spike vers target_neuron avec délai."""
        if delay < 1:
            delay = 1
        arrival = self.current_tick + delay
        self.delayed_spikes[arrival].append((target_neuron, float(weight)))
        self.n_buffered += 1

    def add_spikes(self, target_neurons: np.ndarray, weights: np.ndarray,
                   delay: int = 1) -> None:
        """Version vectorisée — ajoute plusieurs spikes."""
        if delay < 1:
            delay = 1
        arrival = self.current_tick + delay
        for n, w in zip(target_neurons, weights):
            self.delayed_spikes[arrival].append((int(n), float(w)))
        self.n_buffered += len(target_neurons)

    def tick(self) -> np.ndarray:
        """
        Avance d'un tick. Renvoie un array (n_neurons,) avec les poids
        agrégés pour les spikes dus à current_tick.

        Attention: pour utiliser ce retour, il faut connaître n_neurons
        côté appelant. Sinon, utiliser tick_dict() qui renvoie un dict.
        """
        arrival = self.current_tick
        spikes = self.delayed_spikes.pop(arrival, [])
        self.current_tick += 1
        return spikes  # list of (target, weight)

    def tick_aggregated(self, n_neurons: int) -> np.ndarray:
        """
        Tick + agrège les spikes en un array (n_neurons,) de poids.
        Pour injecter directement dans LIFNeuron.step(I=...).
        """
        spikes = self.tick()
        I = np.zeros(n_neurons, dtype=np.float32)
        for target, weight in spikes:
            if 0 <= target < n_neurons:
                I[target] += weight
        return I

    def reset(self) -> None:
        """Vide le buffer et reset le tick."""
        self.delayed_spikes.clear()
        self.current_tick = 0
        self.n_buffered = 0

    def stats(self) -> dict:
        return {
            "current_tick": self.current_tick,
            "n_pending_ticks": len(self.delayed_spikes),
            "n_buffered_total": self.n_buffered,
            "n_current_tick": len(self.delayed_spikes.get(self.current_tick, [])),
        }


# ---------------------------------------------------------------------- #
# Network avec lazy propagation
# ---------------------------------------------------------------------- #

@dataclass
class LazySpikingNetwork:
    """
    Variante de SpikingNetwork avec lazy spike propagation.

    Au lieu de propager les spikes immédiatement à chaque tick, on les
    empile dans un buffer avec délai. Au tick suivant, on dépile et on
    injecte le courant agrégé.

    Gain attendu: ~10x plus rapide pour N > 10000 car on évite la
    multiplication sparse @ dense quand il n'y a pas de spikes.

    La population associative a maintenant un délai hétérogène par
    synapse (entre 1 et max_delay ticks), ce qui ajoute une dynamique
    temporelle riche (réponses oscillatoires, synchronisation, etc.).
    """
    n_sensory: int = 200
    n_associative: int = 1000
    n_motor: int = 100
    max_delay: int = 5      # délai max (en ticks) pour les synapses
    dt: float = 1.0

    # Lazy buffers — un par groupe de synapses
    buffer_sens_assoc: LazySpikeBuffer = field(init=False)
    buffer_assoc_assoc: LazySpikeBuffer = field(init=False)
    buffer_assoc_motor: LazySpikeBuffer = field(init=False)

    def __post_init__(self):
        # On réutilise l'architecture de SpikingNetwork mais avec lazy buffers
        from .network import SpikingNetwork, SynapseGroup, PopulationType
        from .core import LIFNeuron, LIFParams, SimulationClock

        self.sensory = LIFNeuron(LIFParams(
            n_neurons=self.n_sensory, tau_m=10.0, V_thresh=0.8,
            tau_ref=1.0, dt=self.dt, heterogeneity=0.05,
        ))
        self.associative = LIFNeuron(LIFParams(
            n_neurons=self.n_associative, tau_m=20.0, V_thresh=1.0,
            tau_ref=2.0, dt=self.dt, heterogeneity=0.15,
        ))
        self.motor = LIFNeuron(LIFParams(
            n_neurons=self.n_motor, tau_m=15.0, V_thresh=1.2,
            tau_ref=5.0, dt=self.dt, heterogeneity=0.1,
        ))

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

        # Connectivité aléatoire
        self.syn_sens_to_assoc.random_connect(density=0.10, w_mean=1.5, w_std=0.3, seed=1)
        self.syn_assoc_to_assoc.random_connect(density=0.02, w_mean=0.8, w_std=0.2, seed=2)
        # Inhibition
        inh = SynapseGroup("inh", PopulationType.ASSOCIATIVE, PopulationType.ASSOCIATIVE,
                           self.n_associative, self.n_associative, plastic=False)
        inh.random_connect_inhibitory(density=0.01, w_mean=-0.8, seed=3)
        self.syn_assoc_to_assoc.W = self.syn_assoc_to_assoc.W + inh.W
        self.syn_assoc_to_motor.random_connect(density=0.05, w_mean=1.2, w_std=0.3, seed=4)

        # Buffers
        self.buffer_sens_assoc = LazySpikeBuffer()
        self.buffer_assoc_assoc = LazySpikeBuffer()
        self.buffer_assoc_motor = LazySpikeBuffer()

        self.clock = SimulationClock()
        self.last_spikes: dict = {}

    # ---------------------------------------------------------------- #
    def tick(self, I_sensory: np.ndarray, record: bool = False) -> dict:
        """
        Tick lazy:
          1. Récupère les spikes dus au tick courant (depuis les buffers)
          2. Step sensory (avec I externe)
          3. Injecte les spikes sensory→assoc dans le buffer (délai 1)
          4. Step associative (avec I des buffers)
          5. Injecte les spikes assoc→assoc et assoc→motor dans les buffers
          6. Step motor
        """
        t = self.clock.t

        # 1. Récupère les courants dus au tick courant
        I_assoc_from_sens = self.buffer_sens_assoc.tick_aggregated(self.n_associative)
        I_assoc_from_assoc = self.buffer_assoc_assoc.tick_aggregated(self.n_associative)
        I_motor_from_assoc = self.buffer_assoc_motor.tick_aggregated(self.n_motor)

        # 2. Step sensory
        spikes_sens = self.sensory.step(I_sensory, t)

        # 3. Propagation sensory → associative (dans le buffer)
        if spikes_sens.any():
            pre_indices = np.where(spikes_sens)[0]
            # Récupère les poids des neurones pré qui spikent
            # W[pre_indices] est sparse (k, n_post)
            sub_W = self.syn_sens_to_assoc.W[pre_indices].tocoo()
            # Pour chaque synapse non-nulle, on empile (target, weight)
            for pre_local_idx, post, weight in zip(sub_W.row, sub_W.col, sub_W.data):
                # Le délai peut être déterminé par pre + post hash, ou simplementement 1
                # Pour rester simple: délai = 1 + (post % self.max_delay)
                delay = 1 + (int(post) % self.max_delay)
                # target_neuron = post (index global dans la pop associative)
                self.buffer_sens_assoc.add_spike(int(post), float(weight), delay=delay)

        # 4. Step associative (avec I cumulé)
        I_assoc_total = I_assoc_from_sens + I_assoc_from_assoc
        spikes_assoc = self.associative.step(I_assoc_total, t)

        # 5. Propagation assoc → assoc (dans le buffer)
        if spikes_assoc.any():
            pre_indices = np.where(spikes_assoc)[0]
            sub_W = self.syn_assoc_to_assoc.W[pre_indices].tocoo()
            for _, post, weight in zip(sub_W.row, sub_W.col, sub_W.data):
                delay = 1 + (int(post) % self.max_delay)
                self.buffer_assoc_assoc.add_spike(int(post), float(weight), delay=delay)

        # 6. Propagation assoc → motor
        if spikes_assoc.any():
            pre_indices = np.where(spikes_assoc)[0]
            sub_W = self.syn_assoc_to_motor.W[pre_indices].tocoo()
            for _, post, weight in zip(sub_W.row, sub_W.col, sub_W.data):
                delay = 1 + (int(post) % self.max_delay)
                self.buffer_assoc_motor.add_spike(int(post), float(weight), delay=delay)

        # 7. Step motor
        spikes_motor = self.motor.step(I_motor_from_assoc, t)

        # Tick
        self.clock.tick()

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
    def reset(self, soft: bool = False) -> None:
        self.sensory.reset(soft=soft)
        self.associative.reset(soft=soft)
        self.motor.reset(soft=soft)
        self.clock.reset()
        self.buffer_sens_assoc.reset()
        self.buffer_assoc_assoc.reset()
        self.buffer_assoc_motor.reset()
        self.last_spikes = {}

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
            "buffer_sens_assoc": self.buffer_sens_assoc.stats(),
            "buffer_assoc_assoc": self.buffer_assoc_assoc.stats(),
            "buffer_assoc_motor": self.buffer_assoc_motor.stats(),
        }


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time as _time
    print("Test LazySpikeBuffer...")
    buf = LazySpikeBuffer()
    buf.add_spike(5, 1.5, delay=2)
    buf.add_spike(7, 0.8, delay=2)
    buf.add_spike(5, 0.3, delay=1)
    print(f"Tick 0: {buf.tick_aggregated(10)}")  # spike 5 poids 0.3
    print(f"Tick 1: {buf.tick_aggregated(10)}")  # spike 5 (1.5) + spike 7 (0.8)
    print(f"Tick 2: {buf.tick_aggregated(10)}")  # rien

    print("\nTest LazySpikingNetwork...")
    net = LazySpikingNetwork(n_sensory=100, n_associative=500, n_motor=50, max_delay=3)
    I_fn = lambda t: np.where((np.arange(100) < 10) & (10 < t < 50), 3.0, 0.0).astype(np.float32)

    t0 = _time.time()
    for _ in range(100):
        net.tick(I_fn(net.clock.t))
    t1 = _time.time()

    print(f"100 ticks en {(t1-t0)*1000:.2f} ms")
    s = net.stats()
    print(f"Spikes: sensory={s['sensory_total_spikes']}, "
          f"assoc={s['assoc_total_spikes']}, motor={s['motor_total_spikes']}")
    print(f"Buffer sens→assoc: {s['buffer_sens_assoc']}")
