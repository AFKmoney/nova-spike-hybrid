"""
Noyau SNN : neurones LIF, spikes, horloge de simulation.

Le neurone LIF (Leaky Integrate-and-Fire):
    τ_m · dV/dt = -V + R · I(t)
    si V >= V_thresh: spike, V ← V_reset, réfractaire pendant τ_ref

Implémentation vectorisée: on gère TOUS les neurones d'une population
en parallèle avec numpy. L'équation est discrétisée en Euler:

    V(t+dt) = V(t) + dt/τ_m · (-V(t) + R · I(t))

La fuite (leak) est vectorisée: V *= decay où decay = exp(-dt/τ_m).

Le déclenchement (fire) se fait par masque booléen:
    spikes = V >= V_thresh
    V[spikes] = V_reset
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LIFParams:
    """Paramètres d'une population de neurones LIF (vectorisés)."""
    n_neurons: int
    tau_m: float = 20.0       # ms, constante de membrane
    V_thresh: float = 1.0     # seuil de déclenchement
    V_reset: float = 0.0      # potentiel après spike
    V_rest: float = 0.0       # potentiel de repos
    R: float = 1.0            # résistance d'entrée
    tau_ref: float = 2.0      # ms, période réfractaire
    dt: float = 1.0           # ms, pas de simulation

    # Hétérogénéité (optionnel): facteurs multiplicatifs par neurone
    # pour casser la symétrie et enrichir la dynamique
    heterogeneity: float = 0.1  # ±10% de variation


@dataclass
class LIFNeuron:
    """
    État d'une population de neurones LIF.

    Toutes les grandeurs sont des arrays de shape (n_neurons,).
    """
    params: LIFParams
    V: np.ndarray = field(init=False)              # potentiel de membrane
    refractory: np.ndarray = field(init=False)     # temps réfractaire restant
    spike_count: np.ndarray = field(init=False)    # compteur total
    last_spike_time: np.ndarray = field(init=False)  # t du dernier spike

    # Facteurs d'hétérogénéité (par neurone)
    tau_m_factor: np.ndarray = field(init=False)
    thresh_factor: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.params.n_neurons
        rng = np.random.default_rng()
        self.V = np.full(n, self.params.V_rest, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.float32)
        self.spike_count = np.zeros(n, dtype=np.int32)
        self.last_spike_time = np.full(n, -1e6, dtype=np.float32)
        # Hétérogénéité
        h = self.params.heterogeneity
        self.tau_m_factor = 1.0 + rng.uniform(-h, h, n).astype(np.float32)
        self.thresh_factor = 1.0 + rng.uniform(-h, h, n).astype(np.float32)

    # ---------------------------------------------------------------- #
    # Step de simulation — vectorisé
    # ---------------------------------------------------------------- #
    def step(self, I: np.ndarray, t: float) -> np.ndarray:
        """
        Avance d'un pas dt. Applique le courant I, calcule les spikes.

        Args:
            I: courant d'entrée (n_neurons,), peut être négatif (inhibiteur)
            t: temps courant (ms)

        Return:
            Masque booléen (n_neurons,) — True où un spike a été émis.
        """
        dt = self.params.dt

        # 1. Décroissance du temps réfractaire
        active = self.refractory <= 0  # neurones non réfractaires

        # 2. Decay du potentiel pour les neurones actifs (leak)
        # V *= exp(-dt / (tau_m * tau_m_factor))
        decay = np.exp(-dt / (self.params.tau_m * self.tau_m_factor)).astype(np.float32)
        self.V = np.where(active, self.V * decay, self.V)

        # 3. Intégration du courant (only for active neurons)
        # V += R * I * dt / tau_m
        delta = self.params.R * I * dt / (self.params.tau_m * self.tau_m_factor)
        self.V = np.where(active, self.V + delta, self.V)

        # 4. Déclenchement (fire)
        thresh = self.params.V_thresh * self.thresh_factor
        spikes = active & (self.V >= thresh)

        # 5. Reset + réfractaire pour les neurones qui spikent
        self.V[spikes] = self.params.V_reset
        self.refractory[spikes] = self.params.tau_ref
        self.spike_count[spikes] += 1
        self.last_spike_time[spikes] = t

        # 6. Decay du temps réfractaire
        self.refractory = np.maximum(0, self.refractory - dt)

        return spikes

    # ---------------------------------------------------------------- #
    def reset(self, soft: bool = False) -> None:
        """Reset l'état. Si soft=True, garde une trace (V *= 0.5)."""
        if soft:
            self.V *= 0.5
        else:
            self.V = np.full_like(self.V, self.params.V_rest)
        self.refractory = np.zeros_like(self.refractory)
        # On garde spike_count et last_spike_time


@dataclass
class Spike:
    """Événement spike (pour debug/logging)."""
    neuron_id: int
    time: float
    weight: float = 1.0


@dataclass
class SimulationClock:
    """Horloge de simulation — émet des ticks."""
    t: float = 0.0
    dt: float = 1.0
    n_ticks: int = 0

    def tick(self) -> float:
        """Avance d'un pas, renvoie le nouveau temps."""
        self.t += self.dt
        self.n_ticks += 1
        return self.t

    def reset(self) -> None:
        self.t = 0.0
        self.n_ticks = 0


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time as _time
    print("Test neurones LIF (N=1000, 1000 ticks)...")
    params = LIFParams(n_neurons=1000, tau_m=20.0, V_thresh=1.0)
    neuron = LIFNeuron(params)

    # Courant constant sur 100 neurones
    I = np.zeros(1000, dtype=np.float32)
    I[:100] = 1.5  # au-dessus du seuil → devraient spiker régulièrement

    clock = SimulationClock()
    t0 = _time.time()
    spike_log = []
    for _ in range(1000):
        s = neuron.step(I, clock.t)
        if s.any():
            spike_log.append(s.sum())
        clock.tick()
    t1 = _time.time()

    print(f"1000 ticks en {(t1-t0)*1000:.2f} ms")
    print(f"Neurones 0-99 (I=1.5): {neuron.spike_count[:100].mean():.1f} spikes en moyenne")
    print(f"Neurones 100-999 (I=0): {neuron.spike_count[100:].mean():.1f} spikes en moyenne")
    print(f"Total spikes: {neuron.spike_count.sum()}")
