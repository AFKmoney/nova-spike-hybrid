"""
kuramoto.py — Kuramoto oscillator network for cognitive binding.

BRAIN INSPIRATION
-----------------
In the brain, distant neural populations synchronize their firing when
they represent bound features (e.g., the red color and the shape of an
apple are bound by synchronous gamma oscillations ~40Hz). Singer, Gray,
Engel and others showed this experimentally in the 1990s.

The Kuramoto model (Yoshiki Kuramoto, 1984) is the canonical mathematical
model of synchronization:

    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)

where:
  - θ_i is the phase of oscillator i
  - ω_i is its natural frequency
  - K is the global coupling strength
  - N is the number of oscillators

The ORDER PARAMETER r = |Σ e^(iθ_i)| / N measures global synchrony:
  r ≈ 0  → incoherent (chaos)
  r ≈ 1  → fully synchronized (one big cluster)

AETHER'S USE
-----------
Each concept / token / perception is an oscillator. The HD vector of a
concept initializes its phase:

    θ_i = angle(Σ_j HD_vector_i[j] * exp(i * 2π * j / D))

Two concepts with similar HD vectors start with similar phases → they
synchronize faster → they are "bound" together.

CLUSTERS of synchronized oscillators = BOUND IDEAS.

When the system runs, you observe:
  - Phase clusters forming (bound concepts)
  - Order parameter rising (comprehension forming)
  - Frequency locking (stable ideas)
  - Phase slips (idea shifts / "aha!" moments)

This is the dynamical layer beneath the HD algebra.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from dataclasses import dataclass, field

from .hd import HDVector, DIM


# ---------------------------------------------------------------------------
# Phase extraction from HD vectors
# ---------------------------------------------------------------------------

def hd_to_phase(vec: HDVector, n_oscillators: int = 32) -> np.ndarray:
    """Extract N oscillator phases from an HD vector.

    The HD vector is split into N chunks; the angle of each chunk's
    "complex representation" gives a phase. Two HD vectors with similar
    bits produce similar phase patterns.
    """
    chunk_size = len(vec.data) // n_oscillators
    phases = np.zeros(n_oscillators, dtype=np.float64)
    for i in range(n_oscillators):
        chunk = vec.data[i * chunk_size:(i + 1) * chunk_size].astype(np.float64)
        # Map -1/+1 bipolar to angle: +1 → 0, -1 → π
        # Plus deterministic noise from chunk content for richer dynamics
        mean = chunk.mean()
        # Spread: -1 to +1 maps to π to 0
        phases[i] = (1.0 - mean) * np.pi / 2.0
        # Add small offset from chunk variance
        if len(chunk) > 1:
            phases[i] += np.sin(chunk.sum()) * 0.3
    return phases


def hd_to_frequency(vec: HDVector, n_oscillators: int = 32,
                    base_freq: float = 1.0, freq_spread: float = 0.5) -> np.ndarray:
    """Extract N natural frequencies from an HD vector.

    Frequencies are drawn from a narrow band around base_freq. The HD
    vector determines the offset, so different concepts have different
    intrinsic rhythms (matching biology: different cortical areas have
    different characteristic frequencies).
    """
    chunk_size = len(vec.data) // n_oscillators
    freqs = np.zeros(n_oscillators, dtype=np.float64)
    for i in range(n_oscillators):
        chunk = vec.data[i * chunk_size:(i + 1) * chunk_size].astype(np.float64)
        # Use chunk variance as frequency offset
        offset = (chunk.std() - 1.0) * freq_spread  # std of bipolar ±1 is 1.0
        freqs[i] = base_freq + offset
    return freqs


# ---------------------------------------------------------------------------
# Kuramoto network
# ---------------------------------------------------------------------------

@dataclass
class KuramotoState:
    """State of the Kuramoto network at a single time step."""
    phases: np.ndarray           # θ_i for each oscillator
    frequencies: np.ndarray      # ω_i (natural frequencies)
    order_parameter: float       # r (global synchrony)
    mean_phase: float            # ψ (mean field phase)
    clusters: List[List[int]]    # synchronized clusters
    time: float = 0.0


class KuramotoNetwork:
    """A Kuramoto oscillator network for cognitive binding.

    Each "concept" or "percept" is represented as one or more oscillators.
    Concepts with similar HD vectors start with similar phases → they
    synchronize faster → they are "bound" together.

    The network evolves over time; synchronized clusters = bound ideas.
    """

    def __init__(
        self,
        n_oscillators: int = 64,
        coupling: float = 0.6,
        dt: float = 0.05,
        base_freq: float = 1.0,
        freq_spread: float = 0.8,
        phase_noise: float = 0.02,
    ):
        self.N = n_oscillators
        self.K = coupling
        self.dt = dt
        self.base_freq = base_freq
        self.freq_spread = freq_spread
        self.phase_noise = phase_noise

        # State
        self.phases: np.ndarray = np.random.uniform(0, 2 * np.pi, n_oscillators)
        self.frequencies: np.ndarray = np.full(n_oscillators, base_freq, dtype=np.float64)
        # Which oscillators belong to which concept (for interpretation)
        self.concept_labels: List[str] = [""] * n_oscillators
        # History for analysis
        self.history: List[KuramotoState] = []

    # ------------------------------------------------------------------ #
    # Initialization from HD vectors
    # ------------------------------------------------------------------ #
    def add_concept(self, label: str, vec: HDVector, n_osc: int = 4) -> int:
        """Add a concept (HD vector) to the network. Returns the start index."""
        # Find free slots
        free_indices = [i for i, lbl in enumerate(self.concept_labels) if not lbl]
        if len(free_indices) < n_osc:
            # Expand the network
            old_N = self.N
            self.N += n_osc
            self.phases = np.concatenate([self.phases, np.random.uniform(0, 2*np.pi, n_osc)])
            self.frequencies = np.concatenate([self.frequencies, np.full(n_osc, self.base_freq)])
            self.concept_labels.extend([""] * n_osc)
            free_indices = list(range(old_N, self.N))

        # Assign n_osc oscillators to this concept
        start_idx = free_indices[0]
        for k in range(n_osc):
            idx = free_indices[k]
            self.concept_labels[idx] = label
        # Initialize their phases and frequencies from the HD vector
        phases = hd_to_phase(vec, n_oscillators=n_osc)
        freqs = hd_to_frequency(vec, n_oscillators=n_osc,
                                base_freq=self.base_freq,
                                freq_spread=self.freq_spread)
        for k, idx in enumerate(free_indices[:n_osc]):
            self.phases[idx] = phases[k]
            self.frequencies[idx] = freqs[k]
        return start_idx

    def reset(self) -> None:
        """Reset all phases to random."""
        self.phases = np.random.uniform(0, 2 * np.pi, self.N)
        self.history.clear()

    def clear_concepts(self) -> None:
        """Remove all concept labels (keeps oscillator count)."""
        self.concept_labels = [""] * self.N
        self.history.clear()

    # ------------------------------------------------------------------ #
    # Evolution (the Kuramoto equation)
    # ------------------------------------------------------------------ #
    def step(self) -> KuramotoState:
        """Evolve one time step using the Kuramoto equation.

        dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i) + noise
        """
        # Compute pairwise phase differences (vectorized)
        # sin(θ_j - θ_i) for all j → mean → coupling term
        phase_diffs = np.sin(self.phases[np.newaxis, :] - self.phases[:, np.newaxis])
        coupling_term = (self.K / self.N) * phase_diffs.sum(axis=1)
        # Update
        noise = np.random.normal(0, self.phase_noise, self.N)
        self.phases = self.phases + self.dt * (self.frequencies + coupling_term + noise)
        # Wrap to [0, 2π]
        self.phases = self.phases % (2 * np.pi)

        # Compute order parameter and clusters
        state = self._compute_state()
        self.history.append(state)
        return state

    def run(self, n_steps: int = 100, record: bool = True) -> List[KuramotoState]:
        """Run the network for n_steps."""
        states = []
        for _ in range(n_steps):
            state = self.step()
            if record:
                states.append(state)
        return states

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def _compute_state(self) -> KuramotoState:
        """Compute order parameter, mean phase, and synchronized clusters."""
        # Order parameter r and mean phase ψ
        z = np.mean(np.exp(1j * self.phases))
        r = abs(z)
        psi = np.angle(z)

        # Find synchronized clusters (oscillators with similar phases)
        clusters = self._find_clusters()

        return KuramotoState(
            phases=self.phases.copy(),
            frequencies=self.frequencies.copy(),
            order_parameter=r,
            mean_phase=psi,
            clusters=clusters,
            time=len(self.history) * self.dt,
        )

    def _find_clusters(self, phase_threshold: float = 0.5) -> List[List[int]]:
        """Group oscillators with similar phases into clusters.

        Two oscillators are in the same cluster if their phase difference
        (modulo 2π) is less than phase_threshold radians.
        """
        n = len(self.phases)
        # Compute pairwise phase distances (on the circle)
        diff = np.abs(self.phases[np.newaxis, :] - self.phases[:, np.newaxis])
        diff = np.minimum(diff, 2 * np.pi - diff)
        # Union-find clustering
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if diff[i, j] < phase_threshold:
                    union(i, j)

        # Collect clusters
        cluster_map: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            cluster_map.setdefault(r, []).append(i)
        return list(cluster_map.values())

    # ------------------------------------------------------------------ #
    # Concept-level analysis
    # ------------------------------------------------------------------ #
    def concept_synchrony(self) -> Dict[Tuple[str, str], float]:
        """Compute pairwise synchrony between concepts.

        Returns dict {(concept_a, concept_b): synchrony_score}.
        Synchrony is the average phase coherence between oscillators of
        the two concepts.
        """
        # Group oscillator indices by concept
        concept_osc: Dict[str, List[int]] = {}
        for i, label in enumerate(self.concept_labels):
            if label:
                concept_osc.setdefault(label, []).append(i)

        result: Dict[Tuple[str, str], float] = {}
        concepts = list(concept_osc.keys())
        for a in range(len(concepts)):
            for b in range(a, len(concepts)):
                ca, cb = concepts[a], concepts[b]
                oscs_a = concept_osc[ca]
                oscs_b = concept_osc[cb]
                # Average phase coherence: |mean(exp(i*(θ_a - θ_b)))|
                phases_a = self.phases[oscs_a]
                phases_b = self.phases[oscs_b]
                # Pair all-vs-all and average
                diff = phases_a[np.newaxis, :] - phases_b[:, np.newaxis]
                coherence = abs(np.mean(np.exp(1j * diff)))
                result[(ca, cb)] = float(coherence)
        return result

    def bound_concepts(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Return pairs of concepts that are bound (synchronized > threshold)."""
        sync = self.concept_synchrony()
        bound = []
        for (a, b), score in sync.items():
            if a != b and score >= threshold:
                bound.append((a, b, score))
        bound.sort(key=lambda x: -x[2])
        return bound

    # ------------------------------------------------------------------ #
    # Convergence detection
    # ------------------------------------------------------------------ #
    def has_converged(self, window: int = 10, r_threshold: float = 0.5,
                      stability: float = 0.05) -> bool:
        """Check if the network has converged to a stable state.

        Convergence = order parameter r is above threshold AND stable
        (low variance over recent history).
        """
        if len(self.history) < window:
            return False
        recent = [s.order_parameter for s in self.history[-window:]]
        mean_r = np.mean(recent)
        std_r = np.std(recent)
        return mean_r >= r_threshold and std_r <= stability

    def comprehension_score(self) -> float:
        """A 0-1 score for how 'understood' the current state is.

        Combines:
          - order parameter (global synchrony)
          - number of stable clusters (multiple ideas bound)
          - convergence stability
        """
        if not self.history:
            return 0.0
        recent = self.history[-10:]
        mean_r = np.mean([s.order_parameter for s in recent])
        std_r = np.std([s.order_parameter for s in recent])
        n_clusters = np.mean([len(s.clusters) for s in recent])
        # Normalize: 1 cluster = bad (no diversity), 4-6 clusters = optimal
        cluster_score = 1.0 - abs(n_clusters - 5) / 10.0
        cluster_score = max(0, cluster_score)
        stability = max(0, 1 - std_r * 10)
        return float(mean_r * stability * cluster_score)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Kuramoto Network Test ===\n")

    net = KuramotoNetwork(n_oscillators=64, coupling=1.5, dt=0.05)

    # Add 4 concepts (HD vectors)
    concepts = [
        ("Paris",   HDVector.from_text_seed("Paris",   4096)),
        ("France",  HDVector.from_text_seed("France",  4096)),
        ("Tokyo",   HDVector.from_text_seed("Tokyo",   4096)),
        ("Japan",   HDVector.from_text_seed("Japan",   4096)),
    ]
    for label, vec in concepts:
        net.add_concept(label, vec, n_osc=8)
        print(f"  Added concept: {label}")

    print(f"\n  Network size: {net.N} oscillators")
    print(f"  Initial order parameter r = {net._compute_state().order_parameter:.4f}")

    # Run
    print("\n  Running 200 steps...")
    states = net.run(200)

    print(f"\n  Final state:")
    print(f"    Order parameter r = {states[-1].order_parameter:.4f}")
    print(f"    Mean phase ψ      = {states[-1].mean_phase:.4f}")
    print(f"    Number of clusters: {len(states[-1].clusters)}")

    # Show concept synchrony
    print("\n  Concept synchrony (after 200 steps):")
    sync = net.concept_synchrony()
    for (a, b), score in sorted(sync.items(), key=lambda x: -x[1])[:8]:
        print(f"    sync({a!r:8s}, {b!r:8s}) = {score:.4f}")

    # Bound concepts
    print("\n  Bound concepts (sync > 0.7):")
    bound = net.bound_concepts(threshold=0.7)
    if bound:
        for a, b, score in bound:
            print(f"    {a!r:8s} <-> {b!r:8s}  (score={score:.4f})")
    else:
        print("    (no bound concepts at this threshold)")

    print(f"\n  Comprehension score: {net.comprehension_score():.4f}")
    print(f"  Converged: {net.has_converged()}")

    # Show trajectory of order parameter
    print("\n  Order parameter trajectory (every 20 steps):")
    for i in range(0, len(states), 20):
        r = states[i].order_parameter
        bar = "#" * int(r * 40)
        print(f"    t={i:3d}: r={r:.3f} |{bar}")
