"""
attractor.py — Attractor networks for cognition.

BRAIN INSPIRATION
-----------------
In the brain, persistent activity states in recurrent neural networks
form "attractors" — stable patterns of activity that the network relaxes
into. These attractors are thought to underlie:
  - Working memory (persistent firing in prefrontal cortex)
  - Decision making (winner-take-all attractor basins)
  - Pattern completion (Hopfield-style auto-associative memory)
  - Spatial cognition (continuous attractor networks in hippocampus)

AETHER'S USE
------------
We build TWO kinds of attractor networks on top of HD vectors:

  1. DISCRETE attractor (Hopfield-style):
     - Each stored memory is a stable fixed point.
     - Given a noisy/partial input, the network relaxes to the nearest stored memory.
     - Energy: E(v) = -Σ w_ij v_i v_j
     - Update: v_i = sign(Σ w_ij v_j)
     - We use the SDM's counter array as the weight matrix W.

  2. CONTINUOUS attractor:
     - For spatial / continuous domains (place cells, head direction).
     - Activity bump that can move smoothly.
     - We implement a 1D ring attractor and a 2D sheet attractor.

When the cognitive loop has a "thought" (an HD vector), it is fed into
the attractor network. The network either:
  - Snaps to a stored memory (discrete attractor → recognition)
  - Smoothly drifts toward a related memory (continuous attractor → analogical reasoning)

This is what gives AETHER "real comprehension" — inputs are not just
matched, they are STABILIZED into coherent attractor states.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from dataclasses import dataclass, field

from .hd import HDVector, DIM, bundle, _sign
from .memory import AssociativeMemory, SparseDistributedMemory


# ---------------------------------------------------------------------------
# Discrete attractor network (Hopfield on HD)
# ---------------------------------------------------------------------------

@dataclass
class AttractorState:
    """State of the attractor network during relaxation."""
    current_vector: HDVector
    energy: float
    iteration: int
    converged: bool
    settled_at: Optional[str] = None  # label of memory it settled on


class DiscreteAttractorNetwork:
    """Hopfield-style discrete attractor network for HD vectors.

    Stored memories are stable fixed points. Given a noisy input, the
    network iteratively relaxes to the nearest fixed point.

    We use the SDM's counter array as the weight matrix:
      - Write each memory with itself as both address and data
        (auto-associative: addr = data = memory).
      - Read at the current state's address; the result is the next state.
      - Iterate until convergence.
    """

    def __init__(self, dim: int = DIM, n_locations: int = 3000, k: int = 15,
                 max_iterations: int = 20, convergence_threshold: float = 0.95):
        self.dim = dim
        self.sdm = SparseDistributedMemory(dim=dim, n_locations=n_locations, k=k)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        # Labeled memories (for reporting which one we settled on)
        self.labeled_memories: Dict[str, HDVector] = {}

    def store(self, vector: HDVector, label: Optional[str] = None, n_reinforce: int = 5) -> None:
        """Store a vector as a stable fixed point (auto-associative).

        The vector is written n_reinforce times (with slight noise) to
        strengthen its basin of attraction — multiple writes accumulate
        in the SDM counters, making the fixed point more stable.
        """
        # Auto-associative: addr = data = vector
        # Write multiple times to strengthen the attractor basin
        for _ in range(n_reinforce):
            self.sdm.write(vector, vector)
        if label:
            self.labeled_memories[label] = vector.copy()

    def store_labeled(self, label: str, vector: HDVector) -> None:
        self.store(vector, label)

    def relax(self, input_vector: HDVector) -> AttractorState:
        """Iteratively relax the input toward a stored attractor.

        Returns the final AttractorState.
        """
        current = input_vector.copy()
        prev = None
        for it in range(self.max_iterations):
            # Read from SDM at the current address
            retrieved = self.sdm.read(current)
            if retrieved is None:
                # No nearby stored memory — return as-is
                return AttractorState(
                    current_vector=current,
                    energy=self._energy(current),
                    iteration=it,
                    converged=False,
                )
            # Check convergence
            if prev is not None:
                sim = current.similarity(retrieved)
                if sim >= self.convergence_threshold:
                    # Converged — find the closest labeled memory
                    settled = self._closest_label(retrieved)
                    return AttractorState(
                        current_vector=retrieved,
                        energy=self._energy(retrieved),
                        iteration=it,
                        converged=True,
                        settled_at=settled,
                    )
            prev = current
            current = retrieved
        # Did not converge within max_iterations
        settled = self._closest_label(current)
        return AttractorState(
            current_vector=current,
            energy=self._energy(current),
            iteration=self.max_iterations,
            converged=False,
            settled_at=settled,
        )

    def _energy(self, vec: HDVector) -> float:
        """Hopfield energy: E = -<v, W*v> = -<v, retrieved_at_v>."""
        retrieved = self.sdm.read(vec)
        if retrieved is None:
            return 0.0
        return -vec.similarity(retrieved)

    def _closest_label(self, vec: HDVector) -> Optional[str]:
        """Find the labeled memory most similar to vec."""
        if not self.labeled_memories:
            return None
        best_label, best_sim = None, -1.0
        for label, mem_vec in self.labeled_memories.items():
            sim = vec.similarity(mem_vec)
            if sim > best_sim:
                best_sim, best_label = sim, label
        if best_sim >= 0.5:
            return best_label
        return None

    def stats(self) -> Dict[str, int]:
        return {
            "dim": self.dim,
            "n_memories": len(self.labeled_memories),
            "n_labeled": len(self.labeled_memories),
            "sdm_writes": int(self.sdm.write_count.sum()),
        }


# ---------------------------------------------------------------------------
# Continuous attractor network (1D ring + 2D sheet)
# ---------------------------------------------------------------------------

class RingAttractor:
    """1D ring attractor for circular variables (head direction, time of day).

    The activity bump is a Gaussian centered at angle θ. The bump can
    be shifted by external input (e.g., from an HD vector encoding a
    direction). The bump is stable — small perturbations relax back.
    """

    def __init__(self, n_units: int = 64, sigma: float = 4.0,
                 recurrent_strength: float = 1.0):
        self.N = n_units
        self.sigma = sigma
        self.w = recurrent_strength
        # Pre-compute recurrent weight matrix (Gaussian on the ring)
        self.W = np.zeros((n_units, n_units), dtype=np.float64)
        for i in range(n_units):
            for j in range(n_units):
                # Distance on the ring (0 to N/2)
                d = min(abs(i - j), n_units - abs(i - j))
                self.W[i, j] = np.exp(-0.5 * (d / sigma) ** 2)
        # Activity bump (initialized at angle 0)
        self.activity = self._bump_at(0.0)

    def _bump_at(self, angle: float) -> np.ndarray:
        """Return a Gaussian bump centered at `angle` (in radians, [0, 2π))."""
        angle = angle % (2 * np.pi)
        idx = int(angle / (2 * np.pi) * self.N)
        bump = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            d = min(abs(i - idx), self.N - abs(i - idx))
            bump[i] = np.exp(-0.5 * (d / self.sigma) ** 2)
        return bump

    def step(self, external_input: Optional[np.ndarray] = None, dt: float = 0.3) -> np.ndarray:
        """One relaxation step. External input shifts the bump.

        With persistent external input (not dropped), the bump will
        migrate toward the external input's center.
        """
        # Recurrent dynamics + external input (persistent during relax)
        recurrent = self.w * self.W @ self.activity
        if external_input is not None:
            recurrent = recurrent + 2.0 * external_input  # strong external pull
        # Nonlinearity (ReLU-like)
        recurrent = np.maximum(recurrent, 0)
        # Normalize to keep total activity bounded
        total = recurrent.sum()
        if total > 0:
            recurrent = recurrent / total * self.N
        self.activity = recurrent
        return self.activity

    def relax(self, external_input: Optional[np.ndarray] = None,
              n_steps: int = 50, dt: float = 0.3) -> np.ndarray:
        """Run multiple relaxation steps (external input PERSISTS)."""
        for _ in range(n_steps):
            self.step(external_input, dt)
        return self.activity

    def current_angle(self) -> float:
        """Decode the current bump center to an angle in [0, 2π)."""
        # Use circular mean
        angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        z = np.sum(self.activity * np.exp(1j * angles))
        return float(np.angle(z)) % (2 * np.pi)

    def reset(self, angle: float = 0.0) -> None:
        self.activity = self._bump_at(angle)


class SheetAttractor:
    """2D sheet attractor for spatial cognition (place cells, grid cells).

    A 2D Gaussian bump that can be moved around a 2D sheet. Used for
    spatial reasoning and analogical distance.
    """

    def __init__(self, width: int = 16, height: int = 16, sigma: float = 2.0):
        self.W = width
        self.H = height
        self.sigma = sigma
        # Activity map (2D)
        self.activity = self._bump_at(width / 2, height / 2)
        # Pre-compute 2D Gaussian recurrent weights
        self.W_mat = np.zeros((width * height, width * height), dtype=np.float64)
        for i in range(width * height):
            xi, yi = i % width, i // width
            for j in range(width * height):
                xj, yj = j % width, j // width
                d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                self.W_mat[i, j] = np.exp(-0.5 * d2 / sigma ** 2)

    def _bump_at(self, x: float, y: float) -> np.ndarray:
        bump = np.zeros(self.W * self.H, dtype=np.float64)
        for i in range(self.W):
            for j in range(self.H):
                d2 = (i - x) ** 2 + (j - y) ** 2
                bump[i + j * self.W] = np.exp(-0.5 * d2 / self.sigma ** 2)
        return bump

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        recurrent = self.W_mat @ self.activity
        if external_input is not None:
            recurrent = recurrent + 2.0 * external_input  # strong external pull
        recurrent = np.maximum(recurrent, 0)
        total = recurrent.sum()
        if total > 0:
            recurrent = recurrent / total * (self.W * self.H)
        self.activity = recurrent
        return self.activity

    def relax(self, external_input: Optional[np.ndarray] = None,
              n_steps: int = 50) -> np.ndarray:
        """Run multiple relaxation steps (external input PERSISTS)."""
        for _ in range(n_steps):
            self.step(external_input)
        return self.activity

    def current_position(self) -> Tuple[float, float]:
        """Decode the bump center to (x, y)."""
        total = self.activity.sum()
        if total == 0:
            return (self.W / 2, self.H / 2)
        # Weighted mean
        xs = np.arange(self.W)
        ys = np.arange(self.H)
        activity_2d = self.activity.reshape(self.H, self.W)
        x = np.sum(xs * activity_2d.sum(axis=0)) / total
        y = np.sum(ys * activity_2d.sum(axis=1)) / total
        return (float(x), float(y))

    def reset(self, x: float = None, y: float = None) -> None:
        if x is None:
            x = self.W / 2
        if y is None:
            y = self.H / 2
        self.activity = self._bump_at(x, y)


# ---------------------------------------------------------------------------
# Pattern completion (attractor-based)
# ---------------------------------------------------------------------------

class PatternCompleter:
    """Use a discrete attractor network to complete partial/noisy patterns.

    Given a partial HD vector (some bits zeroed out, or noise added),
    run the attractor network to complete it to the nearest stored memory.
    """

    def __init__(self, attractor: DiscreteAttractorNetwork):
        self.attractor = attractor

    def complete(self, partial_vec: HDVector, noise_level: float = 0.0) -> Tuple[HDVector, AttractorState]:
        """Complete a partial/noisy HD vector.

        Args:
            partial_vec: the input vector (may have zeros or noise)
            noise_level: fraction of bits to flip (0 = no noise)

        Returns:
            (completed_vector, attractor_state)
        """
        # Optionally add noise
        if noise_level > 0:
            noisy_data = partial_vec.data.copy()
            n_flip = int(noise_level * len(noisy_data))
            flip_indices = np.random.choice(len(noisy_data), n_flip, replace=False)
            noisy_data[flip_indices] *= -1
            input_vec = HDVector(data=noisy_data, dim=partial_vec.dim)
        else:
            input_vec = partial_vec.copy()

        state = self.attractor.relax(input_vec)
        return state.current_vector, state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Discrete Attractor Network Test ===\n")

    net = DiscreteAttractorNetwork(dim=4096, n_locations=3000, k=15)

    # Store 5 memories
    memories = [
        ("Paris",  HDVector.from_text_seed("Paris",  4096)),
        ("Tokyo",  HDVector.from_text_seed("Tokyo",  4096)),
        ("Python", HDVector.from_text_seed("Python", 4096)),
        ("Dog",    HDVector.from_text_seed("Dog",    4096)),
        ("Red",    HDVector.from_text_seed("Red",    4096)),
    ]
    for label, vec in memories:
        net.store_labeled(label, vec)
    print(f"  Stored {len(memories)} memories.")

    # Test pattern completion with noise
    print("\n  Pattern completion test (20% noise):")
    completer = PatternCompleter(net)
    for label, vec in memories:
        completed, state = completer.complete(vec, noise_level=0.2)
        sim = vec.similarity(completed)
        settled = state.settled_at or "(none)"
        print(f"    {label!r:8s}: settled={settled!r:12s} sim_to_original={sim:.3f} iters={state.iteration}")

    print("\n  Pattern completion test (10% noise):")
    for label, vec in memories:
        completed, state = completer.complete(vec, noise_level=0.1)
        sim = vec.similarity(completed)
        settled = state.settled_at or "(none)"
        print(f"    {label!r:8s}: settled={settled!r:12s} sim_to_original={sim:.3f} iters={state.iteration}")

    # Test with a query that's not a stored memory
    print("\n  Relaxation of unfamiliar input:")
    unfamiliar = HDVector.from_text_seed("R2D2", 4096)
    state = net.relax(unfamiliar)
    print(f"    Input: 'R2D2' -> settled_at={state.settled_at!r} converged={state.converged}")

    print("\n=== Ring Attractor Test ===\n")
    ring = RingAttractor(n_units=64, sigma=4.0)
    print(f"  Initial angle: {np.degrees(ring.current_angle()):.1f}°")

    # Apply external input at 90°
    target_angle = np.pi / 2  # 90°
    print(f"  Applying input at 90°...")
    ring.reset(0.0)
    external = ring._bump_at(target_angle) * 2.0
    ring.relax(external_input=external, n_steps=30)
    print(f"  After relaxation: {np.degrees(ring.current_angle()):.1f}°")

    print("\n=== Sheet Attractor Test ===\n")
    sheet = SheetAttractor(width=16, height=16, sigma=2.0)
    print(f"  Initial position: {sheet.current_position()}")

    # Move bump to (10, 12)
    print(f"  Applying input at (10, 12)...")
    sheet.reset(8, 8)
    external_2d = sheet._bump_at(10, 12) * 2.0
    sheet.relax(external_input=external_2d, n_steps=30)
    print(f"  After relaxation: {sheet.current_position()}")
