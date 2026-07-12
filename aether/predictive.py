"""
predictive.py — Predictive coding (Friston's Free Energy Principle).

BRAIN INSPIRATION
-----------------
Karl Friston's Free Energy Principle (FEP) proposes that the brain is a
prediction machine. At every level of processing, the brain generates
predictions about its inputs and updates its models to minimize
"prediction error" (free energy).

Key ideas:
  - HIERARCHICAL: higher levels predict lower levels; lower levels
    send prediction ERRORS up.
  - TOP-DOWN: predictions flow from abstract to concrete.
  - BOTTOM-UP: only prediction errors propagate up (efficient coding).
  - LEARNING: update generative models to reduce future errors.

Example: when you see a half-occluded dog, your visual cortex predicts
the rest of the dog. The actual input minus the prediction = prediction
error. The brain uses this error to refine its prediction.

AETHER'S USE
------------
We implement a predictive coding layer where:
  - The agent predicts the next perception (next HD vector)
  - The actual perception is compared to the prediction
  - The PREDICTION ERROR drives learning and attention
  - Large errors = surprise = conscious processing
  - Small errors = expected = unconscious processing

This is the OPPOSITE of pure feedforward — predictions gate what
even gets processed. Surprise is the currency of cognition.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from .hd import HDVector, DIM, bundle, _sign
from .memory import AssociativeMemory, SparseDistributedMemory


# ---------------------------------------------------------------------------
# Prediction error
# ---------------------------------------------------------------------------

@dataclass
class PredictionError:
    """A single prediction error signal."""
    predicted: HDVector
    actual: HDVector
    error_vector: HDVector        # = actual XOR predicted (binding, self-inverse)
    surprise: float               # scalar: 0.5 * (1 - similarity)
    timestamp: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        """Normalized magnitude of the error [0, 1]."""
        return self.surprise


def compute_prediction_error(predicted: HDVector, actual: HDVector) -> PredictionError:
    """Compute the prediction error between a predicted and actual HD vector.

    Error vector = XOR (binding) of predicted and actual. For bipolar
    vectors, this gives a vector that is +1 where they match and -1 where
    they differ. The "surprise" is the fraction of mismatched bits.
    """
    error_vec = predicted.bind(actual)  # +1 on match, -1 on mismatch
    similarity = predicted.similarity(actual)  # in [-1, 1]
    surprise = 0.5 * (1.0 - similarity)  # in [0, 1]
    return PredictionError(
        predicted=predicted,
        actual=actual,
        error_vector=error_vec,
        surprise=surprise,
    )


# ---------------------------------------------------------------------------
# Predictive model
# ---------------------------------------------------------------------------

class PredictiveModel:
    """A predictive coding model that learns to predict the next perception.

    The model maintains:
      - A "generative model" stored in an SDM (predict next from current)
      - A history of recent predictions and errors
      - A "surprise" score (running average of prediction errors)

    Learning is driven by prediction errors: when an error is large, the
    model updates its SDM more strongly.
    """

    def __init__(self, dim: int = DIM, n_locations: int = 3000, k: int = 15,
                 learning_rate: float = 1.0, surprise_threshold: float = 0.3):
        self.dim = dim
        self.sdm = SparseDistributedMemory(dim=dim, n_locations=n_locations, k=k)
        self.learning_rate = learning_rate
        self.surprise_threshold = surprise_threshold
        # State
        self.last_prediction: Optional[HDVector] = None
        self.last_actual: Optional[HDVector] = None
        self.last_error: Optional[PredictionError] = None
        self.history: List[PredictionError] = []
        self.surprise_history: List[float] = []
        self.cycle: int = 0

    def predict(self, current: HDVector) -> HDVector:
        """Predict the next perception given the current one."""
        retrieved = self.sdm.read(current)
        if retrieved is None:
            # No prediction available — return current (assume static)
            return current.copy()
        return retrieved

    def observe(self, actual: HDVector, current: Optional[HDVector] = None) -> PredictionError:
        """Observe an actual perception and compute prediction error.

        If `current` is provided, use it as the address; otherwise use
        the last actual perception.

        Learning: write (current -> actual) to the SDM, weighted by the
        prediction error (more surprise = stronger learning).
        """
        self.cycle += 1
        if current is None:
            current = self.last_actual if self.last_actual is not None else actual.copy()

        # Predict
        prediction = self.predict(current)
        self.last_prediction = prediction

        # Compute error
        error = compute_prediction_error(prediction, actual)
        error.timestamp = self.cycle
        self.last_actual = actual
        self.last_error = error
        self.history.append(error)
        self.surprise_history.append(error.surprise)
        # Keep history bounded
        if len(self.history) > 100:
            self.history = self.history[-100:]
            self.surprise_history = self.surprise_history[-100:]

        # Learning: write the association, weighted by error magnitude
        # Large errors => stronger write (more learning)
        n_writes = 1 + int(error.surprise * 5 * self.learning_rate)
        for _ in range(n_writes):
            self.sdm.write(current, actual)

        return error

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def mean_surprise(self, window: int = 20) -> float:
        """Average surprise over recent cycles."""
        if not self.surprise_history:
            return 0.0
        recent = self.surprise_history[-window:]
        return float(np.mean(recent))

    def surprise_trend(self, window: int = 20) -> float:
        """Trend of surprise (negative = improving, positive = worsening)."""
        if len(self.surprise_history) < 2 * window:
            return 0.0
        old = np.mean(self.surprise_history[-2 * window:-window])
        new = np.mean(self.surprise_history[-window:])
        return float(new - old)

    def is_surprised(self) -> bool:
        """Is the model currently surprised (last error > threshold)?"""
        return self.last_error is not None and self.last_error.surprise >= self.surprise_threshold

    def stats(self) -> Dict[str, float]:
        return {
            "cycle": self.cycle,
            "mean_surprise": self.mean_surprise(),
            "surprise_trend": self.surprise_trend(),
            "last_surprise": self.last_error.surprise if self.last_error else 0.0,
            "is_surprised": self.is_surprised(),
            "sdm_writes": int(self.sdm.write_count.sum()),
        }


# ---------------------------------------------------------------------------
# Sequence predictor (predict next in a sequence of HD vectors)
# ---------------------------------------------------------------------------

class SequencePredictor:
    """Predict the next item in a sequence of HD vectors.

    Uses the SDM to store (current_item -> next_item) associations.
    Optionally uses the last N items as context (n-gram style).
    """

    def __init__(self, dim: int = DIM, context_size: int = 1, n_locations: int = 3000, k: int = 15):
        self.dim = dim
        self.context_size = context_size
        self.sdm = SparseDistributedMemory(dim=dim, n_locations=n_locations, k=k)
        self.context: List[HDVector] = []
        self.last_prediction: Optional[HDVector] = None
        self.last_error: Optional[PredictionError] = None
        self.surprise_history: List[float] = []

    def _encode_context(self) -> HDVector:
        """Encode the current context (last N items) as a single HD vector."""
        if not self.context:
            return HDVector.random(self.dim)
        # Bundle the last context_size items with positional permutation
        vecs = []
        for i, v in enumerate(self.context[-self.context_size:]):
            vecs.append(HDVector(data=np.roll(v.data, i), dim=self.dim))
        return bundle(vecs)

    def predict_next(self) -> Optional[HDVector]:
        """Predict the next item given the current context."""
        if not self.context:
            return None
        ctx = self._encode_context()
        retrieved = self.sdm.read(ctx)
        self.last_prediction = retrieved
        return retrieved

    def observe(self, item: HDVector) -> PredictionError:
        """Observe a new item in the sequence and learn from it."""
        # Predict first (if context exists)
        prediction = self.predict_next()

        # Compute error
        if prediction is not None:
            error = compute_prediction_error(prediction, item)
        else:
            error = PredictionError(
                predicted=item.copy(),
                actual=item.copy(),
                error_vector=HDVector.zero(self.dim),
                surprise=0.5,  # no prediction = max uncertainty
            )

        # Learn: write (context -> item) to the SDM
        if self.context:
            ctx = self._encode_context()
            self.sdm.write(ctx, item)

        # Update context
        self.context.append(item)
        if len(self.context) > self.context_size + 5:
            self.context = self.context[-(self.context_size + 5):]

        self.last_error = error
        self.surprise_history.append(error.surprise)
        if len(self.surprise_history) > 100:
            self.surprise_history = self.surprise_history[-100:]

        return error

    def reset(self) -> None:
        self.context.clear()
        self.last_prediction = None
        self.last_error = None
        self.surprise_history.clear()

    def stats(self) -> Dict[str, float]:
        return {
            "context_size": len(self.context),
            "mean_surprise": float(np.mean(self.surprise_history)) if self.surprise_history else 0.0,
            "sdm_writes": int(self.sdm.write_count.sum()),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Predictive Coding Test ===\n")

    model = PredictiveModel(dim=4096)

    # Train on a sequence of HD vectors (with some pattern)
    rng = np.random.default_rng(42)
    base_vec = HDVector.from_text_seed("base", 4096)
    pattern = [base_vec.copy() for _ in range(5)]
    # Add small variations to make it learnable
    for i, v in enumerate(pattern):
        noise_vec = HDVector.from_text_seed(f"noise_{i}", 4096)
        pattern[i] = bundle([v, noise_vec], weights=[0.8, 0.2])

    print("  Training on a 5-item pattern (repeated 3 times)...")
    for trial in range(3):
        for v in pattern:
            error = model.observe(v)
        print(f"    Trial {trial+1}: mean_surprise={model.mean_surprise():.3f}")

    print(f"\n  Final stats: {model.stats()}")

    # Test with a "surprising" input
    print("\n  Testing with a surprising input...")
    surprise_vec = HDVector.from_text_seed("SURPRISE!", 4096)
    error = model.observe(surprise_vec)
    print(f"    Surprise: {error.surprise:.3f}")
    print(f"    Is surprised: {model.is_surprised()}")

    # Test with an expected input
    print("\n  Testing with an expected input...")
    error = model.observe(pattern[0])
    print(f"    Surprise: {error.surprise:.3f}")
    print(f"    Is surprised: {model.is_surprised()}")

    print("\n=== Sequence Predictor Test ===\n")
    sp = SequencePredictor(dim=4096, context_size=2)
    # Train on a sequence A -> B -> C -> A -> B -> C -> ...
    seq = [HDVector.from_text_seed(f"item_{i % 3}", 4096) for i in range(15)]
    print("  Training on sequence A->B->C->A->B->C... (15 items)")
    for i, item in enumerate(seq):
        error = sp.observe(item)
        if i >= 2:
            print(f"    item {i+1}: surprise={error.surprise:.3f}")

    print(f"\n  Final stats: {sp.stats()}")
