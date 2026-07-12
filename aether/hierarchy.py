"""
hierarchy.py — Hierarchical predictive coding cortex.

BRAIN INSPIRATION
-----------------
The cortex is a 6-layered hierarchy. Lower levels process sensory
details (pixels, phonemes); higher levels process abstractions
(objects, words, concepts, plans). Information flows:
  - BOTTOM-UP: prediction errors (what didn't match the prediction)
  - TOP-DOWN: predictions (what should be perceived next)

Friston's predictive coding formalizes this: each level predicts the
level below, and only the RESIDUAL (prediction error) propagates up.

AETHER'S USE
------------
We implement a 4-level hierarchy:

  Level 0: SENSORY   — raw input HD vector (text/image/audio encoding)
  Level 1: FEATURES  — local patterns (n-grams, edges, phonemes)
  Level 2: CONCEPTS  — bound features (words, objects, relations)
  Level 3: ABSTRACT  — abstract concepts (sentences, scenes, plans)

Each level has its own predictive model. Top-down predictions flow
from L3 to L0; bottom-up errors flow from L0 to L3.

The hierarchy LEARNS by minimizing prediction error at every level.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from .hd import HDVector, DIM, bundle, _sign, ngram_encode
from .predictive import PredictiveModel, PredictionError, compute_prediction_error
from .encoder import TextEncoder


# ---------------------------------------------------------------------------
# Hierarchy level
# ---------------------------------------------------------------------------

@dataclass
class LevelState:
    """State of one hierarchical level at a single cycle."""
    level: int
    name: str
    prediction: Optional[HDVector]
    actual: Optional[HDVector]
    error: Optional[PredictionError]
    surprise: float


class HierarchyLevel:
    """One level of the predictive coding hierarchy.

    Each level:
      - Has its own PredictiveModel (predicts next perception at this level)
      - Receives bottom-up errors from the level below
      - Sends top-down predictions to the level below
      - Encodes its input via a level-specific encoder
    """

    def __init__(self, level: int, name: str, encoder: TextEncoder,
                 n_locations: int = 2000, k: int = 15):
        self.level = level
        self.name = name
        self.encoder = encoder
        self.predictor = PredictiveModel(
            dim=encoder.dim, n_locations=n_locations, k=k,
            learning_rate=1.0, surprise_threshold=0.3,
        )
        self.last_state: Optional[LevelState] = None
        # Bottom-up error from the level below (set externally)
        self.bottom_up_error: Optional[PredictionError] = None
        # Top-down prediction from the level above (set externally)
        self.top_down_prediction: Optional[HDVector] = None

    def encode_input(self, raw_input: Any) -> HDVector:
        """Encode raw input at this level."""
        # Level-specific encoding
        if self.level == 0:
            # Sensory: encode the full text/image/audio
            if isinstance(raw_input, str):
                return self.encoder.encode_text(raw_input)
            elif isinstance(raw_input, HDVector):
                return raw_input.copy()
            else:
                return HDVector.random(self.encoder.dim)
        elif self.level == 1:
            # Features: encode as n-gram superposition (more local)
            if isinstance(raw_input, str):
                tokens = self.encoder.encode_tokens(raw_input)
                if not tokens:
                    return HDVector.zero(self.encoder.dim)
                # Use shorter n-grams for feature-level encoding
                return ngram_encode(tokens, n=2)
            elif isinstance(raw_input, HDVector):
                return raw_input.copy()
            else:
                return HDVector.random(self.encoder.dim)
        elif self.level == 2:
            # Concepts: bundle of token vectors (word-level)
            if isinstance(raw_input, str):
                tokens = self.encoder.encode_tokens(raw_input)
                if not tokens:
                    return HDVector.zero(self.encoder.dim)
                return bundle(tokens)
            elif isinstance(raw_input, HDVector):
                return raw_input.copy()
            else:
                return HDVector.random(self.encoder.dim)
        else:
            # Abstract: encode the full text again (top-level abstraction)
            if isinstance(raw_input, str):
                return self.encoder.encode_text(raw_input)
            elif isinstance(raw_input, HDVector):
                return raw_input.copy()
            else:
                return HDVector.random(self.encoder.dim)

    def process(self, raw_input: Any) -> LevelState:
        """Process input at this level: encode, predict, observe, compute error."""
        # Encode the input
        actual = self.encode_input(raw_input)

        # Get prediction (from top-down if available, else from local predictor)
        if self.top_down_prediction is not None:
            prediction = self.top_down_prediction
        else:
            prediction = self.predictor.predict(actual) if self.predictor.last_actual else None

        # Observe and learn
        error = self.predictor.observe(actual)

        # If we had a top-down prediction, use that for the error too
        if self.top_down_prediction is not None:
            error = compute_prediction_error(self.top_down_prediction, actual)

        state = LevelState(
            level=self.level,
            name=self.name,
            prediction=prediction,
            actual=actual,
            error=error,
            surprise=error.surprise,
        )
        self.last_state = state
        return state


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class PredictiveHierarchy:
    """A multi-level predictive coding hierarchy.

    Information flows:
      - BOTTOM-UP: raw_input -> L0 -> L1 -> L2 -> L3 (each level encodes)
      - TOP-DOWN: L3 predicts L2 predicts L1 predicts L0 (predictions flow down)
      - BOTTOM-UP ERRORS: L0 error -> L1 error -> L2 error -> L3 error

    Each level runs its own PredictiveModel and learns to predict the
    next perception at its level of abstraction.
    """

    def __init__(self, encoder: TextEncoder, n_levels: int = 4):
        self.encoder = encoder
        self.n_levels = n_levels
        # Create levels
        names = ["sensory", "features", "concepts", "abstract"][:n_levels]
        self.levels: List[HierarchyLevel] = [
            HierarchyLevel(i, names[i], encoder) for i in range(n_levels)
        ]
        self.cycle: int = 0
        self.global_surprise_history: List[float] = []

    def process(self, raw_input: Any) -> List[LevelState]:
        """Process input through the full hierarchy.

        Returns the list of LevelState (one per level), bottom-up.
        """
        self.cycle += 1

        # BOTTOM-UP: encode at each level
        states: List[LevelState] = []
        for level in self.levels:
            state = level.process(raw_input)
            states.append(state)

        # TOP-DOWN: predictions flow from higher to lower levels
        # The prediction at level L is the actual at level L+1 (or its own prediction if L is top)
        for i in range(self.n_levels - 2, -1, -1):
            # Top-down prediction: higher level's actual becomes the prediction for lower level
            self.levels[i].top_down_prediction = states[i + 1].actual

        # Compute global surprise (average across levels)
        global_surprise = float(np.mean([s.surprise for s in states]))
        self.global_surprise_history.append(global_surprise)
        if len(self.global_surprise_history) > 100:
            self.global_surprise_history = self.global_surprise_history[-100:]

        return states

    def mean_surprise(self, window: int = 20) -> float:
        if not self.global_surprise_history:
            return 0.0
        recent = self.global_surprise_history[-window:]
        return float(np.mean(recent))

    def surprise_trend(self, window: int = 20) -> float:
        if len(self.global_surprise_history) < 2 * window:
            return 0.0
        old = np.mean(self.global_surprise_history[-2 * window:-window])
        new = np.mean(self.global_surprise_history[-window:])
        return float(new - old)

    def is_surprised(self) -> bool:
        """Is the hierarchy currently in a state of surprise?"""
        if not self.global_surprise_history:
            return False
        return self.global_surprise_history[-1] >= 0.4

    def stats(self) -> Dict[str, any]:
        return {
            "cycle": self.cycle,
            "n_levels": self.n_levels,
            "mean_surprise": self.mean_surprise(),
            "surprise_trend": self.surprise_trend(),
            "is_surprised": self.is_surprised(),
            "level_names": [l.name for l in self.levels],
            "level_surprises": [l.last_state.surprise if l.last_state else 0.0 for l in self.levels],
        }


# Need Any import
from typing import Any


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Predictive Hierarchy Test ===\n")

    from .memory import AssociativeMemory
    from .encoder import TextEncoder

    assoc = AssociativeMemory(dim=4096)
    encoder = TextEncoder(dim=4096)
    encoder.assoc = assoc

    # Pre-seed some vocabulary
    for word in ["paris", "france", "tokyo", "japan", "capital", "the", "of", "is"]:
        assoc.get_symbol(word)

    hierarchy = PredictiveHierarchy(encoder, n_levels=4)
    print(f"  Created {hierarchy.n_levels}-level hierarchy:")
    for level in hierarchy.levels:
        print(f"    L{level.level}: {level.name}")
    print()

    # Process a sequence of inputs (some repeated, some novel)
    inputs = [
        "Paris is the capital of France",
        "Paris is the capital of France",  # repeat
        "Paris is the capital of France",  # repeat
        "Tokyo is the capital of Japan",   # novel
        "Tokyo is the capital of Japan",   # repeat
        "Tokyo is the capital of Japan",   # repeat
        "Water is a liquid",                # very novel
        "Water is a liquid",                # repeat
    ]

    print("  Processing sequence:")
    for i, inp in enumerate(inputs):
        states = hierarchy.process(inp)
        global_surprise = np.mean([s.surprise for s in states])
        print(f"    cycle {i+1}: {inp!r:50s} surprise={global_surprise:.3f}")
        print(f"             level surprises: {[f'{s.surprise:.3f}' for s in states]}")

    print(f"\n  Final stats: {hierarchy.stats()}")
    print(f"\n  Mean surprise (last 5): {hierarchy.mean_surprise(5):.3f}")
    print(f"  Surprise trend: {hierarchy.surprise_trend():.3f}  (negative = improving)")
    print(f"  Is currently surprised: {hierarchy.is_surprised()}")
