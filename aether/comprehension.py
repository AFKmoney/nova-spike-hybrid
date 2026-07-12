"""
comprehension.py — Real comprehension integrator.

WHAT IS REAL COMPREHENSION?
---------------------------
In a transformer LLM, "comprehension" is just pattern-matching to
training data. There's no test of whether the model actually understands.

In a brain, comprehension is a DYNAMIC STATE involving:
  1. STABLE ATTRACTOR: the neural activity settles into a stable pattern
     (the "thought" is held in working memory by persistent firing).
  2. PREDICTION MATCH: the brain predicts what should happen next, and
     the prediction matches reality (low prediction error).
  3. GLOBAL BROADCAST: the thought wins the global workspace competition
     and is broadcast to all specialist modules (it's conscious).
  4. OSCILLATORY BINDING: the relevant features are bound by synchronous
     oscillations (Kuramoto-style phase locking).
  5. LOW SURPRISE: the predictive hierarchy reports low surprise at all
     levels (the input is well-explained by the internal model).

AETHER'S USE
------------
The ComprehensionIntegrator takes inputs from:
  - AttractorNetwork (stable attractor state?)
  - PredictiveModel (low prediction error?)
  - GlobalWorkspace (broadcast happened?)
  - KuramotoNetwork (oscillators synchronized?)
  - PredictiveHierarchy (low global surprise?)

It outputs a COMPREHENSION SCORE in [0, 1] that measures how well the
agent currently "understands" its input. This is the REAL difference
from a transformer — we have a measurable notion of comprehension.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from .hd import HDVector, DIM


# ---------------------------------------------------------------------------
# Comprehension state
# ---------------------------------------------------------------------------

@dataclass
class ComprehensionState:
    """Snapshot of all the indicators of comprehension at one moment."""
    attractor_stability: float    # 0-1: how stable is the current attractor?
    prediction_match: float       # 0-1: how well does prediction match reality?
    broadcast_active: float       # 0-1: is the global workspace broadcasting?
    oscillator_sync: float        # 0-1: how synchronized are the oscillators?
    hierarchy_calm: float         # 0-1: how low is the hierarchy surprise?
    neuromodulator_balance: float # 0-1: how balanced are the modulators?

    # Aggregate scores
    comprehension_score: float = 0.0   # weighted combination
    confidence: float = 0.0            # confidence in the comprehension
    is_comprehending: bool = False     # boolean: above threshold?

    # Diagnostic
    cycle: int = 0
    notes: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {
            "attractor_stability": self.attractor_stability,
            "prediction_match": self.prediction_match,
            "broadcast_active": self.broadcast_active,
            "oscillator_sync": self.oscillator_sync,
            "hierarchy_calm": self.hierarchy_calm,
            "neuromodulator_balance": self.neuromodulator_balance,
            "comprehension_score": self.comprehension_score,
            "confidence": self.confidence,
            "is_comprehending": self.is_comprehending,
            "cycle": self.cycle,
        }


# ---------------------------------------------------------------------------
# Comprehension integrator
# ---------------------------------------------------------------------------

class ComprehensionIntegrator:
    """Integrates signals from all cognitive subsystems into a
    comprehension score.

    Usage:
        integrator = ComprehensionIntegrator()
        integrator.connect_attractor(attractor_network)
        integrator.connect_predictive(predictive_model)
        integrator.connect_workspace(global_workspace)
        integrator.connect_kuramoto(kuramoto_network)
        integrator.connect_hierarchy(predictive_hierarchy)
        integrator.connect_neuromodulators(neuromodulator_system)

        state = integrator.assess()
        if state.is_comprehending:
            print("AETHER understands!")
    """

    def __init__(
        self,
        comprehension_threshold: float = 0.55,
        confidence_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.comprehension_threshold = comprehension_threshold
        self.confidence_threshold = confidence_threshold
        # Weights for each indicator (must sum to 1.0)
        self.weights = weights or {
            "attractor_stability": 0.20,
            "prediction_match":    0.25,
            "broadcast_active":    0.20,
            "oscillator_sync":     0.10,
            "hierarchy_calm":      0.15,
            "neuromodulator_balance": 0.10,
        }

        # Connected subsystems (set via connect_* methods)
        self.attractor = None
        self.predictive = None
        self.workspace = None
        self.kuramoto = None
        self.hierarchy = None
        self.neuromodulators = None

        # History
        self.history: List[ComprehensionState] = []
        self.cycle: int = 0

    # ------------------------------------------------------------------ #
    # Connection methods
    # ------------------------------------------------------------------ #
    def connect_attractor(self, attractor) -> None:
        self.attractor = attractor

    def connect_predictive(self, predictive) -> None:
        self.predictive = predictive

    def connect_workspace(self, workspace) -> None:
        self.workspace = workspace

    def connect_kuramoto(self, kuramoto) -> None:
        self.kuramoto = kuramoto

    def connect_hierarchy(self, hierarchy) -> None:
        self.hierarchy = hierarchy

    def connect_neuromodulators(self, nm) -> None:
        self.neuromodulators = nm

    # ------------------------------------------------------------------ #
    # Assessment
    # ------------------------------------------------------------------ #
    def assess(self) -> ComprehensionState:
        """Compute the current comprehension state from all subsystems."""
        self.cycle += 1

        # 1. Attractor stability: did the network converge?
        attractor_stability = self._assess_attractor()

        # 2. Prediction match: 1 - surprise
        prediction_match = self._assess_prediction_match()

        # 3. Broadcast active: is the global workspace ignited?
        broadcast_active = self._assess_broadcast()

        # 4. Oscillator sync: order parameter of the Kuramoto network
        oscillator_sync = self._assess_oscillator_sync()

        # 5. Hierarchy calm: 1 - global surprise
        hierarchy_calm = self._assess_hierarchy_calm()

        # 6. Neuromodulator balance: how close to balanced (0.5)?
        neuromodulator_balance = self._assess_neuromodulator_balance()

        # Aggregate comprehension score
        score = (
            self.weights["attractor_stability"] * attractor_stability +
            self.weights["prediction_match"] * prediction_match +
            self.weights["broadcast_active"] * broadcast_active +
            self.weights["oscillator_sync"] * oscillator_sync +
            self.weights["hierarchy_calm"] * hierarchy_calm +
            self.weights["neuromodulator_balance"] * neuromodulator_balance
        )

        # Confidence: how consistent are the indicators?
        indicators = [attractor_stability, prediction_match, broadcast_active,
                      oscillator_sync, hierarchy_calm, neuromodulator_balance]
        mean_ind = float(np.mean(indicators))
        std_ind = float(np.std(indicators))
        # High confidence = high mean, low std (all indicators agree)
        confidence = max(0.0, mean_ind * (1.0 - std_ind))

        # Boolean: is comprehending?
        is_comp = score >= self.comprehension_threshold and confidence >= self.confidence_threshold

        # Diagnostic note
        note = self._diagnose(attractor_stability, prediction_match, broadcast_active,
                              oscillator_sync, hierarchy_calm, neuromodulator_balance)

        state = ComprehensionState(
            attractor_stability=attractor_stability,
            prediction_match=prediction_match,
            broadcast_active=broadcast_active,
            oscillator_sync=oscillator_sync,
            hierarchy_calm=hierarchy_calm,
            neuromodulator_balance=neuromodulator_balance,
            comprehension_score=score,
            confidence=confidence,
            is_comprehending=is_comp,
            cycle=self.cycle,
            notes=note,
        )
        self.history.append(state)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return state

    def _assess_attractor(self) -> float:
        """Did the attractor network converge to a stable state?"""
        if self.attractor is None:
            return 0.5  # no attractor connected → neutral
        # Look at the most recent relaxation
        # If attractor has history of recent states, check if they stabilized
        if hasattr(self.attractor, 'history') and self.attractor.history:
            recent = self.attractor.history[-5:]
            if hasattr(recent[-1], 'converged'):
                # DiscreteAttractorNetwork case
                converged_count = sum(1 for s in recent if s.converged)
                return converged_count / len(recent)
            elif hasattr(recent[-1], 'order_parameter'):
                # Continuous attractor: use order parameter
                return float(np.mean([s.order_parameter for s in recent]))
        return 0.5

    def _assess_prediction_match(self) -> float:
        """How well does the prediction match reality? (1 - surprise)"""
        if self.predictive is None:
            return 0.5
        if hasattr(self.predictive, 'mean_surprise'):
            return max(0.0, 1.0 - self.predictive.mean_surprise())
        return 0.5

    def _assess_broadcast(self) -> float:
        """Is the global workspace broadcasting?"""
        if self.workspace is None:
            return 0.5
        if hasattr(self.workspace, 'ignition_rate'):
            return self.workspace.ignition_rate()
        return 0.5

    def _assess_oscillator_sync(self) -> float:
        """How synchronized are the Kuramoto oscillators?"""
        if self.kuramoto is None:
            return 0.5
        if hasattr(self.kuramoto, 'history') and self.kuramoto.history:
            recent = self.kuramoto.history[-5:]
            return float(np.mean([s.order_parameter for s in recent]))
        return 0.5

    def _assess_hierarchy_calm(self) -> float:
        """How calm is the predictive hierarchy? (1 - surprise)"""
        if self.hierarchy is None:
            return 0.5
        if hasattr(self.hierarchy, 'mean_surprise'):
            return max(0.0, 1.0 - self.hierarchy.mean_surprise())
        return 0.5

    def _assess_neuromodulator_balance(self) -> float:
        """How balanced are the neuromodulators? (closer to 0.5 = better)."""
        if self.neuromodulators is None:
            return 0.5
        levels = self.neuromodulators.levels
        # Average deviation from 0.5 (the "balanced" baseline)
        deviations = [
            abs(levels.dopamine - 0.5),
            abs(levels.serotonin - 0.5),
            abs(levels.acetylcholine - 0.5),
            abs(levels.norepinephrine - 0.5),
        ]
        avg_dev = float(np.mean(deviations))
        # Less deviation = more balanced = higher score
        return max(0.0, 1.0 - 2.0 * avg_dev)

    def _diagnose(self, attr, pred, bcast, osc, hier, nm) -> str:
        """Generate a diagnostic note explaining the comprehension state."""
        issues = []
        if attr < 0.4: issues.append(f"unstable attractor ({attr:.2f})")
        if pred < 0.4: issues.append(f"high prediction error ({1-pred:.2f})")
        if bcast < 0.4: issues.append(f"no global broadcast ({bcast:.2f})")
        if osc < 0.4: issues.append(f"oscillators desynchronized ({osc:.2f})")
        if hier < 0.4: issues.append(f"hierarchy surprised ({1-hier:.2f})")
        if nm < 0.4: issues.append(f"neuromodulator imbalance ({nm:.2f})")
        if not issues:
            return "all indicators aligned"
        return "issues: " + "; ".join(issues)

    # ------------------------------------------------------------------ #
    # History analysis
    # ------------------------------------------------------------------ #
    def comprehension_trend(self, window: int = 10) -> float:
        """Trend of comprehension score (positive = improving)."""
        if len(self.history) < 2 * window:
            return 0.0
        old = np.mean([s.comprehension_score for s in self.history[-2*window:-window]])
        new = np.mean([s.comprehension_score for s in self.history[-window:]])
        return float(new - old)

    def time_above_threshold(self, window: int = 50) -> float:
        """Fraction of recent cycles where the agent was comprehending."""
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return sum(1 for s in recent if s.is_comprehending) / max(len(recent), 1)

    def stats(self) -> Dict[str, any]:
        if not self.history:
            return {"cycle": self.cycle, "comprehension_score": 0.0}
        last = self.history[-1]
        return {
            "cycle": self.cycle,
            "comprehension_score": last.comprehension_score,
            "confidence": last.confidence,
            "is_comprehending": last.is_comprehending,
            "comprehension_trend": self.comprehension_trend(),
            "time_above_threshold": self.time_above_threshold(),
            "last_note": last.notes,
        }
