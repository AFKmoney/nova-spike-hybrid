"""
self_modify.py — Self-modification: adjust parameters based on performance.

PROBLEM
-------
Transformer LLMs have fixed weights after training. They can't adapt their
own cognitive parameters (temperature, attention, learning rate) based on
how well they're doing.

SOLUTION
--------
AETHER monitors its own performance (comprehension scores, prediction
errors, IQ trend) and adjusts:

  1. Comprehension threshold (how strict is "understanding"?)
  2. SDM activation k (broader or narrower retrieval)
  3. Attractor convergence threshold (faster or more careful)
  4. Kuramoto coupling (more or less binding)
  5. Cognitive loop max_cycles (deeper thinking when needed)
  6. Neuromodulator baselines (boost DA when learning well, etc.)

This is real self-modification — AETHER literally tunes its own brain.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ParamSnapshot:
    """Snapshot of adjustable parameters at one moment."""
    comprehension_threshold: float
    sdm_k: int
    attractor_convergence: float
    kuramoto_coupling: float
    cogloop_max_cycles: int
    nm_baseline_dopamine: float
    nm_baseline_serotonin: float
    timestamp: float


@dataclass
class ModificationReport:
    """Report of one self-modification cycle."""
    old_params: ParamSnapshot
    new_params: ParamSnapshot
    changes: List[str] = field(default_factory=list)
    reason: str = ""


class SelfModifier:
    """Adjusts AETHER's own parameters based on performance."""

    def __init__(self, agent):
        self.agent = agent
        self.history: List[ParamSnapshot] = [self._snapshot()]
        self.performance_history: List[float] = []

    def _snapshot(self) -> ParamSnapshot:
        return ParamSnapshot(
            comprehension_threshold=self.agent.comprehension.comprehension_threshold,
            sdm_k=self.agent.assoc.kb_store.k,
            attractor_convergence=self.agent.attractor.convergence_threshold,
            kuramoto_coupling=self.agent.kuramoto.K,
            cogloop_max_cycles=self.agent.cogloop.max_cycles,
            nm_baseline_dopamine=self.agent.neuromodulators.levels.dopamine,
            nm_baseline_serotonin=self.agent.neuromodulators.levels.serotonin,
            timestamp=time.time(),
        )

    def record_performance(self, comprehension_score: float) -> None:
        """Record a comprehension score (call after each ask)."""
        self.performance_history.append(comprehension_score)
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

    def mean_performance(self, window: int = 20) -> float:
        if not self.performance_history: return 0.5
        recent = self.performance_history[-window:]
        return float(np.mean(recent))

    def performance_trend(self, window: int = 20) -> float:
        """Trend of comprehension (positive = improving)."""
        if len(self.performance_history) < 2 * window: return 0.0
        old = float(np.mean(self.performance_history[-2*window:-window]))
        new = float(np.mean(self.performance_history[-window:]))
        return new - old

    # ------------------------------------------------------------------ #
    # Self-modification rules
    # ------------------------------------------------------------------ #
    def modify(self) -> ModificationReport:
        """Adjust parameters based on recent performance."""
        old = self._snapshot()
        report = ModificationReport(old_params=old, new_params=old, changes=[])
        new_params = ParamSnapshot(
            old.comprehension_threshold, old.sdm_k, old.attractor_convergence,
            old.kuramoto_coupling, old.cogloop_max_cycles,
            old.nm_baseline_dopamine, old.nm_baseline_serotonin, time.time(),
        )

        perf = self.mean_performance()
        trend = self.performance_trend()

        # Rule 1: if performance is low and falling, INCREASE cognitive depth
        if perf < 0.4 and trend < 0:
            new_params.cogloop_max_cycles = min(20, old.cogloop_max_cycles + 2)
            report.changes.append(f"cogloop_max_cycles: {old.cogloop_max_cycles} -> {new_params.cogloop_max_cycles} (deeper thinking)")
            report.reason = "low and falling performance → think deeper"

        # Rule 2: if performance is high and rising, be FASTER (less cycles)
        elif perf > 0.7 and trend > 0:
            new_params.cogloop_max_cycles = max(3, old.cogloop_max_cycles - 1)
            report.changes.append(f"cogloop_max_cycles: {old.cogloop_max_cycles} -> {new_params.cogloop_max_cycles} (faster)")
            report.reason = "high and rising performance → be faster"

        # Rule 3: if performance is moderate but unstable, INCREASE attractor convergence
        if 0.3 < perf < 0.6 and abs(trend) > 0.05:
            new_params.attractor_convergence = min(0.99, old.attractor_convergence + 0.02)
            report.changes.append(f"attractor_convergence: {old.attractor_convergence:.3f} -> {new_params.attractor_convergence:.3f} (more careful)")
            if not report.reason: report.reason = "unstable performance → more careful convergence"

        # Rule 4: if comprehension is too strict, LOWER the threshold
        if perf < 0.3:
            new_params.comprehension_threshold = max(0.3, old.comprehension_threshold - 0.05)
            report.changes.append(f"comprehension_threshold: {old.comprehension_threshold:.3f} -> {new_params.comprehension_threshold:.3f} (more lenient)")
            if not report.reason: report.reason = "very low performance → more lenient threshold"

        # Rule 5: if very high performance, RAISE the threshold (be stricter)
        if perf > 0.8:
            new_params.comprehension_threshold = min(0.8, old.comprehension_threshold + 0.02)
            report.changes.append(f"comprehension_threshold: {old.comprehension_threshold:.3f} -> {new_params.comprehension_threshold:.3f} (stricter)")
            if not report.reason: report.reason = "very high performance → stricter threshold"

        # Rule 6: if Kuramoto synchronization is poor, INCREASE coupling
        if self.agent.kuramoto.history:
            recent_sync = float(np.mean([s.order_parameter for s in self.agent.kuramoto.history[-10:]]))
            if recent_sync < 0.5:
                new_params.kuramoto_coupling = min(2.0, old.kuramoto_coupling + 0.1)
                report.changes.append(f"kuramoto_coupling: {old.kuramoto_coupling:.3f} -> {new_params.kuramoto_coupling:.3f} (stronger binding)")

        # Apply changes
        self.agent.comprehension.comprehension_threshold = new_params.comprehension_threshold
        self.agent.assoc.kb_store.k = new_params.sdm_k
        self.agent.attractor.convergence_threshold = new_params.attractor_convergence
        self.agent.kuramoto.K = new_params.kuramoto_coupling
        self.agent.cogloop.max_cycles = new_params.cogloop_max_cycles

        report.new_params = new_params
        self.history.append(new_params)
        if len(self.history) > 50:
            self.history = self.history[-50:]

        if report.changes:
            log.info(f"self-modification: {report.reason}; " + "; ".join(report.changes))
        return report

    def summary(self) -> Dict[str, Any]:
        return {
            "modifications_run": len(self.history) - 1,
            "current_params": self.history[-1].__dict__,
            "mean_performance": self.mean_performance(),
            "performance_trend": self.performance_trend(),
        }
