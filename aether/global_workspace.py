"""
global_workspace.py — Global Workspace Theory (Baars 1988, Dehaene 2014).

BRAIN INSPIRATION
-----------------
Bernard Baars proposed that consciousness arises from a "global workspace"
— a central memory buffer where specialized modules (vision, language,
motor, memory, etc.) COMPETE to broadcast their content. The winning
module's content becomes globally available to all other modules.

This explains:
  - Why only one thought is conscious at a time (winner-take-all)
  - Why conscious content is broadcast (massive parallel access)
  - The "ignition" phenomenon: sudden synchronized activity across cortex
    when a stimulus crosses the consciousness threshold

Dehaene's "global neuronal workspace" (GNW) formalized this:
  - Sub-threshold processing: localized, parallel, unconscious
  -Cross-threshold ignition: long-range synchrony, broadcast, conscious
  - Threshold depends on attention, arousal, signal strength

AETHER'S USE
------------
We implement a GlobalWorkspace with:
  - N specialist modules (each submits a candidate + confidence)
  - Competition: winner-take-all with softmax weighting
  - Ignition threshold: only broadcast if winner's confidence > threshold
  - Broadcast: winning content goes to ALL specialists as new input

This gives AETHER a conscious/unconscious distinction:
  - Below threshold: parallel specialist processing (unconscious)
  - Above threshold: one winner broadcasts (conscious "thought")

Crucially, the workspace holds the CURRENT THOUGHT — the agent's
awareness at time t. The cognitive loop reads from the workspace to
decide what to do next.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import math

from .hd import HDVector, DIM, bundle, _sign


# ---------------------------------------------------------------------------
# Specialist module
# ---------------------------------------------------------------------------

@dataclass
class SpecialistOutput:
    """Output from one specialist module."""
    specialist_name: str
    content_vector: HDVector          # the proposed thought content
    confidence: float                  # 0-1, how confident this specialist is
    content_text: Optional[str] = None  # human-readable interpretation
    metadata: Dict[str, Any] = field(default_factory=dict)


class Specialist:
    """A specialist module that submits content to the global workspace.

    Each specialist has:
      - A name (e.g., "language", "vision", "memory", "motor")
      - A process() function that takes the current workspace state + raw input
        and returns a SpecialistOutput
    """

    def __init__(self, name: str, process_fn: Callable[[Optional[HDVector], Any], SpecialistOutput]):
        self.name = name
        self.process_fn = process_fn
        self.last_output: Optional[SpecialistOutput] = None
        self.activation_history: List[float] = []

    def process(self, current_workspace: Optional[HDVector], raw_input: Any) -> SpecialistOutput:
        """Run the specialist, get its output for this cycle."""
        output = self.process_fn(current_workspace, raw_input)
        self.last_output = output
        self.activation_history.append(output.confidence)
        # Keep history bounded
        if len(self.activation_history) > 50:
            self.activation_history = self.activation_history[-50:]
        return output


# ---------------------------------------------------------------------------
# Global Workspace
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceState:
    """State of the global workspace at a single cycle."""
    cycle: int
    winner: Optional[str]                          # name of winning specialist
    winning_content: Optional[HDVector]            # broadcast content
    winning_confidence: float                      # winner's confidence
    ignition: bool                                 # did ignition happen?
    all_outputs: List[SpecialistOutput] = field(default_factory=list)
    broadcast_active: bool = False                 # is content being broadcast?


class GlobalWorkspace:
    """The central workspace where specialists compete for broadcast.

    Implements:
      - Specialist competition (softmax-weighted by confidence)
      - Ignition threshold (broadcast only above threshold)
      - Broadcast (winning content goes to all specialists next cycle)
      - Persistent state (workspace holds current thought)
    """

    def __init__(
        self,
        dim: int = DIM,
        ignition_threshold: float = 0.4,
        softmax_temperature: float = 0.3,
        broadcast_decay: float = 0.7,
    ):
        self.dim = dim
        self.ignition_threshold = ignition_threshold
        self.softmax_temperature = softmax_temperature
        self.broadcast_decay = broadcast_decay

        self.specialists: List[Specialist] = []
        self.current_content: Optional[HDVector] = None
        self.cycle: int = 0
        self.history: List[WorkspaceState] = []
        self.ignition_count: int = 0
        # Attention modulation (can boost specific specialists)
        self.attention_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Specialist registration
    # ------------------------------------------------------------------ #
    def register_specialist(self, specialist: Specialist) -> None:
        self.specialists.append(specialist)
        self.attention_weights.setdefault(specialist.name, 1.0)

    def set_attention(self, specialist_name: str, weight: float) -> None:
        """Modulate attention to a specialist (1.0 = neutral, >1 = boost, <1 = suppress)."""
        self.attention_weights[specialist_name] = max(0.0, weight)

    # ------------------------------------------------------------------ #
    # Cycle: specialists compete + winner broadcasts
    # ------------------------------------------------------------------ #
    def cycle_step(self, raw_input: Any = None) -> WorkspaceState:
        """Run one workspace cycle.

        1. Each specialist processes the current workspace state + raw input.
        2. Confidence values are combined with attention weights.
        3. Softmax selects a winner (with stochasticity).
        4. If winner's confidence > ignition_threshold, broadcast happens.
        5. Broadcast content becomes the new current_content (decayed).
        """
        self.cycle += 1

        # 1. Specialists process
        outputs: List[SpecialistOutput] = []
        for spec in self.specialists:
            output = spec.process(self.current_content, raw_input)
            # Apply attention modulation
            att = self.attention_weights.get(spec.name, 1.0)
            output.confidence *= att
            outputs.append(output)

        # 2. Filter out zero-confidence outputs
        candidates = [o for o in outputs if o.confidence > 0]
        if not candidates:
            # No specialist produced output — decay the workspace
            self._decay_workspace()
            state = WorkspaceState(
                cycle=self.cycle,
                winner=None,
                winning_content=None,
                winning_confidence=0.0,
                ignition=False,
                all_outputs=outputs,
            )
            self.history.append(state)
            return state

        # 3. Softmax selection over confidences
        # Lower temperature = more winner-take-all
        confs = np.array([o.confidence for o in candidates], dtype=np.float64)
        # Normalize to non-negative
        confs = np.maximum(confs, 1e-6)
        # Softmax with temperature
        scaled = confs / max(self.softmax_temperature, 0.01)
        scaled = scaled - scaled.max()
        exp_s = np.exp(scaled * 3.0)  # scale up for sharper distribution
        probs = exp_s / exp_s.sum()

        # Sample a winner
        winner_idx = int(np.random.choice(len(candidates), p=probs))
        winner = candidates[winner_idx]

        # 4. Ignition check
        ignition = winner.confidence >= self.ignition_threshold

        # 5. Broadcast
        broadcast_active = False
        if ignition:
            # Broadcast: winning content becomes the new workspace content
            if self.current_content is None:
                self.current_content = winner.content_vector.copy()
            else:
                # Blend: workspace = decay(old) + winner
                self.current_content = bundle(
                    [self.current_content, winner.content_vector],
                    weights=[self.broadcast_decay, 1.0 - self.broadcast_decay],
                )
            self.ignition_count += 1
            broadcast_active = True
        else:
            # Below threshold: workspace decays slowly
            self._decay_workspace()

        state = WorkspaceState(
            cycle=self.cycle,
            winner=winner.specialist_name,
            winning_content=self.current_content.copy() if self.current_content else None,
            winning_confidence=winner.confidence,
            ignition=ignition,
            all_outputs=outputs,
            broadcast_active=broadcast_active,
        )
        self.history.append(state)
        # Keep history bounded
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return state

    def _decay_workspace(self) -> None:
        """Apply decay to the workspace content (toward zero)."""
        if self.current_content is not None:
            # Decay: blend with random noise vector
            noise = HDVector.random(self.dim)
            self.current_content = bundle(
                [self.current_content, noise],
                weights=[self.broadcast_decay, 1.0 - self.broadcast_decay],
            )

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def current_thought(self) -> Optional[HDVector]:
        """Return the current conscious content (or None if workspace is empty)."""
        return self.current_content

    def is_conscious(self) -> bool:
        """Is the workspace currently in an ignited (conscious) state?"""
        if not self.history:
            return False
        return self.history[-1].ignition

    def dominant_specialist(self) -> Optional[str]:
        """Which specialist has been dominating recent cycles?"""
        if not self.history:
            return None
        # Count winners over last 10 cycles
        recent = self.history[-10:]
        winners = [s.winner for s in recent if s.winner]
        if not winners:
            return None
        # Most common
        from collections import Counter
        return Counter(winners).most_common(1)[0][0]

    def ignition_rate(self, window: int = 20) -> float:
        """Fraction of recent cycles that resulted in ignition."""
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return sum(1 for s in recent if s.ignition) / max(len(recent), 1)

    def stats(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "n_specialists": len(self.specialists),
            "ignition_count": self.ignition_count,
            "ignition_rate": self.ignition_rate(),
            "dominant_specialist": self.dominant_specialist(),
            "is_conscious": self.is_conscious(),
            "current_thought_present": self.current_content is not None,
        }

    def reset(self) -> None:
        self.current_content = None
        self.cycle = 0
        self.history.clear()
        self.ignition_count = 0


# ---------------------------------------------------------------------------
# Built-in specialists for AETHER
# ---------------------------------------------------------------------------

def make_language_specialist(assoc) -> Specialist:
    """Language specialist: detects linguistic patterns and produces text-based content."""
    from .encoder import TextEncoder
    encoder = TextEncoder(dim=assoc.dim)
    encoder.assoc = assoc

    def process(workspace: Optional[HDVector], raw_input: Any) -> SpecialistOutput:
        if isinstance(raw_input, str):
            # Encode the text input
            vec = encoder.encode_text(raw_input)
            # Confidence is high for clear text
            return SpecialistOutput(
                specialist_name="language",
                content_vector=vec,
                confidence=0.9,
                content_text=raw_input,
                metadata={"type": "text_input"},
            )
        # No text input — language specialist stays quiet
        return SpecialistOutput(
            specialist_name="language",
            content_vector=HDVector.zero(assoc.dim),
            confidence=0.0,
        )

    return Specialist("language", process)


def make_memory_specialist(assoc) -> Specialist:
    """Memory specialist: retrieves from the KB based on current workspace content."""
    def process(workspace: Optional[HDVector], raw_input: Any) -> SpecialistOutput:
        if workspace is None:
            return SpecialistOutput(
                specialist_name="memory",
                content_vector=HDVector.zero(assoc.dim),
                confidence=0.0,
            )
        # Query the KB at the current workspace content
        retrieved = assoc.kb_store.read(workspace)
        if retrieved is None:
            return SpecialistOutput(
                specialist_name="memory",
                content_vector=HDVector.zero(assoc.dim),
                confidence=0.0,
            )
        # Confidence based on how strongly the memory was retrieved
        # (we don't have a direct similarity score from SDM.read, so use 0.5)
        return SpecialistOutput(
            specialist_name="memory",
            content_vector=retrieved,
            confidence=0.6,
            metadata={"type": "kb_retrieval"},
        )

    return Specialist("memory", process)


def make_tool_specialist(assoc, tools) -> Specialist:
    """Tool specialist: detects tool-trigger patterns in the workspace."""
    def process(workspace: Optional[HDVector], raw_input: Any) -> SpecialistOutput:
        if workspace is None:
            return SpecialistOutput(
                specialist_name="tool",
                content_vector=HDVector.zero(assoc.dim),
                confidence=0.0,
            )
        # Match workspace against tool triggers
        best_name, best_sim = None, -1.0
        for name, (vec, thr) in tools.triggers_.items():
            sim = workspace.similarity(vec)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None or best_sim < 0.1:
            return SpecialistOutput(
                specialist_name="tool",
                content_vector=HDVector.zero(assoc.dim),
                confidence=0.0,
            )
        # Confidence proportional to similarity above threshold
        trigger_vec, trigger_thr = tools.triggers_[best_name]
        return SpecialistOutput(
            specialist_name="tool",
            content_vector=trigger_vec,
            confidence=min(1.0, best_sim * 2.0),
            content_text=best_name,
            metadata={"tool": best_name, "similarity": best_sim},
        )

    return Specialist("tool", process)


def make_inference_specialist(assoc) -> Specialist:
    """Inference specialist: performs KB lookups when triggered by question words."""
    from .encoder import TextEncoder
    encoder = TextEncoder(dim=assoc.dim)
    encoder.assoc = assoc

    def process(workspace: Optional[HDVector], raw_input: Any) -> SpecialistOutput:
        if workspace is None or not isinstance(raw_input, str):
            return SpecialistOutput(
                specialist_name="inference",
                content_vector=HDVector.zero(assoc.dim),
                confidence=0.0,
            )
        # Check if input looks like a question
        text = raw_input.lower()
        if any(text.startswith(q) for q in ["what", "where", "who", "when", "why", "how"]):
            # Try to parse as a KB query
            vec = encoder.encode_text(raw_input)
            return SpecialistOutput(
                specialist_name="inference",
                content_vector=vec,
                confidence=0.5,
                content_text="query",
                metadata={"type": "question"},
            )
        return SpecialistOutput(
            specialist_name="inference",
            content_vector=HDVector.zero(assoc.dim),
            confidence=0.0,
        )

    return Specialist("inference", process)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Global Workspace Test ===\n")

    from .memory import AssociativeMemory
    from .encoder import TextEncoder
    from .tools import default_tools

    assoc = AssociativeMemory(dim=4096)
    encoder = TextEncoder(dim=4096)
    encoder.assoc = assoc
    tools = default_tools(encoder)

    # Pre-seed some knowledge
    assoc.learn_triple("paris", "capital_of", "france")
    assoc.learn_triple("tokyo", "capital_of", "japan")

    # Create workspace
    gw = GlobalWorkspace(dim=4096, ignition_threshold=0.4)

    # Register specialists
    gw.register_specialist(make_language_specialist(assoc))
    gw.register_specialist(make_memory_specialist(assoc))
    gw.register_specialist(make_tool_specialist(assoc, tools))
    gw.register_specialist(make_inference_specialist(assoc))

    print(f"  Registered {len(gw.specialists)} specialists:")
    for s in gw.specialists:
        print(f"    - {s.name}")
    print()

    # Run cycles with various inputs
    inputs = [
        "What is the capital of France?",
        "calc 2+2",
        "Hello",
        "Tokyo",
        "Where is Montreal located?",
    ]
    for inp in inputs:
        print(f"  Input: {inp!r}")
        # Run a few cycles with this input
        for _ in range(3):
            state = gw.cycle_step(inp)
        print(f"    Cycle {state.cycle}: winner={state.winner!r} "
              f"conf={state.winning_confidence:.3f} ignition={state.ignition}")
        if state.all_outputs:
            print(f"    All specialist confidences:")
            for o in state.all_outputs:
                print(f"      {o.specialist_name:12s}: {o.confidence:.3f}")
        print(f"    Workspace stats: {gw.stats()}")
        print()
