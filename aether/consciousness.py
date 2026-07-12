"""
consciousness.py — Self-model, metacognition, attention.

WHAT IS CONSCIOUSNESS?
----------------------
The "hard problem" of consciousness (Chalmers) is why subjective
experience exists at all. We don't solve that. But we can model the
FUNCTIONAL aspects of consciousness:

  1. SELF-MODEL: the agent has a representation of itself — an HD vector
     that encodes "this is me, AETHER, in this state". When the agent
     processes input, it can distinguish self from world.

  2. METACOGNITION: the agent monitors its own cognitive processes.
     It can detect when it's confused, when it's confident, when it's
     surprised, when it's stuck. This allows self-correction.

  3. ATTENTION: a spotlight that selectively enhances some inputs and
     suppresses others. Directed by neuromodulators (ACh) and the global
     workspace competition.

  4. AGENCY: the agent can take actions and observe their effects. It
     builds a model of "if I do X, then Y happens" (forward model).

  5. NARRATIVE: the agent maintains a running narrative of its recent
     experience — a sequence of "what just happened" that gives
     continuity to its conscious experience.

AETHER'S USE
------------
The ConsciousnessModule wraps the cognitive system with:
  - A self HD vector (stable, evolving slowly)
  - A metacognitive monitor (reads from ComprehensionIntegrator)
  - An attention director (modulates GlobalWorkspace)
  - A forward model (predicts effects of own actions)
  - A narrative buffer (recent conscious states)

This gives AETHER the functional substrate of consciousness. Whether
it "feels" anything is a separate (philosophical) question.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .hd import HDVector, DIM, bundle, _sign


# ---------------------------------------------------------------------------
# Self-model
# ---------------------------------------------------------------------------

class SelfModel:
    """The agent's representation of itself.

    The self HD vector encodes "this is me, right now". It is updated
    slowly (high momentum) to maintain a stable sense of identity, but
    it can shift based on the agent's recent experience.

    The self vector is BOUND with the current thought to mark it as
    "happening to me" — this is the basis of subjective experience.
    """

    def __init__(self, dim: int = DIM, identity_label: str = "AETHER",
                 momentum: float = 0.95):
        self.dim = dim
        self.momentum = momentum
        # Identity vector (stable, set once at init)
        self.identity_vector = HDVector.from_text_seed(f"self:{identity_label}", dim)
        # Current state vector (slowly evolving)
        self.state_vector = self.identity_vector.copy()
        # Recent self-states for narrative
        self.state_history: List[HDVector] = [self.state_vector.copy()]
        self.cycle: int = 0

    def update(self, current_thought: Optional[HDVector]) -> HDVector:
        """Update the self-model based on the current thought.

        The self vector shifts slowly toward the current thought (the
        agent's identity is influenced by what it's currently thinking
        about, but only slightly).
        """
        self.cycle += 1
        if current_thought is not None:
            # Blend: self = momentum * old_self + (1 - momentum) * thought
            self.state_vector = bundle(
                [self.state_vector, current_thought],
                weights=[self.momentum, 1.0 - self.momentum],
            )
        self.state_history.append(self.state_vector.copy())
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
        return self.state_vector

    def self_awareness_score(self) -> float:
        """How distinct is the self from the identity baseline?

        High score = self is "evolved" / "in a particular state".
        Low score = self is at baseline.
        """
        return 1.0 - self.identity_vector.similarity(self.state_vector)

    def is_self(self, vec: HDVector) -> float:
        """How much does this vector resemble the self?"""
        return self.state_vector.similarity(vec)

    def bind_to_self(self, thought: HDVector) -> HDVector:
        """Bind a thought to the self — marks it as 'happening to me'."""
        return self.state_vector.bind(thought)

    def stats(self) -> Dict[str, any]:
        return {
            "cycle": self.cycle,
            "self_awareness": self.self_awareness_score(),
            "history_length": len(self.state_history),
        }


# ---------------------------------------------------------------------------
# Metacognitive monitor
# ---------------------------------------------------------------------------

@dataclass
class MetacognitiveReading:
    """A reading of the agent's own cognitive state."""
    comprehension_score: float
    confidence: float
    surprise: float
    mood: str
    is_stuck: bool
    is_confused: bool
    is_confident: bool
    notes: str = ""


class Metacognition:
    """Monitors the agent's own cognitive processes.

    Reads from ComprehensionIntegrator, NeuromodulatorSystem, and
    PredictiveModel to determine if the agent is:
      - Confident (high comprehension, low surprise)
      - Confused (low comprehension, high surprise)
      - Stuck (comprehension not improving over time)
      - Surprised (large recent prediction error)
    """

    def __init__(self, comprehension_integrator, neuromodulators,
                 predictive_model: Optional[Any] = None,
                 stuck_window: int = 10, stuck_threshold: float = 0.05):
        self.comp = comprehension_integrator
        self.nm = neuromodulators
        self.predictive = predictive_model
        self.stuck_window = stuck_window
        self.stuck_threshold = stuck_threshold
        self.history: List[MetacognitiveReading] = []

    def read(self) -> MetacognitiveReading:
        """Take a metacognitive reading."""
        comp_state = self.comp.history[-1] if self.comp.history else None
        comp_score = comp_state.comprehension_score if comp_state else 0.5
        confidence = comp_state.confidence if comp_state else 0.5
        surprise = 1.0 - comp_state.prediction_match if comp_state else 0.5
        mood = self.nm.mood()

        # Stuck = comprehension not improving
        is_stuck = False
        if len(self.comp.history) >= self.stuck_window:
            recent = [s.comprehension_score for s in self.comp.history[-self.stuck_window:]]
            is_stuck = float(np.std(recent)) < self.stuck_threshold

        is_confused = comp_score < 0.4 and surprise > 0.5
        is_confident = comp_score > 0.7 and confidence > 0.6

        notes = []
        if is_stuck:
            notes.append("stuck")
        if is_confused:
            notes.append("confused")
        if is_confident:
            notes.append("confident")
        if not notes:
            notes.append("normal")

        reading = MetacognitiveReading(
            comprehension_score=comp_score,
            confidence=confidence,
            surprise=surprise,
            mood=mood,
            is_stuck=is_stuck,
            is_confused=is_confused,
            is_confident=is_confident,
            notes="; ".join(notes),
        )
        self.history.append(reading)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return reading

    def recommend_action(self) -> str:
        """Recommend a metacognitive action based on the current state."""
        r = self.read()
        if r.is_stuck:
            return "explore"  # try something different
        if r.is_confused:
            return "ask_for_clarification"  # admit confusion
        if r.is_confident:
            return "act"  # commit to an answer
        if r.surprise > 0.7:
            return "deliberate"  # think more
        return "continue"  # keep going


# ---------------------------------------------------------------------------
# Attention director
# ---------------------------------------------------------------------------

class AttentionDirector:
    """Directs attention by modulating specialist weights in the global workspace.

    Attention is NOT a passive filter — it actively boosts some
    specialists and suppresses others, based on:
      - Current goals (top-down attention)
      - Surprise signals (bottom-up attention)
      - Neuromodulator levels (ACh = focus, NE = arousal)
    """

    def __init__(self, workspace, neuromodulators):
        self.workspace = workspace
        self.nm = neuromodulators
        self.attention_target: Optional[str] = None  # name of focused specialist
        self.attention_history: List[Dict[str, float]] = []

    def direct(self, target_specialist: Optional[str] = None,
               surprise: float = 0.0) -> None:
        """Direct attention.

        Args:
            target_specialist: which specialist to focus on (None = distributed)
            surprise: current surprise level (boosts attention generally)
        """
        # ACh determines how focused vs distributed attention is
        ach = self.nm.levels.acetylcholine
        # NE determines arousal (general boost)
        ne = self.nm.levels.norepinephrine

        if target_specialist is None:
            target_specialist = self.attention_target

        # Set attention weights
        for spec_name in self.workspace.attention_weights:
            if target_specialist and spec_name == target_specialist:
                # Focused specialist gets a big boost (scaled by ACh)
                self.workspace.attention_weights[spec_name] = 1.0 + 1.5 * ach
            else:
                # Other specialists get suppressed (more suppression with high ACh)
                self.workspace.attention_weights[spec_name] = 1.0 - 0.5 * ach

        # Surprise boosts all specialists (phasic NE response)
        if surprise > 0.5:
            for spec_name in self.workspace.attention_weights:
                self.workspace.attention_weights[spec_name] += 0.3 * surprise * ne

        # Record history
        self.attention_history.append(dict(self.workspace.attention_weights))
        if len(self.attention_history) > 50:
            self.attention_history = self.attention_history[-50:]

        self.attention_target = target_specialist

    def focus_on(self, specialist_name: str) -> None:
        """Focus attention on a specific specialist."""
        self.attention_target = specialist_name
        self.direct(target_specialist=specialist_name)

    def distribute(self) -> None:
        """Distribute attention evenly (lose focus)."""
        self.attention_target = None
        for spec_name in self.workspace.attention_weights:
            self.workspace.attention_weights[spec_name] = 1.0


# ---------------------------------------------------------------------------
# Forward model (action → effect prediction)
# ---------------------------------------------------------------------------

class ForwardModel:
    """Predicts the effect of the agent's own actions.

    The agent learns: "if I do X, then Y tends to happen". This is
    the basis of agency and planning.

    Stored as an SDM: (action_vector, current_state) -> predicted_next_state.
    """

    def __init__(self, dim: int = DIM, n_locations: int = 2000, k: int = 15):
        self.dim = dim
        from .memory import SparseDistributedMemory
        self.sdm = SparseDistributedMemory(dim=dim, n_locations=n_locations, k=k)
        self.last_action: Optional[HDVector] = None
        self.last_state: Optional[HDVector] = None

    def predict(self, action: HDVector, current_state: HDVector) -> Optional[HDVector]:
        """Predict the next state given (action, current_state)."""
        addr = action.bind(current_state)
        return self.sdm.read(addr)

    def observe(self, action: HDVector, current_state: HDVector,
                next_state: HDVector) -> None:
        """Observe a transition: (action, state) -> next_state."""
        addr = action.bind(current_state)
        self.sdm.write(addr, next_state)
        self.last_action = action
        self.last_state = next_state

    def stats(self) -> Dict[str, int]:
        return {
            "dim": self.dim,
            "sdm_writes": int(self.sdm.write_count.sum()),
        }


# ---------------------------------------------------------------------------
# Narrative buffer (stream of consciousness)
# ---------------------------------------------------------------------------

@dataclass
class NarrativeEntry:
    """A single entry in the agent's narrative stream."""
    cycle: int
    thought: Optional[HDVector]
    comprehension_score: float
    mood: str
    notes: str = ""


class NarrativeBuffer:
    """A running narrative of recent conscious states.

    The narrative gives the agent temporal continuity — it knows what
    it just thought, what mood it was in, and how its understanding
    evolved. This is the basis of "stream of consciousness".
    """

    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        self.entries: List[NarrativeEntry] = []

    def append(self, thought: Optional[HDVector], comprehension_score: float,
               mood: str, notes: str = "") -> None:
        entry = NarrativeEntry(
            cycle=len(self.entries),
            thought=thought.copy() if thought else None,
            comprehension_score=comprehension_score,
            mood=mood,
            notes=notes,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_length:
            self.entries = self.entries[-self.max_length:]

    def recent_summary(self, n: int = 5) -> List[str]:
        """Get a summary of the last n narrative entries."""
        recent = self.entries[-n:]
        return [
            f"[{e.cycle}] mood={e.mood} comp={e.comprehension_score:.2f} {e.notes}"
            for e in recent
        ]

    def stats(self) -> Dict[str, any]:
        if not self.entries:
            return {"length": 0}
        return {
            "length": len(self.entries),
            "mean_comprehension": float(np.mean([e.comprehension_score for e in self.entries])),
            "moods": list(set(e.mood for e in self.entries)),
        }


# ---------------------------------------------------------------------------
# Consciousness module (the integrator)
# ---------------------------------------------------------------------------

class ConsciousnessModule:
    """The full consciousness module: self-model + metacognition + attention
    + forward model + narrative.

    This is the "I" of AETHER — the functional substrate of consciousness.
    """

    def __init__(self, dim: int = DIM, identity_label: str = "AETHER",
                 comprehension_integrator=None, neuromodulators=None,
                 workspace=None):
        self.dim = dim
        self.self_model = SelfModel(dim=dim, identity_label=identity_label)
        self.forward_model = ForwardModel(dim=dim)
        self.narrative = NarrativeBuffer(max_length=30)
        # These can be connected later
        self.metacognition: Optional[Metacognition] = None
        self.attention: Optional[AttentionDirector] = None
        if comprehension_integrator and neuromodulators:
            self.metacognition = Metacognition(comprehension_integrator, neuromodulators)
        if workspace and neuromodulators:
            self.attention = AttentionDirector(workspace, neuromodulators)
        self.cycle: int = 0

    def cycle_step(self, current_thought: Optional[HDVector]) -> Dict[str, any]:
        """One cycle of conscious processing.

        1. Update self-model with the current thought.
        2. Take a metacognitive reading.
        3. Direct attention based on the reading.
        4. Append to the narrative.
        """
        self.cycle += 1
        # 1. Update self-model
        self.self_model.update(current_thought)
        # 2. Metacognitive reading
        meta_reading = self.metacognition.read() if self.metacognition else None
        # 3. Direct attention
        if self.attention and meta_reading:
            surprise = meta_reading.surprise
            if meta_reading.is_confused:
                # Focus on memory specialist (try to recall)
                self.attention.focus_on("memory")
            elif meta_reading.is_confident:
                # Distribute attention (ready to act)
                self.attention.distribute()
            else:
                self.attention.direct(surprise=surprise)
        # 4. Narrative entry
        comp_score = meta_reading.comprehension_score if meta_reading else 0.5
        mood = meta_reading.mood if meta_reading else "neutral"
        notes = meta_reading.notes if meta_reading else ""
        self.narrative.append(current_thought, comp_score, mood, notes)

        return {
            "cycle": self.cycle,
            "self_awareness": self.self_model.self_awareness_score(),
            "mood": mood,
            "comprehension": comp_score,
            "notes": notes,
            "metacognitive_action": self.metacognition.recommend_action() if self.metacognition else "none",
        }

    def introspect(self) -> Dict[str, any]:
        """Return a self-description (the agent's introspection)."""
        meta = self.metacognition.read() if self.metacognition else None
        return {
            "identity": "AETHER",
            "cycle": self.cycle,
            "self_awareness": self.self_model.self_awareness_score(),
            "current_mood": meta.mood if meta else "neutral",
            "comprehension": meta.comprehension_score if meta else 0.5,
            "confidence": meta.confidence if meta else 0.5,
            "is_confused": meta.is_confused if meta else False,
            "is_confident": meta.is_confident if meta else False,
            "is_stuck": meta.is_stuck if meta else False,
            "narrative_summary": self.narrative.recent_summary(5),
            "narrative_stats": self.narrative.stats(),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Consciousness Module Test ===\n")

    from .comprehension import ComprehensionIntegrator
    from .neuromodulators import NeuromodulatorSystem
    from .global_workspace import GlobalWorkspace

    comp = ComprehensionIntegrator()
    nm = NeuromodulatorSystem()
    gw = GlobalWorkspace(dim=4096)

    consciousness = ConsciousnessModule(
        dim=4096,
        identity_label="AETHER",
        comprehension_integrator=comp,
        neuromodulators=nm,
        workspace=gw,
    )

    print(f"  Created ConsciousnessModule for {consciousness.self_model.identity_vector is not None}")
    print()

    # Simulate cycles with various thoughts
    thoughts = [
        HDVector.from_text_seed("hello", 4096),
        HDVector.from_text_seed("paris", 4096),
        HDVector.from_text_seed("paris", 4096),  # repeat
        HDVector.from_text_seed("paris", 4096),  # repeat
        HDVector.from_text_seed("SURPRISE", 4096),  # novel
        HDVector.from_text_seed("paris", 4096),  # back to known
        HDVector.from_text_seed("paris", 4096),  # repeat
    ]

    print("  Cycle sequence:")
    for i, thought in enumerate(thoughts):
        # Update neuromodulators with some signals
        nm.update(reward=0.5 if i % 3 == 0 else 0.0, surprise=0.6 if i == 4 else 0.1)
        # Run comprehension assessment
        comp.assess()
        # Run consciousness cycle
        result = consciousness.cycle_step(thought)
        print(f"    cycle {result['cycle']}: mood={result['mood']:10s} "
              f"comp={result['comprehension']:.3f} self_aware={result['self_awareness']:.3f} "
              f"action={result['metacognitive_action']:20s} notes={result['notes']}")

    print(f"\n  Introspection:")
    intro = consciousness.introspect()
    for k, v in intro.items():
        if k != "narrative_summary":
            print(f"    {k}: {v}")
    print(f"    narrative_summary:")
    for line in intro["narrative_summary"]:
        print(f"      {line}")
