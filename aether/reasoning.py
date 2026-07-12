"""
reasoning.py — Continuous cognitive loop.

The "thought" is a single HD vector that evolves cycle by cycle. Each cycle:

    PERCEIVE    — encode new input (if any) into perception vector
    RETRIEVE    — query episodic + semantic memory with current thought
    DELIBERATE  — match against tool triggers and rule patterns
    ACT         — emit an action: speak, call tool, or think more

The loop runs until either:
  - The thought stabilizes (similarity between consecutive thoughts > 0.95)
  - A "speak" action is triggered with confidence above threshold
  - A max iteration budget is exhausted

This is the "thinking continuously" part — unlike a transformer, which does
a single forward pass per token, AETHER iterates on its internal state
until it converges or decides to act.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import numpy as np

from .hd import HDVector, DIM, bundle
from .memory import AssociativeMemory
from .encoder import TextEncoder


@dataclass
class ThoughtTrace:
    """A single step of the cognitive loop, for introspection."""
    cycle: int
    perception: Optional[str]
    retrieved: Optional[str]
    thought_similarity_to_prev: float
    action: str
    action_detail: str = ""
    confidence: float = 0.0


@dataclass
class CognitiveState:
    """Full state of the cognitive loop at any moment."""
    working_memory: HDVector
    goal: HDVector
    perception: Optional[HDVector] = None
    last_thought: Optional[HDVector] = None
    cycles_elapsed: int = 0
    trace: List[ThoughtTrace] = field(default_factory=list)
    pending_tool: Optional[str] = None
    pending_tool_args: Optional[str] = None
    final_output: Optional[str] = None
    final_confidence: float = 0.0
    converged: bool = False


class CognitiveLoop:
    """
    The continuous reasoning engine.

    Maintains a working memory (HD vector) and runs cycles of
    PERCEIVE -> RETRIEVE -> DELIBERATE -> ACT until convergence.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        assoc: AssociativeMemory,
        tool_registry=None,
        max_cycles: int = 8,
        convergence_threshold: float = 0.92,
        speak_threshold: float = 0.30,
        decay: float = 0.85,
    ):
        self.encoder = encoder
        self.assoc = assoc
        self.tools = tool_registry
        self.max_cycles = max_cycles
        self.convergence_threshold = convergence_threshold
        self.speak_threshold = speak_threshold
        self.decay = decay

    # ----- public API -----
    def think(
        self,
        user_input: str,
        goal_hint: Optional[str] = None,
    ) -> CognitiveState:
        """Run a full cognitive session in response to user input.

        Returns the final CognitiveState including a trace of every cycle
        for introspection (the agent can "explain its thinking").
        """
        # Initialize state
        perception = self.encoder.encode_text(user_input)
        goal = self.encoder.encode_text(goal_hint) if goal_hint else HDVector.random(self.encoder.dim)
        # Seed working memory: perception dominates, with goal as attractor
        wm = bundle([perception, goal], weights=[0.8, 0.2])

        state = CognitiveState(
            working_memory=wm,
            goal=goal,
            perception=perception,
        )

        for cycle in range(self.max_cycles):
            state.cycles_elapsed = cycle + 1

            # --- RETRIEVE ---
            retrieved_vec = self.assoc.kb_store.read(wm)
            retrieved_text = None
            if retrieved_vec is not None:
                # Find best matching episode (for explanation)
                sims = self.assoc.retrieve_similar(retrieved_vec, top_k=1)
                if sims:
                    retrieved_text = sims[0][0]
                # Integrate retrieved content into working memory
                wm = bundle(
                    [wm, retrieved_vec],
                    weights=[self.decay, 1.0 - self.decay],
                )

            # --- DELIBERATE: check for tool trigger ---
            tool_decision = self._match_tools(wm)
            if tool_decision:
                tool_name, tool_args, trig_sim = tool_decision
                trace = ThoughtTrace(
                    cycle=cycle,
                    perception=user_input if cycle == 0 else None,
                    retrieved=retrieved_text,
                    thought_similarity_to_prev=0.0,
                    action="call_tool",
                    action_detail=f"{tool_name}({tool_args})",
                    confidence=trig_sim,
                )
                state.trace.append(trace)
                state.pending_tool = tool_name
                state.pending_tool_args = tool_args
                state.working_memory = wm
                return state

            # --- DELIBERATE: check for direct KB answer ---
            # Try to interpret input as a question: (subject, predicate) -> ?
            kb_answer = self._try_kb_query(user_input)
            if kb_answer:
                answer_text, conf = kb_answer
                if conf >= self.speak_threshold:
                    trace = ThoughtTrace(
                        cycle=cycle,
                        perception=user_input if cycle == 0 else None,
                        retrieved=retrieved_text,
                        thought_similarity_to_prev=0.0,
                        action="speak",
                        action_detail=answer_text,
                        confidence=conf,
                    )
                    state.trace.append(trace)
                    state.final_output = answer_text
                    state.final_confidence = conf
                    state.converged = True
                    state.working_memory = wm
                    return state

            # --- DELIBERATE: check for episode retrieval (high similarity) ---
            sims = self.assoc.retrieve_similar(wm, top_k=1)
            if sims and sims[0][1] >= self.speak_threshold:
                ans, conf = sims[0]
                trace = ThoughtTrace(
                    cycle=cycle,
                    perception=user_input if cycle == 0 else None,
                    retrieved=retrieved_text,
                    thought_similarity_to_prev=0.0,
                    action="speak",
                    action_detail=ans,
                    confidence=conf,
                )
                state.trace.append(trace)
                state.final_output = ans
                state.final_confidence = conf
                state.converged = True
                state.working_memory = wm
                return state

            # --- Check convergence ---
            if state.last_thought is not None:
                sim = wm.similarity(state.last_thought)
                if sim >= self.convergence_threshold:
                    trace = ThoughtTrace(
                        cycle=cycle,
                        perception=user_input if cycle == 0 else None,
                        retrieved=retrieved_text,
                        thought_similarity_to_prev=sim,
                        action="converged",
                        action_detail="thought stabilized",
                        confidence=sim,
                    )
                    state.trace.append(trace)
                    state.converged = True
                    state.working_memory = wm
                    return state

            # --- Continue thinking ---
            trace = ThoughtTrace(
                cycle=cycle,
                perception=user_input if cycle == 0 else None,
                retrieved=retrieved_text,
                thought_similarity_to_prev=0.0,
                action="think",
                action_detail="deliberating...",
            )
            state.trace.append(trace)
            state.last_thought = wm.copy()
            state.working_memory = wm

        # Out of cycles: emit best-effort answer
        best_output, best_conf = self._best_effort(wm)
        trace = ThoughtTrace(
            cycle=state.cycles_elapsed,
            perception=None,
            retrieved=None,
            thought_similarity_to_prev=0.0,
            action="speak",
            action_detail=best_output or "(no answer)",
            confidence=best_conf,
        )
        state.trace.append(trace)
        state.final_output = best_output
        state.final_confidence = best_conf
        state.working_memory = wm
        return state

    # ----- internals -----
    def _match_tools(self, wm: HDVector):
        """Check if any tool's trigger vector is similar enough to fire."""
        if self.tools is None:
            return None
        best = None
        for name, trig_vec, trig_threshold in self.tools.triggers():
            sim = wm.similarity(trig_vec)
            if sim >= trig_threshold and (best is None or sim > best[2]):
                best = (name, None, sim)  # args extracted later by tool layer
        return best

    def _try_kb_query(self, user_input: str):
        """Heuristic: parse simple 'X is Y' / 'what is X' / 'capital of X'.

        Returns (answer, confidence) or None.
        """
        text = user_input.lower().strip()
        # Strip trailing punctuation
        text = text.rstrip("?.!")

        # Try patterns: (prefix, predicate, trailing_strip)
        # trailing_strip removes predicate-suffix words from the subject.
        patterns = [
            ("what is the capital of", "capital_of", ""),
            ("what is capital of",     "capital_of", ""),
            ("capital of",              "capital_of", ""),
            ("who is",                  "is_a",       ""),
            ("what is",                 "is_a",       ""),
            ("where is",                "located_in", "located"),
            ("where is",                "located_in", "found"),
            ("where is",                "located_in", ""),
            ("what does",               "does",       ""),
        ]
        for prefix, predicate, trailing in patterns:
            if text.startswith(prefix):
                subject = text[len(prefix):].strip()
                # Strip trailing predicate word like "located", "found"
                if trailing and subject.endswith(" " + trailing):
                    subject = subject[:-(len(trailing) + 1)].strip()
                subject = subject.rstrip("?")
                if not subject:
                    continue
                result = self.assoc.query_triple(subject, predicate)
                if result:
                    ans, sim = result
                    return ans, sim
        return None

    def _best_effort(self, wm: HDVector):
        """When out of cycles, return the closest stored episode."""
        sims = self.assoc.retrieve_similar(wm, top_k=1)
        if sims:
            return sims[0]
        return None, 0.0

    # ----- introspection -----
    def explain(self, state: CognitiveState) -> str:
        """Produce a human-readable trace of the cognitive process."""
        lines = [
            f"Cognitive trace ({state.cycles_elapsed} cycles, "
            f"converged={state.converged}, confidence={state.final_confidence:.2f}):"
        ]
        for t in state.trace:
            line = f"  [{t.cycle}] "
            if t.perception:
                line += f"PERCEIVE({t.perception[:40]!r}) -> "
            if t.retrieved:
                line += f"RETRIEVE({t.retrieved[:40]!r}) -> "
            line += f"{t.action.upper()}"
            if t.action_detail:
                line += f": {t.action_detail}"
            if t.confidence > 0:
                line += f"  (conf={t.confidence:.2f})"
            lines.append(line)
        return "\n".join(lines)
