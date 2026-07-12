"""
context_window.py — Long context management with sliding window + summarization.

PROBLEM
-------
AETHER has no context window management. Every conversation turn is
independent. GPT-4 has 128K token context with efficient management.

SOLUTION
--------
ContextWindowManager provides:
  1. Sliding window: keep last N turns in full detail
  2. Auto-summarization: when window is full, summarize oldest turns
  3. HD-vector context: maintain a running HD vector of the conversation
  4. Key info extraction: detect and preserve important facts
  5. Context retrieval: when asked, retrieve relevant past context
"""

from __future__ import annotations
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class ContextTurn:
    """A single conversation turn in the context window."""
    role: str  # "user" or "agent"
    text: str
    timestamp: float
    summary: Optional[str] = None  # filled when summarized
    is_summarized: bool = False
    key_facts: List[str] = field(default_factory=list)


class ContextWindowManager:
    """Sliding window context with auto-summarization."""

    def __init__(self, agent, max_turns: int = 20, summarize_threshold: int = 15):
        self.agent = agent
        self.max_turns = max_turns
        self.summarize_threshold = summarize_threshold
        self.turns: List[ContextTurn] = []
        # Running HD vector of the conversation
        self.conversation_vec = None
        # Extracted key facts (persist even when turns are summarized)
        self.key_facts: List[str] = []

    # ------------------------------------------------------------------ #
    # Adding turns
    # ------------------------------------------------------------------ #
    def add_turn(self, role: str, text: str) -> ContextTurn:
        """Add a conversation turn to the context."""
        turn = ContextTurn(role=role, text=text, timestamp=time.time())
        self.turns.append(turn)

        # Extract key facts from this turn
        facts = self._extract_key_facts(text)
        turn.key_facts = facts
        self.key_facts.extend(facts)

        # Update the conversation HD vector
        turn_vec = self.agent.encoder.encode_text(text)
        if self.conversation_vec is None:
            self.conversation_vec = turn_vec
        else:
            from .hd import bundle
            self.conversation_vec = bundle([self.conversation_vec, turn_vec],
                                           weights=[0.8, 0.2])

        # Check if we need to summarize
        if len(self.turns) > self.summarize_threshold:
            self._summarize_oldest()

        # Enforce max_turns
        if len(self.turns) > self.max_turns:
            # Keep only the most recent (already-summarized turns are kept as summaries)
            self.turns = self.turns[-self.max_turns:]

        return turn

    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key facts from a text."""
        from .learn_from_text import extract_facts
        facts = extract_facts(text)
        result = []
        for f in facts:
            result.append(f"{f.subject} {f.predicate} {f.object}")
        return result

    # ------------------------------------------------------------------ #
    # Summarization
    # ------------------------------------------------------------------ #
    def _summarize_oldest(self) -> None:
        """Summarize the oldest un-summarized turns."""
        # Find the oldest 5 un-summarized turns
        to_summarize = [t for t in self.turns if not t.is_summarized][:5]
        if not to_summarize:
            return

        # Combine their text
        combined_text = " ".join(t.text for t in to_summarize)

        # Generate a summary
        summary = self._generate_summary(combined_text)

        # Mark turns as summarized
        for turn in to_summarize:
            turn.is_summarized = True
            turn.summary = summary

        log.info(f"summarized {len(to_summarize)} turns into: {summary[:80]}...")

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of a text passage."""
        from .learn_from_text import extract_facts
        facts = extract_facts(text)
        if not facts:
            # Fallback: take first 100 chars
            return text[:100] + ("..." if len(text) > 100 else "")

        # Build summary from key facts
        parts = []
        for f in facts[:3]:
            if f.predicate == "capital_of":
                parts.append(f"{f.subject} is the capital of {f.object}")
            elif f.predicate == "located_in":
                parts.append(f"{f.subject} is in {f.object}")
            elif f.predicate == "is_a":
                parts.append(f"{f.subject} is {f.object}")
            else:
                parts.append(f"{f.subject} {f.predicate.replace('_',' ')} {f.object}")
        return ". ".join(parts) + "."

    # ------------------------------------------------------------------ #
    # Context retrieval
    # ------------------------------------------------------------------ #
    def get_context(self, query: Optional[str] = None, max_turns: int = 10) -> str:
        """Get the current context as a string.

        If query is provided, prioritize turns relevant to the query.
        """
        if not self.turns:
            return ""

        if query:
            # Find turns most relevant to the query
            relevant = self._find_relevant_turns(query, max_turns)
        else:
            relevant = self.turns[-max_turns:]

        # Build context string
        parts = []
        for turn in relevant:
            if turn.is_summarized and turn.summary:
                parts.append(f"[{turn.role} (summarized)]: {turn.summary}")
            else:
                parts.append(f"[{turn.role}]: {turn.text}")
        return "\n".join(parts)

    def _find_relevant_turns(self, query: str, max_turns: int) -> List[ContextTurn]:
        """Find turns most relevant to a query."""
        q_vec = self.agent.encoder.encode_text(query)
        scored = []
        for turn in self.turns:
            t_vec = self.agent.encoder.encode_text(turn.text)
            sim = q_vec.similarity(t_vec)
            scored.append((turn, sim))
        scored.sort(key=lambda x: -x[1])
        return [t for t, _ in scored[:max_turns]]

    # ------------------------------------------------------------------ #
    # Key facts
    # ------------------------------------------------------------------ #
    def get_key_facts(self) -> List[str]:
        """Get all extracted key facts from the conversation."""
        return list(self.key_facts)

    def get_relevant_facts(self, query: str, top_k: int = 5) -> List[str]:
        """Find key facts relevant to a query."""
        q_vec = self.agent.encoder.encode_text(query)
        scored = []
        for fact in self.key_facts:
            f_vec = self.agent.encoder.encode_text(fact)
            sim = q_vec.similarity(f_vec)
            scored.append((fact, sim))
        scored.sort(key=lambda x: -x[1])
        return [f for f, _ in scored[:top_k]]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        n_summarized = sum(1 for t in self.turns if t.is_summarized)
        n_active = sum(1 for t in self.turns if not t.is_summarized)
        return {
            "n_turns": len(self.turns),
            "n_summarized": n_summarized,
            "n_active": n_active,
            "n_key_facts": len(self.key_facts),
            "max_turns": self.max_turns,
            "conversation_vec_active": self.conversation_vec is not None,
        }

    def reset(self) -> None:
        """Reset the context window."""
        self.turns.clear()
        self.key_facts.clear()
        self.conversation_vec = None
