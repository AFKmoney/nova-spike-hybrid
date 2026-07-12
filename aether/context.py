"""
context.py — Conversation context and memory.

Maintains state across turns:
  - history of (user, agent) turns
  - recently mentioned entities (for pronoun resolution)
  - current goal / sub-goal stack
  - topic tracking

Pronouns ("it", "he", "she", "they", "that") get resolved to the most
recently mentioned entity of the right type before being sent to the
inference engine.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import re

from .semantic import is_proper_noun, is_number, STOP_WORDS


@dataclass
class Turn:
    """A single conversation turn."""
    user_input: str
    agent_response: str
    entities_mentioned: List[str] = field(default_factory=list)
    timestamp: int = 0  # turn index


class ConversationContext:
    """Stateful conversation memory."""

    # Pronouns and their preferred antecedent types
    PRONOUN_MAP = {
        "it":   "thing",
        "he":   "person_m",
        "she":  "person_f",
        "they": "plural",
        "that": "thing",
        "this": "thing",
    }

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Turn] = []
        self.entity_recency: Dict[str, int] = {}  # entity -> last turn index
        self.current_goal: Optional[str] = None
        self.topic: Optional[str] = None
        self._turn_counter = 0

    def add_turn(self, user_input: str, agent_response: str) -> Turn:
        """Record a turn and update entity recency."""
        entities = self._extract_entities(user_input + " " + agent_response)
        turn = Turn(
            user_input=user_input,
            agent_response=agent_response,
            entities_mentioned=entities,
            timestamp=self._turn_counter,
        )
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        for ent in entities:
            self.entity_recency[ent.lower()] = self._turn_counter
        self._turn_counter += 1
        return turn

    def _extract_entities(self, text: str) -> List[str]:
        """Extract probable entities (proper nouns, numbers) from text."""
        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+|[^\sA-Za-zÀ-ÿ0-9]", text)
        entities = []
        # Multi-word entity detection: merge consecutive proper nouns
        current_entity = []
        for tok in tokens:
            if is_proper_noun(tok):
                current_entity.append(tok)
            else:
                if current_entity:
                    entities.append(" ".join(current_entity))
                    current_entity = []
                if is_number(tok):
                    entities.append(tok)
        if current_entity:
            entities.append(" ".join(current_entity))
        return entities

    # ------------------------------------------------------------------ #
    # Pronoun resolution
    # ------------------------------------------------------------------ #
    def resolve_pronouns(self, text: str) -> str:
        """Replace pronouns with the most recently mentioned entity."""
        tokens = text.split()
        resolved = []
        for tok in tokens:
            clean = tok.lower().rstrip(",.?!")
            if clean in self.PRONOUN_MAP:
                antecedent = self._most_recent_entity(exclude=clean)
                if antecedent:
                    # Preserve trailing punctuation
                    trailing = tok[len(clean):]
                    resolved.append(antecedent + trailing)
                    continue
            resolved.append(tok)
        return " ".join(resolved)

    def _most_recent_entity(self, exclude: str = "") -> Optional[str]:
        """Return the most recently mentioned entity (excluding stop words)."""
        if not self.entity_recency:
            return None
        # Sort by recency (highest timestamp first)
        sorted_entities = sorted(self.entity_recency.items(), key=lambda x: -x[1])
        for ent, _ in sorted_entities:
            if ent.lower() in STOP_WORDS:
                continue
            if ent.lower() == exclude.lower():
                continue
            if len(ent) <= 2:
                continue
            return ent
        return None

    # ------------------------------------------------------------------ #
    # Goal tracking
    # ------------------------------------------------------------------ #
    def set_goal(self, goal: str) -> None:
        self.current_goal = goal

    def clear_goal(self) -> None:
        self.current_goal = None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def recent_entities(self, n: int = 5) -> List[str]:
        """Return the n most recently mentioned entities."""
        sorted_entities = sorted(self.entity_recency.items(), key=lambda x: -x[1])
        return [ent for ent, _ in sorted_entities[:n] if ent.lower() not in STOP_WORDS and len(ent) > 2]

    def last_user_turn(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1].user_input

    def last_agent_turn(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1].agent_response

    def summary(self) -> str:
        """Brief summary of the conversation state."""
        lines = [
            f"Turns: {len(self.history)}",
            f"Recent entities: {self.recent_entities(5)}",
            f"Current goal: {self.current_goal or '(none)'}",
            f"Topic: {self.topic or '(none)'}",
        ]
        return "\n".join(lines)
