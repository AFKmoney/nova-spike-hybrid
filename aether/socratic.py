"""
socratic.py — Socratic dialogue: ask clarifying questions.

PROBLEM
-------
Transformer LLMs always answer, even when the question is ambiguous.
A human expert asks for clarification: "What do you mean by X?" or
"Are you asking about A or B?"

SOLUTION
--------
When AETHER detects ambiguity or lack of knowledge, instead of
hallucinating, it asks a Socratic clarifying question:

  User: "What is the capital?"
  AETHER: "Which country are you asking about? I know the capitals of
          France, Japan, Canada, and 12 others."

  User: "Tell me about Python."
  AETHER: "Do you mean Python the programming language, or python the snake?
          I have information about both."

This is genuinely different from LLMs — AETHER admits uncertainty and
helps the user refine their question.
"""

from __future__ import annotations
import re
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class SocraticResponse:
    """A Socratic clarifying question."""
    should_ask: bool
    question: str
    reason: str
    suggested_clarifications: List[str]


class SocraticDialogue:
    """Detect when to ask clarifying questions instead of answering."""

    def __init__(self, agent):
        self.agent = agent

    def should_ask_clarification(self, question: str) -> SocraticResponse:
        """Determine if AETHER should ask for clarification instead of answering."""
        q = question.strip().lower().rstrip("?.!")

        # Pattern 1: question is too short/vague
        if len(q.split()) <= 2 and not any(kw in q for kw in ["calc", "time", "list", "stats"]):
            # Check if there are multiple things this could refer to
            related = self._find_related_entities(q)
            if len(related) >= 2:
                return SocraticResponse(
                    True,
                    f"What would you like to know about? I see multiple matches: {', '.join(related[:5])}.",
                    "vague question with multiple matches",
                    related,
                )

        # Pattern 2: "What is the capital?" without specifying country
        if q in ("what is the capital", "what is the capital of", "capital of", "capital"):
            known_countries = self._find_entities_with_predicate("capital_of", reverse=True)
            if known_countries:
                return SocraticResponse(
                    True,
                    f"Which country? I know the capitals of: {', '.join(known_countries[:5])}.",
                    "missing country specification",
                    known_countries,
                )

        # Pattern 3: "Where is it?" without prior context
        if q in ("where is it", "where is it located", "what is it"):
            # Check conversation history for context
            if not self.agent.context.history:
                return SocraticResponse(
                    True,
                    "I don't have context for 'it'. Could you specify what you're asking about?",
                    "no prior context for pronoun",
                    [],
                )

        # Pattern 4: ambiguous entity (multiple meanings)
        # E.g., "Bank" = financial or river
        if len(q.split()) <= 3:
            # Check if the entity has multiple is_a values
            tokens = q.split()
            for token in tokens:
                if token in ("what", "is", "the", "of", "where", "a", "an"): continue
                meanings = self._find_meanings(token)
                if len(meanings) >= 2:
                    return SocraticResponse(
                        True,
                        f"Did you mean {token} as {meanings[0]} or as {meanings[1]}?",
                        f"ambiguous entity: {token} has multiple meanings",
                        meanings,
                    )

        # Pattern 5: AETHER doesn't know the answer — admit it
        # (this is handled by the regular response generator, but we can suggest teaching)
        # Check if the question is a KB query that would fail
        m = re.match(r"what is the capital of (.+)", q)
        if m:
            country = m.group(1).strip()
            result = self.agent.inference.lookup(country, "capital_of")
            if not result:
                return SocraticResponse(
                    True,
                    f"I don't know the capital of {country}. Would you like to teach me? (e.g., 'teach X is the capital of {country}')",
                    f"unknown capital for {country}",
                    [],
                )

        # No clarification needed
        return SocraticResponse(False, "", "clear question", [])

    def _find_related_entities(self, query: str) -> List[str]:
        """Find entities in the KB related to the query."""
        if not query: return []
        results = []
        for s, p, o in self.agent.assoc.list_triples():
            if query in s.lower() or query in o.lower():
                results.append(s if query in s else o)
        return list(set(results))[:5]

    def _find_entities_with_predicate(self, predicate: str, reverse: bool = False) -> List[str]:
        """Find all entities that have a value for the given predicate."""
        results = []
        for s, p, o in self.agent.assoc.list_triples():
            if p.lower() == predicate.lower():
                # If reverse, return the object (the thing we'd query for)
                # If forward, return the subject
                results.append(o if reverse else s)
        return list(set(results))[:10]

    def _find_meanings(self, entity: str) -> List[str]:
        """Find all the 'is_a' values for an entity (multiple meanings)."""
        meanings = []
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == entity.lower() and p.lower() == "is_a":
                meanings.append(o)
        return meanings[:3]
