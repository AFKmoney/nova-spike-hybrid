"""
curiosity.py — Curiosity-driven self-questioning.

BRAIN INSPIRATION
-----------------
A child doesn't just answer questions — they ASK them. Curiosity is the
drive to reduce uncertainty about the world. The brain's dopamine system
rewards information gain, not just external reward.

AETHER'S USE
------------
When idle, AETHER:
  1. Picks a random entity from its KB
  2. Asks itself questions about that entity
  3. Checks if it knows the answer
  4. If not, identifies the knowledge gap and asks the user (or explores)

This drives autonomous learning. AETHER becomes genuinely curious — it
asks questions to fill its own gaps.
"""

from __future__ import annotations
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class CuriosityQuestion:
    """A self-generated question."""
    question: str
    target_entity: str
    target_predicate: str
    knows_answer: bool
    expected_answer: Optional[str] = None


@dataclass
class CuriosityReport:
    """Report of a curiosity cycle."""
    n_questions_generated: int
    n_known: int
    n_unknown: int
    questions: List[CuriosityQuestion] = field(default_factory=list)
    unknown_gaps: List[Tuple[str, str]] = field(default_factory=list)


# Templates for generating questions
QUESTION_TEMPLATES = {
    "capital_of": "What is the capital of {entity}?",
    "located_in": "Where is {entity} located?",
    "is_a": "What is {entity}?",
    "has": "What does {entity} have?",
    "founded_in": "When was {entity} founded?",
    "born_in": "When was {entity} born?",
    "invented": "What did {entity} invent?",
    "famous_for": "What is {entity} famous for?",
}


class CuriosityEngine:
    """Generate self-questions to identify knowledge gaps."""

    def __init__(self, agent):
        self.agent = agent
        self.rng = random.Random()
        self.last_report: Optional[CuriosityReport] = None

    def explore(self, n_questions: int = 10) -> CuriosityReport:
        """Generate and test n self-questions."""
        report = CuriosityReport(n_questions_generated=0, n_known=0, n_unknown=0)
        # Get all entities in the KB
        entities = set()
        for s, p, o in self.agent.assoc.list_triples():
            entities.add(s)
            entities.add(o)
        entities = list(entities)
        if not entities: return report

        predicates = list(QUESTION_TEMPLATES.keys())

        for _ in range(n_questions):
            entity = self.rng.choice(entities)
            predicate = self.rng.choice(predicates)
            question = QUESTION_TEMPLATES[predicate].format(entity=entity)

            # Does AETHER know the answer?
            result = self.agent.inference.lookup(entity, predicate)
            knows = result is not None
            expected = result[0] if result else None

            cq = CuriosityQuestion(
                question=question,
                target_entity=entity,
                target_predicate=predicate,
                knows_answer=knows,
                expected_answer=expected,
            )
            report.questions.append(cq)
            report.n_questions_generated += 1
            if knows:
                report.n_known += 1
            else:
                report.n_unknown += 1
                report.unknown_gaps.append((entity, predicate))

        self.last_report = report
        log.info(f"curiosity: {report.n_known} known, {report.n_unknown} unknown out of {report.n_questions_generated}")
        return report

    def most_curious_about(self) -> List[Tuple[str, str, int]]:
        """Return entities with the most knowledge gaps."""
        if not self.last_report: return []
        gap_counts: Dict[str, int] = {}
        for entity, predicate in self.last_report.unknown_gaps:
            gap_counts[entity] = gap_counts.get(entity, 0) + 1
        return sorted(gap_counts.items(), key=lambda x: -x[1])[:5]

    def suggest_question_to_user(self) -> Optional[str]:
        """Suggest a question the user could answer to fill a gap."""
        if not self.last_report or not self.last_report.unknown_gaps:
            return None
        entity, predicate = self.rng.choice(self.last_report.unknown_gaps)
        template = QUESTION_TEMPLATES.get(predicate, "Tell me about {entity}.")
        return f"I'm curious: {template.format(entity=entity)} (I don't know this yet — could you teach me?)"

    def summary(self) -> Dict[str, int]:
        if self.last_report is None: return {"cycles_run": 0}
        return {
            "cycles_run": 1,
            "questions_generated": self.last_report.n_questions_generated,
            "known": self.last_report.n_known,
            "unknown": self.last_report.n_unknown,
            "top_gaps": len(self.most_curious_about()),
        }
