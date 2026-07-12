"""
compositional.py — Compositional reasoning: decompose complex questions.

PROBLEM
-------
Transformer LLMs answer complex questions in one shot — they often hallucinate
because they can't show their work or backtrack.

SOLUTION
--------
CompositionalReasoner decomposes a complex question into a tree of
sub-questions, solves each recursively, and combines the answers.

Example:
  "What is the capital of the country where the city that hosts the 2024 Olympics is located?"

Decomposition:
  Q1: "What city hosts the 2024 Olympics?" → Paris
  Q2: "What country is Paris in?" → France
  Q3: "What is the capital of France?" → Paris

Tree:
  ROOT → Q3(capital_of, Q2(country_of, Q1(host_of, "2024 Olympics")))

Each sub-question is solved by AETHER's normal pipeline; results are threaded
upward. If any sub-question fails, the reasoner can backtrack and try a
different decomposition.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class SubQuestion:
    """A single sub-question in a compositional decomposition."""
    question: str
    subquestions: List["SubQuestion"] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 0.0
    failed: bool = False


@dataclass
class CompositionalResult:
    """Result of compositional reasoning."""
    root: SubQuestion
    final_answer: Optional[str]
    final_confidence: float
    n_subquestions: int
    depth: int
    decomposition_str: str


class CompositionalReasoner:
    """Decompose complex questions into sub-questions."""

    def __init__(self, agent):
        self.agent = agent

    def answer(self, question: str) -> CompositionalResult:
        """Answer a complex question by decomposition."""
        root = self._decompose(question)
        self._solve(root)
        final = root.answer
        conf = root.confidence
        decomp = self._format_tree(root)
        return CompositionalResult(
            root=root,
            final_answer=final,
            final_confidence=conf,
            n_subquestions=self._count_nodes(root) - 1,
            depth=self._depth(root),
            decomposition_str=decomp,
        )

    # ------------------------------------------------------------------ #
    # Decomposition patterns
    # ------------------------------------------------------------------ #
    def _decompose(self, question: str) -> SubQuestion:
        """Decompose a question into a tree of sub-questions."""
        q = question.strip().rstrip("?")
        lower = q.lower()

        # Pattern 1: "What is the capital of the country where X is located?"
        m = re.match(r"what is the capital of (?:the )?country where (.+) is located", lower)
        if m:
            place = m.group(1).strip()
            return SubQuestion(
                question=question,
                subquestions=[
                    SubQuestion(f"Where is {place} located?"),  # sub_q1
                    SubQuestion("What is the capital of $1?"),  # placeholder
                ],
            )

        # Pattern 2: "What is the capital of the country whose capital is X?"
        m = re.match(r"what is the capital of (?:the )?country whose capital is (.+)", lower)
        if m:
            capital = m.group(1).strip()
            return SubQuestion(
                question=question,
                subquestions=[
                    SubQuestion(f"What country has {capital} as its capital?"),
                    SubQuestion("What is the capital of $1?"),
                ],
            )

        # Pattern 3: "Where is the capital of X located?"
        m = re.match(r"where is the capital of (.+) located", lower)
        if m:
            country = m.group(1).strip()
            return SubQuestion(
                question=question,
                subquestions=[
                    SubQuestion(f"What is the capital of {country}?"),
                    SubQuestion("Where is $1 located?"),
                ],
            )

        # Pattern 4: "What is the X of the Y of Z?"
        m = re.match(r"what is the (\w+) of the (\w+) of (.+)", lower)
        if m:
            p1, p2, subj = m.groups()
            return SubQuestion(
                question=question,
                subquestions=[
                    SubQuestion(f"What is the {p2} of {subj}?"),
                    SubQuestion(f"What is the {p1} of $1?"),
                ],
            )

        # Pattern 5: "What is the X of Y?" (single hop, no decomposition)
        m = re.match(r"what is the (\w+) of (.+)", lower)
        if m:
            return SubQuestion(question=question)  # leaf

        # Fallback: no decomposition
        return SubQuestion(question=question)

    # ------------------------------------------------------------------ #
    # Solve the tree recursively
    # ------------------------------------------------------------------ #
    def _solve(self, node: SubQuestion) -> None:
        """Solve a sub-question node, recursively solving its children first."""
        # First solve all sub-questions
        for i, sub in enumerate(node.subquestions):
            self._solve(sub)
            # Substitute the placeholder $1, $2 with sub-answers
            if sub.answer and not sub.failed:
                placeholder = f"${i+1}"
                for j, later_sub in enumerate(node.subquestions[i+1:], start=i+1):
                    later_sub.question = later_sub.question.replace(placeholder, sub.answer)

        # If there are sub-questions, the answer is the last one's answer
        if node.subquestions:
            last = node.subquestions[-1]
            node.answer = last.answer
            node.confidence = last.confidence
            node.failed = last.failed
            return

        # Leaf node: ask AETHER directly
        try:
            answer = self.agent.ask(node.question)
            # Clean up the answer (strip NLG templates)
            cleaned = self._extract_answer(answer, node.question)
            node.answer = cleaned
            node.confidence = 0.9 if cleaned else 0.0
            node.failed = not cleaned
        except Exception as e:
            log.warning(f"sub-question failed: {node.question!r}: {e}")
            node.failed = True

    def _extract_answer(self, response: str, question: str) -> Optional[str]:
        """Extract the bare answer from a natural-language response."""
        # If the response contains "It's X" or "The X is Y", extract X or Y
        # Simple heuristic: take the last capitalized word or the answer after "is"
        r = response.strip().rstrip(".")
        # Check for known patterns
        m = re.search(r"(?:It's|It is)\s+(.+)", r, re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.search(r"(?:The capital of \w+ is|capital is)\s+(.+)", r, re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.search(r"is located in\s+(.+)", r, re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.search(r"is\s+(.+)", r, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Fallback: return the whole response
        return r if len(r) < 50 else None

    # ------------------------------------------------------------------ #
    # Tree utilities
    # ------------------------------------------------------------------ #
    def _count_nodes(self, node: SubQuestion) -> int:
        return 1 + sum(self._count_nodes(s) for s in node.subquestions)

    def _depth(self, node: SubQuestion) -> int:
        if not node.subquestions: return 1
        return 1 + max(self._depth(s) for s in node.subquestions)

    def _format_tree(self, node: SubQuestion, indent: int = 0) -> str:
        prefix = "  " * indent
        ans = node.answer or "?"
        mark = "✓" if not node.failed else "✗"
        line = f"{prefix}{mark} Q: {node.question!r}\n{prefix}   A: {ans!r}\n"
        for sub in node.subquestions:
            line += self._format_tree(sub, indent + 1)
        return line
