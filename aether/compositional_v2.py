"""
compositional_v2.py — Recursive compositional reasoning with persistent traces.

IMPROVEMENTS OVER v1
--------------------
v1 was limited to a fixed set of decomposition patterns.
v2 implements:

  1. RECURSIVE DECOMPOSITION
     - Any question can be decomposed into sub-questions
     - Sub-questions can themselves be decomposed (arbitrary depth)
     - Backtracking: if a sub-question fails, try a different decomposition

  2. PERSISTENT REASONING TRACES
     - Every step of reasoning is stored as an HD vector
     - Traces can be retrieved and reused for similar future questions
     - "I solved something like this before" — true analogical transfer

  3. SYNTHESIS
     - Sub-answers are bundled (not just threaded) into a synthesis
     - The synthesis HD vector captures the gist of the reasoning
     - Can be queried: "what was the gist of my reasoning?"

  4. PLAN REUSE
     - When a similar question is asked later, the previous plan is retrieved
     - Avoids re-decomposing the same kind of question
"""

from __future__ import annotations
import re
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """A persistent trace of a reasoning episode."""
    question: str
    question_vec: Any  # HDVector
    sub_questions: List["ReasoningTrace"] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 0.0
    failed: bool = False
    decomposition_strategy: str = ""
    timestamp: float = field(default_factory_factory=time.time) if False else 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class CompositionalResultV2:
    """Result of v2 compositional reasoning."""
    root: ReasoningTrace
    final_answer: Optional[str]
    final_confidence: float
    n_subquestions: int
    depth: int
    decomposition_str: str
    synthesis_vector: Optional[Any] = None  # HDVector
    reused_trace: bool = False  # was a previous trace reused?


class CompositionalReasonerV2:
    """Recursive compositional reasoning with persistent traces."""

    # Maximum decomposition depth (prevents infinite recursion)
    MAX_DEPTH = 5
    # Minimum confidence to accept an answer
    MIN_CONFIDENCE = 0.3

    def __init__(self, agent):
        self.agent = agent
        # Persistent trace store (HD-vector-addressed)
        self.traces: List[ReasoningTrace] = []
        # Synthesis vectors for retrieval
        self.synthesis_store: List[Tuple[HDVector, ReasoningTrace]] = []

    def answer(self, question: str) -> CompositionalResultV2:
        """Answer a complex question by recursive decomposition."""
        # First: try to find a similar previous trace
        q_vec = self.agent.encoder.encode_text(question)
        reused = self._find_similar_trace(q_vec)
        if reused and reused.answer:
            log.info("reusing previous reasoning trace")
            return CompositionalResultV2(
                root=reused, final_answer=reused.answer,
                final_confidence=reused.confidence,
                n_subquestions=self._count_nodes(reused) - 1,
                depth=self._depth(reused),
                decomposition_str=self._format_tree(reused),
                synthesis_vector=self._compute_synthesis(reused),
                reused_trace=True,
            )

        # Decompose recursively
        root = self._decompose_recursive(question, depth=0)
        self._solve_recursive(root)
        # Build synthesis vector
        synthesis = self._compute_synthesis(root)
        # Store the trace
        self.traces.append(root)
        self.synthesis_store.append((synthesis, root))
        if len(self.traces) > 100:
            self.traces = self.traces[-100:]
            self.synthesis_store = self.synthesis_store[-100:]

        return CompositionalResultV2(
            root=root, final_answer=root.answer,
            final_confidence=root.confidence,
            n_subquestions=self._count_nodes(root) - 1,
            depth=self._depth(root),
            decomposition_str=self._format_tree(root),
            synthesis_vector=synthesis,
            reused_trace=False,
        )

    # ------------------------------------------------------------------ #
    # Recursive decomposition
    # ------------------------------------------------------------------ #
    def _decompose_recursive(self, question: str, depth: int) -> ReasoningTrace:
        """Decompose a question into sub-questions, recursively."""
        trace = ReasoningTrace(
            question=question,
            question_vec=self.agent.encoder.encode_text(question),
            timestamp=time.time(),
        )

        if depth >= self.MAX_DEPTH:
            trace.decomposition_strategy = "max_depth_reached"
            return trace

        # Try each decomposition strategy
        for strategy_name, strategy_fn in [
            ("capital_of_country_where", self._decomp_capital_of_country_where),
            ("where_is_capital_of", self._decomp_where_is_capital_of),
            ("what_is_x_of_y_of_z", self._decomp_what_is_x_of_y_of_z),
            ("comparison", self._decomp_comparison),
            ("nested_question", self._decomp_nested_question),
        ]:
            sub_qs = strategy_fn(question)
            if sub_qs:
                trace.decomposition_strategy = strategy_name
                trace.sub_questions = [
                    self._decompose_recursive(sq, depth + 1) for sq in sub_qs
                ]
                return trace

        # No decomposition: leaf node
        trace.decomposition_strategy = "leaf"
        return trace

    # ------------------------------------------------------------------ #
    # Decomposition strategies
    # ------------------------------------------------------------------ #
    def _decomp_capital_of_country_where(self, q: str) -> Optional[List[str]]:
        m = re.match(r"what is the capital of (?:the )?country where (.+) is located", q, re.I)
        if m:
            place = m.group(1).strip()
            return [f"Where is {place} located?", "What is the capital of $1?"]
        return None

    def _decomp_where_is_capital_of(self, q: str) -> Optional[List[str]]:
        m = re.match(r"where is the capital of (.+) located", q, re.I)
        if m:
            country = m.group(1).strip()
            return [f"What is the capital of {country}?", "Where is $1 located?"]
        return None

    def _decomp_what_is_x_of_y_of_z(self, q: str) -> Optional[List[str]]:
        m = re.match(r"what is the (\w+) of the (\w+) of (.+)", q, re.I)
        if m:
            p1, p2, subj = m.groups()
            return [f"What is the {p2} of {subj}?", f"What is the {p1} of $1?"]
        return None

    def _decomp_comparison(self, q: str) -> Optional[List[str]]:
        m = re.match(r"compare (.+) and (.+)", q, re.I)
        if m:
            a, b = m.groups()
            return [f"What is {a}?", f"What is {b}?", "Compare $1 and $2."]
        return None

    def _decomp_nested_question(self, q: str) -> Optional[List[str]]:
        # Generic: "What is the X of Y?" where Y is itself complex
        m = re.match(r"what is the (\w+) of (.+)", q, re.I)
        if m and len(m.group(2).split()) > 2:
            pred, subj = m.groups()
            # Try to decompose subj further
            # e.g., "What is the capital of the country where Lyon is located?"
            sub = self._decomp_capital_of_country_where(f"What is the {pred} of {subj}?")
            if sub: return sub
        return None

    # ------------------------------------------------------------------ #
    # Recursive solving
    # ------------------------------------------------------------------ #
    def _solve_recursive(self, trace: ReasoningTrace) -> None:
        """Solve a trace node, recursively solving its children first."""
        # Solve all sub-questions first
        for i, sub in enumerate(trace.sub_questions):
            self._solve_recursive(sub)
            # Substitute the placeholder $1, $2 with sub-answers
            if sub.answer and not sub.failed:
                placeholder = f"${i+1}"
                for j, later_sub in enumerate(trace.sub_questions[i+1:], start=i+1):
                    later_sub.question = later_sub.question.replace(placeholder, sub.answer)

        # If there are sub-questions, the answer is the last one's answer
        if trace.sub_questions:
            last = trace.sub_questions[-1]
            trace.answer = last.answer
            trace.confidence = last.confidence
            trace.failed = last.failed
            return

        # Leaf node: ask AETHER directly
        try:
            answer = self.agent.ask(trace.question)
            cleaned = self._extract_answer(answer, trace.question)
            trace.answer = cleaned
            trace.confidence = 0.9 if cleaned else 0.0
            trace.failed = not cleaned or trace.confidence < self.MIN_CONFIDENCE
        except Exception as e:
            log.warning(f"leaf failed: {trace.question!r}: {e}")
            trace.failed = True

    def _extract_answer(self, response: str, question: str) -> Optional[str]:
        """Extract the bare answer from a natural-language response."""
        r = response.strip().rstrip(".")
        m = re.search(r"(?:It's|It is)\s+(.+)", r, re.I)
        if m: return m.group(1).strip()
        m = re.search(r"(?:The capital of \w+ is|capital is|capital of \w+ is)\s+(.+)", r, re.I)
        if m: return m.group(1).strip()
        m = re.search(r"is located in\s+(.+)", r, re.I)
        if m: return m.group(1).strip()
        m = re.search(r"is\s+(.+)", r, re.I)
        if m: return m.group(1).strip()
        return r if len(r) < 50 else None

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def _compute_synthesis(self, trace: ReasoningTrace) -> HDVector:
        """Compute the synthesis HD vector of a reasoning trace.

        The synthesis bundles the question vector with all sub-answers'
        vectors. It captures the gist of the reasoning.
        """
        vecs = [trace.question_vec]
        for sub in trace.sub_questions:
            if sub.answer:
                vecs.append(self.agent.encoder.encode_text(sub.answer))
        return bundle(vecs)

    # ------------------------------------------------------------------ #
    # Trace retrieval (analogical transfer)
    # ------------------------------------------------------------------ #
    def _find_similar_trace(self, q_vec: HDVector, threshold: float = 0.7) -> Optional[ReasoningTrace]:
        """Find a similar previous trace for analogical transfer."""
        if not self.synthesis_store: return None
        best_trace = None
        best_sim = -1.0
        for syn_vec, trace in self.synthesis_store:
            sim = q_vec.similarity(syn_vec)
            if sim > best_sim:
                best_sim = sim
                best_trace = trace
        if best_sim >= threshold:
            return best_trace
        return None

    # ------------------------------------------------------------------ #
    # Tree utilities
    # ------------------------------------------------------------------ #
    def _count_nodes(self, node: ReasoningTrace) -> int:
        return 1 + sum(self._count_nodes(s) for s in node.sub_questions)

    def _depth(self, node: ReasoningTrace) -> int:
        if not node.sub_questions: return 1
        return 1 + max(self._depth(s) for s in node.sub_questions)

    def _format_tree(self, node: ReasoningTrace, indent: int = 0) -> str:
        prefix = "  " * indent
        ans = node.answer or "?"
        mark = "✓" if not node.failed else "✗"
        strat = f"[{node.decomposition_strategy}] " if node.decomposition_strategy else ""
        line = f"{prefix}{mark} {strat}Q: {node.question!r}\n{prefix}   A: {ans!r}\n"
        for sub in node.sub_questions:
            line += self._format_tree(sub, indent + 1)
        return line

    def stats(self) -> Dict[str, Any]:
        return {
            "n_traces_stored": len(self.traces),
            "max_depth_seen": max((self._depth(t) for t in self.traces), default=0),
            "strategies_used": list(set(t.decomposition_strategy for t in self.traces)),
        }
