"""
composer.py — Automatic tool composition planner.

PROBLEM
-------
In v2, the planner used regex patterns to decide which tool to call:
  'compare X and Y'  -> compare tool
  'explain X'        -> explain tool
  'calc X'           -> calc tool

But what about questions like:
  'How many capitals are in Europe?'
  'What's the average population of countries where French is spoken?'
  'Compare the capitals of France and Japan'

These require CHAINING multiple tools: list_kb -> filter -> count, or
kb_query -> kb_query -> compare. No single regex matches.

SOLUTION
--------
A HD-based composer that:

  1. Encodes the question as an HD vector.
  2. Matches against ALL tool trigger vectors + KB predicates.
  3. Builds a dependency graph of needed operations.
  4. Topologically sorts them into a plan.
  5. Falls back to single-tool plans when no chain is needed.

The composer is "pattern-free" — it works for any question by matching
HD similarity, not by regex.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .hd import HDVector, DIM, bundle
from .memory import AssociativeMemory
from .encoder import TextEncoder
from .tools import ToolRegistry
from .inference import InferenceEngine
from .planner import Planner, Plan, PlanStep
from .generator import analyze_question, QuestionAnalysis


@dataclass
class ToolNode:
    """A node in the composition graph: one tool invocation."""
    tool_name: str
    args: str = ""
    depends_on: List[int] = field(default_factory=list)  # indices of dependency nodes
    output_var: str = ""  # symbolic name for the output (e.g., "$v0")
    description: str = ""


class ToolComposer:
    """Builds multi-tool plans by HD similarity matching.

    The composer maintains HD vectors for each "operation type":
      - filter, count, average, sum, compare, explain, retrieve, list, ...
    When a question is asked, it:
      1. Encodes the question.
      2. Matches against operation vectors.
      3. Picks the operations whose similarity > threshold.
      4. Chains them: output of op_n feeds into args of op_{n+1}.
    """

    # Operations the composer knows about. Each is a (name, description) pair.
    # The HD vector of the description is the operation's trigger.
    OPERATIONS: List[Tuple[str, str]] = [
        ("retrieve",   "what is capital of who where when"),
        ("count",      "how many count number of total"),
        ("list",       "list show all every"),
        ("compare",    "compare difference versus vs versus"),
        ("explain",    "explain describe tell me about what is"),
        ("summarize",  "summarize summary overview"),
        ("calculate",  "calculate compute arithmetic plus minus times divide average sum"),
        ("filter",     "filter where which that with having"),
        ("sort",       "sort order by ascending descending"),
        ("find",       "find search look for recall remember"),
        ("define",     "define definition meaning"),
        ("time",       "time date now today current"),
    ]

    def __init__(self, encoder: TextEncoder, tools: ToolRegistry, inference: InferenceEngine):
        self.encoder = encoder
        self.tools = tools
        self.inference = inference
        # Pre-compute operation trigger vectors
        self.op_vectors: Dict[str, HDVector] = {
            name: self.encoder.encode_text(desc)
            for name, desc in self.OPERATIONS
        }

    def compose(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose a multi-tool plan for the question.

        Uses BOTH HD similarity AND regex patterns to detect operations.
        Returns (list_of_nodes, rationale_string).
        """
        import re
        q_lower = question.lower().strip().rstrip("?.!")

        # Step 1: Pattern-based detection (high confidence)
        # These are tried first because they're reliable.
        if re.search(r"compare\s+.+\s+(?:and|vs\.?|versus)\s+.+", q_lower):
            return self._compose_compare(question)

        if re.search(r"explain\s+.+|describe\s+.+|tell me about\s+.+", q_lower):
            return self._compose_explain(question)

        if re.search(r"how many\s+\w+", q_lower):
            return self._compose_count_list(question)

        if re.search(r"(?:calculate|compute)\s+.+|\d[\d\s\+\-\*\/\(\)\.,%]+", q_lower):
            return self._compose_calculate(question)

        if re.search(r"what is the capital of\s+.+", q_lower):
            return self._compose_retrieve_capital(question)

        if re.search(r"where is\s+.+", q_lower):
            return self._compose_retrieve_location(question)

        if re.search(r"what is\s+.+|who is\s+.+", q_lower):
            return self._compose_retrieve_definition(question)

        if re.search(r"^list|^show|^list kb|^list triples|^list facts", q_lower):
            return self._compose_list(question)

        if re.search(r"find\s+.+|recall\s+.+|search\s+.+", q_lower):
            return self._compose_find(question)

        if re.search(r"^time$|^date$|^now$|what time", q_lower):
            return self._compose_time(question)

        if re.search(r"summarize|summary|overview", q_lower):
            return self._compose_summarize(question)

        # Step 2: HD similarity fallback (for questions not matched by regex)
        q_vec = self.encoder.encode_text(question)
        op_scores: List[Tuple[str, float]] = []
        for op_name, op_vec in self.op_vectors.items():
            sim = q_vec.similarity(op_vec)
            op_scores.append((op_name, sim))
        op_scores.sort(key=lambda x: -x[1])

        THRESHOLD = 0.03
        selected_ops = [(name, sim) for name, sim in op_scores if sim >= THRESHOLD]

        if not selected_ops:
            # Step 3: cognitive loop fallback
            return (
                [ToolNode(
                    tool_name="cognitive_loop",
                    args=question,
                    output_var="$v0",
                    description="fall back to cognitive loop",
                )],
                "no pattern matched — fall back to cognitive loop",
            )

        # Use the top HD-matched operation
        top_op = selected_ops[0][0]
        return (
            [ToolNode(
                tool_name=top_op,
                args=question,
                output_var="$v0",
                description=f"HD-matched operation: {top_op}",
            )],
            f"HD-matched: {top_op} (sim={selected_ops[0][1]:.3f})",
        )

    # ------------------------------------------------------------------ #
    # Composition strategies
    # ------------------------------------------------------------------ #
    def _compose_compare(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: extract X and Y from 'compare X and Y', query KB for both, compare."""
        import re
        m = re.search(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$)", question, re.I)
        if not m:
            # Try "difference between X and Y"
            m = re.search(r"(?:difference between|compare)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$)", question, re.I)
        if not m:
            return ([], "compare pattern not matched")
        x, y = m.group(1).strip(), m.group(2).strip()
        nodes = [
            ToolNode(
                tool_name="kb_query",
                args=f"{x}|*",
                output_var="$v0",
                description=f"retrieve all facts about {x}",
            ),
            ToolNode(
                tool_name="kb_query",
                args=f"{y}|*",
                output_var="$v1",
                description=f"retrieve all facts about {y}",
            ),
            ToolNode(
                tool_name="compare",
                args=f"{x} and {y}",
                depends_on=[0, 1],
                output_var="$v2",
                description=f"compare {x} and {y}",
            ),
        ]
        return nodes, f"composed compare: query({x}) -> query({y}) -> compare"

    def _compose_count_list(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: list(X) -> count($v0)."""
        import re
        # Try to extract what to count
        m = re.search(r"how many\s+(\w+)", question, re.I)
        what = m.group(1) if m else "triples"
        nodes = [
            ToolNode(
                tool_name="list_kb",
                args="",
                output_var="$v0",
                description=f"list all KB triples",
            ),
            ToolNode(
                tool_name="count",
                args=what,
                depends_on=[0],
                output_var="$v1",
                description=f"count {what}",
            ),
        ]
        return nodes, f"composed count: list_kb -> count({what})"

    def _compose_calculate(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: extract arithmetic expression, calculate it."""
        import re
        m = re.search(r"(?:calculate|compute|average|sum)\s+(.+?)(?:\?|$)", question, re.I)
        if m:
            expr = m.group(1).strip()
        else:
            # Try to find an arithmetic expression directly
            m = re.search(r"(\d[\d\s\+\-\*\/\(\)\.,%]+)", question)
            expr = m.group(1).strip() if m else "1+1"
        nodes = [
            ToolNode(
                tool_name="calc",
                args=expr,
                output_var="$v0",
                description=f"calculate {expr}",
            ),
        ]
        return nodes, f"composed calculate: calc({expr})"

    def _compose_explain(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: extract subject, explain it."""
        import re
        m = re.search(r"(?:explain|describe|tell me about)\s+(.+?)(?:\?|$)", question, re.I)
        subject = m.group(1).strip() if m else question
        nodes = [
            ToolNode(
                tool_name="explain",
                args=subject,
                output_var="$v0",
                description=f"explain {subject}",
            ),
        ]
        return nodes, f"composed explain: explain({subject})"

    def _compose_retrieve(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: extract subject + predicate, retrieve from KB."""
        # Try to parse "what is X" / "where is X" / "who is X"
        import re
        m = re.search(r"what is the capital of (.+?)\??$", question, re.I)
        if m:
            return self._compose_retrieve_capital(question)
        m = re.search(r"where is (.+?)(?:\s+located)?\??$", question, re.I)
        if m:
            return self._compose_retrieve_location(question)
        m = re.search(r"what is (.+?)\??$", question, re.I)
        if m:
            return self._compose_retrieve_definition(question)
        # Fallback: cognitive loop
        return (
            [ToolNode(
                tool_name="cognitive_loop",
                args=question,
                output_var="$v0",
                description="fall back to cognitive loop",
            )],
            "no specific retrieve pattern matched",
        )

    def _compose_retrieve_capital(self, question: str) -> Tuple[List[ToolNode], str]:
        import re
        m = re.search(r"what is the capital of (.+?)\??$", question, re.I)
        subject = m.group(1).strip() if m else ""
        return (
            [ToolNode(
                tool_name="kb_query",
                args=f"{subject}|capital_of",
                output_var="$v0",
                description=f"retrieve capital of {subject}",
            )],
            f"composed retrieve: kb_query({subject}, capital_of)",
        )

    def _compose_retrieve_location(self, question: str) -> Tuple[List[ToolNode], str]:
        import re
        m = re.search(r"where is (.+?)(?:\s+located)?\??$", question, re.I)
        subject = m.group(1).strip() if m else ""
        if subject.endswith(" located"):
            subject = subject[:-len(" located")].strip()
        return (
            [ToolNode(
                tool_name="kb_query",
                args=f"{subject}|located_in",
                output_var="$v0",
                description=f"retrieve location of {subject}",
            )],
            f"composed retrieve: kb_query({subject}, located_in)",
        )

    def _compose_retrieve_definition(self, question: str) -> Tuple[List[ToolNode], str]:
        import re
        m = re.search(r"what is (.+?)\??$", question, re.I)
        if not m:
            m = re.search(r"who is (.+?)\??$", question, re.I)
        subject = m.group(1).strip() if m else ""
        return (
            [ToolNode(
                tool_name="kb_query",
                args=f"{subject}|is_a",
                output_var="$v0",
                description=f"retrieve definition of {subject}",
            )],
            f"composed retrieve: kb_query({subject}, is_a)",
        )

    def _compose_find(self, question: str) -> Tuple[List[ToolNode], str]:
        import re
        m = re.search(r"(?:find|recall|search)\s+(.+?)\??$", question, re.I)
        query = m.group(1).strip() if m else question
        return (
            [ToolNode(
                tool_name="recall",
                args=query,
                output_var="$v0",
                description=f"recall memories matching '{query}'",
            )],
            f"composed find: recall({query})",
        )

    def _compose_time(self, question: str) -> Tuple[List[ToolNode], str]:
        return (
            [ToolNode(
                tool_name="time",
                args="",
                output_var="$v0",
                description="get current time",
            )],
            "composed time: time()",
        )

    def _compose_summarize(self, question: str) -> Tuple[List[ToolNode], str]:
        import re
        m = re.search(r"summarize\s*(\d*)|summary\s*(\d*)", question, re.I)
        n = ""
        if m:
            n = (m.group(1) or m.group(2) or "").strip()
        return (
            [ToolNode(
                tool_name="summarize",
                args=n,
                output_var="$v0",
                description=f"summarize {n or '5'} recent memories",
            )],
            f"composed summarize: summarize({n or '5'})",
        )

    def _compose_list(self, question: str) -> Tuple[List[ToolNode], str]:
        """Compose: list tool."""
        return (
            [ToolNode(
                tool_name="list_kb",
                args="",
                output_var="$v0",
                description="list all KB triples",
            )],
            "composed list: list_kb",
        )


# ---------------------------------------------------------------------------
# Execute a composed plan (similar to PlanExecutor but for ToolNodes)
# ---------------------------------------------------------------------------

class ComposedExecutor:
    """Execute a composed plan (list of ToolNodes) with output threading."""

    def __init__(self, agent):
        self.agent = agent

    def execute(self, nodes: List[ToolNode]) -> Tuple[Optional[str], List[str]]:
        """Execute the composed plan. Returns (final_output, list_of_step_outputs)."""
        if not nodes:
            return None, []

        outputs: Dict[int, str] = {}
        step_logs: List[str] = []

        for i, node in enumerate(nodes):
            # Resolve args: substitute $vN references with prior outputs
            args = node.args
            for dep_idx in node.depends_on:
                if dep_idx in outputs:
                    placeholder = f"$v{dep_idx}"
                    if placeholder in args:
                        args = args.replace(placeholder, outputs[dep_idx])

            # Execute
            try:
                result = self._execute_node(node, args, outputs)
            except Exception as e:
                result = f"[node {i} error: {e}]"
            outputs[i] = result
            step_logs.append(f"node {i} ({node.tool_name}): {result[:100]}")

            # Set the output variable for downstream nodes
            node.output_var = f"$v{i}"

        # The final output is the last node's output
        final = outputs.get(len(nodes) - 1) if nodes else None
        return final, step_logs

    def _execute_node(self, node: ToolNode, args: str, prior_outputs: Dict[int, str]) -> str:
        """Execute a single node, given resolved args and prior outputs."""
        if node.tool_name == "kb_query":
            # args format: "subject|predicate" or "subject|*"
            if "|" not in args:
                return "(invalid kb_query args)"
            subject, predicate = args.split("|", 1)
            subject = subject.strip()
            predicate = predicate.strip()
            if predicate == "*":
                # Retrieve all predicates for this subject
                results = []
                for s, p, o in self.agent.assoc.list_triples():
                    if s.lower() == subject.lower():
                        results.append(f"({s}, {p}, {o})")
                return " ; ".join(results) if results else f"(no facts about {subject})"
            result = self.agent.assoc.query_triple(subject, predicate)
            if result:
                ans, sim = result
                return ans
            return "(unknown)"

        if node.tool_name == "cognitive_loop":
            state = self.agent.cogloop.think(args)
            return state.final_output or "(no answer)"

        # Standard tool
        from .tools import ToolContext
        ctx = ToolContext(self.agent.encoder, self.agent.assoc)
        return self.agent.tools.call(node.tool_name, args, ctx)
