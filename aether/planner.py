"""
planner.py — Agentic task planning and decomposition.

For complex queries that no single tool or pattern can answer, the planner
decomposes the request into a sequence of subtasks, each of which is either:
  - a tool call
  - a KB query
  - a sub-question to the cognitive loop

Plans are executed step-by-step; each step's output is fed as input to the
next step (chain-of-thought in hyperdimensional space).

Supported plan types:
  - "direct"         : 1 step, no decomposition
  - "multi_hop"      : chained KB lookups (n hops)
  - "tool_then_think": run a tool, feed result to cognitive loop
  - "compare"        : query both sides of a comparison, then summarize
  - "explain"        : retrieve a fact, generate a natural-language explanation
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .inference import InferenceEngine, Proof
from .generator import analyze_question, QuestionAnalysis


@dataclass
class PlanStep:
    """One step in an execution plan."""
    kind: str  # "kb_query" | "tool" | "sub_question" | "explain" | "compare"
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Plan:
    """A complete execution plan."""
    steps: List[PlanStep] = field(default_factory=list)
    rationale: str = ""
    expected_complexity: str = "low"  # low | medium | high


class Planner:
    """Decompose a user request into executable steps."""

    def __init__(self, inference: InferenceEngine):
        self.inference = inference

    def plan(self, question: str, analysis: Optional[QuestionAnalysis] = None) -> Plan:
        """Decide on a plan for the question."""
        if analysis is None:
            analysis = analyze_question(question)

        # 'teach' is intercepted by the agent itself, but if we get here,
        # pass it through as a teach step
        if analysis.qtype == "teach":
            fact = analysis.slots.get("fact", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="teach",
                    args={"fact": fact},
                    description=f"learn fact: {fact}",
                )],
                rationale="teach command",
            )

        # Multi-hop capital query: "What is the capital of the country where X is located?"
        if analysis.qtype == "multi_hop_capital":
            place = analysis.slots.get("place", "").strip()
            return Plan(
                steps=[
                    PlanStep(
                        kind="kb_query",
                        args={"subject": place, "predicate": "located_in"},
                        description=f"hop 1: where is {place} located?",
                    ),
                    PlanStep(
                        kind="kb_query",
                        args={"subject": "$prev", "predicate": "capital_of"},
                        description="hop 2: capital of that country?",
                    ),
                ],
                rationale=f"two-hop reasoning ({place} -> country -> capital)",
                expected_complexity="medium",
            )

        # Multi-hop location query: "Where is the capital of X located?"
        if analysis.qtype == "multi_hop_location":
            country = analysis.slots.get("country", "").strip()
            return Plan(
                steps=[
                    PlanStep(
                        kind="kb_query",
                        args={"subject": country, "predicate": "capital_of"},
                        description=f"hop 1: what is the capital of {country}?",
                    ),
                    PlanStep(
                        kind="kb_query",
                        args={"subject": "$prev", "predicate": "located_in"},
                        description="hop 2: where is that capital located?",
                    ),
                ],
                rationale=f"two-hop reasoning ({country} -> capital -> location)",
                expected_complexity="medium",
            )

        # Comparison query
        if analysis.qtype == "compare":
            x = analysis.slots.get("x", "").strip()
            y = analysis.slots.get("y", "").strip()
            args = f"{x} and {y}"
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "compare", "args": args},
                    description=f"compare {x} and {y}",
                )],
                rationale="compare tool",
            )

        # Direct question patterns — single-step plans
        if analysis.qtype == "capital_of":
            country = analysis.slots.get("country", "").strip()
            if not country:
                return Plan(steps=[], rationale="no country extracted")
            return Plan(
                steps=[PlanStep(
                    kind="kb_query",
                    args={"subject": country, "predicate": "capital_of"},
                    description=f"look up capital of {country}",
                )],
                rationale="direct capital-of lookup",
            )

        if analysis.qtype == "located_in":
            subject = analysis.slots.get("subject", "").strip()
            if subject.endswith(" located"):
                subject = subject[:-len(" located")].strip()
            if not subject:
                return Plan(steps=[], rationale="no subject extracted")
            return Plan(
                steps=[PlanStep(
                    kind="kb_query",
                    args={"subject": subject, "predicate": "located_in"},
                    description=f"look up location of {subject}",
                )],
                rationale="direct located-in lookup",
            )

        if analysis.qtype == "definition":
            subject = analysis.slots.get("subject", "").strip()
            if not subject:
                return Plan(steps=[], rationale="no subject extracted")
            return Plan(
                steps=[PlanStep(
                    kind="kb_query",
                    args={"subject": subject, "predicate": "is_a"},
                    description=f"look up definition of {subject}",
                )],
                rationale="direct is_a lookup",
            )

        if analysis.qtype == "predicate_of":
            predicate = analysis.slots.get("predicate", "").strip()
            subject = analysis.slots.get("subject", "").strip()
            pred_map = {
                "capital": "capital_of",
                "location": "located_in",
                "type": "is_a",
                "color": "color",
                "size": "size",
            }
            predicate = pred_map.get(predicate, predicate)
            return Plan(
                steps=[PlanStep(
                    kind="kb_query",
                    args={"subject": subject, "predicate": predicate},
                    description=f"look up {predicate} of {subject}",
                )],
                rationale="direct predicate lookup",
            )

        # Tool calls — single-step plans
        if analysis.qtype == "calc":
            expr = analysis.slots.get("expr", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "calc", "args": expr},
                    description=f"calculate {expr}",
                )],
                rationale="calculator tool",
            )

        if analysis.qtype == "time":
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "time", "args": ""},
                    description="get current time",
                )],
                rationale="time tool",
            )

        if analysis.qtype == "recall":
            query = analysis.slots.get("query", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "recall", "args": query},
                    description=f"recall memories matching {query!r}",
                )],
                rationale="recall tool",
            )

        if analysis.qtype == "list_kb":
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "list_kb", "args": ""},
                    description="list knowledge base",
                )],
                rationale="list_kb tool",
            )

        if analysis.qtype == "explain":
            subject = analysis.slots.get("subject", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "explain", "args": subject},
                    description=f"explain {subject}",
                )],
                rationale="explain tool",
            )

        if analysis.qtype == "compare":
            x = analysis.slots.get("x", "").strip()
            y = analysis.slots.get("y", "").strip()
            args = f"{x} and {y}"
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "compare", "args": args},
                    description=f"compare {x} and {y}",
                )],
                rationale="compare tool",
            )

        if analysis.qtype == "summarize":
            n = analysis.slots.get("n", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "summarize", "args": n},
                    description="summarize recent memories",
                )],
                rationale="summarize tool",
            )

        if analysis.qtype == "count":
            what = analysis.slots.get("what", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "count", "args": what},
                    description=f"count {what}",
                )],
                rationale="count tool",
            )

        if analysis.qtype == "define":
            subject = analysis.slots.get("subject", "").strip()
            return Plan(
                steps=[PlanStep(
                    kind="tool",
                    args={"tool": "define", "args": subject},
                    description=f"define {subject}",
                )],
                rationale="define tool",
            )

        # Meta question types — direct response, no plan
        if analysis.qtype in ("greeting", "farewell", "thanks", "help", "stats",
                              "identity", "capabilities", "self_explain"):
            return Plan(
                steps=[PlanStep(
                    kind="meta",
                    args={"qtype": analysis.qtype},
                    description=f"respond to {analysis.qtype}",
                )],
                rationale="meta question",
            )

        # Multi-hop detection (already handled above via analyze_question — kept as fallback)
        # (no-op; multi_hop_capital and multi_hop_location are handled at the top of this method)

        # Fallback: try cognitive loop
        return Plan(
            steps=[PlanStep(
                kind="cognitive_loop",
                args={"question": question},
                description="run cognitive loop",
            )],
            rationale="no pattern matched — fall back to cognitive loop",
            expected_complexity="high",
        )


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------

class PlanExecutor:
    """Execute a plan step by step, threading results between steps."""

    def __init__(self, agent):
        # agent is the AETHER instance — passed in to avoid circular imports
        self.agent = agent

    def execute(self, plan: Plan) -> Tuple[Optional[str], List[str]]:
        """Execute the plan. Returns (final_answer, list_of_step_outputs)."""
        if not plan.steps:
            return None, []

        prev_output: Optional[str] = None
        outputs: List[str] = []

        for i, step in enumerate(plan.steps):
            try:
                step_output = self._execute_step(step, prev_output)
            except Exception as e:
                step_output = f"[step {i+1} error: {e}]"
            outputs.append(f"step {i+1} ({step.kind}): {step_output}")
            prev_output = step_output

        return prev_output, outputs

    def _execute_step(self, step: PlanStep, prev_output: Optional[str]) -> str:
        if step.kind == "kb_query":
            subject = step.args.get("subject", "")
            predicate = step.args.get("predicate", "")
            # Substitute $prev from previous step
            if subject == "$prev" and prev_output:
                subject = prev_output
            if predicate == "*":
                # Retrieve all known predicates for this subject
                results = []
                for s, p, o in self.agent.assoc.list_triples():
                    if s.lower() == subject.lower():
                        results.append(f"({s}, {p}, {o})")
                return " ; ".join(results) if results else f"(no facts about {subject})"
            # Normal lookup — use inference.lookup which verifies against
            # the explicit triples list (avoids SDM hallucination)
            result = self.agent.inference.lookup(subject, predicate)
            if result:
                ans, sim = result
                return ans
            return "(unknown)"

        if step.kind == "teach":
            fact = step.args.get("fact", "")
            msg = self.agent.teach(fact, silent=True)
            return msg

        if step.kind == "tool":
            from .tools import ToolContext
            ctx = ToolContext(self.agent.encoder, self.agent.assoc)
            return self.agent.tools.call(step.args["tool"], step.args.get("args", ""), ctx)

        if step.kind == "meta":
            # Handled by the generator with templates
            return step.args.get("qtype", "")

        if step.kind == "compare":
            # prev_output already contains the retrieved facts for both
            return prev_output or "(nothing to compare)"

        if step.kind == "cognitive_loop":
            state = self.agent.cogloop.think(step.args["question"])
            return state.final_output or "(no answer)"

        if step.kind == "explain":
            return prev_output or ""

        return f"(unknown step kind: {step.kind})"
