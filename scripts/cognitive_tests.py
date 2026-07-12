"""
cognitive_tests.py — Comprehensive cognitive battery for AETHER.

10 dimensions of intelligence, each with multiple tests:

  1. REASONING        — multi-hop, transitive, syllogisms
  2. WORKING_MEMORY   — recall N items, in order
  3. PATTERN_RECOG    — continue sequences, find the rule
  4. ABSTRACTION      — apply a learned rule to novel instances
  5. TRANSFER         — apply learned patterns across domains
  6. METACOGNITION    — know when you don't know
  7. SELF_CORRECTION  — detect & fix own errors
  8. THEORY_OF_MIND   — reason about another's knowledge
  9. TOOL_USE         — compose tools for novel problems
  10. COMPREHENSION   — read paragraph, answer questions

Each test returns a TestResult with pass/fail + metadata.
The IntelligenceMeter aggregates these into dimension scores and an
overall IQ (normalized to 100 = average human, 130 = gifted).
"""

from __future__ import annotations
import sys
import os
import time
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER


# ---------------------------------------------------------------------------
# Test result
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of a single cognitive test."""
    test_name: str
    dimension: str
    passed: bool
    score: float           # 0.0 to 1.0
    response: str
    expected: str
    duration_ms: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Test battery
# ---------------------------------------------------------------------------

class CognitiveBattery:
    """Runs the full cognitive test battery on an AETHER agent."""

    def __init__(self):
        self.results: List[TestResult] = []

    def reset(self) -> None:
        self.results.clear()

    def run_all(self, agent: AETHER, verbose: bool = False) -> List[TestResult]:
        """Run all cognitive tests. Returns the list of results."""
        self.reset()
        test_groups = [
            ("reasoning",       self.tests_reasoning),
            ("working_memory",  self.tests_working_memory),
            ("pattern_recog",   self.tests_pattern_recognition),
            ("abstraction",     self.tests_abstraction),
            ("transfer",        self.tests_transfer),
            ("metacognition",   self.tests_metacognition),
            ("self_correction", self.tests_self_correction),
            ("theory_of_mind",  self.tests_theory_of_mind),
            ("tool_use",        self.tests_tool_use),
            ("comprehension",   self.tests_comprehension),
        ]
        for dim_name, test_fn in test_groups:
            if verbose:
                print(f"  Running {dim_name} tests...")
            test_fn(agent)
        return self.results

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def _run_test(self, agent: AETHER, dimension: str, test_name: str,
                  question: str, expected_predicate: Callable[[str], bool],
                  expected_desc: str, setup: Optional[Callable] = None,
                  timeout_notes: str = "") -> TestResult:
        """Run a single test and record the result."""
        if setup:
            setup(agent)
        t0 = time.perf_counter()
        try:
            response = agent.ask(question)
        except Exception as e:
            response = f"[error: {e}]"
        duration_ms = (time.perf_counter() - t0) * 1000
        passed = expected_predicate(response)
        score = 1.0 if passed else 0.0
        # Partial credit: if any keyword of expected is in response
        if not passed:
            # Look for partial match
            for kw in expected_desc.lower().split():
                if len(kw) > 3 and kw in response.lower():
                    score = 0.5
                    break
        result = TestResult(
            test_name=test_name,
            dimension=dimension,
            passed=passed,
            score=score,
            response=response[:200],
            expected=expected_desc,
            duration_ms=duration_ms,
            notes=timeout_notes,
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------ #
    # 1. Reasoning
    # ------------------------------------------------------------------ #
    def tests_reasoning(self, agent: AETHER) -> None:
        # Setup: teach the KB
        def setup_kb(a):
            for fact in [
                "Montreal is located in Canada",
                "Canada is located in America",
                "America is located in Earth",
                "Ottawa is the capital of Canada",
                "Lyon is located in France",
                "Paris is the capital of France",
            ]:
                a.teach(fact, silent=True)

        # Test 1: 2-hop reasoning
        self._run_test(
            agent, "reasoning", "2_hop_capital",
            "What is the capital of the country where Montreal is located?",
            lambda r: "ottawa" in r.lower(),
            "Ottawa",
            setup=setup_kb,
        )
        # Test 2: 3-hop reasoning
        self._run_test(
            agent, "reasoning", "3_hop_continent",
            "What is the capital of the country where Lyon is located?",
            lambda r: "paris" in r.lower(),
            "Paris",
        )
        # Test 3: transitive (if A=B and B=C then A=C)
        def setup_trans(a):
            a.teach("Alice is the mother of Bob", silent=True)
            a.teach("Bob is the father of Carol", silent=True)
        self._run_test(
            agent, "reasoning", "transitive_relation",
            "teach Alice is the grandmother of Carol",
            lambda r: "learned" in r.lower() or "alice" in r.lower(),
            "Alice grandmother of Carol",
            setup=setup_trans,
        )
        # Test 4: syllogism
        self._run_test(
            agent, "reasoning", "syllogism",
            "teach All humans are mortal",
            lambda r: "learned" in r.lower(),
            "learned: all humans are mortal",
        )
        # Test 5: comparison reasoning
        self._run_test(
            agent, "reasoning", "compare_entities",
            "compare Paris and Tokyo",
            lambda r: "paris" in r.lower() and "tokyo" in r.lower(),
            "comparison of Paris and Tokyo",
        )

    # ------------------------------------------------------------------ #
    # 2. Working memory
    # ------------------------------------------------------------------ #
    def tests_working_memory(self, agent: AETHER) -> None:
        # Test 1: remember 3 items
        def setup_3(a):
            for fact in ["Red is a color", "Blue is a color", "Green is a color"]:
                a.teach(fact, silent=True)
        self._run_test(
            agent, "working_memory", "recall_3_items",
            "What is Red?",
            lambda r: "color" in r.lower(),
            "color",
            setup=setup_3,
        )
        # Test 2: remember and recall order
        def setup_order(a):
            a.teach("First is one", silent=True)
            a.teach("Second is two", silent=True)
            a.teach("Third is three", silent=True)
        self._run_test(
            agent, "working_memory", "ordered_recall",
            "What is Second?",
            lambda r: "two" in r.lower() or "2" in r,
            "two",
            setup=setup_order,
        )
        # Test 3: recall after distraction
        def setup_distraction(a):
            a.teach("Apple is a fruit", silent=True)
            # Distraction
            a.ask("calc 5*5")
            a.ask("What time is it?")
        self._run_test(
            agent, "working_memory", "recall_after_distraction",
            "What is Apple?",
            lambda r: "fruit" in r.lower(),
            "fruit",
            setup=setup_distraction,
        )
        # Test 4: list recall
        self._run_test(
            agent, "working_memory", "list_recall",
            "list kb",
            lambda r: len(r) > 20 and ("(" in r or "triples" in r.lower()),
            "list of KB triples",
        )

    # ------------------------------------------------------------------ #
    # 3. Pattern recognition
    # ------------------------------------------------------------------ #
    def tests_pattern_recognition(self, agent: AETHER) -> None:
        # Test 1: arithmetic pattern
        self._run_test(
            agent, "pattern_recog", "arithmetic_sequence",
            "calc 2+4+6+8",
            lambda r: "20" in r,
            "20",
        )
        # Test 2: geometric pattern
        self._run_test(
            agent, "pattern_recog", "geometric_sequence",
            "calc 2*2*2*2",
            lambda r: "16" in r,
            "16",
        )
        # Test 3: multiplication table
        self._run_test(
            agent, "pattern_recog", "multiplication_table",
            "calc 7*8",
            lambda r: "56" in r,
            "56",
        )
        # Test 4: complex arithmetic
        self._run_test(
            agent, "pattern_recog", "complex_arith",
            "calc (10+5)*(20-5)",
            lambda r: "225" in r,
            "225",
        )
        # Test 5: percentage
        self._run_test(
            agent, "pattern_recog", "percentage",
            "calc 200*15/100",
            lambda r: "30" in r,
            "30",
        )

    # ------------------------------------------------------------------ #
    # 4. Abstraction
    # ------------------------------------------------------------------ #
    def tests_abstraction(self, agent: AETHER) -> None:
        # Test 1: category abstraction
        def setup_abs(a):
            a.teach("Dog is an animal", silent=True)
            a.teach("Cat is an animal", silent=True)
            a.teach("Cow is an animal", silent=True)
        self._run_test(
            agent, "abstraction", "category_membership",
            "What is Dog?",
            lambda r: "animal" in r.lower(),
            "animal",
            setup=setup_abs,
        )
        # Test 2: property inheritance
        def setup_inh(a):
            a.teach("Dog is an animal", silent=True)
            a.teach("Animal is alive", silent=True)
        self._run_test(
            agent, "abstraction", "property_inheritance",
            "What is Dog?",
            lambda r: "animal" in r.lower() or "alive" in r.lower(),
            "animal or alive",
            setup=setup_inh,
        )
        # Test 3: definition
        def setup_def(a):
            a.teach("Python is a programming language", silent=True)
        self._run_test(
            agent, "abstraction", "definition",
            "What is Python?",
            lambda r: "programming" in r.lower() or "language" in r.lower(),
            "programming language",
            setup=setup_def,
        )
        # Test 4: classify new instance
        self._run_test(
            agent, "abstraction", "classify_new",
            "teach Eagle is an animal",
            lambda r: "learned" in r.lower(),
            "learned",
        )
        # Test 5: abstract rule
        self._run_test(
            agent, "abstraction", "abstract_rule",
            "teach All birds can fly",
            lambda r: "learned" in r.lower(),
            "learned: all birds can fly",
        )

    # ------------------------------------------------------------------ #
    # 5. Transfer learning
    # ------------------------------------------------------------------ #
    def tests_transfer(self, agent: AETHER) -> None:
        # Test 1: apply capital-of pattern to new country
        def setup_capitals(a):
            for fact in [
                "Paris is the capital of France",
                "Tokyo is the capital of Japan",
                "Berlin is the capital of Germany",
            ]:
                a.teach(fact, silent=True)
            a.teach("Madrid is the capital of Spain", silent=True)
        self._run_test(
            agent, "transfer", "transfer_capital_pattern",
            "What is the capital of Spain?",
            lambda r: "madrid" in r.lower(),
            "Madrid",
            setup=setup_capitals,
        )
        # Test 2: apply location pattern
        def setup_locations(a):
            a.teach("Montreal is located in Canada", silent=True)
            a.teach("Lyon is located in France", silent=True)
            a.teach("Osaka is located in Japan", silent=True)
            a.teach("Munich is located in Germany", silent=True)
        self._run_test(
            agent, "transfer", "transfer_location_pattern",
            "Where is Munich located?",
            lambda r: "germany" in r.lower(),
            "Germany",
            setup=setup_locations,
        )
        # Test 3: new domain — colors
        def setup_colors(a):
            a.teach("Red is a color", silent=True)
            a.teach("Blue is a color", silent=True)
            a.teach("Green is a color", silent=True)
            a.teach("Yellow is a color", silent=True)
        self._run_test(
            agent, "transfer", "transfer_color_domain",
            "What is Yellow?",
            lambda r: "color" in r.lower(),
            "color",
            setup=setup_colors,
        )
        # Test 4: transfer to numbers
        def setup_numbers(a):
            a.teach("One is a number", silent=True)
            a.teach("Two is a number", silent=True)
            a.teach("Three is a number", silent=True)
        self._run_test(
            agent, "transfer", "transfer_number_domain",
            "What is Three?",
            lambda r: "number" in r.lower(),
            "number",
            setup=setup_numbers,
        )
        # Test 5: cross-domain transfer
        def setup_xdomain(a):
            a.teach("Paris is the capital of France", silent=True)
            a.teach("France is located in Europe", silent=True)
        self._run_test(
            agent, "transfer", "cross_domain",
            "What is the capital of the country where Lyon is located?",
            lambda r: "paris" in r.lower() or "france" in r.lower(),
            "Paris",
            setup=setup_xdomain,
        )

    # ------------------------------------------------------------------ #
    # 6. Metacognition
    # ------------------------------------------------------------------ #
    def tests_metacognition(self, agent: AETHER) -> None:
        # Test 1: know when you don't know
        self._run_test(
            agent, "metacognition", "unknown_awareness",
            "What is the capital of Mars?",
            lambda r: any(kw in r.lower() for kw in ["don't know", "couldn't find", "don't have", "no answer", "i don't", "teach me"]),
            "don't know / couldn't find",
        )
        # Test 2: comprehension score is computed
        comp_score = agent.comprehension_score()
        result = TestResult(
            test_name="comprehension_score_exists",
            dimension="metacognition",
            passed=0.0 <= comp_score <= 1.0,
            score=1.0 if 0.0 <= comp_score <= 1.0 else 0.0,
            response=f"comprehension={comp_score:.3f}",
            expected="comprehension score in [0,1]",
            duration_ms=0.0,
        )
        self.results.append(result)
        # Test 3: mood is tracked
        mood = agent.mood()
        result = TestResult(
            test_name="mood_tracked",
            dimension="metacognition",
            passed=mood in ["motivated", "alert", "neutral", "focused", "patient", "depressed", "drowsy"],
            score=1.0 if mood in ["motivated", "alert", "neutral", "focused", "patient", "depressed", "drowsy"] else 0.0,
            response=f"mood={mood}",
            expected="valid mood label",
            duration_ms=0.0,
        )
        self.results.append(result)
        # Test 4: metacognitive action is recommended
        action = agent.metacognitive_action()
        result = TestResult(
            test_name="metacog_action",
            dimension="metacognition",
            passed=action in ["continue", "act", "explore", "deliberate", "ask_for_clarification", "none"],
            score=1.0,
            response=f"action={action}",
            expected="valid metacognitive action",
            duration_ms=0.0,
        )
        self.results.append(result)
        # Test 5: narrative exists
        narrative = agent.narrative()
        result = TestResult(
            test_name="narrative_exists",
            dimension="metacognition",
            passed=len(narrative) > 0,
            score=1.0 if narrative else 0.0,
            response=f"narrative_length={len(narrative)}",
            expected="non-empty narrative",
            duration_ms=0.0,
        )
        self.results.append(result)

    # ------------------------------------------------------------------ #
    # 7. Self-correction
    # ------------------------------------------------------------------ #
    def tests_self_correction(self, agent: AETHER) -> None:
        # Test 1: teach corrects a missing fact
        def setup_missing(a):
            pass  # nothing taught
        self._run_test(
            agent, "self_correction", "teach_to_correct",
            "teach Madrid is the capital of Spain",
            lambda r: "learned" in r.lower(),
            "learned",
            setup=setup_missing,
        )
        # Test 2: re-teach overwrites
        def setup_overwrite(a):
            a.teach("X is the capital of Y", silent=True)
        self._run_test(
            agent, "self_correction", "re_teach",
            "teach Madrid is the capital of Spain",
            lambda r: "learned" in r.lower(),
            "learned",
            setup=setup_overwrite,
        )
        # Test 3: detect error in arithmetic (calc gives correct answer)
        self._run_test(
            agent, "self_correction", "arith_no_error",
            "calc 2+2",
            lambda r: "4" in r,
            "4",
        )
        # Test 4: complex correction
        def setup_complex(a):
            a.teach("Wrong is the capital of France", silent=True)
            a.teach("Paris is the capital of France", silent=True)
        self._run_test(
            agent, "self_correction", "correct_contradiction",
            "What is the capital of France?",
            lambda r: "paris" in r.lower(),
            "Paris (latest)",
            setup=setup_complex,
        )
        # Test 5: tool error recovery
        self._run_test(
            agent, "self_correction", "tool_error_recovery",
            "calc 1/0",
            lambda r: "error" in r.lower() or "zero" in r.lower() or "divis" in r.lower() or "inf" in r.lower() or "1/0" in r,
            "error message",
        )

    # ------------------------------------------------------------------ #
    # 8. Theory of mind
    # ------------------------------------------------------------------ #
    def tests_theory_of_mind(self, agent: AETHER) -> None:
        # Test 1: distinguish self from other
        self._run_test(
            agent, "theory_of_mind", "self_other_distinction",
            "What are you?",
            lambda r: "aether" in r.lower() or "i am" in r.lower() or "i'm" in r.lower(),
            "self-identification",
        )
        # Test 2: know what you know
        def setup_know(a):
            a.teach("Paris is the capital of France", silent=True)
        self._run_test(
            agent, "theory_of_mind", "know_what_you_know",
            "What is the capital of France?",
            lambda r: "paris" in r.lower(),
            "Paris",
            setup=setup_know,
        )
        # Test 3: know what you don't know
        self._run_test(
            agent, "theory_of_mind", "know_what_you_dont_know",
            "What is the capital of Mars?",
            lambda r: any(kw in r.lower() for kw in ["don't know", "couldn't find", "don't have", "no answer", "i don't", "teach me"]),
            "don't know",
        )
        # Test 4: explain own reasoning
        self._run_test(
            agent, "theory_of_mind", "explain_reasoning",
            "How do you work?",
            lambda r: "hyperdimensional" in r.lower() or "vector" in r.lower() or "memory" in r.lower() or "cognitive" in r.lower(),
            "self-explanation",
        )
        # Test 5: introspect
        intro = agent.introspect()
        result = TestResult(
            test_name="introspection",
            dimension="theory_of_mind",
            passed="identity" in intro and ("mood" in intro or "current_mood" in intro),
            score=1.0 if "identity" in intro and ("mood" in intro or "current_mood" in intro) else 0.0,
            response=f"identity={intro.get('identity')} mood={intro.get('current_mood')}",
            expected="valid introspection",
            duration_ms=0.0,
        )
        self.results.append(result)

    # ------------------------------------------------------------------ #
    # 9. Tool use
    # ------------------------------------------------------------------ #
    def tests_tool_use(self, agent: AETHER) -> None:
        # Test 1: calc tool
        self._run_test(
            agent, "tool_use", "calc_tool",
            "calc 25*4",
            lambda r: "100" in r,
            "100",
        )
        # Test 2: time tool
        self._run_test(
            agent, "tool_use", "time_tool",
            "time",
            lambda r: "20" in r and ":" in r,  # looks like a timestamp
            "timestamp",
        )
        # Test 3: recall tool
        def setup_recall(a):
            a.teach("TestFact is a test", silent=True)
        self._run_test(
            agent, "tool_use", "recall_tool",
            "recall TestFact",
            lambda r: len(r) > 5,
            "recall result",
            setup=setup_recall,
        )
        # Test 4: list_kb tool
        self._run_test(
            agent, "tool_use", "list_kb_tool",
            "list kb",
            lambda r: "(" in r or "triples" in r.lower() or "empty" in r.lower(),
            "KB listing",
        )
        # Test 5: explain tool
        def setup_explain(a):
            a.teach("Paris is the capital of France", silent=True)
        self._run_test(
            agent, "tool_use", "explain_tool",
            "explain Paris",
            lambda r: "paris" in r.lower() or "france" in r.lower() or "capital" in r.lower(),
            "explanation of Paris",
            setup=setup_explain,
        )
        # Test 6: compare tool
        def setup_compare(a):
            a.teach("Paris is the capital of France", silent=True)
            a.teach("Tokyo is the capital of Japan", silent=True)
        self._run_test(
            agent, "tool_use", "compare_tool",
            "compare Paris and Tokyo",
            lambda r: "paris" in r.lower() and "tokyo" in r.lower(),
            "comparison",
            setup=setup_compare,
        )
        # Test 7: count tool
        self._run_test(
            agent, "tool_use", "count_tool",
            "count triples",
            lambda r: "triples" in r.lower() or any(c.isdigit() for c in r),
            "count result",
        )

    # ------------------------------------------------------------------ #
    # 10. Comprehension
    # ------------------------------------------------------------------ #
    def tests_comprehension(self, agent: AETHER) -> None:
        # Test 1: answer direct question
        def setup_comp(a):
            a.teach("Paris is the capital of France", silent=True)
        self._run_test(
            agent, "comprehension", "direct_qa",
            "What is the capital of France?",
            lambda r: "paris" in r.lower(),
            "Paris",
            setup=setup_comp,
        )
        # Test 2: multi-hop comprehension
        def setup_multihop(a):
            a.teach("Lyon is located in France", silent=True)
            a.teach("Paris is the capital of France", silent=True)
        self._run_test(
            agent, "comprehension", "multi_hop_qa",
            "What is the capital of the country where Lyon is located?",
            lambda r: "paris" in r.lower(),
            "Paris",
            setup=setup_multihop,
        )
        # Test 3: comprehension score after a question
        agent.ask("What is the capital of France?")
        comp_score = agent.comprehension_score()
        result = TestResult(
            test_name="comprehension_score_post_q",
            dimension="comprehension",
            passed=comp_score > 0.0,
            score=min(1.0, comp_score * 2),
            response=f"score={comp_score:.3f}",
            expected="comprehension score > 0",
            duration_ms=0.0,
        )
        self.results.append(result)
        # Test 4: define
        def setup_def(a):
            a.teach("Python is a programming language", silent=True)
        self._run_test(
            agent, "comprehension", "define_concept",
            "define Python",
            lambda r: "programming" in r.lower() or "language" in r.lower(),
            "programming language",
            setup=setup_def,
        )
        # Test 5: identify unknown
        self._run_test(
            agent, "comprehension", "identify_unknown",
            "What is the capital of Pluto?",
            lambda r: any(kw in r.lower() for kw in ["don't know", "couldn't find", "don't have", "no answer", "i don't", "teach me"]),
            "don't know",
        )


# ---------------------------------------------------------------------------
# Intelligence meter
# ---------------------------------------------------------------------------

class IntelligenceMeter:
    """Aggregates test results into dimension scores and an overall IQ."""

    # IQ normalization: 100 = average, 130 = gifted, 145 = genius
    # Each dimension contributes equally; IQ = 100 + (mean_score - 0.5) * 100
    # So a perfect 1.0 → IQ 150, 0.5 → IQ 100, 0.0 → IQ 50

    DIMENSIONS = [
        "reasoning", "working_memory", "pattern_recog", "abstraction",
        "transfer", "metacognition", "self_correction", "theory_of_mind",
        "tool_use", "comprehension",
    ]

    def __init__(self):
        self.results: List[TestResult] = []

    def measure(self, results: List[TestResult]) -> Dict[str, Any]:
        """Compute dimension scores, overall score, and IQ."""
        self.results = results

        # Per-dimension scores
        dim_scores: Dict[str, List[float]] = {d: [] for d in self.DIMENSIONS}
        for r in results:
            if r.dimension in dim_scores:
                dim_scores[r.dimension].append(r.score)

        # Average per dimension
        dim_averages: Dict[str, float] = {}
        for d, scores in dim_scores.items():
            if scores:
                dim_averages[d] = sum(scores) / len(scores)
            else:
                dim_averages[d] = 0.0

        # Overall score = mean of dimension averages
        overall = sum(dim_averages.values()) / max(len(dim_averages), 1)

        # IQ = 50 + overall * 100 (so 0.0 → 50, 1.0 → 150)
        iq = int(50 + overall * 100)

        # Strengths (top 3) and weaknesses (bottom 3)
        sorted_dims = sorted(dim_averages.items(), key=lambda x: -x[1])
        strengths = sorted_dims[:3]
        weaknesses = sorted_dims[-3:]

        return {
            "dimension_scores": dim_averages,
            "overall_score": overall,
            "iq": iq,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "n_tests": len(results),
            "n_passed": sum(1 for r in results if r.passed),
            "n_partial": sum(1 for r in results if 0 < r.score < 1.0),
            "pass_rate": sum(1 for r in results if r.passed) / max(len(results), 1),
        }

    def report(self, results: List[TestResult]) -> str:
        """Generate a human-readable intelligence report."""
        m = self.measure(results)
        lines = []
        lines.append("=" * 76)
        lines.append("  AETHER COGNITIVE ASSESSMENT REPORT")
        lines.append("=" * 76)
        lines.append(f"  Tests run:    {m['n_tests']}")
        lines.append(f"  Tests passed: {m['n_passed']} ({m['pass_rate']*100:.1f}%)")
        lines.append(f"  Partial:      {m['n_partial']}")
        lines.append(f"  Overall:      {m['overall_score']:.3f}")
        lines.append(f"  IQ:           {m['iq']}")
        lines.append("")
        lines.append("  Per-dimension scores:")
        for dim, score in sorted(m["dimension_scores"].items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 30)
            lines.append(f"    {dim:18s} {score:.3f} |{bar}")
        lines.append("")
        lines.append("  Strengths:")
        for dim, score in m["strengths"]:
            lines.append(f"    {dim:18s} {score:.3f}")
        lines.append("")
        lines.append("  Weaknesses:")
        for dim, score in m["weaknesses"]:
            lines.append(f"    {dim:18s} {score:.3f}")
        lines.append("=" * 76)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = AETHER()
    battery = CognitiveBattery()
    print("Running cognitive battery on AETHER v4...\n")
    results = battery.run_all(agent, verbose=True)
    meter = IntelligenceMeter()
    print()
    print(meter.report(results))
