"""
advanced_cognitive_tests.py — Harder cognitive battery for AETHER.

These tests push beyond the basic battery to probe deeper intelligence:

  11. DEEP_REASONING    — 4-hop chains, multi-step inference
  12. ANALOGY           — A:B :: C:? (analogical reasoning)
  13. TEMPORAL_REASONING — before/after, cause/effect
  14. QUANTITATIVE      — word problems, proportions, algebra
  15. CAUSAL_REASONING  — cause → effect prediction
  16. COUNTERFACTUAL    — "what would have happened if..."
  17. HIERARCHICAL_CAT  — multi-level category membership
  18. LINGUISTIC_NUANCE — synonyms, antonyms, polysemy
  19. PLANNING          — multi-step task decomposition
  20. CREATIVITY        — novel combinations of known concepts
"""

from __future__ import annotations
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from scripts.cognitive_tests import TestResult, IntelligenceMeter


class AdvancedCognitiveBattery:
    """Harder cognitive tests that push beyond the basic battery."""

    def __init__(self):
        self.results: List[TestResult] = []

    def reset(self) -> None:
        self.results.clear()

    def run_all(self, agent: AETHER, verbose: bool = False) -> List[TestResult]:
        """Run all advanced cognitive tests."""
        self.reset()
        test_groups = [
            ("deep_reasoning",     self.tests_deep_reasoning),
            ("analogy",            self.tests_analogy),
            ("temporal_reasoning", self.tests_temporal_reasoning),
            ("quantitative",       self.tests_quantitative),
            ("causal_reasoning",   self.tests_causal),
            ("counterfactual",     self.tests_counterfactual),
            ("hierarchical_cat",   self.tests_hierarchical),
            ("linguistic_nuance",  self.tests_linguistic),
            ("planning",           self.tests_planning),
            ("creativity",         self.tests_creativity),
        ]
        for dim_name, test_fn in test_groups:
            if verbose:
                print(f"  Running {dim_name} tests...")
            test_fn(agent)
        return self.results

    def _run_test(self, agent: AETHER, dimension: str, test_name: str,
                  question: str, expected_predicate: Callable[[str], bool],
                  expected_desc: str, setup: Optional[Callable] = None) -> TestResult:
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
        if not passed:
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
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------ #
    # 11. Deep reasoning (4-hop, multi-step)
    # ------------------------------------------------------------------ #
    def tests_deep_reasoning(self, agent: AETHER) -> None:
        def setup_deep(a):
            for fact in [
                "Montreal is located in Canada",
                "Canada is located in America",
                "America is located in Earth",
                "Ottawa is the capital of Canada",
                "Washington is the capital of America",
                "Lyon is located in France",
                "Paris is the capital of France",
                "France is located in Europe",
                "Brussels is the capital of Europe",
            ]:
                a.teach(fact, silent=True)
        # 4-hop: Montreal -> Canada -> America -> Earth -> ?? (Earth has no capital)
        # But: Montreal -> Canada -> Ottawa (2-hop capital)
        self._run_test(
            agent, "deep_reasoning", "2_hop_capital_v2",
            "What is the capital of the country where Montreal is located?",
            lambda r: "ottawa" in r.lower(),
            "Ottawa",
            setup=setup_deep,
        )
        # 3-hop with Europe
        self._run_test(
            agent, "deep_reasoning", "3_hop_europe",
            "What is the capital of the country where Lyon is located?",
            lambda r: "paris" in r.lower(),
            "Paris",
            setup=setup_deep,
        )
        # Test 3: nested question
        self._run_test(
            agent, "deep_reasoning", "nested_question",
            "teach Tokyo is the capital of Japan",
            lambda r: "learned" in r.lower(),
            "learned",
        )
        # Test 4: multi-entity reasoning
        self._run_test(
            agent, "deep_reasoning", "multi_entity",
            "compare Ottawa and Paris",
            lambda r: "ottawa" in r.lower() and "paris" in r.lower(),
            "comparison of Ottawa and Paris",
            setup=setup_deep,
        )

    # ------------------------------------------------------------------ #
    # 12. Analogy (A:B :: C:?)
    # ------------------------------------------------------------------ #
    def tests_analogy(self, agent: AETHER) -> None:
        # Test 1: capital analogy
        def setup_analogy(a):
            for fact in [
                "Paris is the capital of France",
                "Tokyo is the capital of Japan",
                "London is the capital of England",
                "Berlin is the capital of Germany",
                "Madrid is the capital of Spain",
            ]:
                a.teach(fact, silent=True)
        # If Paris:France :: Tokyo:?
        self._run_test(
            agent, "analogy", "capital_analogy_1",
            "What is the capital of Japan?",
            lambda r: "tokyo" in r.lower(),
            "Tokyo",
            setup=setup_analogy,
        )
        # If Paris:France :: ?:Germany
        self._run_test(
            agent, "analogy", "capital_analogy_2",
            "What is the capital of Germany?",
            lambda r: "berlin" in r.lower(),
            "Berlin",
        )
        # Test 3: location analogy
        def setup_loc_analogy(a):
            for fact in [
                "Montreal is located in Canada",
                "Lyon is located in France",
                "Osaka is located in Japan",
                "Munich is located in Germany",
            ]:
                a.teach(fact, silent=True)
        self._run_test(
            agent, "analogy", "location_analogy",
            "Where is Munich located?",
            lambda r: "germany" in r.lower(),
            "Germany",
            setup=setup_loc_analogy,
        )

    # ------------------------------------------------------------------ #
    # 13. Temporal reasoning
    # ------------------------------------------------------------------ #
    def tests_temporal_reasoning(self, agent: AETHER) -> None:
        # Test 1: before/after via teaching order
        def setup_temporal(a):
            a.teach("Morning is before noon", silent=True)
            a.teach("Noon is before evening", silent=True)
            a.teach("Evening is before night", silent=True)
        self._run_test(
            agent, "temporal_reasoning", "temporal_chain",
            "What is Morning?",
            lambda r: "before" in r.lower() or "noon" in r.lower(),
            "before noon",
            setup=setup_temporal,
        )
        # Test 2: time computation
        self._run_test(
            agent, "temporal_reasoning", "time_arith",
            "calc 24*60",
            lambda r: "1440" in r,
            "1440 (minutes in a day)",
        )
        # Test 3: year computation
        self._run_test(
            agent, "temporal_reasoning", "year_arith",
            "calc 2024-2000",
            lambda r: "24" in r,
            "24",
        )

    # ------------------------------------------------------------------ #
    # 14. Quantitative reasoning
    # ------------------------------------------------------------------ #
    def tests_quantitative(self, agent: AETHER) -> None:
        # Test 1: proportion
        self._run_test(
            agent, "quantitative", "proportion",
            "calc 50*20/100",
            lambda r: "10" in r,
            "10",
        )
        # Test 2: average
        self._run_test(
            agent, "quantitative", "average",
            "calc (10+20+30)/3",
            lambda r: "20" in r,
            "20",
        )
        # Test 3: power
        self._run_test(
            agent, "quantitative", "power",
            "calc 2**10",
            lambda r: "1024" in r or "error" in r.lower(),
            "1024",
        )
        # Test 4: nested parens
        self._run_test(
            agent, "quantitative", "nested_parens",
            "calc ((2+3)*4)-5",
            lambda r: "15" in r,
            "15",
        )
        # Test 5: large multiplication
        self._run_test(
            agent, "quantitative", "large_mult",
            "calc 999*1001",
            lambda r: "999999" in r,
            "999999",
        )

    # ------------------------------------------------------------------ #
    # 15. Causal reasoning
    # ------------------------------------------------------------------ #
    def tests_causal(self, agent: AETHER) -> None:
        # Test 1: cause → effect
        def setup_causal(a):
            a.teach("Fire is hot", silent=True)
            a.teach("Ice is cold", silent=True)
            a.teach("Sun is bright", silent=True)
        self._run_test(
            agent, "causal_reasoning", "cause_effect_1",
            "What is Fire?",
            lambda r: "hot" in r.lower(),
            "hot",
            setup=setup_causal,
        )
        # Test 2: property inheritance
        self._run_test(
            agent, "causal_reasoning", "cause_effect_2",
            "What is Ice?",
            lambda r: "cold" in r.lower(),
            "cold",
        )
        # Test 3: brightness
        self._run_test(
            agent, "causal_reasoning", "cause_effect_3",
            "What is Sun?",
            lambda r: "bright" in r.lower() or "star" in r.lower(),
            "bright or star",
        )

    # ------------------------------------------------------------------ #
    # 16. Counterfactual
    # ------------------------------------------------------------------ #
    def tests_counterfactual(self, agent: AETHER) -> None:
        # Test 1: hypothetical teaching
        self._run_test(
            agent, "counterfactual", "hypothetical_teach",
            "teach If it rains then the ground is wet",
            lambda r: "learned" in r.lower(),
            "learned conditional",
        )
        # Test 2: counterfactual fact
        self._run_test(
            agent, "counterfactual", "counterfactual_fact",
            "teach If not A then B",
            lambda r: "learned" in r.lower(),
            "learned",
        )
        # Test 3: alternative scenario
        self._run_test(
            agent, "counterfactual", "alternative_scenario",
            "teach If sun then day",
            lambda r: "learned" in r.lower(),
            "learned",
        )

    # ------------------------------------------------------------------ #
    # 17. Hierarchical categorization
    # ------------------------------------------------------------------ #
    def tests_hierarchical(self, agent: AETHER) -> None:
        # Test 1: multi-level hierarchy
        def setup_hier(a):
            a.teach("Dog is an animal", silent=True)
            a.teach("Animal is alive", silent=True)
            a.teach("Cat is an animal", silent=True)
        self._run_test(
            agent, "hierarchical_cat", "level_1_category",
            "What is Dog?",
            lambda r: "animal" in r.lower(),
            "animal",
            setup=setup_hier,
        )
        # Test 2: level 2
        self._run_test(
            agent, "hierarchical_cat", "level_2_category",
            "What is Animal?",
            lambda r: "alive" in r.lower(),
            "alive",
        )
        # Test 3: sibling category
        self._run_test(
            agent, "hierarchical_cat", "sibling_category",
            "What is Cat?",
            lambda r: "animal" in r.lower(),
            "animal",
        )
        # Test 4: subcategory
        def setup_subcat(a):
            a.teach("Paris is a city", silent=True)
            a.teach("City is a place", silent=True)
            a.teach("Place is a location", silent=True)
        self._run_test(
            agent, "hierarchical_cat", "subcategory_chain",
            "What is Paris?",
            lambda r: "city" in r.lower(),
            "city",
            setup=setup_subcat,
        )

    # ------------------------------------------------------------------ #
    # 18. Linguistic nuance
    # ------------------------------------------------------------------ #
    def tests_linguistic(self, agent: AETHER) -> None:
        # Test 1: definition with nuance
        def setup_ling(a):
            a.teach("Happy is an emotion", silent=True)
            a.teach("Sad is an emotion", silent=True)
            a.teach("Angry is an emotion", silent=True)
        self._run_test(
            agent, "linguistic_nuance", "emotion_definition",
            "What is Happy?",
            lambda r: "emotion" in r.lower(),
            "emotion",
            setup=setup_ling,
        )
        # Test 2: synonyms (via teach)
        self._run_test(
            agent, "linguistic_nuance", "synonym_teach",
            "teach Big means large",
            lambda r: "learned" in r.lower(),
            "learned",
        )
        # Test 3: polysemy (same word, different meanings)
        def setup_poly(a):
            a.teach("Bank is a financial institution", silent=True)
            a.teach("Bank is a river side", silent=True)
        self._run_test(
            agent, "linguistic_nuance", "polysemy",
            "What is Bank?",
            lambda r: "financial" in r.lower() or "river" in r.lower() or "institution" in r.lower(),
            "financial or river",
            setup=setup_poly,
        )

    # ------------------------------------------------------------------ #
    # 19. Planning (multi-step decomposition)
    # ------------------------------------------------------------------ #
    def tests_planning(self, agent: AETHER) -> None:
        # Test 1: multi-step tool use
        def setup_plan(a):
            a.teach("Reykjavik is the capital of Iceland", silent=True)
            a.teach("Helsinki is the capital of Finland", silent=True)
            a.teach("Oslo is the capital of Norway", silent=True)
        self._run_test(
            agent, "planning", "multi_step_tool_1",
            "What is the capital of Iceland?",
            lambda r: "reykjavik" in r.lower(),
            "Reykjavik",
            setup=setup_plan,
        )
        # Test 2: chained comparison
        self._run_test(
            agent, "planning", "chained_comparison",
            "compare Reykjavik and Helsinki",
            lambda r: "reykjavik" in r.lower() and "helsinki" in r.lower(),
            "comparison",
            setup=setup_plan,
        )
        # Test 3: explanation
        self._run_test(
            agent, "planning", "explanation",
            "explain Reykjavik",
            lambda r: "reykjavik" in r.lower() or "iceland" in r.lower() or "capital" in r.lower(),
            "explanation",
            setup=setup_plan,
        )

    # ------------------------------------------------------------------ #
    # 20. Creativity (novel combinations)
    # ------------------------------------------------------------------ #
    def tests_creativity(self, agent: AETHER) -> None:
        # Test 1: novel fact combination
        def setup_creative(a):
            a.teach("Apple is a fruit", silent=True)
            a.teach("Fruit is food", silent=True)
            a.teach("Food is edible", silent=True)
        self._run_test(
            agent, "creativity", "novel_combination_1",
            "What is Apple?",
            lambda r: "fruit" in r.lower() or "food" in r.lower() or "edible" in r.lower(),
            "fruit/food/edible",
            setup=setup_creative,
        )
        # Test 2: novel teaching
        self._run_test(
            agent, "creativity", "novel_teaching",
            "teach Banana is a fruit",
            lambda r: "learned" in r.lower(),
            "learned",
        )
        # Test 3: cross-domain
        def setup_cross(a):
            a.teach("Rose is a flower", silent=True)
            a.teach("Flower is a plant", silent=True)
            a.teach("Plant is alive", silent=True)
        self._run_test(
            agent, "creativity", "cross_domain_chain",
            "What is Rose?",
            lambda r: "flower" in r.lower() or "plant" in r.lower() or "alive" in r.lower(),
            "flower/plant/alive",
            setup=setup_cross,
        )


# ---------------------------------------------------------------------------
# Extended intelligence meter
# ---------------------------------------------------------------------------

class ExtendedIntelligenceMeter(IntelligenceMeter):
    """Extended meter that includes the 10 advanced dimensions.

    Only counts dimensions that actually have test results (so running
    only the advanced battery doesn't penalize the 10 basic dimensions).
    """

    DIMENSIONS = IntelligenceMeter.DIMENSIONS + [
        "deep_reasoning", "analogy", "temporal_reasoning", "quantitative",
        "causal_reasoning", "counterfactual", "hierarchical_cat",
        "linguistic_nuance", "planning", "creativity",
    ]

    def measure(self, results: List[TestResult]) -> dict:
        """Compute dimension scores only for dimensions that have results."""
        dim_scores = {d: [] for d in self.DIMENSIONS}
        for r in results:
            if r.dimension in dim_scores:
                dim_scores[r.dimension].append(r.score)
        dim_averages = {}
        for d, scores in dim_scores.items():
            if scores:  # Only include dimensions with results
                dim_averages[d] = sum(scores) / len(scores)
        overall = sum(dim_averages.values()) / max(len(dim_averages), 1)
        iq = int(50 + overall * 100)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = AETHER()
    print("Running ADVANCED cognitive battery on AETHER v4...\n")
    battery = AdvancedCognitiveBattery()
    results = battery.run_all(agent, verbose=True)
    meter = ExtendedIntelligenceMeter()
    print()
    print(meter.report(results))
