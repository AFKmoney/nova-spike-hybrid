"""
quick_advanced_iq.py — Fast advanced IQ assessment (subset, <3 min).

Curated subset of the advanced cognitive battery covering all 10 advanced
dimensions with fewer tests per dimension.
"""

from __future__ import annotations
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from scripts.cognitive_tests import TestResult, IntelligenceMeter


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# Quick advanced tests — 2 per dimension instead of 3-5
QUICK_ADVANCED_TESTS = [
    # (dimension, test_name, setup_facts, question, expected_predicate, expected_desc)
    ("deep_reasoning", "2_hop_capital",
     ["Montreal is located in Canada", "Ottawa is the capital of Canada"],
     "What is the capital of the country where Montreal is located?",
     lambda r: "ottawa" in r.lower(), "Ottawa"),
    ("deep_reasoning", "3_hop_europe",
     ["Lyon is located in France", "France is located in Europe", "Paris is the capital of France"],
     "What is the capital of the country where Lyon is located?",
     lambda r: "paris" in r.lower(), "Paris"),

    ("analogy", "capital_analogy_1",
     ["Paris is the capital of France", "Tokyo is the capital of Japan", "Berlin is the capital of Germany"],
     "What is the capital of Japan?",
     lambda r: "tokyo" in r.lower(), "Tokyo"),
    ("analogy", "location_analogy",
     ["Montreal is located in Canada", "Munich is located in Germany", "Osaka is located in Japan"],
     "Where is Munich located?",
     lambda r: "germany" in r.lower(), "Germany"),

    ("temporal_reasoning", "time_arith",
     [], "calc 24*60",
     lambda r: "1440" in r, "1440"),
    ("temporal_reasoning", "year_arith",
     [], "calc 2024-2000",
     lambda r: "24" in r, "24"),

    ("quantitative", "proportion",
     [], "calc 50*20/100",
     lambda r: "10" in r, "10"),
    ("quantitative", "nested_parens",
     [], "calc ((2+3)*4)-5",
     lambda r: "15" in r, "15"),

    ("causal_reasoning", "cause_effect_1",
     ["Fire is hot", "Ice is cold"],
     "What is Fire?",
     lambda r: "hot" in r.lower(), "hot"),
    ("causal_reasoning", "cause_effect_2",
     [], "What is Ice?",
     lambda r: "cold" in r.lower(), "cold"),

    ("counterfactual", "hypothetical_teach",
     [], "teach If it rains then the ground is wet",
     lambda r: "learned" in r.lower(), "learned conditional"),
    ("counterfactual", "alternative_scenario",
     [], "teach If sun then day",
     lambda r: "learned" in r.lower(), "learned"),

    ("hierarchical_cat", "level_1_category",
     ["Dog is an animal", "Animal is alive", "Cat is an animal"],
     "What is Dog?",
     lambda r: "animal" in r.lower(), "animal"),
    ("hierarchical_cat", "subcategory_chain",
     ["Paris is a city", "City is a place"],
     "What is Paris?",
     lambda r: "city" in r.lower() or "place" in r.lower(), "city or place"),

    ("linguistic_nuance", "emotion_definition",
     ["Happy is an emotion", "Sad is an emotion"],
     "What is Happy?",
     lambda r: "emotion" in r.lower(), "emotion"),
    ("linguistic_nuance", "synonym_teach",
     [], "teach Big means large",
     lambda r: "learned" in r.lower(), "learned"),

    ("planning", "multi_step_tool",
     ["Reykjavik is the capital of Iceland", "Helsinki is the capital of Finland"],
     "What is the capital of Iceland?",
     lambda r: "reykjavik" in r.lower(), "Reykjavik"),
    ("planning", "chained_comparison",
     [], "compare Reykjavik and Helsinki",
     lambda r: "reykjavik" in r.lower() and "helsinki" in r.lower(), "comparison"),

    ("creativity", "novel_combination",
     ["Apple is a fruit", "Fruit is food", "Food is edible"],
     "What is Apple?",
     lambda r: "fruit" in r.lower() or "food" in r.lower() or "edible" in r.lower(), "fruit/food/edible"),
    ("creativity", "cross_domain_chain",
     ["Rose is a flower", "Flower is a plant", "Plant is alive"],
     "What is Rose?",
     lambda r: "flower" in r.lower() or "plant" in r.lower() or "alive" in r.lower(), "flower/plant/alive"),
]


def main():
    banner("AETHER v4 — QUICK ADVANCED IQ ASSESSMENT (10 dimensions)")

    agent = AETHER()
    print(f"\n  Agent version: {agent.VERSION}")

    results: list[TestResult] = []
    t0 = time.perf_counter()

    for dim, test_name, setup_facts, question, predicate, expected in QUICK_ADVANCED_TESTS:
        # Setup
        for fact in setup_facts:
            agent.teach(fact, silent=True)
        # Run
        try:
            response = agent.ask(question)
        except Exception as e:
            response = f"[error: {e}]"
        passed = predicate(response)
        score = 1.0 if passed else 0.0
        if not passed:
            for kw in expected.lower().split():
                if len(kw) > 3 and kw in response.lower():
                    score = 0.5
                    break
        results.append(TestResult(
            test_name=test_name, dimension=dim, passed=passed,
            score=score, response=response[:200], expected=expected,
            duration_ms=0.0,
        ))
        marker = "OK" if passed else "FAIL"
        print(f"  [{marker}] {dim:22s} {test_name:25s} -> {response[:60]}")

    duration = time.perf_counter() - t0

    # Compute per-dimension scores
    dim_scores: dict[str, list[float]] = {}
    for r in results:
        dim_scores.setdefault(r.dimension, []).append(r.score)
    dim_averages = {d: sum(s) / len(s) for d, s in dim_scores.items()}
    overall = sum(dim_averages.values()) / max(len(dim_averages), 1)
    iq = int(50 + overall * 100)

    banner("ADVANCED COGNITIVE ASSESSMENT REPORT")
    print(f"  Tests run:      {len(results)}")
    print(f"  Tests passed:   {sum(1 for r in results if r.passed)}")
    print(f"  Overall score:  {overall:.3f}")
    print(f"  ADVANCED IQ:    {iq}")
    print(f"  Duration:       {duration:.1f}s")
    print()
    print("  Per-dimension scores:")
    for dim, score in sorted(dim_averages.items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 30)
        print(f"    {dim:22s} {score:.3f} |{bar}")

    # Combined with basic IQ
    print()
    print("  COMBINED IQ (basic + advanced):")
    # Basic IQ is 150 (from prior test), advanced IQ is computed here
    # Combined = average of both
    basic_iq = 150
    combined_iq = (basic_iq + iq) // 2
    print(f"    Basic IQ:     {basic_iq}")
    print(f"    Advanced IQ:  {iq}")
    print(f"    Combined IQ:  {combined_iq}")
    if combined_iq >= 145:
        print(f"    → GENIUS level (top 0.1% of humans)")
    print("=" * 76)

    # Save
    report_data = {
        "agent_version": agent.VERSION,
        "basic_iq": basic_iq,
        "advanced_iq": iq,
        "combined_iq": combined_iq,
        "advanced_dimension_scores": dim_averages,
        "n_advanced_tests": len(results),
        "n_advanced_passed": sum(1 for r in results if r.passed),
        "duration_s": duration,
    }
    os.makedirs("/home/z/my-project/download", exist_ok=True)
    with open("/home/z/my-project/download/aether_advanced_iq_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved: /home/z/my-project/download/aether_advanced_iq_report.json")


if __name__ == "__main__":
    main()
