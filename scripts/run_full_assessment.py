"""
run_full_assessment.py — Run both basic + advanced batteries and report combined IQ.

This is the "intelligence meter" — runs all 20 dimensions of cognitive tests
and produces a comprehensive report.
"""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from scripts.cognitive_tests import CognitiveBattery, IntelligenceMeter
from scripts.advanced_cognitive_tests import AdvancedCognitiveBattery, ExtendedIntelligenceMeter


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def main():
    banner("AETHER v4 — FULL COGNITIVE ASSESSMENT (20 dimensions, 85 tests)")

    agent = AETHER()
    print(f"\n  Agent version: {agent.VERSION}")
    print(f"  Tools: {len(agent.list_tools())}")
    print(f"  KB: {len(agent.assoc.triples)} triples")
    print(f"  Attractor memories: {len(agent.attractor.labeled_memories)}")
    print(f"  Kuramoto oscillators: {agent.kuramoto.N}")
    print(f"  Specialists: {len(agent.global_workspace.specialists)}")
    print(f"  Hierarchy levels: {agent.hierarchy.n_levels}")

    # Run basic battery
    banner("Phase 1 — Basic cognitive battery (10 dimensions, 51 tests)")
    basic_battery = CognitiveBattery()
    t0 = time.perf_counter()
    basic_results = basic_battery.run_all(agent, verbose=True)
    basic_duration = time.perf_counter() - t0
    print(f"\n  Basic battery completed in {basic_duration:.1f}s")

    # Run advanced battery
    banner("Phase 2 — Advanced cognitive battery (10 dimensions, 34 tests)")
    advanced_battery = AdvancedCognitiveBattery()
    t0 = time.perf_counter()
    advanced_results = advanced_battery.run_all(agent, verbose=True)
    advanced_duration = time.perf_counter() - t0
    print(f"\n  Advanced battery completed in {advanced_duration:.1f}s")

    # Combine results
    all_results = basic_results + advanced_results

    # Measure with extended meter
    meter = ExtendedIntelligenceMeter()
    report = meter.measure(all_results)

    # Print full report
    banner("FINAL COGNITIVE ASSESSMENT REPORT")
    print(f"  Total tests:    {report['n_tests']}")
    print(f"  Tests passed:   {report['n_passed']} ({report['pass_rate']*100:.1f}%)")
    print(f"  Partial:        {report['n_partial']}")
    print(f"  Overall score:  {report['overall_score']:.3f}")
    print(f"  ESTIMATED IQ:   {report['iq']}")
    print()
    print("  Per-dimension scores (sorted):")
    for dim, score in sorted(report["dimension_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 30)
        print(f"    {dim:22s} {score:.3f} |{bar}")
    print()
    print("  Top strengths:")
    for dim, score in report["strengths"]:
        print(f"    {dim:22s} {score:.3f}")
    print()
    print("  Weakest dimensions:")
    for dim, score in report["weaknesses"]:
        print(f"    {dim:22s} {score:.3f}")

    # IQ interpretation
    iq = report["iq"]
    print()
    print("  IQ interpretation:")
    if iq >= 145:
        print(f"    {iq} → GENIUS level (top 0.1% of humans)")
    elif iq >= 130:
        print(f"    {iq} → GIFTED (top 2% of humans)")
    elif iq >= 115:
        print(f"    {iq} → ABOVE AVERAGE (top 16% of humans)")
    elif iq >= 85:
        print(f"    {iq} → AVERAGE")
    else:
        print(f"    {iq} → BELOW AVERAGE")
    print("=" * 76)

    # Save the report
    import json
    report_data = {
        "agent_version": agent.VERSION,
        "timestamp": time.time(),
        "iq": iq,
        "overall_score": report["overall_score"],
        "pass_rate": report["pass_rate"],
        "n_tests": report["n_tests"],
        "n_passed": report["n_passed"],
        "dimension_scores": report["dimension_scores"],
        "failed_tests": [
            {"dimension": r.dimension, "test": r.test_name,
             "response": r.response, "expected": r.expected}
            for r in all_results if not r.passed
        ],
    }
    with open("/home/z/my-project/download/aether_iq_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved to: /home/z/my-project/download/aether_iq_report.json")


if __name__ == "__main__":
    main()
