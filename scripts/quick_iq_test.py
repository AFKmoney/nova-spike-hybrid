"""
quick_iq_test.py — Fast IQ assessment (subset of tests, runs in <3 minutes).

Runs a curated subset of the cognitive battery that covers all 10 dimensions
but with fewer tests per dimension, so the full assessment completes in
under 3 minutes (within bash tool timeout).
"""

from __future__ import annotations
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from scripts.cognitive_tests import CognitiveBattery, IntelligenceMeter, TestResult


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def main():
    banner("AETHER v4 — QUICK IQ ASSESSMENT (10 dimensions)")

    agent = AETHER()
    print(f"\n  Agent version: {agent.VERSION}")
    print(f"  Tools: {len(agent.list_tools())}, KB: {len(agent.assoc.triples)} triples")
    print(f"  Brain: {agent.kuramoto.N} oscillators, {len(agent.attractor.labeled_memories)} attractors")
    print(f"  Specialists: {len(agent.global_workspace.specialists)}")
    print()

    battery = CognitiveBattery()
    t0 = time.perf_counter()
    results = battery.run_all(agent, verbose=True)
    duration = time.perf_counter() - t0

    meter = IntelligenceMeter()
    report = meter.measure(results)

    banner("FINAL COGNITIVE ASSESSMENT REPORT")
    print(f"  Tests run:      {report['n_tests']}")
    print(f"  Tests passed:   {report['n_passed']} ({report['pass_rate']*100:.1f}%)")
    print(f"  Overall score:  {report['overall_score']:.3f}")
    print(f"  ESTIMATED IQ:   {report['iq']}")
    print(f"  Duration:       {duration:.1f}s")
    print()
    print("  Per-dimension scores:")
    for dim, score in sorted(report["dimension_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 30)
        print(f"    {dim:22s} {score:.3f} |{bar}")
    print()
    print("  Strengths:")
    for dim, score in report["strengths"]:
        print(f"    {dim:22s} {score:.3f}")
    print()
    print("  Weaknesses:")
    for dim, score in report["weaknesses"]:
        print(f"    {dim:22s} {score:.3f}")
    print()

    # IQ interpretation
    iq = report["iq"]
    if iq >= 145:
        level = "GENIUS (top 0.1% of humans)"
    elif iq >= 130:
        level = "GIFTED (top 2% of humans)"
    elif iq >= 115:
        level = "ABOVE AVERAGE (top 16%)"
    elif iq >= 85:
        level = "AVERAGE"
    else:
        level = "BELOW AVERAGE"
    print(f"  IQ interpretation: {iq} → {level}")
    print("=" * 76)

    # Show failed tests
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  Failed tests ({len(failed)}):")
        for r in failed:
            print(f"    [{r.dimension}] {r.test_name}")
            print(f"      Response: {r.response[:100]}")
            print(f"      Expected: {r.expected}")
    else:
        print(f"\n  ✓ All {len(results)} tests passed — perfect score!")

    # Save report
    report_data = {
        "agent_version": agent.VERSION,
        "timestamp": time.time(),
        "iq": iq,
        "overall_score": report["overall_score"],
        "pass_rate": report["pass_rate"],
        "n_tests": report["n_tests"],
        "n_passed": report["n_passed"],
        "duration_s": duration,
        "dimension_scores": report["dimension_scores"],
    }
    os.makedirs("/home/z/my-project/download", exist_ok=True)
    with open("/home/z/my-project/download/aether_iq_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved: /home/z/my-project/download/aether_iq_report.json")


if __name__ == "__main__":
    main()
