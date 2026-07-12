"""
demo_advanced.py — Advanced AETHER demonstrations.

Shows capabilities beyond basic Q&A:
  1. Multi-hop reasoning (transitive KB queries)
  2. Tool chaining (tool output fed into reasoning)
  3. Noise-robust retrieval (fuzzy queries)
  4. Live learning + immediate query (zero-shot)
  5. Cognitive trace introspection
  6. Comparison: instant learning vs transformer-style retraining
"""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def demo_multihop():
    banner("1. Multi-hop Reasoning (transitive KB query)")

    agent = AETHER()
    # Build a small knowledge graph
    print("  Building knowledge graph:")
    facts = [
        "Montreal is located in Canada",
        "Toronto is located in Canada",
        "Vancouver is located in Canada",
        "Paris is located in France",
        "Lyon is located in France",
        "Tokyo is located in Japan",
        "Osaka is located in Japan",
        # Capitals
        "Ottawa is the capital of Canada",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
    ]
    for f in facts:
        agent.teach(f, silent=True)
        print(f"    + {f}")

    print("\n  Multi-hop question:")
    q = "What is the capital of the country where Montreal is located?"
    print(f"  > {q}")
    ans = agent.ask(q, explain=True)
    print(f"  => {ans}")

    print()
    q = "What is the capital of the country where Osaka is located?"
    print(f"  > {q}")
    ans = agent.ask(q, explain=True)
    print(f"  => {ans}")


def demo_tool_chaining():
    banner("2. Tool Chaining (calc output fed back into reasoning)")

    agent = AETHER()
    queries = [
        "calc 5 * 8",                                # 40
        "calc (10 + 5) * 3",                         # 45
        "teach Gravity is 9.8",                      # learn a fact
        "What is gravity?",                          # retrieve
        "python [1, 2, 3]",                          # safe list eval
        "time",                                      # current time
    ]
    for q in queries:
        ans = agent.ask(q)
        print(f"  > {q}")
        print(f"    => {ans}")


def demo_noise_robust():
    banner("3. Noise-Robust Retrieval (fuzzy queries)")

    agent = AETHER()
    # The agent already has 'Paris is the capital of France' from bootstrap

    queries = [
        "What is the capital of France?",
        "What is the capital of France",        # missing ?
        "what is capital of france",            # missing 'the'
        "capital of France",                    # terse
        "Capital of FRANCE",                    # case
        "what is the capital of france??",      # extra punctuation
    ]
    for q in queries:
        ans = agent.ask(q)
        print(f"  > {q!r:50s} => {ans!r}")


def demo_live_learning():
    banner("4. Live Learning + Immediate Query (zero-shot, no retraining)")

    agent = AETHER()
    print(f"  Initial KB size: {len(agent.assoc.triples)} triples\n")

    # Teach new facts one at a time and query them immediately
    new_facts = [
        ("teach Reykjavik is the capital of Iceland",    "What is the capital of Iceland?"),
        ("teach Ulaanbaatar is the capital of Mongolia", "What is the capital of Mongolia?"),
        ("teach Antananarivo is the capital of Madagascar", "What is the capital of Madagascar?"),
        ("teach Pluto is a dwarf planet",                "What is Pluto?"),
        ("teach Helium is a noble gas",                  "What is Helium?"),
    ]
    for teach_cmd, query in new_facts:
        # Teach
        t0 = time.perf_counter()
        agent.ask(teach_cmd)
        teach_ms = (time.perf_counter() - t0) * 1000
        # Query
        t0 = time.perf_counter()
        ans = agent.ask(query)
        query_ms = (time.perf_counter() - t0) * 1000
        print(f"  {teach_cmd}")
        print(f"    teach: {teach_ms:.1f}ms | {query}")
        print(f"    query: {query_ms:.1f}ms | => {ans}")
        print()


def demo_introspection():
    banner("5. Cognitive Trace Introspection")

    agent = AETHER()
    agent.teach("Lisbon is the capital of Portugal", silent=True)
    agent.teach("Stockholm is the capital of Sweden", silent=True)

    q = "What is the capital of Portugal?"
    print(f"  Question: {q}\n")
    agent.ask(q, explain=True)
    print()
    print("  Full cognitive trace:")
    print(agent.explain_last())


def demo_paradigm_comparison():
    banner("6. Paradigm Comparison: AETHER vs Transformer-style")

    agent = AETHER()
    print("  Scenario: learn 10 new facts, query each immediately.\n")

    facts = [
        ("Reykjavik is the capital of Iceland",     "What is the capital of Iceland?"),
        ("Ulaanbaatar is the capital of Mongolia",  "What is the capital of Mongolia?"),
        ("Antananarivo is the capital of Madagascar","What is the capital of Madagascar?"),
        ("Suva is the capital of Fiji",             "What is the capital of Fiji?"),
        ("Port Vila is the capital of Vanuatu",     "What is the capital of Vanuatu?"),
        ("Honiara is the capital of Solomon Islands","What is the capital of Solomon?"),
        ("Palikir is the capital of Micronesia",    "What is the capital of Micronesia?"),
        ("Yaren is the capital of Nauru",           "What is the capital of Nauru?"),
        ("Tarawa is the capital of Kiribati",       "What is the capital of Kiribati?"),
        ("Funafuti is the capital of Tuvalu",       "What is the capital of Tuvalu?"),
    ]

    correct = 0
    total_teach_ms = 0.0
    total_query_ms = 0.0

    for fact, query in facts:
        # Teach directly via the API (instant learning)
        t0 = time.perf_counter()
        agent.teach(fact, silent=True)
        total_teach_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ans = agent.ask(query)
        total_query_ms += (time.perf_counter() - t0) * 1000

        # Check correctness (case-insensitive substring match)
        expected = fact.split(" is the capital of ")[0].lower()
        ok = expected in ans.lower() or ans.lower() in expected
        if ok:
            correct += 1
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {fact:55s} -> {ans}")

    print(f"\n  Results:")
    print(f"    Accuracy:    {correct}/{len(facts)} ({100*correct/len(facts):.0f}%)")
    print(f"    Avg teach:   {total_teach_ms/len(facts):.1f} ms  (transformer: minutes to hours for fine-tune)")
    print(f"    Avg query:   {total_query_ms/len(facts):.1f} ms  (transformer: ~50-500ms on GPU)")
    print(f"    GPU:         NOT USED  (pure CPU numpy)")
    print(f"    Transformer: NOT USED  (VSA + SDM + cognitive loop)")
    print(f"    External LLM: NOT USED (fully local)")


def main():
    demos = [
        demo_multihop,
        demo_tool_chaining,
        demo_noise_robust,
        demo_live_learning,
        demo_introspection,
        demo_paradigm_comparison,
    ]
    for d in demos:
        try:
            d()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {d.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 72)
    print("  All advanced demos complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
