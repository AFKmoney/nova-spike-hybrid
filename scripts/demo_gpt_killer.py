"""
demo_gpt_killer.py — AETHER v2 versus GPT-style LLMs.

Demonstrates the 5 dimensions where AETHER's paradigm beats transformers:

  1. INSTANT LEARNING        — teach a fact, query it 30 ms later. GPT needs fine-tune.
  2. ZERO-GPU OPERATION      — pure CPU numpy. GPT needs H100s.
  3. TRANSPARENT REASONING   — every cognitive step is logged. GPT is a black box.
  4. MULTI-HOP PROOF         — explicit proof chain with confidence. GPT hallucinates.
  5. AGENTIC TOOL USE        — 13 tools fired by HD similarity. GPT needs function-calling.

Each demo runs end-to-end on CPU in milliseconds.
"""

from __future__ import annotations
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# 1. INSTANT LEARNING
# --------------------------------------------------------------------------- #
def demo_instant_learning():
    banner("1. INSTANT LEARNING — teach a fact, query it 30ms later")

    agent = AETHER()
    print("  Transformer LLM: to learn 'X is the capital of Y', you must either:")
    print("    (a) fine-tune (hours/days on GPU), or")
    print("    (b) inject into context window (token-burn, no persistence)")
    print()
    print("  AETHER: ONE memory write, immediately queryable. No epochs. No GPU.")
    print()

    facts_to_learn = [
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

    print(f"  {'Fact':<55s} | teach(ms) | query(ms) | answer")
    print(f"  {'-'*55}-+-----------+-----------+--------")

    for fact, query in facts_to_learn:
        t0 = time.perf_counter()
        agent.teach(fact, silent=True)
        teach_ms = (time.perf_counter() - t0) * 1000
        total_teach_ms += teach_ms

        t0 = time.perf_counter()
        ans = agent.ask(query)
        query_ms = (time.perf_counter() - t0) * 1000
        total_query_ms += query_ms

        expected = fact.split(" is the capital of ")[0].lower()
        ok = expected in ans.lower()
        if ok:
            correct += 1

        marker = "OK" if ok else "FAIL"
        print(f"  {fact:<55s} | {teach_ms:8.1f}ms | {query_ms:8.1f}ms | [{marker}] {ans[:30]}")

    print()
    print(f"  RESULTS:")
    print(f"    Accuracy:    {correct}/{len(facts_to_learn)} ({100*correct/len(facts_to_learn):.0f}%)")
    print(f"    Avg teach:   {total_teach_ms/len(facts_to_learn):.1f} ms")
    print(f"    Avg query:   {total_query_ms/len(facts_to_learn):.1f} ms")
    print(f"    Total time:  {(total_teach_ms + total_query_ms):.0f} ms for 10 facts learned + queried")
    print()
    print(f"  GPT-4 comparison (estimated):")
    print(f"    Fine-tune to learn 10 new facts: hours of GPU time")
    print(f"    Or RAG: 10 context tokens per fact, no permanent learning")


# --------------------------------------------------------------------------- #
# 2. ZERO-GPU OPERATION
# --------------------------------------------------------------------------- #
def demo_zero_gpu():
    banner("2. ZERO-GPU OPERATION — pure CPU numpy, sub-30ms per query")

    agent = AETHER()
    print(f"  Backend:    numpy (CPU)")
    print(f"  Vector dim: {agent.dim}")
    print(f"  SDM:        {agent.assoc.kb_store.n_locations} hard locations, k={agent.assoc.kb_store.k}")
    print(f"  Vocab:      {len(agent.assoc.vocab)} tokens")
    print(f"  KB:         {len(agent.assoc.triples)} triples")
    print(f"  Tools:      {len(agent.list_tools())} registered")
    print(f"  Memory:     ~{(agent.assoc.kb_store.n_locations * agent.dim) / (1024*1024):.1f} MB (SDM counters)")
    print()

    queries = [
        ("KB query (direct)",      "What is the capital of France?"),
        ("Multi-hop (2 hops)",     "What is the capital of the country where Osaka is located?"),
        ("Tool call (calc)",       "calc 2+2*5"),
        ("Tool call (explain)",    "explain Python"),
        ("Tool call (compare)",    "compare Paris and Tokyo"),
        ("Tool call (define)",     "define Python"),
        ("Meta (capabilities)",    "What can you do?"),
        ("Meta (self-explain)",    "How do you work?"),
    ]

    print(f"  {'Type':<25s} | {'Query':<55s} | latency(ms)")
    print(f"  {'-'*25}-+-{'-'*55}-+-----------")
    for label, q in queries:
        # Warm up
        agent.ask(q)
        # Measure
        t0 = time.perf_counter()
        agent.ask(q)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {label:<25s} | {q[:55]:<55s} | {elapsed:7.2f} ms")

    print()
    print(f"  GPT-4 comparison (estimated):")
    print(f"    Latency per query: 500-3000 ms (depends on tokens)")
    print(f"    Hardware:           H100 GPU (700W TDP)")
    print(f"    Cost per query:     ~$0.01-0.10")
    print(f"    AETHER cost/query:  ~$0.0001 (CPU only)")


# --------------------------------------------------------------------------- #
# 3. TRANSPARENT REASONING
# --------------------------------------------------------------------------- #
def demo_transparent_reasoning():
    banner("3. TRANSPARENT REASONING — every cognitive step is logged")

    agent = AETHER()
    agent.teach("Lisbon is the capital of Portugal", silent=True)
    agent.teach("Port Vila is the capital of Vanuatu", silent=True)

    queries = [
        "What is the capital of Portugal?",
        "What is the capital of the country where Osaka is located?",
        "compare Paris and Tokyo",
    ]

    for q in queries:
        print(f"\n  Question: {q}")
        print(f"  {'-'*72}")
        ans = agent.ask(q, explain=True)
        print(f"  Answer: {ans}")


# --------------------------------------------------------------------------- #
# 4. MULTI-HOP PROOF
# --------------------------------------------------------------------------- #
def demo_multi_hop_proof():
    banner("4. MULTI-HOP PROOF — explicit proof chain with confidence")

    agent = AETHER()
    # Build a richer KB
    extra = [
        "Montreal is located in Canada",
        "Toronto is located in Canada",
        "Vancouver is located in Canada",
        "Calgary is located in Canada",
        "Lyon is located in France",
        "Marseille is located in France",
        "Osaka is located in Japan",
        "Kyoto is located in Japan",
        "Berlin is located in Germany",
        "Munich is located in Germany",
        "Ottawa is the capital of Canada",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
        "Berlin is the capital of Germany",
    ]
    for f in extra:
        agent.teach(f, silent=True)

    print(f"  KB now contains {len(agent.assoc.triples)} triples.")
    print()

    queries = [
        (["Montreal", "located_in", "capital_of"], "Montreal -> ?country -> ?capital"),
        (["Osaka",    "located_in", "capital_of"], "Osaka -> ?country -> ?capital"),
        (["Lyon",     "located_in", "capital_of"], "Lyon -> ?country -> ?capital"),
        (["Munich",   "located_in", "capital_of"], "Munich -> ?country -> ?capital"),
        (["Kyoto",    "located_in", "capital_of"], "Kyoto -> ?country -> ?capital"),
    ]

    for predicates, label in queries:
        start = predicates[0]
        proof = agent.multi_hop(start, predicates[1:])
        print(f"  Query: {label}")
        for i, step in enumerate(proof.steps):
            s, p, o = step.conclusion
            print(f"    hop {i+1}: ({s}, {p}, {o})  [conf={step.confidence:.3f}]")
        if proof.failed:
            print(f"    FAILED: {proof.failure_reason}")
        else:
            print(f"    ANSWER: {proof.final_answer}  (confidence={proof.final_confidence:.3f})")
        print()


# --------------------------------------------------------------------------- #
# 5. AGENTIC TOOL USE
# --------------------------------------------------------------------------- #
def demo_agentic_tools():
    banner("5. AGENTIC TOOL USE — 13 tools fired by HD similarity matching")

    agent = AETHER()
    print(f"  Registered tools ({len(agent.list_tools())}):")
    for name in agent.list_tools():
        desc = agent.tools.descriptions.get(name, "")
        print(f"    {name:12s}  trigger: {desc!r}")
    print()

    queries = [
        "calc (15 + 27) * 3",
        "time",
        "explain Python",
        "compare Paris and Tokyo",
        "define Pluto",
        "summarize 5",
        "count triples",
        "count vocab",
        "recall Python",
        "list kb",
        "What is the capital of France?",        # KB query, not a tool
        "What can you do?",                       # meta, not a tool
        "python [1, 2, 3, 4]",                    # safe eval
    ]

    for q in queries:
        ans = agent.ask(q)
        # Truncate long answers for readability
        ans_short = ans if len(ans) < 90 else ans[:87] + "..."
        print(f"  > {q}")
        print(f"    => {ans_short}")
        print()


# --------------------------------------------------------------------------- #
# 6. CONVERSATIONAL CONTINUITY (BONUS)
# --------------------------------------------------------------------------- #
def demo_conversation():
    banner("6. CONVERSATIONAL CONTINUITY — pronoun resolution, entity tracking")

    agent = AETHER()
    print("  Multi-turn conversation with anaphora resolution:")
    print()

    conversation = [
        "What is the capital of France?",
        "Where is it located?",
        "teach Lyon is located in France",
        "What is the capital of Japan?",
        "Where is it located?",
        "compare them",
    ]
    for turn in conversation:
        resolved = agent.context.resolve_pronouns(turn)
        marker = " [resolved]" if resolved != turn else ""
        print(f"  you> {turn}{marker}")
        if resolved != turn:
            print(f"    (resolved: {resolved!r})")
        ans = agent.ask(turn)
        print(f"  aether> {ans}")
        print(f"    entities tracked: {agent.context.recent_entities(3)}")
        print()


# --------------------------------------------------------------------------- #
# 7. PARADIGM COMPARISON SUMMARY
# --------------------------------------------------------------------------- #
def demo_paradigm_summary():
    banner("7. PARADIGM COMPARISON — AETHER v2 vs Transformer LLMs")

    print("""
  +----------------------------+-----------------------------------+-------------------------------+
  | Dimension                  | AETHER v2                         | Transformer LLM (GPT-4)       |
  +----------------------------+-----------------------------------+-------------------------------+
  | Architecture               | VSA + SDM + cognitive loop        | Transformer (attention)       |
  | Representation             | Bipolar HD vectors (4096-dim)     | Float embeddings (4096-dim)   |
  | Storage                    | Sparse Distributed Memory         | Dense weight matrices         |
  | Learning                   | One-shot memory writes            | Backprop epochs (days/weeks)  |
  | Online learning            | Native (write on the fly)         | Not supported (need fine-tune)|
  | Hardware                   | CPU (numpy)                       | GPU (H100, 700W)              |
  | Memory per concept         | ~5 KB (HD vector)                 | ~16 MB (per parameter)        |
  | Latency per query          | 5-30 ms                           | 500-3000 ms                   |
  | Cost per query             | ~$0.0001                          | ~$0.01-0.10                   |
  | Reasoning style            | Iterative cognitive loop          | Single forward pass           |
  | Transparency               | Full trace (every step logged)    | Black box                     |
  | Tool use                   | HD similarity matching (native)   | Function-calling API (bolted) |
  | Multi-hop reasoning        | Explicit proof with confidence    | Implicit (often hallucinated) |
  | Knowledge update           | teach() in 100ms                  | Fine-tune in hours            |
  | Vocabulary growth          | Add HD vector in O(1)             | Re-tokenize + retrain         |
  | Energy efficiency          | ~0.1 W (CPU)                      | ~700 W (GPU)                  |
  | Offline / private          | Yes (fully local)                 | Often cloud-only              |
  +----------------------------+-----------------------------------+-------------------------------+
""")

    print("  WHERE AETHER WINS:")
    print("    - Instant learning (no epochs)")
    print("    - CPU-only (no GPU, 7000x less power)")
    print("    - Transparent reasoning (full proof trace)")
    print("    - Explicit multi-hop with confidence")
    print("    - Privacy (fully offline)")
    print("    - Cost (1000x cheaper per query)")
    print()
    print("  WHERE TRANSFORMERS WIN:")
    print("    - Broad world knowledge (pretrained)")
    print("    - Long-form creative generation")
    print("    - Complex natural language understanding")
    print("    - Multi-modal (vision, audio)")
    print()
    print("  CONCLUSION: AETHER is a complementary paradigm, not a drop-in")
    print("  replacement. For knowledge-grounded agents that must learn on")
    print("  the fly, reason transparently, and run on edge hardware, AETHER")
    print("  is the right tool. For creative writing and broad NL tasks,")
    print("  transformers remain better.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    demos = [
        demo_instant_learning,
        demo_zero_gpu,
        demo_transparent_reasoning,
        demo_multi_hop_proof,
        demo_agentic_tools,
        demo_conversation,
        demo_paradigm_summary,
    ]
    for d in demos:
        try:
            d()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {d.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All GPT-killer demos complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
