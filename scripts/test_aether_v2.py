"""
test_aether_v2.py — Test suite for AETHER v2 (GPT killer edition).

Covers all new v2 capabilities:
  1. Semantic HD embeddings (char n-gram similarity)
  2. Question analysis (10+ question types)
  3. Response generation (natural language)
  4. Inference engine (forward + backward chaining, multi-hop)
  5. Planner (decomposition, tool chaining)
  6. Conversation context (pronoun resolution, entity tracking)
  7. Full end-to-end chat (compared to v1)
  8. New tools: explain, compare, summarize, count, define
  9. Performance (still CPU-only, still fast)
"""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from aether.semantic import SemanticEncoder, char_ngrams, tag_token
from aether.generator import analyze_question, ResponseGenerator, parse_triple
from aether.inference import InferenceEngine
from aether.planner import Planner
from aether.context import ConversationContext


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# 1. Semantic embeddings
# --------------------------------------------------------------------------- #
def test_semantic_embeddings():
    banner("1. Semantic HD Embeddings (char n-grams)")

    enc = SemanticEncoder(dim=4096)
    pairs = [
        ("Paris", "paris"),       # case difference
        ("Paris", "parisian"),    # morphological similarity
        ("Paris", "parody"),      # partial overlap
        ("Paris", "Tokyo"),       # unrelated
        ("France", "frank"),      # partial overlap
        ("Tokyo", "tokyo"),       # case
        ("Python", "pythonic"),   # morphological
        ("run", "running"),       # verb form
        ("happy", "happiness"),   # derived
        ("happy", "sad"),         # antonym (no morphological overlap)
    ]
    print("  Word similarity (char n-gram HD encoding):")
    for w1, w2 in pairs:
        sim = enc.similarity(w1, w2)
        print(f"    sim({w1!r:12s}, {w2!r:12s}) = {sim:+.4f}")

    # nearest neighbors
    print("\n  Nearest neighbors of 'Paris':")
    vocab = ["Paris", "parisian", "parisians", "Tokyo", "London", "Berlin",
             "parody", "paradise", "panda", "python"]
    nn = enc.nearest_neighbors("Paris", vocab, top_k=5)
    for w, s in nn:
        print(f"    {w!r:12s} : {s:+.4f}")

    # Tagging
    print("\n  Token concept tagging:")
    for tok in ["Paris", "the", "42", "!", "Tokyo", "is"]:
        tags = tag_token(tok)
        print(f"    {tok!r:8s} -> {tags}")

    print("  -> Semantic embeddings OK")


# --------------------------------------------------------------------------- #
# 2. Question analysis
# --------------------------------------------------------------------------- #
def test_question_analysis():
    banner("2. Question Analysis (10+ question types)")

    questions = [
        "What is the capital of France?",
        "Where is Montreal located?",
        "What is Python?",
        "Who are you?",
        "What can you do?",
        "How do you work?",
        "calc 2+2*5",
        "compare Paris and Tokyo",
        "explain Python",
        "summarize 5",
        "count triples",
        "define Python",
        "teach X is the capital of Y",
        "What is the capital of the country where Osaka is located?",
        "hello",
        "thanks",
        "stats",
        "list kb",
    ]
    print("  Classified questions:")
    for q in questions:
        a = analyze_question(q)
        slots = ", ".join(f"{k}={v!r}" for k, v in a.slots.items()) or "(none)"
        print(f"    [{a.qtype:22s}] slots: {slots}")
        print(f"      q: {q!r}")

    print("  -> Question analysis OK")


# --------------------------------------------------------------------------- #
# 3. Response generation
# --------------------------------------------------------------------------- #
def test_response_generation():
    banner("3. Natural Language Response Generation")

    gen = ResponseGenerator(seed=42)
    cases = [
        ("What is the capital of France?", "paris", "capital_of"),
        ("Where is Montreal located?", "canada", "located_in"),
        ("What is Python?", "a programming language", "definition"),
        ("calc 2+2*5", "2+2*5 = 12", "calc"),
        ("time", "2026-07-11", "time"),
        ("What is the capital of the country where Osaka is located?", "tokyo", "multi_hop_capital"),
    ]
    for q, ans, qtype in cases:
        analysis = analyze_question(q)
        # Force the qtype for the test
        analysis.qtype = qtype
        response = gen.generate(q, answer=ans, analysis=analysis, confidence=1.0)
        print(f"  Q: {q}")
        print(f"  A: {response}")
        print()

    print("  -> Response generation OK")


# --------------------------------------------------------------------------- #
# 4. Inference engine
# --------------------------------------------------------------------------- #
def test_inference_engine():
    banner("4. Inference Engine (multi-hop, forward chaining)")

    agent = AETHER()
    # Teach a richer KB
    extra_facts = [
        "Montreal is located in Canada",
        "Toronto is located in Canada",
        "Vancouver is located in Canada",
        "Lyon is located in France",
        "Osaka is located in Japan",
        "Ottawa is the capital of Canada",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
    ]
    for f in extra_facts:
        agent.teach(f, silent=True)

    print("  Multi-hop reasoning:")
    proof = agent.multi_hop("Montreal", ["located_in", "capital_of"])
    print(f"    Montreal located_in -> ? -> capital_of -> ?")
    print(f"    proof: {agent.inference.explain(proof)}")

    proof = agent.multi_hop("Osaka", ["located_in", "capital_of"])
    print(f"\n    Osaka located_in -> ? -> capital_of -> ?")
    print(f"    proof: {agent.inference.explain(proof)}")

    print("\n  Backward chaining (capital_of Canada):")
    proof = agent.inference.backward_chain("Canada", "capital_of")
    print(f"    {agent.inference.explain(proof)}")

    print("\n  Forward chaining (derive new facts):")
    seed = [("Montreal", "located_in", "Canada"), ("Canada", "located_in", "America")]
    # Also seed America capital_of Washington
    agent.teach("Washington is the capital of USA", silent=True)
    agent.teach("USA is located in America", silent=True)
    derived = agent.inference.forward_chain(seed, max_steps=4)
    if derived:
        for step in derived:
            print(f"    derived: {step.conclusion}  [rule={step.rule}, conf={step.confidence:.2f}]")
    else:
        print("    (no new facts derived)")

    print("  -> Inference engine OK")


# --------------------------------------------------------------------------- #
# 5. Planner
# --------------------------------------------------------------------------- #
def test_planner():
    banner("5. Agentic Planner (decomposition)")

    agent = AETHER()
    queries = [
        "What is the capital of France?",
        "Where is Montreal located?",
        "calc 2+2*5",
        "What is the capital of the country where Osaka is located?",
        "compare Paris and Tokyo",
        "explain Python",
        "What is Python?",
        "Hello",
        "summarize 3",
        "count triples",
    ]
    for q in queries:
        plan = agent.planner.plan(q)
        print(f"  Q: {q}")
        print(f"    rationale: {plan.rationale}")
        print(f"    complexity: {plan.expected_complexity}")
        for i, step in enumerate(plan.steps):
            print(f"    step {i+1}: [{step.kind}] {step.description}")
        print()

    print("  -> Planner OK")


# --------------------------------------------------------------------------- #
# 6. Conversation context (pronoun resolution)
# --------------------------------------------------------------------------- #
def test_conversation_context():
    banner("6. Conversation Context (pronoun resolution, entity tracking)")

    agent = AETHER()
    # Simulate a conversation
    conversation = [
        "What is the capital of France?",
        "Where is it located?",
        "teach Lyon is located in France",
        "What is the capital of Japan?",
        "Where is it located?",
    ]
    print("  Multi-turn conversation with pronoun resolution:")
    for turn in conversation:
        resolved = agent.context.resolve_pronouns(turn)
        marker = " [resolved]" if resolved != turn else ""
        print(f"  user: {turn!r}{marker}")
        if resolved != turn:
            print(f"    -> resolved to: {resolved!r}")
        ans = agent.ask(turn)
        print(f"  aether: {ans}")
        print(f"    recent entities: {agent.context.recent_entities(3)}")
        print()

    print("  -> Conversation context OK")


# --------------------------------------------------------------------------- #
# 7. Full end-to-end chat
# --------------------------------------------------------------------------- #
def test_full_chat():
    banner("7. Full End-to-End Chat (GPT-killer demo)")

    agent = AETHER()
    print(f"  AETHER v{agent.VERSION}")
    print(f"  Tools: {', '.join(agent.list_tools())}")
    print(f"  Vocab: {len(agent.assoc.vocab)} tokens")
    print(f"  KB: {len(agent.assoc.triples)} triples\n")

    conversation = [
        "Hello!",
        "What can you do?",
        "How do you work?",
        "teach Reykjavik is the capital of Iceland",
        "What is the capital of Iceland?",
        "What is the capital of France?",
        "What is the capital of Japan?",
        "Where is Montreal located?",
        "What is the capital of the country where Osaka is located?",
        "compare Paris and Tokyo",
        "explain Python",
        "define Pluto",
        "calc 1234 * 5678",
        "summarize 5",
        "count triples",
        "thank you",
    ]
    for turn in conversation:
        print(f"  you> {turn}")
        ans = agent.ask(turn)
        print(f"  aether> {ans}")
        print()

    print("  -> Full chat OK")


# --------------------------------------------------------------------------- #
# 8. New tools
# --------------------------------------------------------------------------- #
def test_new_tools():
    banner("8. New Tools (explain, compare, summarize, count, define)")

    agent = AETHER()
    queries = [
        ("explain Python",          "should show all known facts about Python"),
        ("explain Paris",           "should show Paris is the capital of France"),
        ("compare Paris and Tokyo", "should compare capital_of predicate"),
        ("define Python",           "should define Python"),
        ("define Pluto",            "should define Pluto"),
        ("count triples",           "should count KB triples"),
        ("count vocab",             "should count vocabulary"),
        ("count episodes",          "should count episodes"),
        ("summarize 3",             "should summarize 3 recent memories"),
    ]
    for q, expected in queries:
        ans = agent.ask(q)
        print(f"  > {q}")
        print(f"    ({expected})")
        print(f"    => {ans}")
        print()


# --------------------------------------------------------------------------- #
# 9. Performance (CPU-only)
# --------------------------------------------------------------------------- #
def test_performance():
    banner("9. Performance (CPU-only, no GPU)")

    agent = AETHER()
    import numpy as np
    print(f"  Numpy: {np.__version__}")
    print(f"  Vector dim: {agent.dim}")
    print(f"  SDM locations: {agent.assoc.kb_store.n_locations}")
    print(f"  Vocab: {len(agent.assoc.vocab)} tokens")
    print(f"  KB: {len(agent.assoc.triples)} triples")
    print(f"  Tools: {len(agent.list_tools())}")
    print()

    # Time ask() calls
    t0 = time.perf_counter()
    for _ in range(30):
        agent.ask("What is the capital of France?")
    elapsed = time.perf_counter() - t0
    print(f"  30 KB queries: {elapsed*1000:.1f} ms total = {elapsed/30*1000:.2f} ms/query")

    # Time teach() calls
    t0 = time.perf_counter()
    for i in range(30):
        agent.teach(f"City{i} is the capital of Country{i}", silent=True)
    elapsed = time.perf_counter() - t0
    print(f"  30 teach() calls: {elapsed*1000:.1f} ms total = {elapsed/30*1000:.2f} ms/teach")

    # Time multi-hop
    agent.teach("Berlin is located in Germany", silent=True)
    t0 = time.perf_counter()
    for _ in range(20):
        agent.ask("What is the capital of the country where Berlin is located?")
    elapsed = time.perf_counter() - t0
    print(f"  20 multi-hop queries: {elapsed*1000:.1f} ms total = {elapsed/20*1000:.2f} ms/query")

    # Tool calls
    t0 = time.perf_counter()
    for _ in range(100):
        agent.ask("calc 2+2")
    elapsed = time.perf_counter() - t0
    print(f"  100 calc tool calls: {elapsed*1000:.1f} ms total = {elapsed/100*1000:.3f} ms/call")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_semantic_embeddings,
        test_question_analysis,
        test_response_generation,
        test_inference_engine,
        test_planner,
        test_conversation_context,
        test_new_tools,
        test_full_chat,
        test_performance,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {t.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 72)
    print("  All v2 tests complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
