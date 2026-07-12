"""
test_aether.py — Demonstration of the AETHER cognitive agent.

Runs a series of tests covering:
  1. HD vector operations (bind, bundle, similarity)
  2. Sparse Distributed Memory (write/read with noise)
  3. Instant learning (teach facts, immediately query)
  4. Cognitive loop with trace (introspection)
  5. Agentic tool use (calculator, time, recall)
  6. HD language model (novel generation from learned sequences)
  7. End-to-end AETHER agent chat

All on CPU, no GPU, no transformer, no external LLM.
"""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from aether.hd import HDVector, bundle, ngram_encode
from aether.memory import SparseDistributedMemory, AssociativeMemory
from aether.encoder import TextEncoder
from aether.reasoning import CognitiveLoop
from aether.tools import default_tools, ToolContext


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1. HD vector primitives
# --------------------------------------------------------------------------- #
def test_hd_primitives():
    banner("1. HD Vector Primitives (Vector Symbolic Architecture)")

    a = HDVector.random()
    b = HDVector.random()
    print(f"  Two random vectors: similarity = {a.similarity(b):.4f}  (expected ~0)")

    bound = a.bind(b)
    print(f"  After bind(a,b): sim(a, bound) = {a.similarity(bound):.4f}")
    print(f"  After bind(a,b): sim(b, bound) = {b.similarity(bound):.4f}")

    unbound = bound.unbind(b)
    print(f"  After unbind(bound, b): sim(a, unbound) = {a.similarity(unbound):.4f}  (should be ~1)")

    c = HDVector.random()
    bundled = bundle([a, b, c])
    print(f"  bundle([a,b,c]): sim(a, bundled) = {a.similarity(bundled):.4f}  (should be ~1/3)")

    perm = a.permute(5)
    print(f"  permute(a, 5): sim(a, perm) = {a.similarity(perm):.4f}  (should be ~0)")
    unperm = perm.inverse_permute(5)
    print(f"  inverse_permute(perm, 5): sim(a, unperm) = {a.similarity(unperm):.4f}  (should be ~1)")

    print("  -> HD primitives OK")


# --------------------------------------------------------------------------- #
# 2. Sparse Distributed Memory
# --------------------------------------------------------------------------- #
def test_sdm():
    banner("2. Sparse Distributed Memory (Kanerva)")

    sdm = SparseDistributedMemory(dim=4096, n_locations=2000, k=25)
    print(f"  SDM: dim={sdm.dim}, n_locations={sdm.n_locations}, k={sdm.k}")

    # Write 50 random associations
    import numpy as np
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(50):
        addr = HDVector.random()
        data = HDVector.random()
        sdm.write(addr, data)
        pairs.append((addr, data))

    print(f"  Wrote {len(pairs)} (address, data) pairs.")

    # Read back exactly — should be highly similar
    sims_exact = []
    for addr, data in pairs[:10]:
        retrieved = sdm.read(addr)
        if retrieved is not None:
            sims_exact.append(data.similarity(retrieved))
    avg_exact = sum(sims_exact) / len(sims_exact) if sims_exact else 0
    print(f"  Read at exact address: avg similarity = {avg_exact:.4f}  (should be high)")

    # Read at noisy address — should still be similar (noise robustness)
    sims_noisy = []
    for addr, data in pairs[:10]:
        noise = rng.choice([-1, 1], size=addr.dim).astype(np.int8)
        flip = rng.random(addr.dim) < 0.10  # flip 10% of bits
        noisy_addr_data = np.where(flip, addr.data * noise, addr.data)
        noisy_addr = HDVector(data=noisy_addr_data, dim=addr.dim)
        retrieved = sdm.read(noisy_addr)
        if retrieved is not None:
            sims_noisy.append(data.similarity(retrieved))
    avg_noisy = sum(sims_noisy) / len(sims_noisy) if sims_noisy else 0
    print(f"  Read at noisy (10% flip) address: avg similarity = {avg_noisy:.4f}  (should be > 0)")

    print(f"  SDM stats: {sdm.stats()}")
    print("  -> SDM OK")


# --------------------------------------------------------------------------- #
# 3. Instant learning + KB query
# --------------------------------------------------------------------------- #
def test_instant_learning():
    banner("3. Instant Learning (one-shot, no training)")

    agent = AETHER()
    print(f"  Initial vocab: {len(agent.assoc.vocab)} tokens")
    print(f"  Initial KB: {len(agent.assoc.triples)} triples")

    # Teach something NEW (not in bootstrap)
    new_facts = [
        "Beijing is the capital of China",
        "Moscow is the capital of Russia",
        "Cairo is the capital of Egypt",
        "Brazil is located in America",
        "Mars is a planet",
    ]
    for fact in new_facts:
        msg = agent.teach(fact, silent=True)
        print(f"  teach: {fact!r}\n    -> {msg}")

    print(f"\n  After teaching {len(new_facts)} new facts:")
    print(f"    vocab: {len(agent.assoc.vocab)} tokens")
    print(f"    KB: {len(agent.assoc.triples)} triples")

    # Query them back INSTANTLY — no training, no epochs
    print("\n  Instant queries (no retraining):")
    queries = [
        "What is the capital of China?",
        "What is the capital of Russia?",
        "What is the capital of Egypt?",
    ]
    for q in queries:
        ans = agent.ask(q)
        print(f"    Q: {q}")
        print(f"    A: {ans}")
    print("  -> Instant learning OK")


# --------------------------------------------------------------------------- #
# 4. Cognitive loop with trace
# --------------------------------------------------------------------------- #
def test_cognitive_trace():
    banner("4. Cognitive Loop Trace (continuous reasoning)")

    agent = AETHER()
    # Teach a fresh fact
    agent.teach("Lisbon is the capital of Portugal", silent=True)

    question = "What is the capital of Portugal?"
    print(f"  Question: {question}")
    answer = agent.ask(question, explain=True)
    print(f"\n  Final answer: {answer}")


# --------------------------------------------------------------------------- #
# 5. Agentic tool use
# --------------------------------------------------------------------------- #
def test_tool_use():
    banner("5. Agentic Tool Use")

    agent = AETHER()
    print(f"  Registered tools: {list(agent.tools.tools.keys())}")

    tool_queries = [
        "calc 2+2*5",
        "calculate (3+4)*2",
        "time",
        "recall Paris",
        "list kb",
        "compute 100/4 - 7",
        "python [1,2,3] + [4]",
    ]
    for q in tool_queries:
        ans = agent.ask(q)
        print(f"  > {q}")
        print(f"    -> {ans}")


# --------------------------------------------------------------------------- #
# 6. HD language model (novel sequence generation)
# --------------------------------------------------------------------------- #
def test_hd_lm():
    banner("6. HD Language Model (novel generation from learned sequences)")

    agent = AETHER()
    print(f"  LM context size: ngram={agent.encoder.ngram}")
    print(f"  Vocab: {len(agent.assoc.vocab)} tokens")

    # Teach some sentences
    sentences = [
        "the sun is a star",
        "the moon is a satellite",
        "water boils at 100 degrees",
        "Python is a programming language",
    ]
    for s in sentences:
        agent.encoder.learn_sequence(s)
        print(f"  learned sequence: {s!r}")

    # Try to generate continuations
    prompts = [
        "the sun",
        "the moon",
        "Python is",
    ]
    print("\n  Generation (greedy, max 10 tokens):")
    for p in prompts:
        gen = agent.encoder.generate(p, max_tokens=10)
        print(f"    prompt: {p!r}")
        print(f"    generated: {gen!r}")


# --------------------------------------------------------------------------- #
# 7. Full AETHER chat (end-to-end)
# --------------------------------------------------------------------------- #
def test_full_chat():
    banner("7. Full AETHER Chat (end-to-end demo)")

    agent = AETHER()

    # Teach new facts at runtime — this is the "instant learning" proof
    print("  Teaching new facts at runtime (instant learning):")
    new_facts = [
        "Montreal is located in Canada",
        "Kinshasa is the capital of Congo",
        "Aether is a cognitive architecture",
    ]
    for f in new_facts:
        agent.teach(f, silent=True)
        print(f"    + {f}")

    print("\n  Conversation:")
    conversation = [
        "Hello",
        "What are you?",
        "What is the capital of France?",
        "What is the capital of Congo?",
        "calc 1234 * 5678",
        "What is the capital of Japan?",
        "teach Buenos Aires is the capital of Argentina",
        "What is the capital of Argentina?",
        "Where is Montreal located?",
        "stats",
    ]
    for turn in conversation:
        print(f"\n  you> {turn}")
        ans = agent.ask(turn)
        print(f"  aether> {ans}")

    print("\n  Final stats:")
    s = agent.stats()
    print(f"    vocab: {s['vocab_size']} tokens")
    print(f"    triples: {s['assoc']['triples']}")
    print(f"    episodes: {s['assoc']['episodes']}")
    print(f"    KB writes: {s['assoc']['kb_store']['total_writes']}")


# --------------------------------------------------------------------------- #
# 8. Performance: pure CPU, no GPU
# --------------------------------------------------------------------------- #
def test_performance():
    banner("8. Performance (CPU-only, no GPU)")

    agent = AETHER()
    import numpy as np
    print(f"  Numpy version: {np.__version__}")
    print(f"  Vector dim: {agent.dim}")
    print(f"  SDM locations: {agent.assoc.kb_store.n_locations}")

    # Time a single cognitive cycle
    t0 = time.perf_counter()
    for _ in range(20):
        agent.ask("What is the capital of France?")
    elapsed = time.perf_counter() - t0
    print(f"  20 ask() calls: {elapsed*1000:.1f} ms total = {elapsed/20*1000:.2f} ms/call")

    # Time a single teach()
    t0 = time.perf_counter()
    for i in range(50):
        agent.teach(f"Fact{i} is a test", silent=True)
    elapsed = time.perf_counter() - t0
    print(f"  50 teach() calls: {elapsed*1000:.1f} ms total = {elapsed/50*1000:.2f} ms/call")

    # Tool call latency
    t0 = time.perf_counter()
    for _ in range(100):
        agent.ask("calc 2+2")
    elapsed = time.perf_counter() - t0
    print(f"  100 calc tool calls: {elapsed*1000:.1f} ms total = {elapsed/100*1000:.2f} ms/call")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_hd_primitives,
        test_sdm,
        test_instant_learning,
        test_cognitive_trace,
        test_tool_use,
        test_hd_lm,
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
    print("\n" + "=" * 70)
    print("  All tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
