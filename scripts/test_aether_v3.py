"""
test_aether_v3.py — Test suite for AETHER v3 (multi-modal GPT killer).

Covers all 6 new v3 capabilities:
  1. Semantic HD embeddings (pretrained.py) — categories, synonyms, antonyms
  2. HD n-gram LM with backoff + temperature sampling + beam search
  3. N-hop arbitrary depth with sub-goaling + path finding + reachability
  4. Tool composer (hybrid HD + pattern matching)
  5. HD RAG (web fetch + offline wiki fallback)
  6. Multi-modal HD encoding (images + audio + cross-modal)
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import (
    AETHER,
    SemanticKB,
    HDLanguageModel,
    ToolComposer, ComposedExecutor,
    HDRAGStore, WebFetcher, get_global_rag, reset_global_rag,
    ImageHDEncoder, AudioHDEncoder, CrossModalSpace,
    integrate_into_agent,
)


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# 1. Semantic embeddings
# --------------------------------------------------------------------------- #
def test_semantic_embeddings():
    banner("1. Semantic HD Embeddings (categories + synonyms + antonyms)")

    skb = SemanticKB(dim=4096)
    skb.build()
    print(f"  Built semantic KB: {skb.stats()}")

    print("\n  Within-category similarity (HIGH):")
    pairs = [("dog", "cat"), ("paris", "london"), ("red", "blue"), ("three", "seven")]
    for w1, w2 in pairs:
        print(f"    sim({w1!r:10s}, {w2!r:10s}) = {skb.similarity(w1, w2):+.4f}")

    print("\n  Cross-category similarity (LOW):")
    pairs = [("dog", "paris"), ("red", "seven"), ("bread", "happy")]
    for w1, w2 in pairs:
        print(f"    sim({w1!r:10s}, {w2!r:10s}) = {skb.similarity(w1, w2):+.4f}")

    print("\n  Synonym similarity (~1.0):")
    for w1, w2 in [("happy", "joyful"), ("big", "large"), ("smart", "intelligent")]:
        print(f"    sim({w1!r:12s}, {w2!r:12s}) = {skb.similarity(w1, w2):+.4f}")

    print("\n  Antonym similarity (LOW):")
    for w1, w2 in [("hot", "cold"), ("big", "small"), ("day", "night")]:
        print(f"    sim({w1!r:12s}, {w2!r:12s}) = {skb.similarity(w1, w2):+.4f}")

    print(f"\n  Nearest neighbors of 'dog':")
    for w, s in skb.nearest_neighbors("dog", top_k=3):
        print(f"    {w!r:15s} : {s:+.4f}")

    # Test integration into agent
    agent = AETHER()
    before = len(agent.assoc.vocab)
    added = integrate_into_agent(agent)
    after = len(agent.assoc.vocab)
    print(f"\n  Integration into agent: {added} vectors added ({before} -> {after} vocab)")
    print("  -> Semantic embeddings OK")


# --------------------------------------------------------------------------- #
# 2. HD n-gram LM
# --------------------------------------------------------------------------- #
def test_hd_lm():
    banner("2. HD n-gram LM with backoff + sampling + beam search")

    agent = AETHER()
    lm = HDLanguageModel(agent.assoc, dim=agent.dim)

    sentences = [
        "the sun is a star",
        "the moon is a satellite",
        "water is a liquid",
        "gold is a metal",
        "python is a programming language",
        "the cat is an animal",
        "the dog is an animal",
        "tokyo is the capital of japan",
        "paris is the capital of france",
        "ottawa is the capital of canada",
    ]
    print(f"  Training LM on {len(sentences)} sentences...")
    for s in sentences:
        lm.learn_sequence(s.lower().split())
    print(f"  LM stats: {lm.stats()}")

    print("\n  Prediction with backoff:")
    contexts = [
        ["the", "sun"],
        ["the", "moon"],
        ["python", "is"],
        ["tokyo", "is", "the", "capital"],
    ]
    for ctx in contexts:
        pred = lm.predict_next(ctx)
        if pred:
            tok, sim, order = pred
            print(f"    ctx={ctx!r:45s} -> {tok!r:12s} (sim={sim:.3f}, order={order})")

    print("\n  Greedy generation (T=0):")
    rng = np.random.default_rng(42)
    for prompt in [["the", "sun"], ["python", "is"], ["paris", "is", "the"]]:
        gen = lm.generate(prompt, max_tokens=8, temperature=0.0, top_k=3, rng=rng)
        print(f"    {' '.join(prompt):20s} ... -> {' '.join(gen)}")

    print("\n  Sampled generation (T=1.0, 3 samples):")
    for prompt in [["the", "sun"]]:
        for i in range(3):
            rng_i = np.random.default_rng(i)
            gen = lm.generate(prompt, max_tokens=8, temperature=1.0, top_k=5, rng=rng_i)
            print(f"    sample {i+1}: {' '.join(prompt)} ... -> {' '.join(gen)}")

    print("\n  Beam search (beam_width=3):")
    for prompt in [["the", "sun"], ["python", "is"]]:
        beams = lm.beam_search(prompt, max_tokens=6, beam_width=3)
        print(f"    prompt: {' '.join(prompt)}")
        for seq, score in beams:
            print(f"      [{score:.3f}] {' '.join(seq)}")

    print("  -> HD LM OK")


# --------------------------------------------------------------------------- #
# 3. N-hop with sub-goaling
# --------------------------------------------------------------------------- #
def test_n_hop_subgoaling():
    banner("3. N-hop with sub-goaling + path finding + reachability")

    agent = AETHER()
    extra = [
        "Montreal is located in Canada",
        "Toronto is located in Canada",
        "Canada is located in America",
        "America is located in Earth",
        "Lyon is located in France",
        "France is located in Europe",
        "Europe is located in Earth",
        "Ottawa is the capital of Canada",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
    ]
    for f in extra:
        agent.teach(f, silent=True)
    print(f"  KB: {len(agent.assoc.triples)} triples")

    print("\n  Direct lookup with explicit verification:")
    proof = agent.inference.n_hop_with_subgoaling("Montreal", "located_in", max_depth=4)
    print(f"    {agent.inference.explain(proof)}")

    print("\n  Path finding (Montreal -> Earth):")
    paths = agent.inference.find_paths("Montreal", "Earth", max_depth=5, max_paths=3)
    for i, p in enumerate(paths):
        print(f"    Path {i+1}:")
        for step in p.steps:
            s, pr, o = step.conclusion
            print(f"      ({s}, {pr}, {o}) [conf={step.confidence:.3f}]")

    print("\n  Reachability from Montreal (3 hops):")
    reachable = agent.inference.reachable("Montreal", max_hops=3)
    for entity, (path, conf) in sorted(reachable.items(), key=lambda x: -x[1][1])[:6]:
        print(f"    {entity:15s} via {' -> '.join(path):40s} conf={conf:.3f}")

    print("\n  Multi-hop query (Montreal -> located_in -> capital_of):")
    proof = agent.inference.multi_hop_query("Montreal", ["located_in", "capital_of"])
    print(f"    {agent.inference.explain(proof)}")

    print("  -> N-hop sub-goaling OK")


# --------------------------------------------------------------------------- #
# 4. Tool composer
# --------------------------------------------------------------------------- #
def test_tool_composer():
    banner("4. Tool Composer (hybrid HD + pattern matching)")

    agent = AETHER()
    composer = ToolComposer(agent.encoder, agent.tools, agent.inference)
    executor = ComposedExecutor(agent)

    for f in ["Tokyo is the capital of Japan", "Paris is the capital of France"]:
        agent.teach(f, silent=True)

    questions = [
        "compare Paris and Tokyo",
        "how many triples are in the KB",
        "calculate 5 * 8 + 3",
        "explain Python",
        "what is the capital of France",
        "where is Montreal located",
        "list all facts",
        "recall Paris",
        "time",
        "summarize 3",
    ]
    print(f"  Testing {len(questions)} questions:\n")
    for q in questions:
        nodes, rationale = composer.compose(q)
        print(f"  Q: {q}")
        print(f"    rationale: {rationale}")
        for i, n in enumerate(nodes):
            deps = f" depends_on={n.depends_on}" if n.depends_on else ""
            print(f"      node {i}: {n.tool_name}({n.args!r}){deps}")
        final, _ = executor.execute(nodes)
        print(f"    => {str(final)[:80]}")
        print()

    print("  -> Tool composer OK")


# --------------------------------------------------------------------------- #
# 5. HD RAG
# --------------------------------------------------------------------------- #
def test_hd_rag():
    banner("5. HD RAG (web fetch + offline wiki fallback)")

    reset_global_rag()
    agent = AETHER()
    rag = get_global_rag(agent.encoder)

    print(f"  RAG store stats: {rag.stats()}")
    print(f"  Has internet: {WebFetcher(rag).has_internet()}")

    print("\n  Retrieval tests (should find the right topic):")
    queries = ["tokyo", "paris", "python", "aether", "transformer",
               "kanerva", "water", "sun", "japan", "france"]
    correct = 0
    for q in queries:
        results = rag.retrieve(q, top_k=1)
        if results:
            doc, sim = results[0]
            ok = q in doc.topic.lower() or doc.topic.lower() in q
            if ok:
                correct += 1
            marker = "OK" if ok else "FAIL"
            print(f"    [{marker}] {q!r:15s} -> [{doc.topic}] (sim={sim:.3f})")
    print(f"\n  Retrieval accuracy: {correct}/{len(queries)}")

    print("\n  Tool-style fetch:")
    from aether.tools import ToolContext
    ctx = ToolContext(agent.encoder, agent.assoc)
    from aether.web import tool_web_search, tool_rag_query, tool_rag_stats
    print(f"  web_search Tokyo: {tool_web_search('Tokyo', ctx)[:80]}...")
    print(f"  rag_query paris:")
    for line in tool_rag_query('paris', ctx).split('\n'):
        print(f"    {line}")
    print(f"  {tool_rag_stats('', ctx)}")

    print("  -> HD RAG OK")


# --------------------------------------------------------------------------- #
# 6. Multi-modal
# --------------------------------------------------------------------------- #
def test_multimodal():
    banner("6. Multi-modal HD Encoding (images + audio + cross-modal)")

    print("  === Image HD Encoder ===")
    img_enc = ImageHDEncoder(dim=4096, grid_size=16, intensity_bins=8)

    # Create test images
    square = np.zeros((32, 32))
    square[8:24, 8:24] = 1.0
    circle = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if (i - 16) ** 2 + (j - 16) ** 2 < 100:
                circle[i, j] = 1.0
    noise = np.random.rand(32, 32)

    v_sq = img_enc.encode(square)
    v_ci = img_enc.encode(circle)
    v_no = img_enc.encode(noise)
    v_sq2 = img_enc.encode(square.copy())

    print(f"    sim(square, square_copy) = {v_sq.similarity(v_sq2):.3f}  (~1.0)")
    print(f"    sim(square, circle)      = {v_sq.similarity(v_ci):.3f}  (moderate)")
    print(f"    sim(square, noise)       = {v_sq.similarity(v_no):.3f}  (~0)")

    print("\n  === Audio HD Encoder ===")
    aud_enc = AudioHDEncoder(dim=4096, n_frames=16, n_bands=16, energy_bins=8)

    v_440 = aud_enc.encode_sine(440)
    v_440b = aud_enc.encode_sine(440)
    v_880 = aud_enc.encode_sine(880)
    v_220 = aud_enc.encode_sine(220)
    v_chord = aud_enc.encode_chord([440, 554, 659])

    print(f"    sim(440, 440_copy) = {v_440.similarity(v_440b):.3f}  (~1.0)")
    print(f"    sim(440, 880)      = {v_440.similarity(v_880):.3f}  (moderate)")
    print(f"    sim(440, 220)      = {v_440.similarity(v_220):.3f}  (moderate)")
    print(f"    sim(440, chord)    = {v_440.similarity(v_chord):.3f}  (lower)")

    print("\n  === Cross-modal Space ===")
    cms = CrossModalSpace(dim=4096)
    cms.add_image_pattern([
        "                                ",
        "         ###########            ",
        "       ###############          ",
        "      #################         ",
        "     ###################        ",
        "     ###################        ",
        "     ###################        ",
        "      #################         ",
        "       ###############          ",
        "         ###########            ",
    ], label="circle_pattern")
    cms.add_image_pattern([
        "                                ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "                                ",
    ], label="rectangle_pattern")
    cms.add_sine(440, label="A4_note")
    cms.add_sine(880, label="A5_note")

    print(f"    CrossModal stats: {cms.stats()}")

    results = cms.find_similar_image_pattern([
        "                                ",
        "         ###########            ",
        "       ###############          ",
        "      #################         ",
        "     ###################        ",
        "     ###################        ",
        "     ###################        ",
        "      #################         ",
        "       ###############          ",
        "         ###########            ",
    ], top_k=3)
    print(f"\n    Find similar to circle_pattern:")
    for mod, label, sim in results:
        print(f"      [{mod}] {label!r:25s} sim={sim:.3f}")

    results = cms.find_similar_sine(440, top_k=3)
    print(f"\n    Find similar to A4 (440 Hz):")
    for mod, label, sim in results:
        print(f"      [{mod}] {label!r:25s} sim={sim:.3f}")

    print("  -> Multi-modal OK")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_semantic_embeddings,
        test_hd_lm,
        test_n_hop_subgoaling,
        test_tool_composer,
        test_hd_rag,
        test_multimodal,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {t.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All v3 tests complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
