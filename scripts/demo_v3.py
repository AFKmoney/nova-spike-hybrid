"""
demo_v3.py — AETHER v3 multi-modal GPT-killer demonstrations.

Showcases all 6 new v3 capabilities:
  1. Semantic HD embeddings — knows 'dog' is more like 'cat' than 'paris'
  2. HD n-gram LM — backoff + sampling + beam search
  3. N-hop reasoning — find paths through the KB with sub-goaling
  4. Tool composer — chains tools automatically (compare X and Y → kb_query × 2 → compare)
  5. HD RAG — retrieve Wikipedia-style docs in HD space (offline fallback)
  6. Multi-modal — encode images + audio as HD vectors, cross-modal similarity
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import (
    AETHER, SemanticKB, HDLanguageModel,
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
def demo_semantic():
    banner("1. Semantic HD Embeddings — AETHER knows what's similar to what")

    print("  Transformer LLM: needs word2vec/GloVe (GB of data, days of training)")
    print("  AETHER: builds a 621-word semantic KB in pure Python, no training.\n")

    skb = SemanticKB(dim=4096)
    skb.build()
    print(f"  Built: {skb.stats()}")

    print("\n  Within-category similarity (animals vs animals):")
    for w1 in ["dog", "cat", "horse", "lion"]:
        sims = [(w2, skb.similarity(w1, w2)) for w2 in ["dog", "cat", "horse", "lion", "paris", "red"] if w2 != w1]
        sims.sort(key=lambda x: -x[1])
        top = sims[0]
        print(f"    {w1!r:8s} ~ {top[0]!r:8s} : {top[1]:+.3f}")

    print("\n  Cross-category similarity (should be near zero):")
    for w1, w2 in [("dog", "paris"), ("red", "seven"), ("bread", "happy")]:
        print(f"    sim({w1!r:8s}, {w2!r:8s}) = {skb.similarity(w1, w2):+.4f}")

    print("\n  Synonyms (should be ~1.0):")
    for w1, w2 in [("happy", "joyful"), ("smart", "intelligent"), ("buy", "purchase")]:
        print(f"    sim({w1!r:12s}, {w2!r:12s}) = {skb.similarity(w1, w2):+.4f}")

    print("\n  Nearest neighbors of 'dog':")
    for w, s in skb.nearest_neighbors("dog", top_k=5):
        print(f"    {w!r:15s} : {s:+.4f}")

    print("\n  Integration into AETHER agent:")
    agent = AETHER()
    before = len(agent.assoc.vocab)
    added = integrate_into_agent(agent)
    print(f"    Added {added} semantic vectors to vocab ({before} -> {len(agent.assoc.vocab)})")


# --------------------------------------------------------------------------- #
# 2. HD LM
# --------------------------------------------------------------------------- #
def demo_lm():
    banner("2. HD n-gram Language Model — backoff + sampling + beam search")

    agent = AETHER()
    lm = HDLanguageModel(agent.assoc, dim=agent.dim)

    sentences = [
        "the sun is a star",
        "the moon is a satellite",
        "water is a liquid",
        "gold is a metal",
        "python is a programming language",
        "tokyo is the capital of japan",
        "paris is the capital of france",
        "ottawa is the capital of canada",
    ]
    print(f"  Training on {len(sentences)} sentences...")
    for s in sentences:
        lm.learn_sequence(s.lower().split())

    print(f"\n  Backoff prediction:")
    for ctx in [["the", "sun"], ["python", "is"], ["tokyo", "is", "the", "capital"]]:
        pred = lm.predict_next(ctx)
        if pred:
            print(f"    ctx={ctx!r:40s} -> {pred[0]!r:12s} (sim={pred[1]:.3f}, order={pred[2]})")

    print(f"\n  Greedy generation (T=0):")
    rng = np.random.default_rng(42)
    for prompt in [["the", "sun"], ["paris", "is", "the"]]:
        gen = lm.generate(prompt, max_tokens=8, temperature=0.0, rng=rng)
        print(f"    {' '.join(prompt)} ... -> {' '.join(gen)}")

    print(f"\n  Beam search (beam_width=3):")
    for prompt in [["the", "sun"], ["python", "is"]]:
        beams = lm.beam_search(prompt, max_tokens=6, beam_width=3)
        print(f"    prompt: {' '.join(prompt)}")
        for seq, score in beams:
            print(f"      [{score:.3f}] {' '.join(seq)}")


# --------------------------------------------------------------------------- #
# 3. N-hop reasoning
# --------------------------------------------------------------------------- #
def demo_nhop():
    banner("3. N-hop Reasoning — path finding + reachability + sub-goaling")

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
        "Japan is located in Asia",
        "Asia is located in Earth",
    ]
    for f in extra:
        agent.teach(f, silent=True)
    print(f"  KB: {len(agent.assoc.triples)} triples")

    print(f"\n  Multi-hop reasoning chains:")
    chains = [
        (["Montreal", "located_in", "capital_of"],        "Montreal -> ?country -> ?capital"),
        (["Lyon",     "located_in", "capital_of"],        "Lyon -> ?country -> ?capital"),
        (["Toronto",  "located_in", "located_in", "located_in"], "Toronto -> ?country -> ?continent -> ?planet"),
    ]
    for predicates, label in chains:
        proof = agent.inference.multi_hop_query(predicates[0], predicates[1:])
        print(f"    {label}")
        for i, step in enumerate(proof.steps):
            s, p, o = step.conclusion
            print(f"      hop {i+1}: ({s}, {p}, {o}) [conf={step.confidence:.3f}]")
        if not proof.failed:
            print(f"      ANSWER: {proof.final_answer} (conf={proof.final_confidence:.3f})")
        print()

    print(f"  Path finding (Montreal -> Earth):")
    paths = agent.inference.find_paths("Montreal", "Earth", max_depth=5, max_paths=3)
    for i, p in enumerate(paths):
        path_str = " -> ".join(f"{s}|{pr}|{o}" for s, pr, o in [step.conclusion for step in p.steps])
        print(f"    Path {i+1}: {path_str}")

    print(f"\n  Reachability from Montreal (3 hops):")
    reachable = agent.inference.reachable("Montreal", max_hops=3)
    for entity, (path, conf) in sorted(reachable.items(), key=lambda x: -x[1][1])[:6]:
        print(f"    {entity:15s} via {' -> '.join(path):40s} conf={conf:.3f}")


# --------------------------------------------------------------------------- #
# 4. Tool composer
# --------------------------------------------------------------------------- #
def demo_composer():
    banner("4. Tool Composer — automatic multi-tool plans")

    agent = AETHER()
    composer = ToolComposer(agent.encoder, agent.tools, agent.inference)
    executor = ComposedExecutor(agent)

    for f in ["Tokyo is the capital of Japan", "Paris is the capital of France"]:
        agent.teach(f, silent=True)

    complex_queries = [
        ("compare Paris and Tokyo",
         "Should compose: kb_query(Paris) -> kb_query(Tokyo) -> compare"),
        ("how many triples are in the KB",
         "Should compose: list_kb -> count(triples)"),
        ("calculate (15 + 27) * 3",
         "Should compose: calc((15+27)*3)"),
        ("explain Python",
         "Should compose: explain(Python)"),
    ]
    print(f"  Complex queries that need tool composition:\n")
    for q, expected in complex_queries:
        print(f"  Q: {q}")
        print(f"  Expected: {expected}")
        nodes, rationale = composer.compose(q)
        print(f"  Plan: {rationale}")
        for i, n in enumerate(nodes):
            deps = f" [depends on node {n.depends_on}]" if n.depends_on else ""
            print(f"    node {i}: {n.tool_name}({n.args!r}){deps}")
        final, _ = executor.execute(nodes)
        print(f"  Result: {str(final)[:90]}")
        print()


# --------------------------------------------------------------------------- #
# 5. HD RAG
# --------------------------------------------------------------------------- #
def demo_rag():
    banner("5. HD RAG — retrieval-augmented generation in HD space")

    reset_global_rag()
    agent = AETHER()
    rag = get_global_rag(agent.encoder)

    print(f"  RAG store: {rag.stats()}")
    fetcher = WebFetcher(rag)
    print(f"  Internet available: {fetcher.has_internet()}")
    print(f"  (Will use offline wiki fallback — 15 curated topics)\n")

    print(f"  Retrieval quality test:")
    queries = ["tokyo", "paris", "python", "aether", "transformer",
               "kanerva", "water", "sun", "japan", "france",
               "canada", "moon", "earth", "hyperdimensional", "ai"]
    correct = 0
    for q in queries:
        results = rag.retrieve(q, top_k=1)
        if results:
            doc, sim = results[0]
            ok = q in doc.topic.lower() or doc.topic.lower() in q
            if ok:
                correct += 1
            marker = "OK" if ok else "FAIL"
            print(f"    [{marker}] {q!r:18s} -> [{doc.topic}] (sim={sim:.3f})")
    print(f"\n  Retrieval accuracy: {correct}/{len(queries)} ({100*correct/len(queries):.0f}%)")

    print(f"\n  Multi-document retrieval (top 3 for 'capital'):")
    results = rag.retrieve("capital", top_k=3)
    for i, (doc, sim) in enumerate(results, 1):
        text_short = doc.text if len(doc.text) < 80 else doc.text[:77] + "..."
        print(f"    {i}. [{doc.topic}] (sim={sim:.3f}) {text_short}")


# --------------------------------------------------------------------------- #
# 6. Multi-modal
# --------------------------------------------------------------------------- #
def demo_multimodal():
    banner("6. Multi-modal HD Encoding — images + audio in HD space")

    print("  Transformer LLM: needs separate ViT (vision) + Whisper (audio) + CLIP (alignment)")
    print("  AETHER: encodes images AND audio in the SAME HD space as text. No neural nets.\n")

    # Images
    print("  === Image encoding ===")
    img_enc = ImageHDEncoder(dim=4096, grid_size=16, intensity_bins=8)
    # Three shapes: square, circle, triangle
    square = np.zeros((32, 32))
    square[8:24, 8:24] = 1.0
    circle = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if (i - 16) ** 2 + (j - 16) ** 2 < 100:
                circle[i, j] = 1.0
    triangle = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if i > 8 and i < 24 and j > 16 - (i - 8) and j < 16 + (i - 8):
                triangle[i, j] = 1.0
    noise = np.random.rand(32, 32)

    v_sq = img_enc.encode(square)
    v_ci = img_enc.encode(circle)
    v_tr = img_enc.encode(triangle)
    v_no = img_enc.encode(noise)

    print(f"    sim(square, square_copy) = {v_sq.similarity(img_enc.encode(square.copy())):.3f}")
    print(f"    sim(square, circle)      = {v_sq.similarity(v_ci):.3f}")
    print(f"    sim(square, triangle)    = {v_sq.similarity(v_tr):.3f}")
    print(f"    sim(square, noise)       = {v_sq.similarity(v_no):.3f}")
    print(f"    sim(circle, triangle)    = {v_ci.similarity(v_tr):.3f}")

    # Audio
    print("\n  === Audio encoding ===")
    aud_enc = AudioHDEncoder(dim=4096)
    # Different musical notes
    notes = {"A3": 220, "A4": 440, "A5": 880, "C4": 261, "C5": 523}
    vecs = {name: aud_enc.encode_sine(freq) for name, freq in notes.items()}
    chord_C = aud_enc.encode_chord([261, 329, 392])  # C major
    chord_A = aud_enc.encode_chord([440, 554, 659])  # A major

    print(f"    sim(A4, A4_copy) = {vecs['A4'].similarity(aud_enc.encode_sine(440)):.3f}")
    print(f"    sim(A4, A5)      = {vecs['A4'].similarity(vecs['A5']):.3f}  (octave)")
    print(f"    sim(A4, A3)      = {vecs['A4'].similarity(vecs['A3']):.3f}  (octave)")
    print(f"    sim(A4, C4)      = {vecs['A4'].similarity(vecs['C4']):.3f}  (different note)")
    print(f"    sim(C_major, A_major) = {chord_C.similarity(chord_A):.3f}")

    # Cross-modal
    print("\n  === Cross-modal space (text + image + audio in same HD space) ===")
    cms = CrossModalSpace(dim=4096)

    # Add images as ASCII patterns
    cms.add_image_pattern([
        "                                ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "                                ",
    ], label="rectangle")

    cms.add_image_pattern([
        "                                ",
        "         ###########            ",
        "       ###############          ",
        "      #################         ",
        "     ###################        ",
        "     ###################        ",
        "      #################         ",
        "       ###############          ",
        "         ###########            ",
        "                                ",
    ], label="circle")

    cms.add_sine(440, label="A4_note")
    cms.add_sine(880, label="A5_note")

    print(f"    Cross-modal store: {cms.stats()}")

    print(f"\n    Find similar to rectangle pattern:")
    rect = [
        "                                ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "                                ",
    ]
    for mod, label, sim in cms.find_similar_image_pattern(rect, top_k=4):
        print(f"      [{mod:6s}] {label!r:20s} sim={sim:.3f}")

    print(f"\n    Find similar to A4 (440 Hz):")
    for mod, label, sim in cms.find_similar_sine(440, top_k=4):
        print(f"      [{mod:6s}] {label!r:20s} sim={sim:.3f}")

    print(f"\n    Find similar to 450 Hz (close to A4):")
    for mod, label, sim in cms.find_similar_sine(450, top_k=4):
        print(f"      [{mod:6s}] {label!r:20s} sim={sim:.3f}")


# --------------------------------------------------------------------------- #
# 7. Performance summary
# --------------------------------------------------------------------------- #
def demo_performance():
    banner("7. Performance — v3 still CPU-only, still fast")

    agent = AETHER()
    print(f"  Numpy: {np.__version__}")
    print(f"  Vector dim: {agent.dim}")
    print(f"  SDM: {agent.assoc.kb_store.n_locations} locations, k={agent.assoc.kb_store.k}")
    print(f"  Vocab: {len(agent.assoc.vocab)} tokens")
    print(f"  KB: {len(agent.assoc.triples)} triples")
    print(f"  Tools: {len(agent.list_tools())}")
    print()

    # Time key operations
    t0 = time.perf_counter()
    for _ in range(30):
        agent.ask("What is the capital of France?")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  30 KB queries: {elapsed:.0f}ms total = {elapsed/30:.2f}ms/query")

    t0 = time.perf_counter()
    for _ in range(20):
        agent.ask("What is the capital of the country where Osaka is located?")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  20 multi-hop queries: {elapsed:.0f}ms total = {elapsed/20:.2f}ms/query")

    # Image encoding perf
    img_enc = ImageHDEncoder(dim=4096)
    img = np.random.rand(64, 64)
    t0 = time.perf_counter()
    for _ in range(20):
        img_enc.encode(img)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  20 image encodings (64x64): {elapsed:.0f}ms total = {elapsed/20:.2f}ms/encode")

    # Audio encoding perf
    aud_enc = AudioHDEncoder(dim=4096)
    t0 = time.perf_counter()
    for _ in range(20):
        aud_enc.encode_sine(440)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  20 audio encodings (1s sine): {elapsed:.0f}ms total = {elapsed/20:.2f}ms/encode")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    demos = [
        demo_semantic,
        demo_lm,
        demo_nhop,
        demo_composer,
        demo_rag,
        demo_multimodal,
        demo_performance,
    ]
    for d in demos:
        try:
            d()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {d.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All v3 demos complete.")
    print("  AETHER v3.0.0 — multi-modal GPT killer edition")
    print("=" * 76)


if __name__ == "__main__":
    main()
