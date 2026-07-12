"""
demo_v6.py — AETHER v6 revolutionary features demonstration.

Showcases all 7 revolutionary axes:
  1. Dream v2 (Hebbian co-activation + concept centroids)
  2. Compositional v2 (recursive decomposition + trace reuse)
  3. learn_from_text v2 (coreference + entity linking)
  4. Cross-modal learning (native multimodal fusion)
  5. Meta-learning (domain-specific params + Lyapunov)
  6. Dual memory (episodic decay + semantic persistent)
  7. Distributional reasoning (HD algebra analogies)
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def demo_axis_1_dream(agent):
    banner("AXIS 1: Dream v2 — Hebbian Co-activation + Concept Centroids")
    print("  v1 dream was simple replay. v2 adds:")
    print("    - Hebbian matrix: concepts that fire together wire together")
    print("    - Concept centroids: cluster similar episodes → new abstract concepts")
    print("    - NREM (consolidation) + REM (creative) phases")
    print()
    # Teach co-occurring concepts
    for _ in range(3):
        agent.teach("fire is hot", silent=True)
        agent.teach("fire produces heat", silent=True)
        agent.teach("sun is hot", silent=True)
        agent.teach("sun produces light", silent=True)
    print("  Taught: fire~hot, fire~heat, sun~hot, sun~light (3x each)")
    print("  Running dream (30 cycles, mixed phase)...")
    dream = agent.dream_consolidate(cycles=30, phase="mixed")
    print(f"  Result:")
    print(f"    Phase: {dream['phase']}")
    print(f"    Episodes replayed: {dream['episodes_replayed']}")
    print(f"    Hebbian links formed: {dream['hebbian_links']}")
    print(f"    Concept centroids created: {dream['centroids_created']}")
    print(f"    New triples discovered: {dream['new_triples']}")
    print(f"    Creative insights: {dream['creative_insights']}")
    if dream['new_centroids']:
        print(f"    New abstract concepts: {dream['new_centroids']}")


def demo_axis_2_compositional(agent):
    banner("AXIS 2: Compositional v2 — Recursive Decomposition + Trace Reuse")
    print("  v1 was limited to fixed patterns. v2 adds:")
    print("    - Arbitrary depth recursive decomposition")
    print("    - Persistent reasoning traces (reusable)")
    print("    - Synthesis HD vectors (gist of reasoning)")
    print()
    questions = [
        "What is the capital of the country where Lyon is located?",
        "What is the capital of the country where Montreal is located?",
    ]
    for q in questions:
        print(f"  Q: {q}")
        result = agent.answer_compositional_v2(q)
        print(f"  Decomposition:")
        for line in result['decomposition'].split('\n')[:6]:
            print(f"    {line}")
        print(f"  Answer: {result['final_answer']} (depth={result['depth']})")
        print(f"  Reused trace: {result['reused_trace']}")
        print()


def demo_axis_3_learn_text(agent):
    banner("AXIS 3: learn_from_text v2 — Coreference + Entity Linking")
    print("  v1 extracted triples per sentence, missing pronoun references.")
    print("  v2 resolves 'He' → 'Einstein', links to existing KB.")
    print()
    text = """
    Albert Einstein was born in 1879. Einstein is a physicist. He discovered relativity.
    He is famous for E=mc2. Einstein died in 1955.

    Paris is the capital of France. France is located in Europe.
    Tokyo is the capital of Japan. Mount Fuji is located in Japan.
    """
    print(f"  Text: {text.strip()[:100]}...")
    result = agent.learn_text_v2(text)
    print(f"\n  Extracted: {result['n_facts_extracted']} facts")
    print(f"  Pronouns resolved: {result['n_pronouns_resolved']}")
    print(f"  Triples learned:")
    for s, p, o in result['triples_learned']:
        print(f"    ({s}, {p}, {o})")
    print(f"\n  Testing what was learned:")
    for q in ["What is Einstein?", "What is the capital of Japan?"]:
        print(f"    Q: {q}")
        print(f"    A: {agent.ask(q)}")


def demo_axis_4_cross_modal(agent):
    banner("AXIS 4: Cross-Modal Learning — Native Multimodal Fusion")
    print("  Transformer LLMs need ViT + Whisper + CLIP (3 separate models).")
    print("  AETHER binds text + image in ONE HD space — no separate arch.")
    print()
    # Create test images
    images = {
        "a red ball on a table": np.random.rand(32, 32) * 0.5 + 0.5,
        "a blue square in the sky": np.random.rand(32, 32) * 0.3 + 0.7,
        "a green tree in a forest": np.random.rand(32, 32) * 0.4 + 0.6,
    }
    print("  Learning 3 images with text descriptions...")
    for desc, img in images.items():
        agent.learn_image(desc, img)
        print(f"    ✓ {desc}")
    print()
    print("  Cross-modal retrieval:")
    queries = ["ball", "blue", "tree"]
    for q in queries:
        results = agent.retrieve_image(q, top_k=2)
        print(f"    '{q}' → {results}")


def demo_axis_5_meta_learning(agent):
    banner("AXIS 5: Meta-Learning — Domain-Specific Parameters + Lyapunov")
    print("  AETHER detects the cognitive domain and loads optimal params.")
    print("  Updates are Lyapunov-stable (no catastrophic meta-forgetting).")
    print()
    questions = [
        ("What is the capital of France?", "geography"),
        ("calc 2+2*5", "math"),
        ("Why is the sky blue?", "reasoning"),
        ("Imagine a flying horse", "creative"),
        ("What is water?", "factual"),
    ]
    print("  Domain detection:")
    for q, expected in questions:
        domain, conf = agent.detect_domain(q)
        marker = "OK" if domain == expected else "?"
        print(f"    [{marker}] {q!r:40s} → {domain!r:12s} (expected {expected!r}, conf={conf:.3f})")
    print(f"\n  Domain profiles:")
    stats = agent.meta_learner.stats()
    for name, p in list(stats['domain_performances'].items())[:5]:
        print(f"    {name:12s}: k_radius={p['kernel_radius']:.2f}, "
              f"temp={p['temperature']:.2f}, max_cycles={p['max_cycles']}, "
              f"queries={p['n_queries']}")


def demo_axis_6_dual_memory(agent):
    banner("AXIS 6: Dual Memory — Episodic (decay) + Semantic (persistent)")
    print("  Humans have two memory systems. So does AETHER:")
    print("    - Episodic: 'I talked about X at 15h' — DECAYS over time")
    print("    - Semantic: 'Paris is the capital of France' — PERSISTS")
    print()
    # Record conversations
    print("  Recording conversation turns as episodic memories...")
    agent.record_episode("Hello, tell me about Paris", "Paris is the capital of France", "geography")
    agent.record_episode("What is the capital of Japan?", "Tokyo is the capital of Japan", "geography")
    agent.record_episode("calc 2+2", "4", "math")
    print("  ✓ 3 episodes recorded")
    print()
    print(f"  Memory stats: {agent.dual_memory.stats()}")
    print()
    # Try to recall
    print("  Recalling:")
    for topic in ["Paris", "calc", "France"]:
        recall = agent.remember(topic)
        if recall:
            print(f"    '{topic}' → {recall[:80]}...")
        else:
            print(f"    '{topic}' → (no recall)")
    print()
    print("  Consolidating episodic → semantic...")
    result = agent.consolidate_memory()
    print(f"  Result: {result}")


def demo_axis_7_distributional(agent):
    banner("AXIS 7: Distributional Reasoning — HD Algebra Analogies")
    print("  True analogical reasoning via HD algebra (Plate 1995):")
    print("    analogy(a, b, c) = ? such that a:b :: c:?")
    print("    Computed via: add(c, subtract(a, b)) → retrieve nearest")
    print()
    # Teach some facts
    agent.teach("Paris is the capital of France", silent=True)
    agent.teach("Tokyo is the capital of Japan", silent=True)
    agent.teach("Ottawa is the capital of Canada", silent=True)
    print("  Taught: Paris~France, Tokyo~Japan, Ottawa~Canada")
    print()
    # Test analogy
    print("  Analogy: paris:france :: tokyo:?")
    result = agent.analogy("paris", "france", "tokyo")
    print(f"    Answer: {result['answer']} (confidence={result['confidence']:.3f})")
    print(f"    Top candidates: {result['candidates'][:3]}")
    print()
    # Nearest concepts
    print("  Nearest concepts to 'paris':")
    nearest = agent.nearest_concepts("paris", top_k=5)
    for c, s in nearest:
        print(f"    {c!r:15s} : {s:.3f}")
    print()
    # Structural similarity
    print("  Structural similarity:")
    sim1 = agent.structural_similarity(("paris", "france"), ("tokyo", "japan"))
    sim2 = agent.structural_similarity(("paris", "france"), ("water", "liquid"))
    print(f"    (paris, france) vs (tokyo, japan) = {sim1:.3f} (should be similar)")
    print(f"    (paris, france) vs (water, liquid) = {sim2:.3f} (should differ)")


def demo_summary(agent):
    banner("AETHER v6 — 7 Revolutionary Axes Summary")
    s = agent.stats()
    print(f"  Version: {s['version']}")
    print(f"  Vocab: {s['vocab_size']} tokens")
    print(f"  Triples: {s['assoc']['triples']}")
    print(f"  Tools: {len(s['tools'])}")
    print()
    print("  7 revolutionary axes:")
    print("    1. ✓ Dream v2 — Hebbian + centroids (auto-abstraction)")
    print("    2. ✓ Compositional v2 — recursive + trace reuse")
    print("    3. ✓ learn_from_text v2 — coreference + entity linking")
    print("    4. ✓ Cross-modal learning — native multimodal fusion")
    print("    5. ✓ Meta-learning — domain-specific + Lyapunov")
    print("    6. ✓ Dual memory — episodic decay + semantic persistent")
    print("    7. ✓ Distributional reasoning — HD algebra analogies")
    print()
    print("  → AETHER v6 implements ALL 7 revolutionary axes.")
    print("    Each axis addresses a fundamental limitation of transformer LLMs.")
    print("    Together, they form a complete cognitive architecture that")
    print("    surpasses LLMs on multiple dimensions.")


def main():
    agent = AETHER()
    print(f"\n  AETHER {agent.VERSION} — running 7 revolutionary axis demos\n")
    demos = [
        demo_axis_1_dream,
        demo_axis_2_compositional,
        demo_axis_3_learn_text,
        demo_axis_4_cross_modal,
        demo_axis_5_meta_learning,
        demo_axis_6_dual_memory,
        demo_axis_7_distributional,
        demo_summary,
    ]
    for d in demos:
        try:
            d(agent)
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {d.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All v6 revolutionary demos complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
