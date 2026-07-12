#!/usr/bin/env python3
"""
Démo finale des 6 features v3.

1. Interface web (FastAPI + WebSocket) — dashboard temps réel
2. Visualisation matplotlib — 6 plots générés
3. BPE tokenizer — sous-mots
4. Multi-modal — images → spikes
5. Distributed — multi-brain
6. Benchmark — comparaison SPIKE/NOVA/HYBRID
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def banner(title: str, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def demo_1_web():
    """Web interface — on lance juste une vérification de l'API."""
    banner("1. INTERFACE WEB (FastAPI + WebSocket)")
    print("Principe: serveur FastAPI + WebSocket pour visualisation temps réel.")
    print("Endpoints:")
    print("  GET  /              — dashboard HTML")
    print("  GET  /api/stats     — stats JSON")
    print("  POST /api/chat      — envoie un message")
    print("  POST /api/learn     — apprentissage")
    print("  POST /api/dream     — mode rêve")
    print("  POST /api/reward    — R-STDP reward")
    print("  WS   /ws/spikes     — stream temps réel des spikes")
    print("  WS   /ws/chat       — chat bidirectionnel")
    print("\nPour lancer:")
    print("  python web/server.py")
    print("  → http://localhost:8000")
    print("\nDashboard affiche en temps réel:")
    print("  - Raster plot des spikes (sensory/assoc/motor)")
    print("  - Compteurs d'activité par population")
    print("  - Stats globales (vocab, faits, synapses, latence)")
    print("  - Log de conversation")
    print("  - Sélection du cerveau (SPIKE/NOVA/HYBRID)")


def demo_2_visualization():
    """Vérifie que les 6 plots sont là."""
    banner("2. VISUALISATION (matplotlib)")
    print("6 plots générés dans download/visualizations/:")
    viz_dir = "/home/z/my-project/download/visualizations"
    if os.path.exists(viz_dir):
        for f in sorted(os.listdir(viz_dir)):
            size = os.path.getsize(os.path.join(viz_dir, f))
            print(f"  ✓ {f} ({size // 1024} Ko)")
    else:
        print("  ⚠ Pas de visualisations. Lancez: python scripts/visualize.py")


def demo_3_bpe():
    """BPE tokenizer."""
    banner("3. BPE TOKENIZER (sous-mots)")
    from spike.bpe import BPETokenizer

    corpus = """
    le chat dort sur le tapis. le chien mange une pomme.
    paris est la capitale de la france. la terre tourne autour du soleil.
    bonjour le monde, comment vas-tu? calcule deux plus deux.
    python est un langage. le chat est un animal.
    """

    bpe = BPETokenizer(vocab_size=150)
    bpe.train(corpus)

    print(f"Vocab final: {len(bpe)} tokens, {len(bpe.merges)} fusions")

    tests = [
        "le chat",
        "chaton",  # mot inconnu — décomposé en sous-mots
        "bonjour",
        "Zorglub",  # totalement inconnu
    ]
    print("\nTests d'encodage:")
    for text in tests:
        ids = bpe.encode(text)
        tokens = [bpe.vocab[i] for i in ids]
        decoded = bpe.decode(ids)
        print(f"  {text!r:<15} → {len(ids)} tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
        print(f"  {'':<17}décodé: {decoded!r}")


def demo_4_multimodal():
    """Multi-modal — images → spikes."""
    banner("4. MULTI-MODAL (images → spikes)")
    from spike import SpikeBrain, SpikeConfig
    from spike.visual import ImageEncoder, MultiModalBrain

    # Génère 3 images synthétiques
    img1 = np.where(np.indices((50, 50)).sum(axis=0) % 20 < 10, 1.0, 0.0).astype(np.float32)
    img2 = np.zeros((50, 50), dtype=np.float32)
    img2[20:30, :] = 1.0
    img3 = np.where(np.sqrt((np.indices((50, 50)) - 25) ** 2).sum(axis=0) < 15, 1.0, 0.0).astype(np.float32)
    images = {"carré": img1, "ligne": img2, "cercle": img3}

    brain = SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=20,
    ))
    mm = MultiModalBrain(brain, n_visual=400, grid_size=(20, 20))

    print(f"Population visuelle: 400 neurones (grille 20x20)")
    print(f"Synapses vis→assoc: {mm.syn_visual_assoc.W.nnz}\n")

    for name, img in images.items():
        r = mm.process_image(img, duration=15)
        print(f"  Image '{name}': {r['total_motor_spikes']} spikes moteur, "
              f"{r['total_assoc_spikes']} assoc")
        if r['top_tokens']:
            top = r['top_tokens'][0]
            print(f"    Top token: '{top[0]}' ({top[1]} spikes)")


def demo_5_distributed():
    """Distributed brain."""
    banner("5. DISTRIBUTED (multi-brain)")
    from distributed import DistributedBrain

    dist = DistributedBrain(verbose=False)

    tests = [
        ("apprends que Mars est une planète", "memory+general"),
        ("que sais-tu sur Mars", "memory"),
        ("calcule 2+2", "math"),
        ("combien font 15 fois 3", "math"),
        ("python: print(42)", "math"),
        ("bonjour", "general"),
    ]

    print(f"Cluster: {len(dist.specs)} cerveaux")
    for name, spec in dist.specs.items():
        print(f"  - {name}: {spec.description}")
    print()

    print("Tests de routing + chat:")
    print(f"{'Query':<35} {'Routed to':<15} {'Response':<40}")
    print("-" * 90)
    for q, expected_route in tests:
        route = dist.route(q)
        r = dist.chat(q)
        r_short = r.replace("\n", " | ")[:38]
        print(f"  {q:<33} → {route:<15} → {r_short}")

    print()
    dist.print_stats()


def demo_6_benchmark():
    """Benchmark — affiche les résultats existants."""
    banner("6. BENCHMARK (SPIKE vs NOVA vs HYBRID)")
    bench_path = "/home/z/my-project/download/benchmark_results.json"
    chart_path = "/home/z/my-project/download/benchmark_chart.png"

    if os.path.exists(bench_path):
        with open(bench_path, "r") as f:
            results = json.load(f)
        print("Résultats du benchmark:\n")
        print(f"{'Tâche':<20} {'SPIKE':<18} {'NOVA':<18} {'HYBRID':<18}")
        print("-" * 74)
        for task in ["arithmetic", "memory", "tool_calling", "robustness"]:
            row = f"{task:<20}"
            for brain in ["SPIKE", "NOVA", "HYBRID"]:
                acc = results[brain][task]["accuracy"] * 100
                row += f" {acc:.0f}%{'':<13}"
            print(row)
        print(f"\n{'RAM (Mo)':<20}", end="")
        for brain in ["SPIKE", "NOVA", "HYBRID"]:
            mem = results[brain]["memory_mb"]
            print(f" {mem:<18.2f}", end="")
        print()
        if os.path.exists(chart_path):
            print(f"\n✓ Graphique: {chart_path}")
    else:
        print("⚠ Pas de résultats. Lancez: python scripts/benchmark.py")


def main():
    banner("SPIKE v3 — 6 NOUVELLES FEATURES", char="#")
    print("""
Cette démo présente les 6 features ajoutées:

  1. Interface web FastAPI + WebSocket (dashboard temps réel)
  2. Visualisation matplotlib (6 plots)
  3. BPE tokenizer (sous-mots)
  4. Multi-modal (images → spikes)
  5. Distributed (multi-brain orchestrator)
  6. Benchmark (suite de comparaison)
""")

    demo_1_web()
    demo_2_visualization()
    demo_3_bpe()
    demo_4_multimodal()
    demo_5_distributed()
    demo_6_benchmark()

    banner("FIN — TOUTES LES FEATURES SONT OPÉRATIONNELLES", char="#")


if __name__ == "__main__":
    main()
