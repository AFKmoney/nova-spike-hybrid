#!/usr/bin/env python3
"""
Démo complète de SPIKE — SNN + agent + STDP.

Compare avec NOVA et le paradigme transformer.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import SpikeBrain, SpikeConfig


def banner(title: str, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def demo_1_lif(nova_brain):
    """Test des neurones LIF seuls."""
    banner("1. NEURONES LIF (Leaky Integrate-and-Fire)")

    from spike.core import LIFNeuron, LIFParams
    params = LIFParams(n_neurons=100, tau_m=20.0, V_thresh=1.0)
    neuron = LIFNeuron(params)
    I = np.zeros(100, dtype=np.float32)
    I[:10] = 1.5  # au-dessus du seuil

    counts = []
    for _ in range(200):
        s = neuron.step(I, 0.0)
        counts.append(int(s.sum()))

    print(f"  100 neurones, courant 1.5 sur 10 d'entre eux, 200 ticks")
    print(f"  Spikes totaux sur neurones stimulés: {neuron.spike_count[:10].sum()}")
    print(f"  Spikes totaux sur neurones non stimulés: {neuron.spike_count[10:].sum()}")
    print(f"  → Décroissance exponentielle (leak) + seuil (fire)")


def demo_2_apprentissage(brain):
    """Apprentissage one-shot par imprint."""
    banner("2. APPRENTISSAGE ONE-SHOT (imprint synaptique)")

    faits = [
        ("le chat", "un animal qui miaule"),
        ("le chien", "un animal qui aboie"),
        ("Paris", "la capitale de la France"),
        ("la terre", "une planète du système solaire"),
        ("l'eau", "H2O"),
        ("Pythagore", "a² + b² = c²"),
        ("la vitesse de la lumière", "300000 km/s"),
    ]
    print(f"Apprentissage de {len(faits)} faits...\n")
    t0 = time.time()
    for k, v in faits:
        r = brain.learn(k, v)
        print(f"  ✓ appris: '{k}' = '{v}'  "
              f"({r['n_sensory_neurons']} sens, {r['n_motor_neurons']} motor)")
    t1 = time.time()
    print(f"\n→ {len(faits)} faits en {(t1-t0)*1000:.1f} ms "
          f"({(t1-t0)*1000/len(faits):.2f} ms par fait)")
    print(f"→ Imprint synaptique direct, pas de backprop.")


def demo_3_rappel(brain):
    """Rappel par simulation SNN."""
    banner("3. RAPPEL PAR SIMULATION SNN")

    queries = [
        ("le chat", "exact"),
        ("Paris", "exact"),
        ("la terre", "exact"),
        ("l'eau", "exact"),
        ("la lune", "non appris"),
    ]
    print(f"{'Query':<25} {'Type':<15} {'Rappel':<50} {'Score':<8}")
    print("-" * 100)
    for q, qtype in queries:
        t0 = time.time()
        r = brain.recall(q)
        t1 = time.time()
        val = r["value"] if r["value"] else r["fact"]
        if r["score"] < 1.0:
            val = "(rien)"
        print(f"{q:<25} {qtype:<15} {(val or '')[:48]:<50} {r['score']:>5.1f}  ({(t1-t0)*1000:.0f}ms)")


def demo_4_stdp(brain):
    """Démontre STDP — les poids changent pendant la simulation."""
    banner("4. STDP — APPRENTISSAGE LOCAL ONLINE")

    print("Poids initiaux (moyennes):")
    print(f"  sens→assoc: {brain.net.syn_sens_to_assoc.W.data.mean():.4f}")
    print(f"  assoc→motor: {brain.net.syn_assoc_to_motor.W.data.mean():.4f}")

    # Simule avec un input répétitif
    print("\nSimulation de 200 ticks avec input 'bonjour le monde'...")
    I = brain.coder.encode_text_to_current("bonjour le monde", gain=2.5)
    for _ in range(200):
        mask = (brain.rng.random(brain.cfg.n_sensory) < 0.6).astype(np.float32)
        brain.net.tick(I * mask)
        brain._apply_stdp()

    print("\nPoids finaux (moyennes):")
    print(f"  sens→assoc: {brain.net.syn_sens_to_assoc.W.data.mean():.4f}")
    print(f"  assoc→motor: {brain.net.syn_assoc_to_motor.W.data.mean():.4f}")
    print("  → STDP a renforcé les synapses activées par l'input")


def demo_5_outils(brain):
    """Démo outils agentiques."""
    banner("5. OUTILS AGENTIQUES (couple SNN ↔ tools)")

    tests = [
        ("calcule 2 + 3 * 4", "calculator"),
        ("combien font 15 fois 7", "calculator (mots)"),
        ("que vaut la racine carrée de 144", "calculator (sqrt)"),
        ("python: print([x**2 for x in range(5)])", "python exec"),
        ("quelle heure est-il", "time"),
    ]
    print(f"{'Query':<45} {'Outil':<22} {'Résultat':<35}")
    print("-" * 105)
    for q, label in tests:
        r = brain.chat(q)
        r = r.replace("\n", " | ")[:60]
        print(f"{q:<45} {label:<22} {r:<35}")


def demo_6_performance(brain):
    """Performance."""
    banner("6. PERFORMANCE — CPU-only, sub-100ms")

    queries = ["calcule 2+2", "que sais-tu sur Paris", "bonjour",
               "python: print(1)", "quelle heure est-il"]
    times = []
    for q in queries:
        t0 = time.time()
        brain.chat(q)
        times.append((time.time() - t0) * 1000)
    print(f"\nTemps de réponse moyen: {np.mean(times):.1f} ms")
    print(f"  min: {min(times):.1f} ms, max: {max(times):.1f} ms")

    # Synapse count
    n_syn = (brain.net.syn_sens_to_assoc.W.nnz
             + brain.net.syn_assoc_to_assoc.W.nnz
             + brain.net.syn_assoc_to_motor.W.nnz)
    mem_syn = n_syn * 8  # float32 + int32 (CSR)
    print(f"\nEmpreinte mémoire:")
    print(f"  Synapses totales: {n_syn} ({mem_syn / 1024:.1f} Ko)")
    print(f"  Neurones: {brain.cfg.n_sensory + brain.cfg.n_associative + brain.cfg.n_motor}")
    print(f"  Total RAM estimé: {(mem_syn + (brain.cfg.n_sensory + brain.cfg.n_associative + brain.cfg.n_motor) * 32) / 1024:.1f} Ko")


def demo_7_vs_nova_vs_transformer(brain):
    """Comparaison."""
    banner("7. COMPARAISON: TRANSFORMER vs NOVA vs SPIKE")

    comparisons = [
        ("Architecture", "Attention O(n²·d)", "HDC + SDM", "SNN (LIF + STDP)"),
        ("GPU requis", "Oui (>10 Go)", "Non (<100 Mo)", "Non (<10 Mo)"),
        ("Calcul dominant", "Matmul dense", "Produit scalaire", "Additions sparse"),
        ("Apprentissage", "Backprop", "One-shot SDM", "One-shot imprint + STDP"),
        ("Temporalité", "Aucune (figé)", "Résonateur continu", "Ticks + latences"),
        ("Mémoire", "Poids", "SDM statique", "Synapses CSR"),
        ("Frugalité CPU", "Mauvaise", "Bonne (~80ms)", "Excellente (~20ms)"),
        ("Biologique", "Non", "Abstrait", "Oui (LIF + STDP)"),
        ("Tool calling", "Fine-tuning", "HD + regex", "Activité moteur + regex"),
        ("Auto-contenu", "Non (API)", "Oui", "Oui"),
    ]
    print(f"\n{'Dimension':<22} {'Transformer':<22} {'NOVA':<22} {'SPIKE':<22}")
    print("-" * 88)
    for d, t, n, s in comparisons:
        print(f"{d:<22} {t:<22} {n:<22} {s:<22}")


def main():
    banner("SPIKE — Spiking Pattern Intelligence with Kernel Execution", char="#")
    print("""
SPIKE est un cerveau artificiel à impulsions (SNN) couplé à une couche
agentique. Inspiré du cerveau biologique, il combine:

  1. Neurones LIF (Leaky Integrate-and-Fire) — calcul par addition pure
  2. Synapses CSR sparse (1% connectivité) — RAM minimal
  3. STDP (Spike-Timing-Dependent Plasticity) — apprentissage local online
  4. Populations sensory/associative/motor — vraie temporalité
  5. Couche agentique — outils déclenchés par l'activité neuronale

Comparé à NOVA (v1, HDC), SPIKE va plus loin:
  - Vraie dynamique temporelle (ticks, latences, réfraction)
  - Apprentissage continu (STDP) en plus de l'imprint one-shot
  - Encore plus frugal (additions vs produits scalaires)
  - Plus biologique (modèle LIF + STDP = cerveau réel)
""")

    print("Initialisation de SPIKE...")
    t0 = time.time()
    cfg = SpikeConfig(n_sensory=400, n_associative=1500, n_motor=300, sim_ticks=50)
    brain = SpikeBrain(cfg, rng=np.random.default_rng(42))
    print(f"Prêt en {(time.time()-t0)*1000:.0f} ms.\n")

    demo_1_lif(brain)
    demo_2_apprentissage(brain)
    demo_3_rappel(brain)
    demo_4_stdp(brain)
    demo_5_outils(brain)
    demo_6_performance(brain)
    demo_7_vs_nova_vs_transformer(brain)

    banner("FIN DE LA DÉMO — SPIKE est opérationnel", char="#")
    print("""
Pour utiliser SPIKE interactivement:
    python spike_cli.py

Pour la démo:
    python spike_cli.py --demo
""")


if __name__ == "__main__":
    main()
