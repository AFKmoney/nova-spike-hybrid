#!/usr/bin/env python3
"""
Démo des 6 nouvelles features SPIKE + HYBRID.

1. R-STDP — reward-modulated STDP
2. Lazy spikes — buffer de propagation asynchrone
3. Synapse direct sens→motor (one-shot plus fort)
4. Save/load pour SPIKE
5. Mode rêve — replay aléatoire
6. Hybride NOVA+SPIKE
"""

import sys
import os
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import SpikeBrain, SpikeConfig
from spike.rstdp import RSTDPTracker, RSTDPConfig
from spike.lazy import LazySpikeBuffer, LazySpikingNetwork
from hybrid import HybridBrain, HybridConfig
from nova import NovaConfig


def banner(title: str, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def demo_1_rstdp():
    """R-STDP: reward-modulated STDP."""
    banner("1. R-STDP — REWARD-MODULATED STDP")
    print("Principe: la STDP calcule des traces d'éligibilité à chaque tick.")
    print("Les poids ne changent QUE quand un signal de récompense arrive.")
    print()

    cfg = SpikeConfig(n_sensory=200, n_associative=500, n_motor=200,
                       sim_ticks=30, rstdp_enabled=True)
    brain = SpikeBrain(cfg)

    # Apprentissage initial
    brain.learn("chat", "un animal")
    print(f"Avant reward: W sens→motor mean = {brain.syn_sens_to_motor.W.data.mean():.4f}")

    # Simule avec input + R-STDP
    I = brain.coder.encode_text_to_current("chat", gain=2.5)
    for _ in range(50):
        mask = (brain.rng.random(cfg.n_sensory) < 0.6).astype(np.float32)
        brain.net.tick(I * mask)
        brain._apply_stdp()

    print(f"Avant reward (après sim): W mean = {brain.syn_sens_to_motor.W.data.mean():.4f}")
    print(f"Eligibility max: {brain.stdp_sens_motor.eligibility.data.max():.4f}")

    # Récompense positive
    r = brain.give_reward(reward=+1.0)
    print(f"\nReward +1.0: {r}")
    print(f"Après reward +1: W mean = {brain.syn_sens_to_motor.W.data.mean():.4f}")

    # Punition
    r = brain.give_reward(reward=-0.5)
    print(f"Reward -0.5: {r}")
    print(f"Après reward -0.5: W mean = {brain.syn_sens_to_motor.W.data.mean():.4f}")


def demo_2_lazy():
    """Lazy spikes."""
    banner("2. LAZY SPIKES — BUFFER ASYNCHRONE")
    print("Principe: au lieu de propager immédiatement, on bufferise les spikes")
    print("avec des délais hétérogènes. Gain: O(spikes_actifs) au lieu de O(N).")
    print()

    # Buffer simple
    buf = LazySpikeBuffer()
    buf.add_spike(5, 1.5, delay=2)
    buf.add_spike(7, 0.8, delay=2)
    buf.add_spike(5, 0.3, delay=1)
    print("Buffer simple — 3 spikes avec délais différents:")
    print(f"  Tick 0 (rien dû): {buf.tick_aggregated(10)}")
    print(f"  Tick 1 (spike 5 poids 0.3): {buf.tick_aggregated(10)}")
    print(f"  Tick 2 (spike 5 + 7): {buf.tick_aggregated(10)}")
    print()

    # Réseau lazy
    import time as _time
    net = LazySpikingNetwork(n_sensory=100, n_associative=500, n_motor=50, max_delay=3)
    I_fn = lambda t: np.where((np.arange(100) < 10) & (10 < t < 50), 3.0, 0.0).astype(np.float32)

    t0 = _time.time()
    for _ in range(100):
        net.tick(I_fn(net.clock.t))
    t1 = _time.time()

    s = net.stats()
    print(f"LazySpikingNetwork: 100 ticks en {(t1-t0)*1000:.2f} ms")
    print(f"  Spikes: sensory={s['sensory_total_spikes']}, "
          f"assoc={s['assoc_total_spikes']}, motor={s['motor_total_spikes']}")
    print(f"  Buffer sens→assoc: {s['buffer_sens_assoc']}")


def demo_3_direct_synapse():
    """Synapse direct sens→motor."""
    banner("3. SYnapse DIRECT sens→motor (one-shot plus fort)")
    print("Principe: bypass de la couche associative pour les faits explicites.")
    print("Le chemin direct permet un rappel plus fort et plus rapide.")
    print()

    # Compare avec et sans synapse directe
    cfg_with = SpikeConfig(n_sensory=300, n_associative=800, n_motor=300,
                            sim_ticks=30, direct_sens_motor=True)
    cfg_without = SpikeConfig(n_sensory=300, n_associative=800, n_motor=300,
                               sim_ticks=30, direct_sens_motor=False)

    brain_with = SpikeBrain(cfg_with)
    brain_without = SpikeBrain(cfg_without)

    faits = [
        ("le chat", "un animal"),
        ("Paris", "la capitale de la France"),
        ("la terre", "une planète"),
    ]
    for k, v in faits:
        brain_with.learn(k, v)
        brain_without.learn(k, v)

    print(f"{'Query':<20} {'Sans directe':<25} {'Avec directe':<25} {'Gain':<10}")
    print("-" * 80)
    for k, v in faits:
        r1 = brain_without.recall(k)
        r2 = brain_with.recall(k)
        gain = r2["score"] / max(1.0, r1["score"])
        print(f"{k:<20} {r1['score']:<25.1f} {r2['score']:<25.1f} {gain:<10.1f}x")


def demo_4_save_load():
    """Save/load pour SPIKE."""
    banner("4. SAVE/LOAD POUR SPIKE")
    print("Principe: persistance complète (synapses, vocab, faits, history).")
    print()

    save_path = "/tmp/spike_save_demo"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    cfg = SpikeConfig(n_sensory=300, n_associative=500, n_motor=300, sim_ticks=30)
    brain1 = SpikeBrain(cfg)
    brain1.learn("Mars", "la quatrième planète")
    brain1.learn("Jupiter", "la plus grosse planète")
    print(f"Avant save: {len(brain1.facts)} faits, "
          f"{brain1.coder.vocab_size} tokens, "
          f"{brain1.net.syn_sens_to_assoc.W.nnz} synapses")

    brain1.save(save_path)
    print(f"\nSauvegardé dans {save_path}")
    print(f"Fichiers: {sorted(os.listdir(save_path))}")

    # Charge dans un nouveau cerveau
    brain2 = SpikeBrain(cfg)
    brain2.load(save_path)
    print(f"\nAprès load: {len(brain2.facts)} faits, "
          f"{brain2.coder.vocab_size} tokens, "
          f"{brain2.net.syn_sens_to_assoc.W.nnz} synapses")

    print("\nRecall après load:")
    for q in ["Mars", "Jupiter", "Saturne"]:
        r = brain2.recall(q)
        val = r["value"] if r["value"] and r["score"] > 1.5 else "(rien)"
        print(f"  {q!r:<12} -> {val!r:<35} score={r['score']:.1f}")

    # Cleanup
    shutil.rmtree(save_path)


def demo_5_dream():
    """Mode rêve."""
    banner("5. MODE RÊVE — CONSOLIDATION STDP")
    print("Principe: rejoue aléatoirement les faits appris pour renforcer")
    print("les synapses via STDP. Inspiré du sommeil biologique.")
    print()

    cfg = SpikeConfig(n_sensory=300, n_associative=500, n_motor=300, sim_ticks=30)
    brain = SpikeBrain(cfg)
    brain.learn("le chat", "un animal")
    brain.learn("Paris", "la capitale de la France")
    brain.learn("la terre", "une planète")

    w_before = brain.net.syn_sens_to_assoc.W.data.mean()
    print(f"Poids sens→assoc avant dream: {w_before:.4f}")
    print(f"Poids sens→motor avant dream: {brain.syn_sens_to_motor.W.data.mean():.4f}")

    print("\nLancement du dream (5 replays, 20 ticks chacun)...")
    t0 = time.time()
    r = brain.dream(n_replays=5, ticks_per_replay=20)
    t1 = time.time()
    print(f"Résultat: {r}")
    print(f"Temps: {(t1-t0)*1000:.1f} ms")

    w_after = brain.net.syn_sens_to_assoc.W.data.mean()
    print(f"\nPoids sens→assoc après dream: {w_after:.4f}")
    print(f"Poids sens→motor après dream: {brain.syn_sens_to_motor.W.data.mean():.4f}")
    print(f"Delta sens→assoc: {w_after - w_before:+.4f}")
    print(f"Total dreams session: {brain.n_dreams}")


def demo_6_hybrid():
    """Hybride NOVA+SPIKE."""
    banner("6. HYBRIDE NOVA + SPIKE")
    print("Principe: SPIKE pour le temporel + NOVA pour la mémoire HD longue durée.")
    print("Double écriture à l'apprentissage, SPIKE d'abord au rappel, NOVA en fallback.")
    print()

    cfg = HybridConfig(
        spike=SpikeConfig(n_sensory=300, n_associative=500, n_motor=300, sim_ticks=25),
        nova=NovaConfig(D=3000, sdm_locations=5000),
    )
    brain = HybridBrain(cfg)

    print("Apprentissage (double write):")
    for k, v in [("le chat", "un animal qui miaule"),
                  ("Paris", "la capitale de la France"),
                  ("la terre", "une planète du système solaire"),
                  ("Einstein", "physicien, relativité")]:
        r = brain.learn(k, v)
        print(f"  ✓ appris: '{k}' = '{v}' ({r['time_ms']:.1f} ms)")

    print("\nRappel (SPIKE d'abord, NOVA en fallback):")
    for q in ["le chat", "Paris", "Einstein", "la lune"]:
        t0 = time.time()
        r = brain.recall(q)
        t1 = time.time()
        val = r["value"] if r["value"] and r["score"] > 1.5 else "(rien)"
        print(f"  {q!r:<15} -> {val!r:<40} "
              f"score={r['score']:.1f}, source={r['source']}, "
              f"({(t1-t0)*1000:.0f} ms)")

    print("\nOutils (via SPIKE):")
    print(f"  calcule 2+2 → {brain.chat('calcule 2+2')}")
    print(f"  quelle heure → {brain.chat('quelle heure est-il')}")

    print("\nStats finales:")
    s = brain.stats()
    print(f"  SPIKE: {s['spike']['n_facts']} faits, {s['spike']['vocab']} tokens")
    print(f"  NOVA:  {s['nova']['sdm']['writes']} écritures, D={s['nova']['D']}")
    print(f"  Calls: {s['n_calls']}, learns: {s['n_learns']}, fallbacks: {s['n_fallbacks']}")


def main():
    banner("SPIKE v2 — 6 NOUVELLES FEATURES", char="#")
    print("""
Cette démo présente les 6 features ajoutées à SPIKE:

  1. R-STDP — STDP reward-modulated (apprentissage par renforcement)
  2. Lazy spikes — buffer de propagation asynchrone (gros réseaux)
  3. Synapse direct sens→motor (one-shot plus fort)
  4. Save/load — persistance complète
  5. Mode rêve — replay aléatoire pour consolider STDP
  6. Hybride NOVA+SPIKE — best of both worlds
""")

    demo_1_rstdp()
    demo_2_lazy()
    demo_3_direct_synapse()
    demo_4_save_load()
    demo_5_dream()
    demo_6_hybrid()

    banner("FIN — TOUTES LES FEATURES SONT OPÉRATIONNELLES", char="#")


if __name__ == "__main__":
    main()
