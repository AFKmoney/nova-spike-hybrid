#!/usr/bin/env python3
"""
SPIKE CLI — interface interactive pour le cerveau SNN.

Usage:
    python spike_cli.py                  # config par défaut
    python spike_cli.py --small          # config légère (rapide)
    python spike_cli.py --large          # config large (qualité)
    python spike_cli.py --demo           # lance la démo
    python spike_cli.py --no-stdp        # désactive STDP (debug)
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import SpikeBrain, SpikeConfig


BANNER = r"""
  ███████ ██   ██ ██ ███    ██ ██ ███████
  ██      ██  ██  ██ ████   ██ ██ ██
  ███████ █████   ██ ██ ██  ██ ██ █████
       ██ ██  ██  ██ ██  ██ ██ ██ ██
  ███████ ██   ██ ██ ██   ████ ██ ███████

Spiking Pattern Intelligence with Kernel Execution
SNN (LIF + STDP) sur CPU, sans GPU, sans transformer
"""


def parse_args():
    p = argparse.ArgumentParser(description="SPIKE — IA à impulsions")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--small", action="store_true", help="config légère")
    p.add_argument("--large", action="store_true", help="config large")
    p.add_argument("--no-stdp", action="store_true", help="désactive STDP")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true")
    return p.parse_args()


def make_config(args) -> SpikeConfig:
    if args.small:
        cfg = SpikeConfig(n_sensory=200, n_associative=500, n_motor=150,
                          sim_ticks=30, debug=args.debug)
    elif args.large:
        cfg = SpikeConfig(n_sensory=600, n_associative=2500, n_motor=400,
                          sim_ticks=60, debug=args.debug)
    else:
        cfg = SpikeConfig(n_sensory=400, n_associative=1500, n_motor=300,
                          sim_ticks=50, debug=args.debug)
    if args.no_stdp:
        cfg.stdp_enabled = False
    return cfg


def run_demo(brain: SpikeBrain):
    print("\n" + "=" * 60)
    print("  DÉMO SPIKE")
    print("=" * 60)

    demos = [
        ("apprends que le chat est un animal", "apprentissage one-shot"),
        ("apprends que Paris est la capitale de la France", "apprentissage"),
        ("apprends que la terre tourne autour du soleil", "apprentissage"),
        ("que sais-tu sur le chat", "rappel par SNN"),
        ("rappelle Paris", "rappel par SNN"),
        ("calcule 2+2", "outil calculator"),
        ("combien font 15 fois 3", "outil calculator (mots)"),
        ("python: print(sum(range(10)))", "outil python"),
        ("quelle heure est-il", "outil time"),
    ]
    for user, label in demos:
        print(f"\n[{label}]")
        print(f"user > {user}")
        response = brain.chat(user)
        response = response.replace("\x1b", "\\x1b")
        print(f"spike> {response}")


def main():
    args = parse_args()
    print(BANNER)

    cfg = make_config(args)
    print(f"Initialisation (sensory={cfg.n_sensory}, assoc={cfg.n_associative}, "
          f"motor={cfg.n_motor}, STDP={'ON' if cfg.stdp_enabled else 'OFF'})...")

    t0 = __import__("time").time()
    brain = SpikeBrain(cfg, rng=np.random.default_rng(args.seed))
    print(f"Prêt en {(__import__('time').time() - t0)*1000:.0f} ms.")
    print(f"{brain.coder.vocab_size} tokens, "
          f"{brain.net.syn_sens_to_assoc.W.nnz} synapses sens→assoc.\n")

    if args.demo:
        run_demo(brain)
        print("\n[fin de la démo]")
        return

    # REPL
    print("=" * 60)
    print("  SPIKE — Mode interactif")
    print("Commandes: /tools  /stats  /reset  /quit")
    print("=" * 60 + "\n")

    while True:
        try:
            user = input("\n[user] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir!")
            break
        if not user:
            continue

        if user.startswith("/"):
            cmd = user[1:].split(None, 1)
            if cmd[0] == "quit":
                print("Au revoir!")
                break
            elif cmd[0] == "tools":
                print(brain.agent.list_tools())
            elif cmd[0] == "stats":
                brain.print_stats()
            elif cmd[0] == "reset":
                brain.net.reset()
                print("[reset] Réseau réinitialisé.")
            else:
                print(f"[?] Commande inconnue: /{cmd[0]}")
            continue

        import time as _time
        t0 = _time.time()
        response = brain.chat(user)
        dt = (_time.time() - t0) * 1000
        print(f"\n[spike] > {response}")
        if args.debug:
            print(f"        ({dt:.1f} ms)")


if __name__ == "__main__":
    main()
