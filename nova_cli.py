#!/usr/bin/env python3
"""
NOVA CLI — interface interactive en ligne de commande.

Usage:
    python nova_cli.py                    # mode interactif
    python nova_cli.py --debug            # mode debug (sortie JSON)
    python nova_cli.py --small            # config légère (D=2000)
    python nova_cli.py --large            # config large (D=20000)
    python nova_cli.py --load PATH        # charge un état sauvegardé
    python nova_cli.py --repl             # force le mode REPL

Exemples de conversation:
    > apprends que le chat est un animal
    > apprends que Paris est la capitale de la France
    > que sais-tu sur le chat ?
    > rappelle Paris
    > calcule 15 fois 3
    > python: print([i**2 for i in range(5)])
    > /tools
    > /stats
    > /save /tmp/nova_state
"""

import argparse
import sys
import os

# Ajoute le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova import Nova, NovaConfig


BANNER = r"""
 _   _ _____ __  ___   _ ___
| \ | | ____\ \/ / | | / __|
|  \| |  _|  \  /| |_| \__ \
| |\  | |___ /  \|  _  |___/
|_| \_|_____/_/\_\_| |_|_(_)

Neural Oscillatory Vector Architecture
IA sans transformer, sans GPU, apprentissage instantané
"""


def parse_args():
    p = argparse.ArgumentParser(description="NOVA — IA sans transformer")
    p.add_argument("--debug", action="store_true", help="mode debug (sortie JSON)")
    p.add_argument("--small", action="store_true", help="config légère (D=2000, rapide)")
    p.add_argument("--large", action="store_true", help="config large (D=20000, qualité)")
    p.add_argument("--load", type=str, help="charge un état sauvegardé")
    p.add_argument("--repl", action="store_true", help="force le mode REPL")
    p.add_argument("--seed", type=int, default=42, help="graine aléatoire")
    p.add_argument("--demo", action="store_true", help="lance la démo puis quitte")
    return p.parse_args()


def make_config(args) -> NovaConfig:
    if args.small:
        return NovaConfig(D=2000, sdm_locations=5000, debug=args.debug,
                          resonator_steps=20)
    if args.large:
        return NovaConfig(D=20000, sdm_locations=100000, debug=args.debug,
                          resonator_steps=40)
    return NovaConfig(D=10000, sdm_locations=50000, debug=args.debug)


def run_demo(nova: Nova):
    """Démo de 30s qui montre toutes les capacités."""
    print("\n" + "=" * 60)
    print("DÉMO NOVA")
    print("=" * 60)

    demos = [
        ("apprends que le chat est un animal", "apprentissage one-shot"),
        ("apprends que le chien est un animal", "apprentissage one-shot"),
        ("apprends que Paris est la capitale de la France", "apprentissage one-shot"),
        ("apprends que la terre tourne autour du soleil", "apprentissage one-shot"),
        ("que sais-tu sur le chat", "rappel mémoire"),
        ("rappelle Paris", "rappel mémoire"),
        ("que sais-tu sur la terre", "rappel mémoire"),
        ("calcule 2+2", "outil calculator"),
        ("combien font 15 fois 3", "outil calculator (mots)"),
        ("python: print(sum(range(10)))", "outil python"),
        ("quelle heure est-il", "outil time"),
    ]
    for user, label in demos:
        print(f"\n[{label}]")
        print(f"user > {user}")
        response = nova.chat(user)
        # Échappe les codes ANSI pour l'affichage
        response = response.replace("\x1b", "\\x1b")
        print(f"nova > {response}")


def main():
    args = parse_args()
    print(BANNER)

    cfg = make_config(args)
    print(f"Initialisation (D={cfg.D}, SDM={cfg.sdm_locations})...")

    import numpy as np
    nova = Nova(cfg, rng=np.random.default_rng(args.seed))

    if args.load:
        print(f"Chargement depuis {args.load}...")
        nova.load(args.load)

    print(f"Prêt. {nova.tokenizer.vocab_size} tokens en mémoire.\n")

    if args.demo:
        run_demo(nova)
        print("\n[fin de la démo]")
        return

    nova.interactive()


if __name__ == "__main__":
    main()
