#!/usr/bin/env python3
"""
Generative CLI — IA générative interactive sans transformer.

Modes:
  /reason <question>    — raisonnement (cognitive loop)
  /gen <prompt>         — génération libre token-par-token
  /story <theme>        — écrit une histoire
  /poem <topic>         — écrit un poème
  /essay <topic>        — écrit un essai
  /summarize <text>     — résume un texte
  /explain <topic>      — explique un sujet
  /teach <fact>         — apprend un fait
  /train <text>         — entraîne sur un texte
  /stats                — statistiques
  /help                 — aide
  /quit                 — quitter

Sans commande: mode chat automatique (route selon l'input).
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generative import GenerativeBrain, GenerativeConfig


BANNER = r"""
 ╔═══════════════════════════════════════════════════════════╗
 ║   GenerativeBrain — Non-Transformer Generative AI         ║
 ║   Reasoning · Generation · Creative Writing · Analysis    ║
 ║   CPU-only · No GPU · No transformer · No external LLM    ║
 ╚═══════════════════════════════════════════════════════════╝
"""


HELP = """
Commands:
  /reason <question>     Reason about a question (cognitive loop)
  /gen <prompt>          Generate text token-by-token
  /story [theme]         Write a short story
  /poem <topic>          Write a poem
  /essay <topic>         Write an essay
  /summarize <text>      Summarize text
  /explain <topic>       Explain a topic
  /teach <fact>          Teach a fact (one-shot learning)
  /train <text>          Train on a text block
  /stats                 Show brain statistics
  /help                  Show this help
  /quit                  Exit

Without a command, just type naturally — the brain will route automatically:
  - "Tell me about X"      → reasoning
  - "Write a poem about X" → creative writing
  - "Summarize: ..."       → analysis
  - "Why is X?"            → reasoning
  - "X is Y"               → teach
"""


def main():
    parser = argparse.ArgumentParser(description="GenerativeBrain CLI")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip default corpus pre-training")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(BANNER)
    print("Initializing...\n")

    cfg = GenerativeConfig(
        pretrained=not args.no_pretrain,
        verbose=args.verbose,
    )
    t0 = time.time()
    brain = GenerativeBrain(cfg)
    print(f"\nReady in {time.time()-t0:.1f}s. Type /help for commands.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue

        if user.startswith("/"):
            cmd, _, rest = user[1:].partition(" ")
            cmd = cmd.lower()
            rest = rest.strip()

            if cmd == "quit" or cmd == "exit":
                print("Bye!")
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "stats":
                brain.print_stats()
            elif cmd == "reason":
                if not rest:
                    print("Usage: /reason <question>")
                    continue
                t0 = time.time()
                r = brain.reason(rest)
                print(f"ai> {r}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "gen":
                if not rest:
                    print("Usage: /gen <prompt>")
                    continue
                t0 = time.time()
                r = brain.generate(rest, max_tokens=30)
                print(f"ai> {rest} {r}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "story":
                t0 = time.time()
                r = brain.write_story(rest or None)
                print(f"ai> {r.get('text', r)}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "poem":
                if not rest:
                    print("Usage: /poem <topic>")
                    continue
                t0 = time.time()
                r = brain.write_poem(rest)
                print(f"ai> {r.get('text', r)}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "essay":
                if not rest:
                    print("Usage: /essay <topic>")
                    continue
                t0 = time.time()
                r = brain.write_essay(rest)
                print(f"ai> {r.get('text', r)}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "summarize":
                if not rest:
                    print("Usage: /summarize <text>")
                    continue
                t0 = time.time()
                r = brain.summarize(rest)
                print(f"ai> {r}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "explain":
                if not rest:
                    print("Usage: /explain <topic>")
                    continue
                t0 = time.time()
                r = brain.explain(rest)
                print(f"ai> {r}  ({(time.time()-t0)*1000:.0f} ms)")
            elif cmd == "teach":
                if not rest:
                    print("Usage: /teach <fact>")
                    continue
                r = brain.teach(rest)
                print(f"ai> [learned] {rest}")
            elif cmd == "train":
                if not rest:
                    print("Usage: /train <text>")
                    continue
                r = brain.train_on_text(rest)
                print(f"ai> [trained] {r}")
            else:
                print(f"Unknown command: /{cmd}. Type /help.")
        else:
            # Chat automatique
            t0 = time.time()
            r = brain.chat(user)
            print(f"ai> {r}  ({(time.time()-t0)*1000:.0f} ms)")


if __name__ == "__main__":
    main()
