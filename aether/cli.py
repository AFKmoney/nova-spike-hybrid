#!/usr/bin/env python3
"""
cli.py — Interactive REPL for AETHER.

Usage:
    python -m aether.cli
    python -m aether.cli --explain      # show cognitive traces
    python -m aether.cli --seed file.json  # preload knowledge

Commands (typed in the REPL):
    ask <question>      — ask AETHER something
    teach <fact>        — teach a fact instantly
    explain             — show the cognitive trace of the last ask
    stats               — show memory stats
    save <path>         — save AETHER's knowledge to a file
    load <path>         — load knowledge from a file
    list                — list known triples
    exit / quit         — leave the REPL

You can also just type freely; AETHER will figure out whether you're
asking, teaching, or instructing a tool call.
"""

from __future__ import annotations
import sys
import os
import argparse

# Make 'aether' importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER


BANNER = r"""
   ___  _____ _______  _____  _____  _____ ______ _____
  / _ \/  ___/  ___\ \/ / _ \/  ___||  __ \| ___ \_   _|
 / /_\ \ `--.\ `--. \  / /_\ \ `--. | |  \/| |_/ / | |
 |  _  |`--. \`--. \/ /|  _  |`--. \| | __ |    /  | |
 | | | /\__/ /\__/ /  \| | | /\__/ /| |_\ \| |\ \  | |
 \_| |_\____/\____/_/\_\_| |_\____/  \____/\_| \_| \_/

  Adaptive Emergent Thinking Hyperdimensional Engine for Reasoning
  v2.0 — non-transformer, GPU-free, instant-learning, agentic, GPT-killer.

  13 tools, multi-hop reasoning, semantic HD embeddings, transparent proof.
  Type 'help' for commands, or just start chatting.
"""


HELP = """
AETHER v2 commands:
  <free text>         — AETHER decides: ask, teach, or tool call
  teach <fact>        — explicit one-shot learning
  explain             — show cognitive trace of last ask
  stats               — memory + vocab statistics
  list                — list learned triples
  save <path>         — save knowledge
  load <path>         — load knowledge
  compare X and Y     — compare two entities
  explain <subject>   — explain a concept
  define <subject>    — define a word
  summarize <n>       — summarize n most recent memories
  count triples|vocab|episodes
  calc <expr>         — safe arithmetic
  recall <query>      — search episodic memory
  exit / quit         — leave
"""


def main():
    parser = argparse.ArgumentParser(description="AETHER — cognitive agent REPL")
    parser.add_argument("--explain", action="store_true", help="show cognitive trace after each ask")
    parser.add_argument("--seed", type=str, help="preload knowledge from JSON")
    args = parser.parse_args()

    print(BANNER)
    agent = AETHER(verbose=False)

    if args.seed and os.path.exists(args.seed):
        agent.load(args.seed)
        print(f"[loaded knowledge from {args.seed}]")

    print(f"[AETHER ready — vocab={len(agent.assoc.vocab)}, triples={len(agent.assoc.triples)}]\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[goodbye]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit"):
            print("[goodbye]")
            break
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "stats":
            import json
            print(json.dumps(agent.stats(), indent=2))
            continue
        if cmd == "list":
            triples = agent.assoc.list_triples()
            if not triples:
                print("  (KB empty)")
            for t in triples:
                print(f"  {t}")
            continue
        if cmd == "explain":
            print(agent.explain_last())
            continue
        if cmd.startswith("save "):
            path = user_input[5:].strip()
            agent.save(path)
            print(f"[saved to {path}]")
            continue
        if cmd.startswith("load "):
            path = user_input[5:].strip()
            if os.path.exists(path):
                agent.load(path)
                print(f"[loaded from {path}]")
            else:
                print(f"[file not found: {path}]")
            continue
        if cmd.startswith("teach "):
            fact = user_input[6:].strip()
            msg = agent.teach(fact)
            print(f"aether> {msg}")
            continue

        # Default: free input — let AETHER figure it out
        try:
            answer = agent.ask(user_input, explain=args.explain)
            print(f"aether> {answer}")
        except Exception as e:
            print(f"aether> [error: {e}]")


if __name__ == "__main__":
    main()
