"""
Distributed — orchestrateur multi-brain.

Principe:
  - Plusieurs cerveaux tournent en parallèle, chacun spécialisé
  - Un routeur décide quel cerveau traiter la requête
  - Si plusieurs cerveaux sont pertinents, on peut fusionner les réponses

Spécialisations:
  - "math"    → SPIKE configuré pour le calcul (outils calculator + python)
  - "memory"  → NOVA avec grosse SDM (mémoire à long terme)
  - "general" → HYBRID (combinaison)

Le routeur:
  - Pattern matching (regex) sur la requête
  - Si "calcule" → math
  - Si "que sais-tu sur" → memory
  - Sinon → general

Usage:
    from distributed import DistributedBrain
    dist = DistributedBrain()
    response = dist.chat("calcule 2+2")
"""

from __future__ import annotations
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from spike import SpikeBrain, SpikeConfig
from nova import Nova, NovaConfig
from hybrid import HybridBrain, HybridConfig


@dataclass
class BrainSpec:
    """Spécification d'un cerveau spécialisé."""
    name: str
    description: str
    patterns: list[str]
    brain: object = field(init=False)
    n_calls: int = field(init=False, default=0)
    total_time_ms: float = field(init=False, default=0.0)


class DistributedBrain:
    """Orchestrateur multi-brain."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        print("Initialisation du cluster distribué...")
        t0 = time.time()

        math_spec = BrainSpec(
            name="math",
            description="Spécialiste calcul et Python",
            patterns=[
                r"calcule?", r"combien", r"que vaut", r"fois", r"plus",
                r"moins", r"divis", r"puissance", r"racine", r"modul",
                r"python", r"code", r"exécut", r"execut",
                r"\d+\s*[+\-*/]",
            ],
        )
        math_spec.brain = SpikeBrain(SpikeConfig(
            n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25,
        ))
        self.specs: dict[str, BrainSpec] = {"math": math_spec}

        memory_spec = BrainSpec(
            name="memory",
            description="Mémoire associative à long terme",
            patterns=[
                r"que sais-tu", r"rappelle", r"as-tu appris", r"as-tu retenu",
                r"quel(?:le)? est", r"qui est",
            ],
        )
        memory_spec.brain = Nova(NovaConfig(D=2000, sdm_locations=5000))
        self.specs["memory"] = memory_spec

        general_spec = BrainSpec(
            name="general",
            description="Cerveau généraliste (fallback)",
            patterns=[],
        )
        general_spec.brain = HybridBrain(HybridConfig(
            spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25),
            nova=NovaConfig(D=3000, sdm_locations=5000),
        ))
        self.specs["general"] = general_spec

        self.n_calls = 0
        self.n_routed = {name: 0 for name in self.specs}
        self.history: list[dict] = []

        print(f"Prêt en {time.time()-t0:.2f}s — {len(self.specs)} cerveaux: "
              f"{list(self.specs.keys())}")

    def route(self, text: str) -> str:
        for name, spec in self.specs.items():
            if name == "general":
                continue
            for pat in spec.patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return name
        return "general"

    def chat(self, text: str) -> str:
        t0 = time.time()
        self.n_calls += 1
        brain_name = self.route(text)
        self.n_routed[brain_name] += 1
        spec = self.specs[brain_name]
        spec.n_calls += 1

        learn_match = None
        for pat in [r"apprends?\s+(?:que\s+)?(.+)", r"mémorise\s+(?:que\s+)?(.+)"]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                learn_match = m.group(1)
                break
        if learn_match:
            fact = learn_match.strip()
            if " est " in fact:
                k, v = fact.split(" est ", 1)
                results = []
                for name in ["memory", "general"]:
                    self.specs[name].brain.learn(k.strip(), v.strip())
                    results.append(f"{name}: ✓")
                response = f"[dist:learn] {k.strip()} = {v.strip()} ({', '.join(results)})"
            else:
                results = []
                for name in ["memory", "general"]:
                    self.specs[name].brain.learn(fact)
                    results.append(f"{name}: ✓")
                response = f"[dist:learn] {fact} ({', '.join(results)})"
        else:
            response = self.specs[brain_name].brain.chat(text)
            if brain_name != "general" and ("rien appris" in response.lower()
                                              or "erreur" in response.lower()):
                if self.verbose:
                    print(f"  [fallback] {brain_name} → general")
                response = self.specs["general"].brain.chat(text)
                brain_name = "general (fallback)"

        dt = (time.time() - t0) * 1000
        spec.total_time_ms += dt
        self.history.append({
            "input": text,
            "response": response,
            "brain": brain_name,
            "time_ms": dt,
            "timestamp": time.time(),
        })
        return response

    def print_stats(self) -> None:
        print("\n=== Distributed Stats ===")
        print(f"  Total calls: {self.n_calls}")
        print(f"  Routing:")
        for name, n in self.n_routed.items():
            spec = self.specs[name]
            avg = spec.total_time_ms / max(1, spec.n_calls)
            print(f"    {name}: {n} routed, {spec.n_calls} total, "
                  f"avg {avg:.1f} ms/call")
        print("=========================\n")

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_brains": len(self.specs),
            "routing": self.n_routed,
            "brains": {
                name: {
                    "description": spec.description,
                    "n_calls": spec.n_calls,
                    "total_time_ms": spec.total_time_ms,
                    "avg_time_ms": spec.total_time_ms / max(1, spec.n_calls),
                }
                for name, spec in self.specs.items()
            },
        }


if __name__ == "__main__":
    print("=== Test Distributed Brain ===\n")
    dist = DistributedBrain(verbose=True)

    print("\nTests de routing:")
    tests = [
        "calcule 2+2",
        "combien font 15 fois 3",
        "python: print(42)",
        "que sais-tu sur le chat",
        "rappelle Paris",
        "apprends que Mars est une planète",
        "bonjour, comment vas-tu?",
    ]
    for t in tests:
        route = dist.route(t)
        print(f"  '{t[:40]}' → {route}")

    print("\nChat:")
    for t in tests:
        r = dist.chat(t)
        print(f"  user: {t}")
        print(f"  dist: {r}")
        print()

    dist.print_stats()
