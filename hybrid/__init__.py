"""
HYBRID — NOVA + SPIKE combinés.

Architecture:
  - SPIKE: raisonnement temporel, dynamique émergente, tool-calling
    par activité neuronale. Excellent pour la perception immédiate.
  - NOVA: mémoire HD à long terme, rappel associatif robuste,
    one-shot sur des faits statiques. Excellent pour la persistance.

HybridBrain utilise:
  - SPIKE pour traiter l'input, choisir l'outil, générer une réponse immédiate
  - NOVA pour stocker les faits à long terme (mémoire HD distribuée)
  - Si SPIKE ne trouve pas (faible activité moteur), on fallback sur NOVA
  - Quand SPIKE apprend, on l'apprend aussi dans NOVA (double écriture)
  - Le mode rêve peut utiliser NOVA pour rejouer des faits anciens
"""

from __future__ import annotations
import re
import time
import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from spike import SpikeBrain, SpikeConfig
from nova import Nova, NovaConfig


@dataclass
class HybridConfig:
    """Configuration de HYBRID."""
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    nova: NovaConfig = field(default_factory=NovaConfig)
    spike_first: bool = True          # SPIKE d'abord, NOVA en fallback
    double_write: bool = True         # apprend dans les deux
    use_nova_recall: bool = True      # fallback NOVA si SPIKE échoue
    nova_threshold: float = 1.5       # seuil score SPIKE pour fallback NOVA
    debug: bool = False


class HybridBrain:
    """
    HYBRID — combine SPIKE (temporel) et NOVA (mémoire HD).

    Workflow:
      1. Input → SPIKE.think() → réponse + score
      2. Si score < nova_threshold et use_nova_recall → NOVA.recall()
      3. Si learn: double_write → SPIKE.learn() + NOVA.learn()
    """

    def __init__(self, cfg: HybridConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or HybridConfig()
        self.rng = rng or np.random.default_rng(42)

        # Les deux cerveaux
        self.spike = SpikeBrain(self.cfg.spike, rng=self.rng)
        self.nova = Nova(self.cfg.nova, rng=self.rng)

        # Stats hybrides
        self.n_learns = 0
        self.n_calls = 0
        self.n_fallbacks = 0  # nombre de fois où on a basculé sur NOVA
        self.history: list[dict] = []

    # ================================================================ #
    # APPRENTISSAGE — double écriture
    # ================================================================ #
    def learn(self, fact: str, value: str | None = None) -> dict:
        """
        Apprentissage double:
          - SPIKE: imprint synaptique (one-shot direct)
          - NOVA: écriture SDM (mémoire HD distribuée)
        """
        t0 = time.time()
        result_spike = self.spike.learn(fact, value)
        result_nova = self.nova.learn(fact, value)
        self.n_learns += 1
        t1 = time.time()
        return {
            "status": "learned",
            "fact": fact,
            "value": value,
            "spike_writes": result_spike.get("n_sensory_neurons", 0),
            "nova_writes": result_nova.get("writes", 0),
            "time_ms": (t1 - t0) * 1000,
        }

    # ================================================================ #
    # RAPPEL — SPIKE d'abord, NOVA en fallback
    # ================================================================ #
    def recall(self, query: str) -> dict:
        """
        Rappel hybride:
          1. SPIKE simule et décode l'activité moteur
          2. Si score SPIKE < threshold → fallback NOVA
          3. Sinon: retourne le résultat SPIKE
        """
        # 1. SPIKE recall
        r_spike = self.spike.recall(query)
        if r_spike["score"] >= self.cfg.nova_threshold:
            r_spike["source"] = "spike"
            return r_spike

        # 2. Fallback NOVA
        if self.cfg.use_nova_recall:
            r_nova = self.nova.recall(query)
            if r_nova["similarity"] > 0.1:
                self.n_fallbacks += 1
                return {
                    "query": query,
                    "fact": r_nova["fact"],
                    "value": r_nova["value"],
                    "score": r_nova["similarity"] * 100,  # normalise
                    "confidence": r_nova["confidence"],
                    "source": "nova",
                }

        # 3. Aucun des deux n'a trouvé
        return {
            "query": query,
            "fact": None,
            "value": None,
            "score": r_spike["score"],
            "confidence": "low",
            "source": "none",
        }

    # ================================================================ #
    # CYCLE COGNITIF
    # ================================================================ #
    def think(self, input_text: str) -> dict:
        """Cycle cognitif hybride."""
        t_start = time.time()
        self.n_calls += 1

        # 1. Détecte apprentissage explicite
        learn_patterns = [
            r"apprends?\s+(?:que\s+)?(.+)",
            r"mémorise\s+(?:que\s+)?(.+)",
            r"retiens?\s+(?:que\s+)?(.+)",
        ]
        for pat in learn_patterns:
            m = re.search(pat, input_text, re.IGNORECASE)
            if m:
                fact = m.group(1).strip()
                if " est " in fact:
                    k, v = fact.split(" est ", 1)
                    return self.learn(k.strip(), v.strip())
                if "=" in fact:
                    k, v = fact.split("=", 1)
                    return self.learn(k.strip(), v.strip())
                return self.learn(fact)

        # 2. Cycle SPIKE
        result_spike = self.spike.think(input_text)

        # 3. Si SPIKE n'a pas trouvé de réponse convaincante, fallback NOVA
        if (self.cfg.use_nova_recall
            and result_spike.get("motor_total_spikes", 0) < 5
            and not result_spike.get("tool_used")):
            # Tente recall via NOVA
            r_nova = self.nova.recall(input_text)
            if r_nova["similarity"] > 0.15:
                self.n_fallbacks += 1
                val = r_nova["value"] if r_nova["value"] else r_nova["fact"]
                result_spike["response"] = (
                    f"[hybrid:nova] {val} "
                    f"(confiance: {r_nova['confidence']}, "
                    f"sim={r_nova['similarity']:.3f})"
                )
                result_spike["source"] = "nova"

        # Marque la source
        if "source" not in result_spike:
            result_spike["source"] = "spike" if result_spike.get("tool_used") else "spike"

        t_total = (time.time() - t_start) * 1000
        result_spike["time_ms"] = t_total
        result_spike["hybrid"] = True

        self.history.append({
            "input": input_text,
            "response": result_spike.get("response", ""),
            "source": result_spike.get("source", "spike"),
            "tool": result_spike.get("tool_used"),
            "timestamp": time.time(),
        })
        return result_spike

    # ================================================================ #
    # INTERFACE
    # ================================================================ #
    def chat(self, input_text: str) -> str:
        result = self.think(input_text)
        if self.cfg.debug:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if result.get("status") == "learned":
            fact = result.get("fact", "")
            value = result.get("value", "")
            if value:
                return f"[hybrid:appris] {fact} = {value}"
            return f"[hybrid:appris] {fact}"
        return result.get("response", "[erreur]")

    # ================================================================ #
    # MODE RÊVE — utilise les deux mémoires
    # ================================================================ #
    def dream(self, n_replays: int | None = None,
              ticks_per_replay: int = 30) -> dict:
        """
        Mode rêve hybride:
          - Rêve SPIKE (replay des faits)
          - Consolidation NOVA (réécrit les faits pour renforcer la SDM)
        """
        r_spike = self.spike.dream(n_replays, ticks_per_replay)
        # NOVA: réécrit tous les faits (consolidation SDM)
        nova_rewritten = 0
        for entry in self.nova.history:
            if entry.get("type") == "learn":
                # Réécrit (idempotent — la SDM cumule)
                self.nova.learn(entry["fact"], entry.get("value"))
                nova_rewritten += 1
        return {
            "spike_dream": r_spike,
            "nova_consolidated": nova_rewritten,
        }

    # ================================================================ #
    # PERSISTANCE
    # ================================================================ #
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.spike.save(os.path.join(path, "spike"))
        self.nova.save(os.path.join(path, "nova"))
        with open(os.path.join(path, "hybrid_config.json"), "w") as f:
            json.dump({
                "spike_first": self.cfg.spike_first,
                "double_write": self.cfg.double_write,
                "use_nova_recall": self.cfg.use_nova_recall,
                "nova_threshold": self.cfg.nova_threshold,
                "n_learns": self.n_learns,
                "n_calls": self.n_calls,
                "n_fallbacks": self.n_fallbacks,
            }, f, indent=2)

    def load(self, path: str) -> None:
        self.spike.load(os.path.join(path, "spike"))
        self.nova.load(os.path.join(path, "nova"))
        with open(os.path.join(path, "hybrid_config.json"), "r") as f:
            d = json.load(f)
        self.cfg.spike_first = d.get("spike_first", True)
        self.cfg.double_write = d.get("double_write", True)
        self.cfg.use_nova_recall = d.get("use_nova_recall", True)
        self.cfg.nova_threshold = d.get("nova_threshold", 1.5)
        self.n_learns = d.get("n_learns", 0)
        self.n_calls = d.get("n_calls", 0)
        self.n_fallbacks = d.get("n_fallbacks", 0)

    # ================================================================ #
    # STATS
    # ================================================================ #
    def print_stats(self) -> None:
        print("\n=== HYBRID Stats ===")
        print(f"  SPIKE: {self.spike.cfg.n_sensory} sens + "
              f"{self.spike.cfg.n_associative} assoc + "
              f"{self.spike.cfg.n_motor} motor")
        print(f"  NOVA:  D={self.nova.cfg.D}, SDM={self.nova.cfg.sdm_locations}")
        print(f"  Vocab: spike={self.spike.coder.vocab_size}, nova={self.nova.tokenizer.vocab_size}")
        print(f"  Faits: spike={len(self.spike.facts)}, nova={self.nova.memory.writes}")
        print(f"  Calls: {self.n_calls}, learns: {self.n_learns}")
        print(f"  Fallbacks (spike→nova): {self.n_fallbacks}")
        print(f"  Source breakdown:")
        sources = {}
        for h in self.history:
            s = h.get("source", "spike")
            sources[s] = sources.get(s, 0) + 1
        for s, n in sources.items():
            print(f"    {s}: {n}")
        print("=====================\n")

    def stats(self) -> dict:
        return {
            "spike": self.spike.stats(),
            "nova": self.nova.stats(),
            "n_learns": self.n_learns,
            "n_calls": self.n_calls,
            "n_fallbacks": self.n_fallbacks,
        }


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Test HYBRID Brain ===\n")
    cfg = HybridConfig(
        spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=30),
        nova=NovaConfig(D=3000, sdm_locations=5000),
    )
    brain = HybridBrain(cfg)

    # Apprentissage (double write)
    print(brain.chat("apprends que le chat est un animal"))
    print(brain.chat("apprends que Paris est la capitale de la France"))
    print(brain.chat("apprends que la terre tourne autour du soleil"))

    # Rappel — SPIKE d'abord
    print(brain.chat("que sais-tu sur le chat"))
    print(brain.chat("rappelle Paris"))

    # Outils — SPIKE
    print(brain.chat("calcule 2+2"))
    print(brain.chat("quelle heure est-il"))

    # Stats
    brain.print_stats()
