"""
Nova — le cerveau qui orchestre tout.

Le cycle cognitif de NOVA:
  1. PERCEPTION: encode l'input utilisateur en vecteur HD
  2. INTENT: l'Agent détecte s'il faut appeler un outil
     - Si oui: exécute l'outil, le résultat devient un nouvel input
     - Si non: passe à la suite
  3. RECALL: lit dans la SDM à l'adresse du contexte (mémoire à long terme)
  4. RESONATE: injecte (input + recall) dans le résonateur, laisse évoluer
  5. GENERATE: décode l'état du résonateur en tokens
  6. LEARN (optionnel): écrit la paire (input, response) dans la SDM

Toutes ces étapes sont O(D) ou O(N_active * D), CPU-only.

Le système est aussi "agentique": si l'input contient une demande
explicite d'apprentissage ("apprends que X est Y"), il l'écrit dans la SDM
sans passer par le cycle de génération.
"""

from __future__ import annotations
import re
import time
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hd import (
    HDVector, hd_random, hd_bind, hd_bundle, hd_permute,
    hd_similarity, hd_bundle_weighted,
)
from .memory import SparseDistributedMemory
from .tokenizer import SimpleTokenizer
from .encoder import HDEncoder
from .decoder import HDDecoder
from .resonator import Resonator, ResonatorConfig
from .agent import Agent, Tool


# Patterns d'apprentissage explicite
_LEARN_PATTERNS = [
    r"apprends?\s+(?:que\s+)?(.+)",
    r"mémorise\s+(?:que\s+)?(.+)",
    r"retiens?\s+(?:que\s+)?(.+)",
    r"enregistre\s+(?:que\s+)?(.+)",
    r"note\s+(?:que\s+)?(.+)",
]
_QUERY_PATTERNS = [
    r"que\s+(?:sais-tu|as-tu appris|as-tu retenu)\s+(?:sur|de)\s+(.+)",
    r"rappelle(?:-toi)?\s+(.+)",
    r"quel(?:le)?\s+(?:est|est la)\s+(.+)",
]


@dataclass
class NovaConfig:
    """Configuration de NOVA."""
    D: int = 10000
    sdm_locations: int = 50000
    sdm_k_active: int = 32
    sdm_radius: float = 0.0
    resonator_steps: int = 30
    resonator_sparsity: float = 0.01
    max_generation: int = 32
    learn_threshold: float = 0.10
    debug: bool = False


class Nova:
    """
    NOVA — Neural Oscillatory Vector Architecture.

    Un cerveau artificiel:
      - Sans transformer
      - Sans GPU (CPU-only)
      - Apprentissage instantané (one-shot, SDM)
      - Raisonnement continu (résonateur)
      - Mode agentique (outils)
      - Auto-contenu (pas d'LLM externe)
    """

    def __init__(self, cfg: NovaConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or NovaConfig()
        self.rng = rng or np.random.default_rng(42)
        D = self.cfg.D

        # Composants
        self.tokenizer = SimpleTokenizer(D=D)
        self.encoder = HDEncoder(self.tokenizer)
        self.decoder = HDDecoder(self.tokenizer, self.encoder)
        self.memory = SparseDistributedMemory(
            D=D, n_locations=self.cfg.sdm_locations,
            k_active=self.cfg.sdm_k_active,
            activation_radius=self.cfg.sdm_radius,
        )
        self.resonator = Resonator(
            ResonatorConfig(
                D=D,
                sparsity=self.cfg.resonator_sparsity,
                n_steps=self.cfg.resonator_steps,
            ),
            rng=self.rng,
        )
        self.agent = Agent(self.tokenizer, self.encoder)

        # Historique conversationnel (mémoire courte)
        self.history: list[dict] = []

        # Stats
        self.n_learns = 0
        self.n_calls = 0
        self.n_tool_calls = 0

    # ================================================================ #
    # APPRENTISSAGE — instantané, one-shot
    # ================================================================ #
    def learn(self, fact: str, value: str | None = None) -> dict:
        """
        Apprend un fait instantanément.

        Deux modes:
          - `learn("le chat est un animal")`     → clé = phrase, valeur = phrase
          - `learn("chat", "un animal")`          → clé = "chat", valeur = "un animal"

        Écrit dans la SDM: bind(key_vec, value_vec) à l'adresse key_vec.
        Le rappel se fait en passant key_vec à la SDM, puis unbind.
        """
        if value is None:
            # Fait unique: on stocke le fait à sa propre adresse, sans binding
            # (bind(a, a) = vecteur constant, inutile)
            fact_vec = self.encoder.encode_text(fact)
            key_vec = fact_vec
            value_vec = fact_vec
            # Écrit directement le vecteur du fait
            self.memory.write(key_vec, value_vec)
        else:
            key_vec = self.encoder.encode_text(fact)
            value_vec = self.encoder.encode_text(value)
            # Paire clé-valeur: on stocke bind(key, value) à l'adresse key
            # Au rappel: read(key) ≈ bind(key, value), puis unbind avec key → value
            bound = hd_bind(key_vec, value_vec)
            self.memory.write(key_vec, bound)

        # On garde aussi une entrée dans l'index (pour le cleanup)
        self.history.append({
            "type": "learn",
            "fact": fact,
            "value": value,
            "timestamp": time.time(),
        })
        self.n_learns += 1

        return {
            "status": "learned",
            "fact": fact,
            "value": value,
            "writes": self.memory.writes,
        }

    def learn_kv(self, key: str, value: str) -> dict:
        """Shortcut pour learn(key, value)."""
        return self.learn(key, value)

    def learn_facts(self, facts: list[tuple[str, str | None]]) -> dict:
        """Apprend plusieurs faits d'un coup."""
        results = []
        for fact, value in facts:
            results.append(self.learn(fact, value))
        return {"learned": len(results), "facts": results}

    # ================================================================ #
    # RAPPEL — lecture SDM + cleanup
    # ================================================================ #
    def recall(self, query: str) -> dict:
        """
        Rappelle ce qui a été appris en lien avec `query`.

        Étapes:
          1. Encode la query
          2. Lit dans la SDM à cette adresse
          3. Pour les paires clé-valeur: unbind avec la query pour récupérer value
             Pour les faits simples: le vecteur lu est déjà le fait
          4. Cleanup en testant toutes les entrées apprises
        """
        query_vec = self.encoder.encode_text(query)
        bound = self.memory.read(query_vec)

        # Cleanup: teste toutes les valeurs apprises
        # Pour les faits simples, `bound` ≈ fact_vec (pas besoin d'unbind)
        # Pour les paires (k, v), `bound` ≈ bind(k, v) → il faut unbind
        best_fact = None
        best_value = None
        best_sim = -2.0
        for entry in self.history:
            if entry["type"] != "learn":
                continue
            value_text = entry["value"] if entry["value"] else entry["fact"]
            value_vec = self.encoder.encode_text(value_text)
            # Test 1: directement (pour faits simples)
            s_direct = hd_similarity(bound, value_vec)
            # Test 2: après unbind avec la query (pour paires k-v)
            unbound = hd_bind(bound, query_vec)
            s_unbind = hd_similarity(unbound, value_vec)
            s = max(s_direct, s_unbind)
            if s > best_sim:
                best_sim = s
                best_fact = entry["fact"]
                best_value = entry["value"]

        return {
            "query": query,
            "fact": best_fact,
            "value": best_value,
            "similarity": best_sim,
            "confidence": "high" if best_sim > 0.3 else "medium" if best_sim > 0.1 else "low",
        }

    # ================================================================ #
    # RAISONNEMENT — cycle cognitif complet
    # ================================================================ #
    def think(self, input_text: str) -> dict:
        """
        Cycle cognitif:
          1. Encode l'input
          2. Détecte un intent outil (et l'exécute si trouvé)
          3. Lit dans la SDM à l'adresse du contexte
          4. Injecte (input + recall) dans le résonateur
          5. Laisse évoluer le champ
          6. Décode l'état en réponse
        """
        t_start = time.time()
        self.n_calls += 1

        # ----- 1. Perception -----
        input_vec = self.encoder.encode_text(input_text)

        # ----- 2. Intent + tool -----
        tool_name, tool_result, tool_obj = self.agent.maybe_call_tool(input_text)
        tool_used = tool_name is not None
        if tool_used:
            self.n_tool_calls += 1

        # ----- 3. Détecte les patterns d'apprentissage explicite -----
        learn_match = None
        for pat in _LEARN_PATTERNS:
            m = re.search(pat, input_text, re.IGNORECASE)
            if m:
                learn_match = m.group(1)
                break
        if learn_match:
            # Si le fait contient "est" ou "=", on sépare clé/valeur
            if " est " in learn_match:
                k, v = learn_match.split(" est ", 1)
                return self.learn(k.strip(), v.strip())
            if "=" in learn_match:
                k, v = learn_match.split("=", 1)
                return self.learn(k.strip(), v.strip())
            return self.learn(learn_match.strip())

        # ----- 4. Détecte les patterns de query de mémoire -----
        query_match = None
        for pat in _QUERY_PATTERNS:
            m = re.search(pat, input_text, re.IGNORECASE)
            if m:
                query_match = m.group(1)
                break

        # ----- 5. Recall SDM -----
        recall_input = query_match if query_match else input_text
        recall_vec = self.encoder.encode_text(recall_input)
        recalled, conf = self.memory.read_with_confidence(recall_vec)
        # Unbind pour récupérer la valeur
        recalled_value = hd_bind(recalled, recall_vec)

        # ----- 6. Résonateur -----
        self.resonator.reset()
        # Injecte l'input (avec gain fort)
        self.resonator.inject(input_vec, gain=1.0)
        # Injecte le recall (avec gain moyen)
        if conf > 0:
            self.resonator.inject(recalled_value, gain=0.5)
        # Injecte le résultat outil (si présent)
        if tool_result:
            tool_vec = self.encoder.encode_text(tool_result)
            self.resonator.inject(tool_vec, gain=0.8)

        # Évolution
        self.resonator.reason()

        # ----- 7. Décodage -----
        state_bipolar = self.resonator.get_state_bipolar()
        # Soit on décode directement la séquence
        # Soit on génère gloutonnement à partir de l'état
        # Ici on combine: on décode l'état comme un vecteur de "pensée"
        # puis on génère des tokens en lisant dans la SDM

        # Approche 1: décodage direct (cleanup tokens)
        decoded_text = self.decoder.decode_text(state_bipolar,
                                                max_len=self.cfg.max_generation)

        # Approche 2: si un tool a été utilisé, on retourne son résultat
        if tool_used:
            response = f"[outil:{tool_name}] {tool_result}"
        elif query_match and recall_vec is not None:
            # Pour une question de mémoire, on retourne le recall
            r = self.recall(recall_input)
            if r["similarity"] > 0.1:
                val = r["value"] if r["value"] else r["fact"]
                response = f"[mémoire] {val} (confiance: {r['confidence']}, sim={r['similarity']:.3f})"
            else:
                response = f"[mémoire] Je n'ai rien appris sur '{recall_input}'."
        elif decoded_text.strip():
            response = decoded_text
        else:
            # Fallback: on retourne un état de "pensée"
            sim_to_input = hd_similarity(state_bipolar, input_vec)
            response = (f"[résonance] J'ai traité votre input "
                        f"(sim_input={sim_to_input:+.3f}, "
                        f"énergie={self.resonator.energy():.3f}). "
                        f"Memoire: {self.memory.stats()['writes']} faits appris.")

        t_total = (time.time() - t_start) * 1000

        # ----- 8. Historique -----
        result = {
            "input": input_text,
            "response": response,
            "tool_used": tool_name,
            "tool_result": tool_result,
            "recall_query": query_match,
            "confidence": conf,
            "energy": self.resonator.energy(),
            "time_ms": t_total,
            "writes": self.memory.writes,
        }
        self.history.append({
            "type": "exchange",
            "input": input_text,
            "response": response,
            "tool": tool_name,
            "timestamp": time.time(),
        })
        return result

    # ================================================================ #
    # CHAT — interface REPL
    # ================================================================ #
    def chat(self, input_text: str) -> str:
        """Interface simple: input -> réponse textuelle."""
        result = self.think(input_text)
        if self.cfg.debug:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        # Si c'est un apprentissage, on formate proprement
        if result.get("status") == "learned":
            fact = result.get("fact", "")
            value = result.get("value", "")
            if value:
                return f"[appris] {fact} = {value}"
            return f"[appris] {fact}"
        return result.get("response", "[erreur] pas de réponse")

    def interactive(self) -> None:
        """Boucle REPL interactive."""
        print("\n" + "=" * 60)
        print("  NOVA — Neural Oscillatory Vector Architecture")
        print("  IA sans transformer, sans GPU, apprentissage instantané")
        print("=" * 60)
        print("Commandes spéciales:")
        print("  /tools      — liste les outils disponibles")
        print("  /stats      — statistiques du cerveau")
        print("  /save PATH  — sauvegarde l'état")
        print("  /load PATH  — charge un état")
        print("  /reset      — reset la mémoire de travail")
        print("  /quit       — quitter")
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
                    print(self.agent.list_tools())
                elif cmd[0] == "stats":
                    self.print_stats()
                elif cmd[0] == "reset":
                    self.resonator.reset()
                    print("[reset] Mémoire de travail effacée.")
                elif cmd[0] == "save" and len(cmd) > 1:
                    self.save(cmd[1])
                    print(f"[save] État sauvegardé dans {cmd[1]}")
                elif cmd[0] == "load" and len(cmd) > 1:
                    self.load(cmd[1])
                    print(f"[load] État chargé depuis {cmd[1]}")
                else:
                    print(f"[?] Commande inconnue: /{cmd[0]}")
                continue

            t0 = time.time()
            response = self.chat(user)
            dt = (time.time() - t0) * 1000
            print(f"\n[nova] > {response}")
            if self.cfg.debug:
                print(f"       ({dt:.1f} ms)")

    # ================================================================ #
    # DIAGNOSTICS
    # ================================================================ #
    def print_stats(self) -> None:
        """Affiche les statistiques du cerveau."""
        sdm_stats = self.memory.stats()
        print("\n=== NOVA Stats ===")
        print(f"  Dimension HD:        {self.cfg.D}")
        print(f"  Vocabulaire:         {self.tokenizer.vocab_size} tokens")
        print(f"  SDM locations:       {sdm_stats['n_locations']}")
        print(f"  SDM writes:          {sdm_stats['writes']}")
        print(f"  SDM active:          {sdm_stats['active_locations']} "
              f"({sdm_stats['fill_ratio']*100:.1f}%)")
        print(f"  Total learns:        {self.n_learns}")
        print(f"  Total calls:         {self.n_calls}")
        print(f"  Total tool calls:    {self.n_tool_calls}")
        print(f"  History size:        {len(self.history)}")
        print(f"  Resonator D:         {self.resonator.cfg.D}")
        print(f"  Resonator W nnz:     {self.resonator.W.nnz}")
        print("==================\n")

    def stats(self) -> dict:
        return {
            "D": self.cfg.D,
            "vocab": self.tokenizer.vocab_size,
            "sdm": self.memory.stats(),
            "n_learns": self.n_learns,
            "n_calls": self.n_calls,
            "n_tool_calls": self.n_tool_calls,
            "history_size": len(self.history),
            "resonator_nnz": int(self.resonator.W.nnz),
        }

    # ================================================================ #
    # PERSISTANCE
    # ================================================================ #
    def save(self, path: str) -> None:
        """Sauvegarde tokenizer + SDM contents + resonator."""
        import os
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.npz"))
        np.save(os.path.join(path, "sdm_locations.npy"),
                self.memory.locations)
        np.save(os.path.join(path, "sdm_contents.npy"),
                self.memory.contents)
        self.resonator.save(os.path.join(path, "resonator.npz"))
        # History
        with open(os.path.join(path, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        # Config
        cfg_dict = {
            "D": self.cfg.D, "sdm_locations": self.cfg.sdm_locations,
            "sdm_radius": self.cfg.sdm_radius,
            "resonator_steps": self.cfg.resonator_steps,
            "resonator_sparsity": self.cfg.resonator_sparsity,
            "max_generation": self.cfg.max_generation,
            "n_learns": self.n_learns, "n_calls": self.n_calls,
            "n_tool_calls": self.n_tool_calls,
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    def load(self, path: str) -> None:
        """Charge un état sauvegardé."""
        import os
        self.tokenizer.load(os.path.join(path, "tokenizer.npz"))
        self.memory.locations = np.load(os.path.join(path, "sdm_locations.npy"))
        self.memory.contents = np.load(os.path.join(path, "sdm_contents.npy"))
        self.resonator = Resonator.load(os.path.join(path, "resonator.npz"))
        with open(os.path.join(path, "history.json"), "r", encoding="utf-8") as f:
            self.history = json.load(f)
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        self.n_learns = cfg_dict.pop("n_learns", 0)
        self.n_calls = cfg_dict.pop("n_calls", 0)
        self.n_tool_calls = cfg_dict.pop("n_tool_calls", 0)


if __name__ == "__main__":
    # Test de smoke
    nova = Nova(NovaConfig(D=5000, sdm_locations=5000, debug=True))
    print("=== Test NOVA ===\n")

    # Apprentissage
    print(nova.chat("apprends que le chat est un animal"))
    print(nova.chat("apprends que le chien est un animal"))
    print(nova.chat("apprends que Paris est la capitale de la France"))

    # Rappel
    print(nova.chat("que sais-tu sur le chat"))
    print(nova.chat("rappelle Paris"))

    # Outils
    print(nova.chat("calcule 2+2"))
    print(nova.chat("combien font 15 fois 3?"))

    # Stats
    nova.print_stats()
