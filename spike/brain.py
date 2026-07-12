"""
SpikeBrain — orchestrateur complet SPIKE.

Cycle cognitif:
  1. PERCEPTION: encode le texte en courant sensoriel (Poisson sur n_ticks)
  2. SIMULATION: fait tourner le SNN pendant n_ticks, STDP online
  3. INTENT: lit l'activité motrice → déclenche un outil si seuil atteint
     - Sinon: fallback symbolique (regex + intent)
  4. LEARN (optionnel): si "apprends que X est Y", imprint le chemin
     sensory(X) → motor(Y)

Comparé à NOVA:
  - Pas de HD vectors, que des spikes
  - STDP locale au lieu de SDM statique
  - Vraie temporalité (ticks, latences)
  - Poids CSR sparse, 1% connectivité
"""

from __future__ import annotations
import re
import time
import json
import os
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import Optional

from .core import LIFNeuron, LIFParams, SimulationClock
from .network import SpikingNetwork, PopulationType, SynapseGroup
from .stdp import STDPTracker, STDPConfig, imprint_path
from .rstdp import RSTDPTracker, RSTDPConfig
from .coder import SpikeCoder, PopulationDecoder, tokenize
from .agent import SpikeAgent


# Patterns d'apprentissage explicite
_LEARN_PATTERNS = [
    r"apprends?\s+(?:que\s+)?(.+)",
    r"mémorise\s+(?:que\s+)?(.+)",
    r"retiens?\s+(?:que\s+)?(.+)",
    r"enregistre\s+(?:que\s+)?(.+)",
]
_QUERY_PATTERNS = [
    r"que\s+(?:sais-tu|as-tu appris|as-tu retenu)\s+(?:sur|de)\s+(.+)",
    r"rappelle(?:-toi)?\s+(.+)",
]


@dataclass
class SpikeConfig:
    """Configuration de SPIKE."""
    n_sensory: int = 600             # augmenté pour gérer + de tokens
    n_associative: int = 1500
    n_motor: int = 600               # augmenté pour gérer + de tokens
    neurons_per_token: int = 6
    sim_ticks: int = 50              # ticks par cycle cognitif
    input_gain: float = 2.5          # gain du courant sensoriel
    poisson_rate: float = 0.6        # proba de spike par neurone sensoriel actif
    stdp_enabled: bool = True
    rstdp_enabled: bool = False      # R-STDP pour apprentissage par renforcement
    learn_weight: float = 3.0        # poids imprint pour apprentissage
    motor_trigger_threshold: int = 4 # nb min de spikes pour déclencher un outil
    direct_sens_motor: bool = True   # synapse direct sens→motor (one-shot plus fort)
    direct_sens_motor_weight: float = 2.0  # poids de la synapse directe
    dream_enabled: bool = True       # mode rêve activable
    dream_default_replays: int = 5   # nb de replays par défaut
    debug: bool = False


class SpikeBrain:
    """
    SPIKE — Spiking Pattern Intelligence with Kernel Execution.
    """

    def __init__(self, cfg: SpikeConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or SpikeConfig()
        self.rng = rng or np.random.default_rng(42)

        # Réseau SNN
        self.net = SpikingNetwork(
            n_sensory=self.cfg.n_sensory,
            n_associative=self.cfg.n_associative,
            n_motor=self.cfg.n_motor,
        )

        # Codeur (partage la n_sensory/n_motor avec le réseau)
        self.coder = SpikeCoder(
            n_sensory=self.cfg.n_sensory,
            n_motor=self.cfg.n_motor,
            neurons_per_token=self.cfg.neurons_per_token,
        )

        # Agent (réserve des slots moteurs pour les outils)
        self.agent = SpikeAgent(
            coder=self.coder,
            trigger_threshold=self.cfg.motor_trigger_threshold,
        )

        # Décodeur de population
        self.decoder = PopulationDecoder(
            coder=self.coder,
            min_count=2,
            cooldown=3,
        )

        # Synapse direct sens→motor (one-shot plus fort)
        # Bypass la couche associative pour les faits explicites
        if self.cfg.direct_sens_motor:
            self.syn_sens_to_motor = SynapseGroup(
                "sens→motor (direct)", PopulationType.SENSORY, PopulationType.MOTOR,
                self.cfg.n_sensory, self.cfg.n_motor, plastic=True,
            )
            # Init à zéro — on imprint pendant learn()
        else:
            self.syn_sens_to_motor = None

        # Trackers STDP (classique ou R-STDP)
        stdp_cfg = STDPConfig()
        if self.cfg.rstdp_enabled:
            rstdp_cfg = RSTDPConfig()
            self.stdp_sens_assoc = RSTDPTracker(self.net.syn_sens_to_assoc, rstdp_cfg)
            self.stdp_assoc_assoc = RSTDPTracker(self.net.syn_assoc_to_assoc, rstdp_cfg)
            self.stdp_assoc_motor = RSTDPTracker(self.net.syn_assoc_to_motor, rstdp_cfg)
            if self.syn_sens_to_motor is not None:
                self.stdp_sens_motor = RSTDPTracker(self.syn_sens_to_motor, rstdp_cfg)
            else:
                self.stdp_sens_motor = None
        else:
            self.stdp_sens_assoc = STDPTracker(self.net.syn_sens_to_assoc, stdp_cfg)
            self.stdp_assoc_assoc = STDPTracker(self.net.syn_assoc_to_assoc, stdp_cfg)
            self.stdp_assoc_motor = STDPTracker(self.net.syn_assoc_to_motor, stdp_cfg)
            if self.syn_sens_to_motor is not None:
                self.stdp_sens_motor = STDPTracker(self.syn_sens_to_motor, stdp_cfg)
            else:
                self.stdp_sens_motor = None

        # Mémoire associative explicite (pour l'apprentissage one-shot)
        # Format: list de (fact, value, sensory_indices, motor_indices)
        self.facts: list[dict] = []

        # Stats
        self.n_learns = 0
        self.n_calls = 0
        self.n_tool_calls = 0
        self.n_rewards = 0
        self.n_dreams = 0
        self.history: list[dict] = []

    # ================================================================ #
    # APPRENTISSAGE — imprint one-shot + STDP online
    # ================================================================ #
    def learn(self, fact: str, value: str | None = None) -> dict:
        """
        Apprentissage one-shot:
          - Encode `fact` en tokens sensoriels
          - Encode `value` en tokens moteurs
          - Imprint les poids sensory→assoc et assoc→motor
            pour créer un chemin direct.

        Si value est None, on apprend juste le fait (auto-association).
        """
        # Tokenize
        fact_tokens = tokenize(fact)
        value_tokens = tokenize(value) if value else fact_tokens

        # Ajoute au vocab
        fact_ids = [self.coder.token2id.get(t, self.coder._add_token(t)) for t in fact_tokens]
        value_ids = [self.coder.token2id.get(t, self.coder._add_token(t)) for t in value_tokens]

        # Indices sensoriels: tous les neurones des slots des tokens du fact
        sensory_indices = []
        for tid in fact_ids:
            slot = self.coder.get_sensory_slot(tid)
            sensory_indices.extend(range(slot.start, slot.stop))

        # Indices moteurs: tous les neurones des slots des tokens du value
        motor_indices = []
        for tid in value_ids:
            slot = self.coder.get_motor_slot(tid)
            motor_indices.extend(range(slot.start, slot.stop))

        # Imprint: pour chaque neurone sensoriel actif, on force une connexion
        # vers quelques neurones associatifs, qui eux-mêmes connectent vers
        # les neurones moteurs cibles.
        # Pour simplifier (et garder l'approche localisée), on connecte directement
        # sensory → motor via les synapses sens→assoc et assoc→motor,
        # en choisissant un pool de neurones associatifs "dédiés" au fait.
        # On tire K neurones associatifs au hasard et on les wire des deux côtés.
        K = max(20, len(sensory_indices) * 2)
        assoc_indices = self.rng.choice(self.net.n_associative, size=K, replace=False).tolist()

        # Imprint sensory → assoc (force des poids forts)
        imprint_path(self.net.syn_sens_to_assoc,
                     sensory_indices, assoc_indices,
                     weight=self.cfg.learn_weight)
        # Imprint assoc → motor
        imprint_path(self.net.syn_assoc_to_motor,
                     assoc_indices, motor_indices,
                     weight=self.cfg.learn_weight)

        # Imprint direct sens → motor (one-shot plus fort, bypass assoc)
        if self.syn_sens_to_motor is not None:
            imprint_path(self.syn_sens_to_motor,
                         sensory_indices, motor_indices,
                         weight=self.cfg.direct_sens_motor_weight)

        # Enregistre le fait
        entry = {
            "fact": fact,
            "value": value,
            "fact_ids": fact_ids,
            "value_ids": value_ids,
            "sensory_indices": sensory_indices,
            "motor_indices": motor_indices,
            "assoc_indices": assoc_indices,
            "timestamp": time.time(),
        }
        self.facts.append(entry)
        self.n_learns += 1
        self.history.append({"type": "learn", **{k: v for k, v in entry.items()
                                                   if k not in ("sensory_indices", "motor_indices", "assoc_indices")}})

        return {
            "status": "learned",
            "fact": fact,
            "value": value,
            "n_sensory_neurons": len(sensory_indices),
            "n_motor_neurons": len(motor_indices),
            "n_assoc_pool": K,
        }

    # ================================================================ #
    # RAPPEL — simulation + décodage
    # ================================================================ #
    def recall(self, query: str) -> dict:
        """
        Rappelle la valeur associée à `query` en simulant le réseau
        et en décodant l'activité motrice.
        """
        # Reset l'état du réseau (mémoire de travail)
        self.net.reset(soft=False)
        self.decoder.reset()

        # Encode la query en courant sensoriel — gain plus fort pour le recall
        # pour s'assurer que l'activité se propage jusqu'au moteur
        I_static = self.coder.encode_text_to_current(query, gain=self.cfg.input_gain * 2.0)

        # Simule 2x plus de ticks pour le recall (laisser le temps de propager)
        sim_ticks = self.cfg.sim_ticks * 2

        # Simule avec _simulate (gère aussi synapse directe)
        total_motor_counts, motor_spike_log = self._simulate(I_static, sim_ticks)

        # Décode les spikes moteurs cumulés
        total_motor_counts = np.stack(motor_spike_log).sum(axis=0).astype(np.int32)
        # Top tokens
        top_tokens = self.coder.decode_top_k(total_motor_counts, k=5, min_count=1)

        # Cleanup: on cherche le fait appris dont la valeur a les meilleurs tokens
        # On exige un score minimum pour confirmer le rappel
        best_fact = None
        best_value = None
        best_score = -1.0
        for entry in self.facts:
            value_ids = entry["value_ids"]
            # Score: somme des spikes sur les slots des tokens de la valeur
            score = 0
            for tid in value_ids:
                slot = self.coder.get_motor_slot(tid)
                score += int(total_motor_counts[slot].sum())
            # Normalise par le nombre de tokens
            score = score / max(1, len(value_ids))
            # Bonus si le fact lui-même a été activé sensoriellement
            fact_ids = entry["fact_ids"]
            sensory_overlap = 0
            for tid in fact_ids:
                slot = self.coder.get_sensory_slot(tid)
                # On vérifie si les tokens du fait sont dans la query
                pass
            # Pénalité si la query n'a aucun token en commun avec le fact
            # On enlève les stopwords (articles, prépositions courantes)
            stopwords = {"le", "la", "les", "l", "un", "une", "des", "du", "de",
                         "et", "ou", "mais", "donc", "or", "ni", "car",
                         "que", "qui", "quoi", "dont", "où",
                         "sur", "sous", "dans", "avec", "sans", "pour",
                         "par", "vers", "en", "au", "aux", "ce", "cet", "cette"}
            query_tokens = set(tokenize(query)) - stopwords
            fact_tokens = set(tokenize(entry["fact"])) - stopwords
            common = query_tokens & fact_tokens
            if not common:
                score *= 0.02  # pénalité très forte — quasi-élimination
            elif len(common) < min(len(query_tokens), len(fact_tokens)) * 0.5:
                # Pas assez de tokens communs (content words)
                score *= 0.2
            if score > best_score:
                best_score = score
                best_fact = entry["fact"]
                best_value = entry["value"]

        return {
            "query": query,
            "fact": best_fact,
            "value": best_value,
            "score": best_score,
            "top_motor_tokens": top_tokens[:5],
            "motor_total_spikes": int(total_motor_counts.sum()),
            "confidence": "high" if best_score > 5 else "medium" if best_score > 1.5 else "low",
        }

    # ================================================================ #
    # STDP — appelé à chaque tick
    # ================================================================ #
    def _apply_stdp(self) -> None:
        """Applique STDP sur tous les groupes de synapses."""
        ls = self.net.last_spikes
        if not ls:
            return
        spikes_sens = ls["sensory"]
        spikes_assoc = ls["associative"]
        spikes_motor = ls["motor"]
        self.stdp_sens_assoc.update(spikes_sens, spikes_assoc)
        self.stdp_assoc_assoc.update(spikes_assoc, spikes_assoc)
        self.stdp_assoc_motor.update(spikes_assoc, spikes_motor)
        # STDP sur la synapse directe aussi
        if self.stdp_sens_motor is not None:
            self.stdp_sens_motor.update(spikes_sens, spikes_motor)

    def _simulate(self, I_static: np.ndarray, n_ticks: int,
                  record_motor: bool = True) -> tuple[np.ndarray, list]:
        """
        Simule n_ticks avec input Poisson. Applique aussi la synapse directe
        sens→motor à chaque tick (si elle existe).

        Return:
            total_motor_counts (n_motor,) int32
            motor_spike_log: list de masques booléens par tick
        """
        motor_spike_log = []
        sensory_spike_log = []
        for _ in range(n_ticks):
            mask = (self.rng.random(self.cfg.n_sensory) < self.cfg.poisson_rate).astype(np.float32)
            I_tick = I_static * mask
            # Tick standard (propage sens→assoc→motor)
            self.net.tick(I_tick)
            # Propagation directe sens→motor (si activée)
            if self.syn_sens_to_motor is not None:
                spikes_sens = self.net.last_spikes["sensory"]
                if spikes_sens.any():
                    # Calcule le courant direct vers motor
                    I_direct = self.syn_sens_to_motor.propagate(spikes_sens)
                    if I_direct.max() > 0:
                        # Injecte dans motor et re-step motor
                        # On cumule au potentiel motor existant
                        self.net.motor.V += I_direct * 0.5  # gain modéré
                        # Re-fire motor si seuil atteint
                        thresh = self.net.motor.params.V_thresh * self.net.motor.thresh_factor
                        new_spikes = self.net.motor.V >= thresh
                        if new_spikes.any():
                            self.net.motor.V[new_spikes] = self.net.motor.params.V_reset
                            self.net.motor.refractory[new_spikes] = self.net.motor.params.tau_ref
                            self.net.motor.spike_count[new_spikes] += 1
                            self.net.motor.last_spike_time[new_spikes] = self.net.clock.t
                            # Update last_spikes motor
                            self.net.last_spikes["motor"] = (
                                self.net.last_spikes["motor"] | new_spikes
                            )
            # STDP
            if self.cfg.stdp_enabled:
                self._apply_stdp()
            motor_spike_log.append(self.net.last_spikes["motor"].copy())
            sensory_spike_log.append(self.net.last_spikes["sensory"].copy())
        total_motor = np.stack(motor_spike_log).sum(axis=0).astype(np.int32) if motor_spike_log else np.zeros(self.cfg.n_motor, dtype=np.int32)
        return total_motor, motor_spike_log

    # ================================================================ #
    # CYCLE COGNITIF COMPLET
    # ================================================================ #
    def think(self, input_text: str) -> dict:
        """
        Cycle cognitif complet:
          1. Détecte les patterns d'apprentissage explicite
          2. Sinon: simule le réseau, lit l'activité motrice
          3. Déclenche un outil si activité suffisante
          4. Sinon: fallback symbolique
          5. Sinon: recall mémoire
        """
        t_start = time.time()
        self.n_calls += 1

        # ---- 1. Patterns d'apprentissage ----
        for pat in _LEARN_PATTERNS:
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

        # ---- 2. Pattern de query de mémoire ----
        query_match = None
        for pat in _QUERY_PATTERNS:
            m = re.search(pat, input_text, re.IGNORECASE)
            if m:
                query_match = m.group(1)
                break

        # ---- 3. Simule le réseau ----
        self.net.reset(soft=False)
        self.decoder.reset()

        # Encode l'input en courant
        I_static = self.coder.encode_text_to_current(input_text, gain=self.cfg.input_gain)

        # Simule avec _simulate (gère aussi synapse directe)
        total_motor_counts, motor_spike_log = self._simulate(I_static, self.cfg.sim_ticks)
        # Pour stats
        sensory_total = int(self.net.sensory.spike_count.sum())

        # ---- 4. Déclenche un outil par activité motrice ----
        tool_name, _, tool_obj = self.agent.maybe_trigger_tool_by_activity(total_motor_counts)

        # ---- 5. Fallback: détection symbolique ----
        if tool_name is None:
            tool_name, tool_result, tool_obj = self.agent.maybe_trigger_tool_by_intent(input_text)
        else:
            # L'outil a été détecté par activité — on doit quand même extraire l'arg
            # via regex
            arg = None
            if tool_obj and tool_obj.pattern:
                m = re.search(tool_obj.pattern, input_text, re.IGNORECASE | re.DOTALL)
                if m:
                    arg = m.group(1) if m.groups() else m.group(0)
            input_arg = arg if arg else input_text
            try:
                tool_result = tool_obj.executor(input_arg)
            except Exception as e:
                tool_result = f"[{tool_name}] Erreur: {e}"

        # ---- 6. Construit la réponse ----
        if tool_name:
            self.n_tool_calls += 1
            response = f"[outil:{tool_name}] {tool_result}"
        elif query_match:
            r = self.recall(query_match)
            if r["score"] > 1.5:
                val = r["value"] if r["value"] else r["fact"]
                response = f"[mémoire] {val} (confiance: {r['confidence']}, score={r['score']:.1f})"
            else:
                response = f"[mémoire] Je n'ai rien appris sur '{query_match}'."
        else:
            # Pas d'outil, pas de query — décodage symbolique
            top_tokens = self.coder.decode_top_k(total_motor_counts, k=5, min_count=2)
            if top_tokens:
                tokens_str = ", ".join(f"{t}({c})" for t, c in top_tokens)
                response = f"[résonance] activité motrice: {tokens_str}"
            else:
                # Check recall
                r = self.recall(input_text)
                if r["score"] > 1.5:
                    val = r["value"] if r["value"] else r["fact"]
                    response = f"[mémoire] {val} (confiance: {r['confidence']})"
                else:
                    response = (f"[résonance] J'ai traité votre input "
                                f"({int(total_motor_counts.sum())} spikes moteur, "
                                f"{sensory_total} sensory). "
                                f"Memoire: {len(self.facts)} faits appris.")

        t_total = (time.time() - t_start) * 1000

        result = {
            "input": input_text,
            "response": response,
            "tool_used": tool_name,
            "tool_result": tool_result if tool_name else None,
            "query_match": query_match,
            "motor_total_spikes": int(total_motor_counts.sum()),
            "sensory_total_spikes": sensory_total,
            "time_ms": t_total,
            "n_facts": len(self.facts),
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
    # INTERFACE
    # ================================================================ #
    def chat(self, input_text: str) -> str:
        """Interface simple."""
        result = self.think(input_text)
        if self.cfg.debug:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if result.get("status") == "learned":
            fact = result.get("fact", "")
            value = result.get("value", "")
            if value:
                return f"[appris] {fact} = {value}"
            return f"[appris] {fact}"
        return result.get("response", "[erreur]")

    # ================================================================ #
    # R-STDP — récompense/punition
    # ================================================================ #
    def give_reward(self, reward: float = 1.0) -> dict:
        """
        Applique un signal de récompense à tous les trackers R-STDP.
        Ne fonctionne que si cfg.rstdp_enabled = True.

        Args:
            reward: positif pour renforcer, négatif pour punir
        """
        if not self.cfg.rstdp_enabled:
            return {"status": "disabled", "msg": "R-STDP not enabled. Use --rstdp flag."}
        if not isinstance(self.stdp_sens_assoc, RSTDPTracker):
            return {"status": "disabled", "msg": "Trackers are not R-STDP."}
        self.stdp_sens_assoc.apply_reward(reward)
        self.stdp_assoc_assoc.apply_reward(reward)
        self.stdp_assoc_motor.apply_reward(reward)
        if self.stdp_sens_motor is not None:
            self.stdp_sens_motor.apply_reward(reward)
        self.n_rewards += 1
        return {"status": "applied", "reward": reward, "n_rewards": self.n_rewards}

    # ================================================================ #
    # MODE RÊVE — replay aléatoire pour consolider STDP
    # ================================================================ #
    def dream(self, n_replays: int | None = None,
              ticks_per_replay: int = 30) -> dict:
        """
        Mode rêve: rejoue aléatoirement des faits appris pour consolider
        les synapses via STDP.

        Principe biologique: pendant le sommeil, le cerveau "replay" les
        patterns d'activité de la journée. La STDP renforce alors les
        chemains souvent activés.

        Args:
            n_replays: nombre de replays (défaut: cfg.dream_default_replays)
            ticks_per_replay: durée de chaque replay en ticks
        """
        if not self.cfg.dream_enabled:
            return {"status": "disabled"}
        if not self.facts:
            return {"status": "no_facts", "msg": "Rien à rêver."}

        n = n_replays or self.cfg.dream_default_replays
        n = min(n, len(self.facts) * 3)  # max 3x le nombre de faits

        total_spikes = 0
        replays_done = 0

        for i in range(n):
            # Choisit un fait aléatoire
            entry = self.facts[self.rng.integers(0, len(self.facts))]
            fact = entry["fact"]
            # Reset réseau (mémoire de travail fraîche pour chaque replay)
            self.net.reset(soft=False)
            # Encode le fact en courant
            I_static = self.coder.encode_text_to_current(fact, gain=self.cfg.input_gain)
            # Simule sans STDP (le dream consolide, n'apprend pas de nouveau)
            stdp_was_enabled = self.cfg.stdp_enabled
            self.cfg.stdp_enabled = True  # ON pour consolider
            total_motor, _ = self._simulate(I_static, ticks_per_replay)
            total_spikes += int(total_motor.sum())
            replays_done += 1
            self.cfg.stdp_enabled = stdp_was_enabled

        self.n_dreams += 1
        return {
            "status": "dreamed",
            "n_replays": replays_done,
            "total_spikes": total_spikes,
            "n_dreams_session": self.n_dreams,
        }

    # ================================================================ #
    # PERSISTANCE — save/load
    # ================================================================ #
    def save(self, path: str) -> None:
        """Sauvegarde l'état du cerveau (synapses, vocab, facts)."""
        os.makedirs(path, exist_ok=True)
        # Synapses (sparse)
        sp.save_npz(os.path.join(path, "W_sens_assoc.npz"),
                    self.net.syn_sens_to_assoc.W)
        sp.save_npz(os.path.join(path, "W_assoc_assoc.npz"),
                    self.net.syn_assoc_to_assoc.W)
        sp.save_npz(os.path.join(path, "W_assoc_motor.npz"),
                    self.net.syn_assoc_to_motor.W)
        if self.syn_sens_to_motor is not None:
            sp.save_npz(os.path.join(path, "W_sens_motor.npz"),
                        self.syn_sens_to_motor.W)
        # Vocab
        np.save(os.path.join(path, "id2token.npy"),
                np.array(self.coder.id2token, dtype=object), allow_pickle=True)
        # Facts (sans les indices, on reconstruira au load)
        facts_clean = [{k: v for k, v in f.items() if k != "assoc_indices"}
                       for f in self.facts]
        with open(os.path.join(path, "facts.json"), "w", encoding="utf-8") as f:
            json.dump(facts_clean, f, ensure_ascii=False, indent=2)
        # History
        with open(os.path.join(path, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, default=str)
        # Config
        cfg_dict = {
            "n_sensory": self.cfg.n_sensory,
            "n_associative": self.cfg.n_associative,
            "n_motor": self.cfg.n_motor,
            "neurons_per_token": self.cfg.neurons_per_token,
            "sim_ticks": self.cfg.sim_ticks,
            "input_gain": self.cfg.input_gain,
            "poisson_rate": self.cfg.poisson_rate,
            "stdp_enabled": self.cfg.stdp_enabled,
            "rstdp_enabled": self.cfg.rstdp_enabled,
            "learn_weight": self.cfg.learn_weight,
            "motor_trigger_threshold": self.cfg.motor_trigger_threshold,
            "direct_sens_motor": self.cfg.direct_sens_motor,
            "direct_sens_motor_weight": self.cfg.direct_sens_motor_weight,
            "dream_enabled": self.cfg.dream_enabled,
            "dream_default_replays": self.cfg.dream_default_replays,
            "n_learns": self.n_learns,
            "n_calls": self.n_calls,
            "n_tool_calls": self.n_tool_calls,
            "n_rewards": self.n_rewards,
            "n_dreams": self.n_dreams,
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    def load(self, path: str) -> None:
        """Charge un état sauvegardé."""
        # Config
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        # Update cfg
        for k, v in cfg_dict.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        # Restore stats
        self.n_learns = cfg_dict.get("n_learns", 0)
        self.n_calls = cfg_dict.get("n_calls", 0)
        self.n_tool_calls = cfg_dict.get("n_tool_calls", 0)
        self.n_rewards = cfg_dict.get("n_rewards", 0)
        self.n_dreams = cfg_dict.get("n_dreams", 0)
        # Synapses
        self.net.syn_sens_to_assoc.W = sp.load_npz(
            os.path.join(path, "W_sens_assoc.npz"))
        self.net.syn_assoc_to_assoc.W = sp.load_npz(
            os.path.join(path, "W_assoc_assoc.npz"))
        self.net.syn_assoc_to_motor.W = sp.load_npz(
            os.path.join(path, "W_assoc_motor.npz"))
        if os.path.exists(os.path.join(path, "W_sens_motor.npz")):
            self.syn_sens_to_motor.W = sp.load_npz(
                os.path.join(path, "W_sens_motor.npz"))
        # Vocab
        id2token_arr = np.load(os.path.join(path, "id2token.npy"),
                                allow_pickle=True)
        self.coder.id2token = list(id2token_arr)
        self.coder.token2id = {t: i for i, t in enumerate(self.coder.id2token)}
        # Facts
        with open(os.path.join(path, "facts.json"), "r", encoding="utf-8") as f:
            facts_clean = json.load(f)
        # Reconstruit les indices
        self.facts = []
        for entry in facts_clean:
            fact = entry["fact"]
            value = entry["value"]
            fact_tokens = tokenize(fact)
            value_tokens = tokenize(value) if value else fact_tokens
            fact_ids = [self.coder.token2id.get(t, 1) for t in fact_tokens]
            value_ids = [self.coder.token2id.get(t, 1) for t in value_tokens]
            sensory_indices = []
            for tid in fact_ids:
                slot = self.coder.get_sensory_slot(tid)
                sensory_indices.extend(range(slot.start, slot.stop))
            motor_indices = []
            for tid in value_ids:
                slot = self.coder.get_motor_slot(tid)
                motor_indices.extend(range(slot.start, slot.stop))
            entry["fact_ids"] = fact_ids
            entry["value_ids"] = value_ids
            entry["sensory_indices"] = sensory_indices
            entry["motor_indices"] = motor_indices
            entry["assoc_indices"] = []  # perdu, pas grave
            self.facts.append(entry)
        # History
        if os.path.exists(os.path.join(path, "history.json")):
            with open(os.path.join(path, "history.json"), "r", encoding="utf-8") as f:
                self.history = json.load(f)

    # ================================================================ #
    # STATS
    # ================================================================ #
    def print_stats(self) -> None:
        print("\n=== SPIKE Stats ===")
        print(f"  Sensory:      {self.cfg.n_sensory} neurones")
        print(f"  Associative:  {self.cfg.n_associative} neurones")
        print(f"  Motor:        {self.cfg.n_motor} neurones")
        print(f"  Vocab:        {self.coder.vocab_size} tokens")
        print(f"  Faits appris: {len(self.facts)}")
        print(f"  Total calls:  {self.n_calls}")
        print(f"  Tool calls:   {self.n_tool_calls}")
        print(f"  Synapses:")
        for name, syn in [("sens→assoc", self.net.syn_sens_to_assoc),
                          ("assoc→assoc", self.net.syn_assoc_to_assoc),
                          ("assoc→motor", self.net.syn_assoc_to_motor)]:
            print(f"    {name}: {syn.W.nnz} synapses "
                  f"(w_mean={syn.W.data.mean() if syn.W.nnz > 0 else 0:.3f})")
        print("===================\n")

    def stats(self) -> dict:
        return {
            "n_sensory": self.cfg.n_sensory,
            "n_associative": self.cfg.n_associative,
            "n_motor": self.cfg.n_motor,
            "vocab": self.coder.vocab_size,
            "n_facts": len(self.facts),
            "n_calls": self.n_calls,
            "n_tool_calls": self.n_tool_calls,
            "synapses": {
                "sens_to_assoc": self.net.syn_sens_to_assoc.W.nnz,
                "assoc_to_assoc": self.net.syn_assoc_to_assoc.W.nnz,
                "assoc_to_motor": self.net.syn_assoc_to_motor.W.nnz,
            },
        }


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Test SPIKE Brain ===\n")
    cfg = SpikeConfig(n_sensory=300, n_associative=800, n_motor=200,
                       sim_ticks=40, debug=True)
    brain = SpikeBrain(cfg)

    # Apprentissage
    print(brain.chat("apprends que le chat est un animal"))
    print(brain.chat("apprends que Paris est la capitale de la France"))

    # Rappel
    print(brain.chat("que sais-tu sur le chat"))
    print(brain.chat("rappelle Paris"))

    # Outils
    print(brain.chat("calcule 2+2"))
    print(brain.chat("combien font 15 fois 3"))
    print(brain.chat("quelle heure est-il"))

    # Stats
    brain.print_stats()
