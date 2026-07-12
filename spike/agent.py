"""
SpikeAgent — couche agentique sur la population motrice.

Principe:
  - On réserve une partie de la population motrice pour des "neurones d'outils"
    (au lieu de tokens).
  - Quand un neurone d'outil dépasse un seuil d'activité, on déclenche l'outil.
  - Le résultat de l'outil peut être réinjecté comme input sensoriel (boucle
    agentique complète).

C'est la même philosophie que l'agent de NOVA, mais la décision se fait
par l'activité neuronale émergente plutôt que par similarité HD.
"""

from __future__ import annotations
import re
import math
import subprocess
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .coder import SpikeCoder, tokenize


# ---------------------------------------------------------------------- #
# Outils
# ---------------------------------------------------------------------- #

@dataclass
class SpikeTool:
    """Définition d'un outil agentic."""
    name: str
    description: str
    intent_keywords: list[str]
    slot_size: int          # nombre de neurones moteurs réservés
    executor: Callable[[str], str]
    pattern: str | None = None
    motor_start: int = -1   # index de début du slot, assigné à l'init


# Implémentations d'outils (réutilisées de NOVA, adaptées)

def _tool_calculator(expression: str) -> str:
    expr = expression.strip()
    m = re.search(r"(?:calcule|calcul|combien (?:fait|font)?|que vaut|évalue|eval)?\s*(.+)",
                  expr, re.IGNORECASE)
    if m:
        expr = m.group(1).strip().rstrip("?").strip()
    expr = re.sub(r"^(?:le|la|les|l'|un|une|des|du|de)\s+", "", expr,
                  flags=re.IGNORECASE).strip()
    func_subs = [
        (r"\bracine (?:carr[ée]e? )?de\s+(\S+)", r"sqrt(\1)"),
        (r"\bcarr[ée] de\s+(\S+)", r"(\1)**2"),
        (r"\bcube de\s+(\S+)", r"(\1)**3"),
        (r"\bpuissance\s+(\S+)\s+(?:de\s+)?(\S+)", r"(\1)**(\2)"),
    ]
    for pat, sub in func_subs:
        expr = re.sub(pat, sub, expr, flags=re.IGNORECASE)
    word_to_sym = {
        r"\bplus\b": "+", r"\bmoins\b": "-",
        r"\bfois\b|\bmultipli[eé] par\b|\bmultipli[eé]\b": "*",
        r"\bdivis[eé] par\b": "/", r"\bmodulo\b|\bmod\b": "%",
        r"\bpuissance\b|\bexposant\b": "**",
    }
    for pat, sym in word_to_sym.items():
        expr = re.sub(pat, sym, expr, flags=re.IGNORECASE)
    safe = re.sub(r"[^0-9a-zA-Z_+\-*/().,\s%]", "", expr).replace(",", ".")
    safe = safe.replace("^", "**")
    if not safe.strip():
        return f"[calc] Expression vide: {expression}"
    try:
        import ast
        ns = {"abs": abs, "round": round, "min": min, "max": max,
              "sqrt": math.sqrt, "pow": pow,
              "sin": math.sin, "cos": math.cos, "tan": math.tan,
              "log": math.log, "log10": math.log10, "exp": math.exp,
              "pi": math.pi, "e": math.e, "tau": math.tau}
        try:
            tree = ast.parse(safe, mode="eval")
        except SyntaxError as e:
            return f"[calc] Syntaxe invalide: '{safe}' ({e})"
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in ns:
                return f"[calc] Symbole non autorisé: '{node.id}'"
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, ns)
        return f"{safe} = {result}"
    except Exception as e:
        return f"[calc] Erreur: {e}"


def _tool_python_exec(code: str) -> str:
    m = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        m = re.search(r"(?:python|code|exécute|execute)\s*[:>]?\s*(.+)",
                      code, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1)
    code = code.strip()
    if not code:
        return "[python] Aucun code fourni"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "PYTHONPATH": "/home/z/my-project"}
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        return out or err or "[python] (aucune sortie)"
    except subprocess.TimeoutExpired:
        return "[python] Timeout (>5s)"
    except Exception as e:
        return f"[python] Erreur: {e}"


def _tool_time(_: str) -> str:
    return time.strftime("Il est %H:%M:%S le %d/%m/%Y", time.localtime())


def _tool_ls(path: str) -> str:
    m = re.search(r"(?:liste|ls|dir)\s*(.+)?", path, re.IGNORECASE)
    p = (m.group(1).strip() if m and m.group(1) else ".") or "."
    try:
        entries = sorted(os.listdir(p))
        return "\n".join(entries) or "[ls] Répertoire vide"
    except Exception as e:
        return f"[ls] Erreur: {e}"


def _tool_file_read(path: str) -> str:
    m = re.search(r"(?:lire|read|affiche|montre|cat)\s+(.+)", path, re.IGNORECASE)
    if m:
        path = m.group(1).strip().strip('"\'')
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(10000)
    except Exception as e:
        return f"[file] Erreur: {e}"


DEFAULT_TOOLS: list[SpikeTool] = [
    SpikeTool(name="calculator",
              description="Calcule une expression arithmétique",
              intent_keywords=["calcule", "calcul", "combien", "que vaut",
                                "fois", "plus", "moins"],
              slot_size=10,
              executor=_tool_calculator,
              pattern=r"(?:calcule|calcul|combien (?:fait|font)?|que vaut|évalue|eval)\s+(.+)"),
    SpikeTool(name="python",
              description="Exécute du code Python",
              intent_keywords=["python", "code", "exécute", "execute"],
              slot_size=10,
              executor=_tool_python_exec,
              pattern=r"```python\s*(.*?)```|(?:python|code|exécute|execute)\s*[:>]?\s*(.+)"),
    SpikeTool(name="time",
              description="Heure et date actuelles",
              intent_keywords=["heure", "time", "date", "quel jour"],
              slot_size=8,
              executor=_tool_time,
              pattern=r"(?:quelle\s+heure|quel\s+jour|l'heure|la\s+date|aujourd'hui|maintenant)"),
    SpikeTool(name="ls",
              description="Liste les fichiers",
              intent_keywords=["liste", "ls", "dir", "fichiers"],
              slot_size=8,
              executor=_tool_ls,
              pattern=r"(?:liste|ls|dir)\s*(.+)?"),
    SpikeTool(name="file_read",
              description="Lit un fichier (chemin requis)",
              intent_keywords=["lire", "read", "affiche", "montre", "cat"],
              slot_size=8,
              executor=_tool_file_read,
              pattern=r"(?:lire|read|affiche|montre|cat)\s+([/\w][\w./-]+\.\w+)"),
]


# ---------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------- #

@dataclass
class SpikeAgent:
    """
    Couche agentique pour SPIKE.

    Réserve des slots moteurs pour les outils, et surveille l'activité
    pour déclencher les outils.

    Args:
        coder: SpikeCoder
        motor_offset: index à partir duquel on place les slots d'outils
            dans la population motrice (les premiers neurones sont pour
            les tokens).
        trigger_threshold: nombre minimum de spikes dans un slot pour
            déclencher l'outil.
        intent_threshold: similarité HD minimum pour déclencher sans
            activité moteur (fallback symbolique).
    """
    coder: SpikeCoder
    tools: list[SpikeTool] = field(default_factory=lambda: list(DEFAULT_TOOLS))
    motor_offset: int = -1   # assigné par __post_init__
    trigger_threshold: int = 5
    intent_threshold: float = 0.15

    # Vecteurs d'intent par mot-clé (pour fallback symbolique)
    _intent_vectors: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Calcul de l'offset: place les outils APRÈS les tokens
        # motor_offset = (n_tokens_max * neurons_per_token)
        # Mais on veut que ce soit dynamique. On réserve une zone fixe en
        # fin de population motrice.
        total_tool_neurons = sum(t.slot_size for t in self.tools)
        self.motor_offset = self.coder.n_motor - total_tool_neurons
        if self.motor_offset < self.coder.vocab_size * self.coder.neurons_per_token:
            # Pas assez de place — on prend ce qu'on peut
            self.motor_offset = max(self.coder.vocab_size * self.coder.neurons_per_token,
                                     self.coder.n_motor // 2)
        # Assigne les slots
        cursor = self.motor_offset
        for tool in self.tools:
            tool.motor_start = cursor
            cursor += tool.slot_size

        # Vecteurs d'intent (one-hot sparse sur les neurones sensoriels
        # correspondant aux keywords)
        self._build_intent_vectors()

    def _build_intent_vectors(self) -> None:
        """Pour chaque outil, construit un vecteur d'intent = courant
        sensoriel déclenché par les keywords."""
        for tool in self.tools:
            intent = np.zeros(self.coder.n_sensory, dtype=np.float32)
            for kw in tool.intent_keywords:
                # Pour chaque keyword, on ajoute son courant
                kw_current = self.coder.encode_text_to_current(kw, gain=0.5)
                intent += kw_current
            # Normalize
            if intent.max() > 0:
                intent /= intent.max()
            self._intent_vectors[tool.name] = intent

    # ---------------------------------------------------------------- #
    # Détection d'intent
    # ---------------------------------------------------------------- #
    def detect_intent(self, text: str) -> list[tuple[SpikeTool, float, str | None]]:
        """
        Détecte les outils pertinents par similarité symbolique (fallback).
        Renvoie une liste triée (meilleur d'abord) de (tool, similarity, arg).

        IMPORTANT: un outil n'est déclenché QUE si son pattern regex match.
        La similarité sert à départager les outils qui matchent.
        """
        text_vec = self.coder.encode_text_to_current(text, gain=1.0)
        # Normalize
        if text_vec.max() > 0:
            text_vec_norm = text_vec / text_vec.max()
        else:
            text_vec_norm = text_vec
        results = []
        for tool in self.tools:
            intent = self._intent_vectors[tool.name]
            # Cosinus similarité
            denom = np.linalg.norm(text_vec_norm) * np.linalg.norm(intent)
            sim = float(text_vec_norm @ intent) / denom if denom > 0 else 0.0
            # Regex extraction — OBLIGATOIRE pour déclencher
            arg = None
            if tool.pattern:
                m = re.search(tool.pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    # Prend le premier groupe non-None
                    groups = m.groups()
                    arg = next((g for g in groups if g is not None), m.group(0))
                    if arg:
                        sim = max(sim, 0.5)  # bonus significatif
                    else:
                        # Pattern matched mais aucun groupe — on garde sim tel quel
                        sim = max(sim, 0.4)
                else:
                    # Pas de match regex — on force sim bas
                    sim = sim * 0.2
            results.append((tool, sim, arg))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ---------------------------------------------------------------- #
    # Lecture de l'activité moteur pour déclencher un outil
    # ---------------------------------------------------------------- #
    def read_motor_activity(self, spike_counts: np.ndarray) -> list[tuple[SpikeTool, int]]:
        """
        Pour chaque outil, compte les spikes dans son slot moteur.
        Renvoie les outils par activité décroissante.
        """
        results = []
        for tool in self.tools:
            slot = slice(tool.motor_start, tool.motor_start + tool.slot_size)
            count = int(spike_counts[slot].sum())
            results.append((tool, count))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def maybe_trigger_tool_by_activity(self, spike_counts: np.ndarray) -> tuple[str | None, str | None, SpikeTool | None]:
        """
        Si un slot d'outil dépasse le seuil, déclenche l'outil.
        Return: (tool_name, None, tool_obj) ou (None, None, None).
        L'argument de l'outil sera à fournir séparément.
        """
        results = self.read_motor_activity(spike_counts)
        if not results:
            return None, None, None
        tool, count = results[0]
        if count < self.trigger_threshold:
            return None, None, None
        return tool.name, None, tool

    def maybe_trigger_tool_by_intent(self, text: str) -> tuple[str | None, str | None, SpikeTool | None]:
        """
        Détection symbolique (fallback) quand l'activité moteur n'a rien donné.
        OBLIGATOIRE: le pattern regex doit matcher pour déclencher l'outil.
        """
        candidates = self.detect_intent(text)
        if not candidates:
            return None, None, None
        # On prend le meilleur outil QUI A un arg extrait
        for tool, sim, arg in candidates:
            if arg is None or not arg.strip():
                continue
            # L'arg existe — on déclenche
            input_arg = arg if arg else text
            try:
                result = tool.executor(input_arg)
            except Exception as e:
                result = f"[{tool.name}] Erreur: {e}"
            return tool.name, result, tool
        return None, None, None

    # ---------------------------------------------------------------- #
    def list_tools(self) -> str:
        lines = []
        for t in self.tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def get_tool_current_injection(self, tool_name: str, gain: float = 1.0) -> np.ndarray:
        """Renvoie un courant sensoriel qui active le slot d'outil voulu.
        (Pour exécuter un outil volontairement en mode apprentissage.)"""
        I = np.zeros(self.coder.n_sensory, dtype=np.float32)
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            return I
        # Pour activer un outil, on injecte sur les slots sensoriels des keywords
        intent = self._intent_vectors[tool_name]
        I = intent * gain * 3.0
        return I


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    coder = SpikeCoder(n_sensory=300, n_motor=100, neurons_per_token=5)
    agent = SpikeAgent(coder=coder)
    print(f"Tools: {[t.name for t in agent.tools]}")
    print(f"Motor offset: {agent.motor_offset}")
    for t in agent.tools:
        print(f"  {t.name}: slot [{t.motor_start}, {t.motor_start + t.slot_size})")

    # Test intent detection
    tests = [
        "calcule 2+2",
        "python: print(42)",
        "quelle heure est-il",
        "bonjour",
    ]
    for t in tests:
        name, result, tool = agent.maybe_trigger_tool_by_intent(t)
        print(f"  [{t}] -> {name}({result})")
