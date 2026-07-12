"""
Couche agentique — détection d'intent et appel d'outils.

Le module Agent:
  1. Définit un ensemble d'outils (calculator, python_exec, file_read, ...)
  2. Chaque outil a un vecteur d'intent HD (signature)
  3. Pour un input utilisateur, calcule la similarité avec chaque intent
  4. Si un outil "résonne" (sim > seuil), on l'exécute
  5. Le résultat est réinjecté comme contexte

C'est du "tool calling" symbolique — pas besoin de LLM pour décider.
"""

from __future__ import annotations
import re
import math
import json
import subprocess
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from .hd import HDVector, hd_random, hd_similarity
from .tokenizer import SimpleTokenizer
from .encoder import HDEncoder


# ---------------------------------------------------------------------- #
# Outils — signatures et exécuteurs
# ---------------------------------------------------------------------- #

@dataclass
class Tool:
    """Définition d'un outil agentic."""
    name: str
    description: str               # description en langage naturel
    intent_keywords: list[str]    # mots-clés pour l'intent
    intent_vector: HDVector | None = field(init=False, default=None)
    executor: Callable[[str], str] = field(default=lambda x: "")
    pattern: str | None = None    # regex optionnelle pour extract args

    def __post_init__(self):
        # L'intent_vector est construit plus tard par l'Agent
        pass


# ---------------------------------------------------------------- #
# Implémentations d'outils concrets
# ---------------------------------------------------------------- #

def _tool_calculator(expression: str) -> str:
    """Évaluateur arithmétique sécurisé."""
    # Extrait l'expression mathématique
    expr = expression.strip()
    # Pattern: "calcule X" ou "X = ?" ou juste "X"
    m = re.search(r"(?:calcule|calcul|combien (?:fait|font)?|que vaut|évalue|eval)?\s*(.+)",
                  expr, re.IGNORECASE)
    if m:
        expr = m.group(1).strip().rstrip("?").strip()
    # Supprime les articles définis/indéfinis en début d'expression
    expr = re.sub(r"^(?:le|la|les|l'|un|une|des|du|de)\s+", "", expr, flags=re.IGNORECASE).strip()
    # Substitutions de mots mathématiques -> symboles/fonctions
    # Pour les fonctions unaires comme sqrt, on capture l'argument
    func_subs = [
        (r"\bracine (?:carr[ée]e? )?de\s+(\S+)", r"sqrt(\1)"),
        (r"\bcarr[ée] de\s+(\S+)", r"(\1)**2"),
        (r"\bcube de\s+(\S+)", r"(\1)**3"),
        (r"\bpuissance\s+(\S+)\s+(?:de\s+)?(\S+)", r"(\1)**(\2)"),
        (r"\bexposant\s+(\S+)\s+(?:de\s+)?(\S+)", r"(\1)**(\2)"),
    ]
    for pat, sub in func_subs:
        expr = re.sub(pat, sub, expr, flags=re.IGNORECASE)
    # Substitutions mot → symbole
    word_to_sym = {
        r"\bplus\b": "+",
        r"\bmoins\b": "-",
        r"\bfois\b|\bmultipli[eé] par\b|\bmultipli[eé]\b": "*",
        r"\bdivis[eé] par\b": "/",
        r"\bmodulo\b|\bmod\b": "%",
        r"\bpuissance\b|\bexposant\b": "**",
    }
    for pat, sym in word_to_sym.items():
        expr = re.sub(pat, sym, expr, flags=re.IGNORECASE)
    # Sécurise: permet chiffres, opérateurs, fonctions, parenthèses, virgules
    # On autorise les lettres pour les noms de fonctions (sqrt, sin, cos, ...)
    safe = re.sub(r"[^0-9a-zA-Z_+\-*/().,\s%]", "", expr).replace(",", ".")
    # Vérifie qu'on n'a que des identifiers sûrs (pas d'appels systèmes)
    # Les fonctions autorisées sont dans le namespace `ns` ci-dessous
    if not safe.strip():
        return f"[calc] Expression vide ou non reconnue: {expression}"
    # Mappe ^ vers **
    safe = safe.replace("^", "**")
    try:
        # Namespace restreint — aucun __builtins__
        ns = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
            "pi": math.pi, "e": math.e, "tau": math.tau,
        }
        # AST check: rejette tout ce qui n'est pas une expression arithmétique
        import ast
        try:
            tree = ast.parse(safe, mode="eval")
        except SyntaxError as e:
            return f"[calc] Syntaxe invalide: '{safe}' ({e})"
        # Vérifie que tous les Name nodes sont dans ns
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in ns:
                return f"[calc] Fonction/variable non autorisée: '{node.id}'"
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, ns)
        return f"{safe} = {result}"
    except Exception as e:
        return f"[calc] Erreur sur '{safe}': {e}"


def _tool_python_exec(code: str) -> str:
    """Exécute du code Python dans un subprocess isolé (timeout 5s)."""
    # Extrait le code d'un bloc ```python ... ```
    m = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        # Sinon prend tout après "python:" ou "code:"
        m = re.search(r"(?:python|code|exécute|execute)\s*[:>]?\s*(.+)",
                      code, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1)
    code = code.strip()
    if not code:
        return "[python] Aucun code fourni"

    # Capture stdout+stderr
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "PYTHONPATH": "/home/z/my-project"}
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if out and err:
            return f"{out}\n[stderr] {err}"
        return out or err or "[python] (aucune sortie)"
    except subprocess.TimeoutExpired:
        return "[python] Timeout (>5s)"
    except Exception as e:
        return f"[python] Erreur: {e}"


def _tool_file_read(path: str) -> str:
    """Lit un fichier texte (limité à 10Ko)."""
    # Extrait le chemin
    m = re.search(r"(?:lire|read|affiche|montre|cat)\s+(.+)", path, re.IGNORECASE)
    if m:
        path = m.group(1).strip().strip('"\'')
    path = path.strip().strip('"\'')
    if not path:
        return "[file] Aucun chemin fourni"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(10000)
        return content or "[file] Fichier vide"
    except Exception as e:
        return f"[file] Erreur: {e}"


def _tool_file_write(args: str) -> str:
    """Écrit dans un fichier. Format: path ||| contenu"""
    if "|||" in args:
        path, content = args.split("|||", 1)
        path = path.strip().strip('"\'')
        content = content.strip()
    else:
        # Tente de parser "path contenu..."
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            return "[write] Format attendu: path ||| contenu"
        path, content = parts
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[write] Écrit {len(content)} caractères dans {path}"
    except Exception as e:
        return f"[write] Erreur: {e}"


def _tool_time(_: str) -> str:
    """Renvoie l'heure actuelle."""
    return time.strftime("Il est %H:%M:%S le %d/%m/%Y", time.localtime())


def _tool_list_files(path: str) -> str:
    """Liste les fichiers d'un répertoire."""
    m = re.search(r"(?:liste|ls|dir)\s*(.+)?", path, re.IGNORECASE)
    p = (m.group(1).strip() if m and m.group(1) else ".") or "."
    try:
        entries = sorted(os.listdir(p))
        return "\n".join(entries) or "[ls] Répertoire vide"
    except Exception as e:
        return f"[ls] Erreur: {e}"


def _tool_search_memory(query: str) -> str:
    """Ce tool est un placeholder — il sera intercepté par la brain pour
    faire une recherche SDM. La signature existe pour l'intent detection."""
    return f"[memory] (intercepté par la brain) query={query!r}"


# ---------------------------------------------------------------- #
# Registre des outils
# ---------------------------------------------------------------- #

DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name="calculator",
        description="Calcule une expression arithmétique. Ex: 'calcule 2+2', 'que vaut 15*3?'",
        intent_keywords=["calcule", "calcul", "combien", "que vaut", "évalue",
                         "+", "-", "*", "/", "fois", "plus", "moins"],
        executor=_tool_calculator,
        pattern=r"(?:calcule|calcul|combien (?:fait|font)?|que vaut|évalue|eval)\s+(.+)",
    ),
    Tool(
        name="python",
        description="Exécute du code Python. Ex: 'python: print(2**10)'",
        intent_keywords=["python", "code", "exécute", "execute", "script",
                         "def ", "print(", "import "],
        executor=_tool_python_exec,
        pattern=r"```python\s*(.*?)```|(?:python|code|exécute|execute)\s*[:>]?\s*(.+)",
    ),
    Tool(
        name="file_read",
        description="Lit un fichier texte. Ex: 'lire /tmp/test.txt'",
        intent_keywords=["lire", "read", "affiche", "montre", "cat", "fichier"],
        executor=_tool_file_read,
        pattern=r"(?:lire|read|affiche|montre|cat)\s+(.+)",
    ),
    Tool(
        name="file_write",
        description="Écrit dans un fichier. Ex: 'écris /tmp/test.txt ||| contenu'",
        intent_keywords=["écris", "write", "sauve", "enregistre"],
        executor=_tool_file_write,
        pattern=r"(?:écris|write|sauve|enregistre)\s+(.+)",
    ),
    Tool(
        name="time",
        description="Renvoie l'heure actuelle.",
        intent_keywords=["heure", "time", "date", "quel jour", "maintenant",
                         "aujourd"],
        executor=_tool_time,
        pattern=r"(?:quelle\s+heure|quel\s+jour|l'heure|la\s+date|aujourd'hui|maintenant)",
    ),
    Tool(
        name="ls",
        description="Liste les fichiers d'un répertoire.",
        intent_keywords=["liste", "ls", "dir", "fichiers"],
        executor=_tool_list_files,
        pattern=r"(?:liste|ls|dir)\s*(.+)?",
    ),
]


# ---------------------------------------------------------------- #
# Agent — détection d'intent + exécution
# ---------------------------------------------------------------- #

class Agent:
    """
    Couche agentique: détecte l'outil à appeler, l'exécute, renvoie
    le résultat formaté.
    """

    def __init__(self, tokenizer: SimpleTokenizer, encoder: HDEncoder,
                 tools: list[Tool] | None = None,
                 intent_threshold: float = 0.15):
        self.tok = tokenizer
        self.enc = encoder
        self.intent_threshold = intent_threshold
        self.tools: list[Tool] = tools or list(DEFAULT_TOOLS)
        self._build_intent_vectors()

    def _build_intent_vectors(self) -> None:
        """
        Construit un vecteur d'intent HD pour chaque outil en bundle
        des vecteurs de tokens de ses keywords.
        """
        from .hd import hd_bundle
        for tool in self.tools:
            kw_vecs = []
            for kw in tool.intent_keywords:
                # Pour les keywords multi-mots, on encode comme une séquence
                kw_vec = self.enc.encode_text(kw)
                kw_vecs.append(kw_vec)
            if kw_vecs:
                tool.intent_vector = hd_bundle(*kw_vecs)
            else:
                tool.intent_vector = hd_random(self.tok.D)

    # ---------------------------------------------------------------- #
    def detect_intent(self, text: str) -> list[tuple[Tool, float, str | None]]:
        """
        Détecte les outils pertinents pour un texte.
        Renvoie une liste triée (meilleur en premier) de:
            (tool, similarity, extracted_arg)
        """
        text_vec = self.enc.encode_text(text)
        results = []
        for tool in self.tools:
            if tool.intent_vector is None:
                continue
            sim = hd_similarity(text_vec, tool.intent_vector)
            # En plus de la similarité HD, on teste le pattern regex
            arg = None
            if tool.pattern:
                m = re.search(tool.pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    arg = m.group(1) if m.groups() else m.group(0)
                    sim = max(sim, 0.3)  # bonus si le pattern match
            results.append((tool, sim, arg))
        # Tri par similarité décroissante
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def maybe_call_tool(self, text: str) -> tuple[str | None, str | None, Tool | None]:
        """
        Si un outil "résonne" au-dessus du seuil, l'exécute.
        Return: (tool_name, result, tool_obj) ou (None, None, None).

        Règle de déclenchement:
          - Soit la similarité HD > intent_threshold
          - Soit un pattern regex match (détection symbolique forte)
          - Et on exige que l'arg extrait soit non vide si regex match
        """
        candidates = self.detect_intent(text)
        if not candidates:
            return None, None, None
        tool, sim, arg = candidates[0]
        # Déclenchement si HD sim haute OU regex match
        triggered = (sim >= self.intent_threshold) or (arg is not None and arg.strip())
        if not triggered:
            return None, None, None
        # Argument: arg extrait par regex, sinon le texte brut
        input_arg = arg if arg else text
        try:
            result = tool.executor(input_arg)
        except Exception as e:
            result = f"[{tool.name}] Erreur d'exécution: {e}"
        return tool.name, result, tool

    def list_tools(self) -> str:
        """Liste lisible des outils disponibles."""
        lines = []
        for t in self.tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)


if __name__ == "__main__":
    tok = SimpleTokenizer(D=2000)
    enc = HDEncoder(tok)
    agent = Agent(tok, enc)

    tests = [
        "Calcule 2+2",
        "Que vaut 15 fois 3?",
        "python: print(2**10)",
        "Quelle heure est-il?",
        "Bonjour, comment vas-tu?",
    ]
    for t in tests:
        name, result, tool = agent.maybe_call_tool(t)
        if name:
            print(f"[{t}] -> {name}({result})")
        else:
            print(f"[{t}] -> pas d'outil déclenché")
