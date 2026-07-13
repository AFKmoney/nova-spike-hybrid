"""
GenerativeBrain — vraie IA générative sans transformer.

Combine AETHER (cognitive loop + HD language model + creative writer) avec
un pré-entraînement sur corpus pour produire une IA qui sait:
  - Raisonner (chain-of-thought multi-cycle)
  - Comprendre (analyse sémantique)
  - Générer du texte (token-by-token avec température)
  - Écrire créativement (story, poem, essay)
  - Analyser (summarize, explain)
  - Apprendre instantanément (one-shot teach)
  - Utiliser des outils (calculator, python, time)

Tout en CPU, sans GPU, sans transformer, sans LLM externe.
"""

from __future__ import annotations
import os
import sys
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import AETHER
from generative.corpus import get_full_corpus, get_corpus_sentences, get_corpus_stats
from generative.predictor import GenerativePredictor


# ---------------------------------------------------------------------- #
# Legacy small corpus (kept for backward compatibility)
# ---------------------------------------------------------------------- #

DEFAULT_CORPUS = """
The cat is a small domesticated carnivorous mammal. Cats are known for their
agility, independence, and hunting skills.
"""

DEFAULT_CORPUS = get_full_corpus()  # Use the large corpus by default


@dataclass
class GenerativeConfig:
    """Configuration du GenerativeBrain."""
    dim: int = 4096
    sdm_locations: int = 5000
    temperature: float = 0.7
    max_tokens: int = 50
    top_k: int = 5
    pretrained: bool = True
    use_bpe: bool = True             # Active le BPE tokenizer (meilleure couverture sous-mots)
    bpe_vocab_size: int = 2000       # Taille du vocab BPE
    max_pretrain_sentences: int = 150  # Nombre max de phrases à pré-entraîner (150 = ~60s)
    verbose: bool = False


class GenerativeBrain:
    """
    IA générative complète sans transformer.

    Usage:
        brain = GenerativeBrain()
        brain.teach("Python is a programming language")
        print(brain.generate("Tell me about Python"))
        print(brain.reason("If all cats are animals and Tom is a cat, what is Tom?"))
        print(brain.write_story("a lonely robot"))
    """

    def __init__(self, cfg: GenerativeConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or GenerativeConfig()
        self.rng = rng or np.random.default_rng(42)

        print("Initializing GenerativeBrain (AETHER core + fast predictor)...")
        t0 = time.time()

        # 1. Fast predictor (n-gram 1-7 + BPE) — for generation
        print("Building fast n-gram predictor (orders 1-7, BPE)...")
        self.predictor = GenerativePredictor(
            orders=(1, 2, 3, 4, 5, 6, 7),
            use_bpe=self.cfg.use_bpe,
            bpe_vocab_size=self.cfg.bpe_vocab_size,
        )

        if self.cfg.pretrained:
            stats = get_corpus_stats()
            print(f"Pre-training predictor on large corpus "
                  f"({stats['n_sentences']} sentences, {stats['n_words']} words, "
                  f"{stats['n_domains']} domains)...")
            self._pretrain_fast()

        # 2. AETHER (for reasoning — slow, lazy-init)
        self.aether = None
        if self.cfg.verbose:
            print("AETHER will be initialized lazily on first reasoning request.")

        pstats = self.predictor.stats()
        print(f"Ready in {time.time()-t0:.1f}s "
              f"(predictor vocab={pstats['vocab_size']}, "
              f"unigrams={pstats['total_unigrams']}, "
              f"bpe={'on' if pstats['use_bpe'] else 'off'})\n")

    def _ensure_aether(self):
        """Lazily initialize AETHER (only needed for reasoning)."""
        if self.aether is None:
            print("Initializing AETHER (first reasoning request)...")
            t0 = time.time()
            self.aether = AETHER()
            # Quick teach a few key facts so reasoning works
            key_facts = [
                "Paris is the capital of France",
                "Tokyo is the capital of Japan",
                "London is the capital of the United Kingdom",
                "Water is a chemical compound",
                "Einstein was a physicist",
            ]
            for fact in key_facts:
                self.aether.teach(fact, silent=True)
            print(f"  AETHER ready in {time.time()-t0:.1f}s")

    def _pretrain_fast(self) -> None:
        """Pre-train the fast predictor on the entire corpus (seconds, not minutes)."""
        corpus = get_full_corpus()
        stats = self.predictor.train_corpus(corpus)
        if self.cfg.verbose:
            pstats = self.predictor.stats()
            print(f"  Predictor trained: {pstats}")
            print(f"  N-gram counts: {pstats['ngram_counts']}")

    def train_on_text(self, text: str) -> dict:
        """Entraîne sur un texte personnalisé (one-shot)."""
        self.predictor.train_text(text)
        return {
            "sentences_learned": len(re.split(r"[.!?]\s+", text)),
            "predictor_stats": self.predictor.stats(),
        }

    def train_on_file(self, path: str) -> dict:
        """Entraîne sur le contenu d'un fichier texte."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.train_on_text(text)

    # ---------------------------------------------------------------- #
    # Génération libre — utilise le fast predictor (BPE + 7-gram)
    # ---------------------------------------------------------------- #
    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float | None = None,
                 top_k: int | None = None,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2) -> str:
        """
        Génère une continuation du prompt token-par-token.
        Utilise le GenerativePredictor (BPE + n-gram 1-7 + Kneser-Ney + nucleus sampling).
        """
        max_t = max_tokens or self.cfg.max_tokens
        temp = temperature if temperature is not None else self.cfg.temperature
        k = top_k or self.cfg.top_k
        return self.predictor.generate_text(
            prompt, max_tokens=max_t, temperature=temp, top_k=k,
            top_p=top_p, repetition_penalty=repetition_penalty,
            rng=self.rng
        )

    def generate_full(self, prompt: str, max_sentences: int = 3) -> str:
        """Génère une réponse multi-phrases fluide."""
        # Generate longer text by concatenating multiple sentences
        result = self.generate(prompt, max_tokens=40, temperature=0.8)
        # Try to add a second sentence
        if max_sentences > 1 and not result.endswith((".", "!", "?")):
            extra = self.generate(result[-30:], max_tokens=20, temperature=0.9)
            result = result + " " + extra
        return result

    # ---------------------------------------------------------------- #
    # Raisonnement
    # ---------------------------------------------------------------- #
    def reason(self, question: str, max_cycles: int = 8,
               use_aether: bool = False) -> str:
        """
        Raisonne sur une question.

        Args:
            question: la question
            max_cycles: nombre max de cycles (utilisé seulement par AETHER)
            use_aether: si True, utilise AETHER cognitive loop (lent, 15s init)
                        si False (défaut), utilise le predictor (rapide, <50ms)
        """
        if use_aether:
            self._ensure_aether()
            return self.aether.ask(question, explain=False)
        return self._reason_with_predictor(question)

    def _reason_with_predictor(self, question: str) -> str:
        """
        Raisonne via le predictor (rapide, pas d'AETHER).

        Stratégie:
          1. Détecte le type de question (what is, who is, where is, capital of, ...)
          2. Extrait le sujet
          3. Génère une réponse via le predictor
        """
        q_lower = question.lower().strip().rstrip("?")

        # Patterns de questions
        patterns = [
            (r"what is the capital of (\w+)", "capital"),
            (r"who (was|is) (\w+)", "person"),
            (r"what (is|are) (\w+)", "definition"),
            (r"where (is|are) (\w+)", "location"),
            (r"when (was|is|did) (\w+)", "time"),
            (r"how (do|does|did) (\w+)", "how"),
            (r"why (is|are|do|does) (\w+)", "why"),
        ]

        subject = None
        qtype = None
        for pat, t in patterns:
            m = re.search(pat, q_lower)
            if m:
                qtype = t
                if t == "capital":
                    subject = m.group(1)
                elif t in ("definition", "location"):
                    subject = m.group(2)
                elif t == "person":
                    subject = m.group(2)
                elif t == "time":
                    subject = m.group(2)
                break

        # Si on a un sujet, génère une réponse ciblée
        if subject:
            if qtype == "capital":
                # "The capital of X is"
                return self.generate(f"The capital of {subject} is",
                                      max_tokens=8, temperature=0.3,
                                      top_p=0.7, repetition_penalty=1.5).strip()
            elif qtype == "definition":
                # "X is"
                return self.generate(f"The {subject} is",
                                      max_tokens=20, temperature=0.5,
                                      top_p=0.8, repetition_penalty=1.3).strip()
            elif qtype == "person":
                # "X was a"
                return self.generate(f"{subject.capitalize()} was",
                                      max_tokens=15, temperature=0.5,
                                      top_p=0.8, repetition_penalty=1.3).strip()
            elif qtype == "location":
                return self.generate(f"{subject.capitalize()} is located",
                                      max_tokens=12, temperature=0.5,
                                      top_p=0.8, repetition_penalty=1.3).strip()
            else:
                return self.generate(f"The {subject} is",
                                      max_tokens=20, temperature=0.5,
                                      top_p=0.8, repetition_penalty=1.3).strip()

        # Sinon: génère depuis la question directement
        return self.generate(question, max_tokens=20, temperature=0.6,
                              top_p=0.85, repetition_penalty=1.3).strip()

    def reason_with_trace(self, question: str) -> dict:
        """Raisonne et retourne aussi la trace cognitive."""
        self._ensure_aether()
        answer = self.aether.ask(question, explain=True)
        trace = self.aether.explain_last() if hasattr(self.aether, "explain_last") else ""
        return {
            "question": question,
            "answer": answer,
            "trace": trace,
        }

    # ---------------------------------------------------------------- #
    # Apprentissage
    # ---------------------------------------------------------------- #
    def teach(self, fact: str) -> str:
        """Apprend un fait instantanément (predictor + AETHER si dispo)."""
        # Always train the predictor (fast)
        self.predictor.train_text(fact)
        # Also teach AETHER if initialized
        if self.aether is not None:
            self.aether.teach(fact, silent=True)
        return fact

    def learn(self, key: str, value: str) -> str:
        """Apprend une paire clé-valeur."""
        fact = f"{key} is {value}"
        return self.teach(fact)

    # ---------------------------------------------------------------- #
    # Écriture créative
    # ---------------------------------------------------------------- #
    def write_story(self, theme: str | None = None) -> dict:
        """Génère une courte histoire sur un thème."""
        # Use the fast predictor for story generation
        prompt = f"Once upon a time there was {theme}" if theme else "Once upon a time"
        text = self.generate(prompt, max_tokens=50, temperature=0.9)
        return {"text": text, "theme": theme, "method": "predictor"}

    def write_poem(self, topic: str) -> dict:
        """Génère un poème sur un sujet."""
        # Poems are hard for n-gram; use a template + generation
        lines = []
        for template in [f"Roses are red", f"Violets are blue", f"{topic} is true", f"And so are you"]:
            line = self.generate(template, max_tokens=5, temperature=0.8)
            lines.append(line.split(".")[-1].strip() if "." in line else line)
        return {"text": "\n".join(lines), "topic": topic, "method": "template+predictor"}

    def write_essay(self, topic: str) -> dict:
        """Génère un court essai sur un sujet."""
        prompt = f"An essay about {topic}"
        text = self.generate(prompt, max_tokens=60, temperature=0.7)
        return {"text": text, "topic": topic, "method": "predictor"}

    def write_description(self, subject: str) -> dict:
        """Génère une description d'un sujet."""
        prompt = f"The {subject} is"
        text = self.generate(prompt, max_tokens=30, temperature=0.7)
        return {"text": text, "subject": subject, "method": "predictor"}

    # ---------------------------------------------------------------- #
    # Analyse
    # ---------------------------------------------------------------- #
    def summarize(self, text: str) -> str:
        """Résume un texte — utilise le predictor (rapide)."""
        self.teach(text)
        # Generate a summary by completing "The summary is"
        return self.generate(f"The main point is", max_tokens=15, temperature=0.6)

    def explain(self, topic: str) -> str:
        """Explique un sujet — utilise le predictor (rapide)."""
        return self.generate(f"The {topic}", max_tokens=25, temperature=0.6)

    def analyze(self, text: str) -> dict:
        summary = self.summarize(text)
        return {
            "input_length": len(text.split()),
            "summary": summary,
            "key_concepts": list(set(re.findall(r"\b[a-z]{4,}\b", text.lower())))[:5],
        }

    # ---------------------------------------------------------------- #
    # Conversation
    # ---------------------------------------------------------------- #
    def chat(self, user_input: str) -> str:
        """Interface de conversation — route selon le type d'input."""
        user_lower = user_input.lower()

        # Apprentissage
        if any(p in user_lower for p in ["teach me", "remember that", "learn that"]):
            return self.teach(user_input)

        # Écriture créative
        if any(p in user_lower for p in ["write a story", "tell me a story"]):
            theme = re.sub(r"(write a story|tell me a story)\s*(about|on)?\s*", "", user_input, flags=re.IGNORECASE).strip()
            r = self.write_story(theme or None)
            return r.get("text", str(r))
        if any(p in user_lower for p in ["write a poem", "compose a poem"]):
            topic = re.sub(r"write a poem\s*(about|on)?\s*", "", user_lower).strip()
            r = self.write_poem(topic)
            return r.get("text", str(r))
        if any(p in user_lower for p in ["write an essay"]):
            topic = re.sub(r"write an essay\s*(about|on)?\s*", "", user_lower).strip()
            r = self.write_essay(topic)
            return r.get("text", str(r))

        # Analyse
        if any(p in user_lower for p in ["summarize", "summary of"]):
            text = re.sub(r"summarize\s*:?\s*", "", user_input, flags=re.IGNORECASE)
            return self.summarize(text)
        if "explain" in user_lower:
            topic = re.sub(r"explain\s*:?\s*", "", user_input, flags=re.IGNORECASE)
            return self.explain(topic)

        # Génération libre
        if any(p in user_lower for p in ["generate", "continue", "complete"]):
            prompt = re.sub(r"(generate|continue|complete)\s*:?\s*", "", user_input, flags=re.IGNORECASE)
            return self.generate_full(prompt)

        # Raisonnement
        if any(p in user_lower for p in ["why", "how", "what if", "reason about", "think about"]):
            return self.reason(user_input)

        # Outil
        if "calc" in user_lower or (any(c in user_input for c in "+-*/") and any(c.isdigit() for c in user_input)):
            self._ensure_aether()
            return self.aether.ask(user_input)

        # Par défaut: si "tell me about X" → génération depuis le predictor
        if any(p in user_lower for p in ["tell me about", "what is", "what are", "who is", "who are"]):
            # Extract topic and generate
            topic = re.sub(r"(tell me about|what is|what are|who is|who are)\s*:?\s*", "", user_input, flags=re.IGNORECASE).strip().rstrip("?")
            text = self.generate(f"The {topic}", max_tokens=25, temperature=0.6)
            return text

        # Sinon: génération simple
        return self.generate(user_input, max_tokens=25, temperature=0.7)

    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        s = {}
        if self.aether is not None and hasattr(self.aether, "stats"):
            s = self.aether.stats()
        s["generative"] = True
        s["pretrained"] = self.cfg.pretrained
        s["predictor"] = self.predictor.stats()
        s["aether_initialized"] = self.aether is not None
        return s

    def print_stats(self) -> None:
        s = self.stats()
        print("\n=== GenerativeBrain Stats ===")
        for k, v in s.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        print("=============================\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  GenerativeBrain — Test")
    print("=" * 60)

    brain = GenerativeBrain(GenerativeConfig(verbose=True))

    print("\n--- Test 1: Reasoning ---")
    print(brain.reason("What is the capital of France?"))

    print("\n--- Test 2: Generation ---")
    print(brain.generate("The cat is", max_tokens=15, temperature=0.8))

    print("\n--- Test 3: Teach + Recall ---")
    brain.teach("Tokyo is the capital of Japan")
    print(brain.reason("What is the capital of Japan?"))

    print("\n--- Test 4: Chat ---")
    print(brain.chat("Tell me about Python"))
