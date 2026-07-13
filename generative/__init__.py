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

        print("Initializing GenerativeBrain (AETHER core)...")
        t0 = time.time()
        self.aether = AETHER()

        # BPE tokenizer (optionnel)
        self.bpe = None
        if self.cfg.use_bpe:
            print("Training BPE tokenizer on corpus...")
            try:
                from aether.bpe import BPETokenizer
                self.bpe = BPETokenizer(vocab_size=self.cfg.bpe_vocab_size)
                self.bpe.train(DEFAULT_CORPUS, verbose=False)
                if self.cfg.verbose:
                    print(f"  BPE trained: {len(self.bpe.vocab)} tokens")
            except Exception as e:
                print(f"  BPE training failed ({e}), falling back to word tokenizer")
                self.bpe = None

        if self.cfg.pretrained:
            stats = get_corpus_stats()
            print(f"Pre-training on large corpus "
                  f"({stats['n_sentences']} sentences, {stats['n_words']} words, "
                  f"{stats['n_domains']} domains)...")
            self._pretrain()
        print(f"Ready in {time.time()-t0:.1f}s "
              f"(vocab={len(self.aether.assoc.vocab)}, "
              f"episodes={len(self.aether.assoc.episodes)}, "
              f"bpe={'on' if self.bpe else 'off'})\n")

    # ---------------------------------------------------------------- #
    def _pretrain(self) -> None:
        """Pré-entraîne sur le corpus par défaut."""
        sentences = get_corpus_sentences()
        max_pretrain = self.cfg.max_pretrain_sentences
        selected = sentences[:max_pretrain]
        if self.cfg.verbose:
            print(f"  Pre-training on {len(selected)}/{len(sentences)} sentences")
        n = 0
        for sent in selected:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            self.aether.teach(sent, silent=True)
            try:
                self.aether.ngram_predictor.train_text(sent)
            except Exception:
                pass
            n += 1
        if self.cfg.verbose:
            print(f"  Learned {n} sentences from corpus")
            print(f"  N-gram predictor: {self.aether.ngram_predictor.total_unigrams} unigrams, "
                  f"{len(self.aether.ngram_predictor.bigram_counts)} bigrams, "
                  f"{len(self.aether.ngram_predictor.trigram_counts)} trigrams")

    def train_on_text(self, text: str) -> dict:
        """Entraîne sur un texte personnalisé (one-shot)."""
        sentences = re.split(r"[.!?]\s+", text)
        n = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            self.aether.teach(sent, silent=True)
            n += 1
        return {"sentences_learned": n, "total_episodes": len(self.aether.assoc.episodes)}

    def train_on_file(self, path: str) -> dict:
        """Entraîne sur le contenu d'un fichier texte."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.train_on_text(text)

    # ---------------------------------------------------------------- #
    # Génération libre
    # ---------------------------------------------------------------- #
    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float | None = None,
                 top_k: int | None = None) -> str:
        """Génère une continuation du prompt token-par-token."""
        max_t = max_tokens or self.cfg.max_tokens
        # Utilise generate_ngram d'AETHER qui fait la prédiction token-par-token
        return self.aether.generate_ngram(prompt, max_tokens=max_t)

    def generate_full(self, prompt: str, max_sentences: int = 3) -> str:
        """Génère une réponse multi-phrases fluide."""
        try:
            return self.aether.generate_fluent(prompt, max_sentences=max_sentences)
        except Exception:
            return self.generate(prompt, max_tokens=30)

    # ---------------------------------------------------------------- #
    # Raisonnement
    # ---------------------------------------------------------------- #
    def reason(self, question: str, max_cycles: int = 8) -> str:
        """Raisonne via le cognitive loop d'AETHER."""
        return self.aether.ask(question, explain=False)

    def reason_with_trace(self, question: str) -> dict:
        """Raisonne et retourne aussi la trace cognitive."""
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
        """Apprend un fait instantanément."""
        return self.aether.teach(fact, silent=True)

    def learn(self, key: str, value: str) -> str:
        """Apprend une paire clé-valeur."""
        fact = f"{key} is {value}"
        return self.aether.teach(fact, silent=True)

    # ---------------------------------------------------------------- #
    # Écriture créative
    # ---------------------------------------------------------------- #
    def write_story(self, theme: str | None = None) -> dict:
        try:
            result = self.aether.write_story(theme)
            if isinstance(result, dict):
                return result
            return {"text": str(result), "theme": theme}
        except Exception as e:
            return {"text": f"Story error: {e}", "theme": theme}

    def write_poem(self, topic: str) -> dict:
        try:
            result = self.aether.write_poem(topic)
            if isinstance(result, dict):
                return result
            return {"text": str(result), "topic": topic}
        except Exception as e:
            return {"text": f"Poem error: {e}", "topic": topic}

    def write_essay(self, topic: str) -> dict:
        try:
            result = self.aether.write_essay(topic)
            if isinstance(result, dict):
                return result
            return {"text": str(result), "topic": topic}
        except Exception as e:
            return {"text": f"Essay error: {e}", "topic": topic}

    def write_description(self, subject: str) -> dict:
        try:
            result = self.aether.write_description(subject)
            if isinstance(result, dict):
                return result
            return {"text": str(result), "subject": subject}
        except Exception as e:
            return {"text": f"Description error: {e}", "subject": subject}

    # ---------------------------------------------------------------- #
    # Analyse
    # ---------------------------------------------------------------- #
    def summarize(self, text: str) -> str:
        self.aether.teach(text, silent=True)
        return self.aether.ask(f"Summarize: {text[:100]}")

    def explain(self, topic: str) -> str:
        return self.aether.ask(f"Explain {topic}")

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
            return self.aether.ask(user_input)

        # Par défaut
        return self.aether.ask(user_input)

    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        s = self.aether.stats() if hasattr(self.aether, "stats") else {}
        s["generative"] = True
        s["pretrained"] = self.cfg.pretrained
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
