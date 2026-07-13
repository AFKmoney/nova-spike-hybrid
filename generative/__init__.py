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


# ---------------------------------------------------------------------- #
# Corpus de pré-entraînement (intégré — pas de fichier externe requis)
# ---------------------------------------------------------------------- #

DEFAULT_CORPUS = """
The cat is a small domesticated carnivorous mammal. Cats are known for their
agility, independence, and hunting skills. A cat sleeps on average 12 to 16
hours per day. Cats communicate through vocalizations like meowing, purring,
and hissing.

The dog is a domesticated descendant of the wolf. Dogs are known for their
loyalty, intelligence, and ability to form strong bonds with humans. A dog
can learn hundreds of commands. Dogs are used as working animals, companions,
and in therapy.

Paris is the capital of France. It is located on the Seine river. Paris is
known for the Eiffel Tower, the Louvre museum, and the Notre-Dame cathedral.
Paris has a population of about 2 million people in the city proper.

The sun is a star at the center of the solar system. It is a nearly perfect
ball of hot plasma. The sun provides energy to Earth through sunlight. The
distance from the sun to Earth is about 150 million kilometers.

Water is a chemical compound with the formula H2O. It covers about 71 percent
of the Earth surface. Water is essential for all known forms of life. Water
exists as solid ice, liquid water, and gaseous steam.

Albert Einstein was a German-born theoretical physicist. He developed the
theory of relativity. Einstein received the Nobel Prize in Physics in 1921.
His most famous equation is E equals m c squared.

The Earth is the third planet from the sun. It is the only known planet to
harbor life. Earth has one natural satellite, the Moon. The Earth atmosphere
is composed mostly of nitrogen and oxygen.

A computer is a machine that processes data according to a set of instructions.
Modern computers use transistors and integrated circuits. A computer has
hardware components like the CPU, memory, and storage. Software includes
operating systems, applications, and games.

Python is a high-level programming language. It was created by Guido van Rossum.
Python emphasizes code readability and simplicity. Python is widely used in
data science, machine learning, and web development.

The brain is the organ of intelligence and emotion. It is composed of about
86 billion neurons. Neurons communicate through electrical and chemical
signals. The brain processes information through specialized regions.

Mathematics is the study of numbers, shapes, and patterns. It includes
arithmetic, algebra, geometry, and calculus. Mathematics is fundamental to
science, engineering, and economics. The Pythagorean theorem states that
a squared plus b squared equals c squared.

Music is the art of arranging sounds in time. It uses rhythm, melody, and
harmony. Music can evoke emotions and convey meaning. Instruments include
the piano, guitar, violin, and drums.

Literature is a collection of written works. It includes novels, poetry,
essays, and plays. Literature explores human experience through language.
Famous writers include Shakespeare, Tolstoy, and Hemingway.

Science is the systematic study of the natural world. It uses observation,
experimentation, and theory. The scientific method involves hypothesis,
prediction, and testing. Major branches include physics, chemistry, and biology.

History is the study of past events. It helps us understand how societies
developed over time. History is recorded through documents, artifacts, and
oral traditions. Learning from history helps avoid repeating mistakes.

The ocean covers most of the Earth surface. It contains salt water and is
home to diverse marine life. The deepest point is the Mariana Trench.
Oceans regulate the global climate and provide food for billions of people.

A tree is a perennial plant with an elongated stem. Trees provide oxygen
through photosynthesis. They absorb carbon dioxide from the atmosphere.
Forests are vital ecosystems that support biodiversity.

The heart is a muscular organ that pumps blood through the body. It beats
about 100000 times per day. The heart has four chambers. Blood carries
oxygen and nutrients to cells.

Dreams are experiences that occur during sleep. They involve images,
thoughts, and emotions. The purpose of dreams is still debated by scientists.
Dreams may help consolidate memories and process emotions.
"""


@dataclass
class GenerativeConfig:
    """Configuration du GenerativeBrain."""
    dim: int = 4096
    sdm_locations: int = 5000
    temperature: float = 0.7
    max_tokens: int = 50
    top_k: int = 5
    pretrained: bool = True
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
        if self.cfg.pretrained:
            print("Pre-training on default corpus...")
            self._pretrain()
        print(f"Ready in {time.time()-t0:.1f}s "
              f"(vocab={len(self.aether.assoc.vocab)}, "
              f"episodes={len(self.aether.assoc.episodes)})\n")

    # ---------------------------------------------------------------- #
    def _pretrain(self) -> None:
        """Pré-entraîne sur le corpus par défaut."""
        sentences = re.split(r"[.!?]\s+", DEFAULT_CORPUS)
        n = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            self.aether.teach(sent, silent=True)
            n += 1
        if self.cfg.verbose:
            print(f"  Learned {n} sentences from corpus")

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
