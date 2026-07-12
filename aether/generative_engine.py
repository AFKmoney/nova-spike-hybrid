"""
generative_engine.py — Fluent multi-sentence text generation.

PROBLEM
-------
AETHER's current generation is template-bound: "X is the capital of Y."
GPT-4 generates fluent paragraphs. We need a real generative engine.

SOLUTION
--------
GenerativeEngine combines ALL v7 boosters into a single pipeline:

  1. BPE tokenize the prompt
  2. Multi-scale encode (char + word + phrase)
  3. HV-attend to retrieve relevant memories
  4. N-gram boosted prediction (1/2/3-token voting)
  5. Template-guided structuring (when applicable)
  6. Iterative refinement (draft → correct)
  7. Sentence-level coherence (bundle sentence vectors)

The result: fluent multi-sentence generation, not just template filling.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of fluent generation."""
    text: str
    sentences: List[str]
    n_passes: int
    confidence: float
    method: str  # "template", "ngram", "retrieval", "hybrid"


class GenerativeEngine:
    """Fluent multi-sentence text generation combining all boosters."""

    def __init__(self, agent):
        self.agent = agent

    def generate(self, prompt: str, max_sentences: int = 5,
                 max_tokens_per_sentence: int = 15) -> GenerationResult:
        """Generate a fluent multi-sentence response to a prompt.

        Strategy:
          1. Try template-based generation (if KB has the answer)
          2. If template fails, use n-gram boosted generation
          3. If n-gram fails, use retrieval-based generation
          4. Refine the result iteratively
        """
        # 1. Try template-based (most reliable for factual questions)
        result = self._try_template_generation(prompt)
        if result and result.confidence > 0.5:
            result = self._refine_result(prompt, result)
            return result

        # 2. Try retrieval-based (find similar memories and adapt)
        result = self._try_retrieval_generation(prompt, max_sentences)
        if result and result.confidence > 0.4:
            result = self._refine_result(prompt, result)
            return result

        # 3. Try n-gram boosted generation
        result = self._try_ngram_generation(prompt, max_sentences, max_tokens_per_sentence)
        if result:
            result = self._refine_result(prompt, result)
            return result

        # 4. Fallback
        return GenerationResult(
            text="I don't have enough information to generate a response.",
            sentences=[],
            n_passes=0,
            confidence=0.0,
            method="fallback",
        )

    # ------------------------------------------------------------------ #
    # Method 1: Template-based generation
    # ------------------------------------------------------------------ #
    def _try_template_generation(self, prompt: str) -> Optional[GenerationResult]:
        """Try to answer using KB + templates."""
        # Parse the question
        from .generator import analyze_question, parse_triple
        analysis = analyze_question(prompt)

        # Try to find a KB match
        if analysis.qtype == "capital_of":
            country = analysis.slots.get("country", "").strip()
            result = self.agent.inference.lookup(country, "capital_of")
            if result:
                capital, conf = result
                text = self.agent.generate_templated(capital, "capital_of", country)
                return GenerationResult(text=text, sentences=[text], n_passes=1,
                                       confidence=conf, method="template")

        elif analysis.qtype == "located_in":
            subject = analysis.slots.get("subject", "").strip()
            if subject.endswith(" located"):
                subject = subject[:-len(" located")].strip()
            result = self.agent.inference.lookup(subject, "located_in")
            if result:
                location, conf = result
                text = self.agent.generate_templated(subject, "located_in", location)
                return GenerationResult(text=text, sentences=[text], n_passes=1,
                                       confidence=conf, method="template")

        elif analysis.qtype == "definition":
            subject = analysis.slots.get("subject", "").strip()
            result = self.agent.inference.lookup(subject, "is_a")
            if result:
                definition, conf = result
                text = self.agent.generate_templated(subject, "is_a", definition)
                return GenerationResult(text=text, sentences=[text], n_passes=1,
                                       confidence=conf, method="template")

        elif analysis.qtype in ("identity", "capabilities", "self_explain",
                                 "greeting", "farewell", "thanks"):
            # Use the standard ask() for these
            text = self.agent.ask(prompt)
            return GenerationResult(text=text, sentences=[text], n_passes=1,
                                   confidence=0.8, method="template")

        return None

    # ------------------------------------------------------------------ #
    # Method 2: Retrieval-based generation
    # ------------------------------------------------------------------ #
    def _try_retrieval_generation(self, prompt: str, max_sentences: int) -> Optional[GenerationResult]:
        """Generate by retrieving and combining relevant memories."""
        # HV-attend to find relevant memories
        attention_result = self.agent.hv_attention.attend_to_text(prompt)
        retrieved = attention_result.retrieved

        if not retrieved:
            return None

        # Build a response from the top retrieved memories
        sentences = []
        for text, sim in retrieved[:max_sentences]:
            if sim > 0.15 and len(text) > 10:
                # Clean up the memory text
                clean = text.strip().rstrip(".")
                if clean and clean not in sentences:
                    sentences.append(clean + ".")

        if not sentences:
            return None

        # Combine sentences into a paragraph
        text = " ".join(sentences)
        # Compute confidence from retrieval similarities
        avg_sim = sum(s for _, s in retrieved[:len(sentences)]) / max(len(sentences), 1)
        return GenerationResult(
            text=text, sentences=sentences, n_passes=1,
            confidence=avg_sim, method="retrieval",
        )

    # ------------------------------------------------------------------ #
    # Method 3: N-gram boosted generation
    # ------------------------------------------------------------------ #
    def _try_ngram_generation(self, prompt: str, max_sentences: int,
                              max_tokens_per_sentence: int) -> Optional[GenerationResult]:
        """Generate using n-gram boosted prediction."""
        # Ensure n-gram predictor is trained
        if self.agent.ngram_predictor.total_unigrams == 0:
            for ep in self.agent.assoc.episodes:
                self.agent.ngram_predictor.train_text(ep.payload)

        if self.agent.ngram_predictor.total_unigrams == 0:
            return None

        from .encoder import tokenize
        tokens = tokenize(prompt)
        sentences = []
        current_sentence = []

        for _ in range(max_sentences):
            # Generate tokens for one sentence
            generated = self.agent.ngram_predictor.generate(tokens, max_tokens=max_tokens_per_sentence)
            if not generated:
                break
            current_sentence = generated
            sentence_text = " ".join(current_sentence)
            # Capitalize first letter
            if sentence_text:
                sentence_text = sentence_text[0].upper() + sentence_text[1:]
            if not sentence_text.endswith("."):
                sentence_text += "."
            sentences.append(sentence_text)
            # Add to context for next sentence
            tokens = tokens + current_sentence

        if not sentences:
            return None

        text = " ".join(sentences)
        return GenerationResult(
            text=text, sentences=sentences, n_passes=1,
            confidence=0.3, method="ngram",
        )

    # ------------------------------------------------------------------ #
    # Refinement
    # ------------------------------------------------------------------ #
    def _refine_result(self, prompt: str, result: GenerationResult) -> GenerationResult:
        """Apply iterative refinement to the generated result."""
        refined = self.agent.refiner.refine(prompt, result.text)
        result.text = refined.final_text
        result.n_passes = refined.n_passes
        result.confidence = refined.final_confidence
        # Re-split into sentences
        result.sentences = re.split(r'(?<=[.!?])\s+', result.text)
        return result

    # ------------------------------------------------------------------ #
    # Specialized generation modes
    # ------------------------------------------------------------------ #
    def generate_explanation(self, topic: str) -> str:
        """Generate a multi-sentence explanation of a topic."""
        # Use the explain tool as a base, then expand
        base = self.agent.call_tool("explain", topic, )
        if "don't know" in base.lower():
            return self.generate(f"Tell me about {topic}").text
        # Try to generate additional sentences from related memories
        attention = self.agent.hv_attention.attend_to_text(topic)
        extra_sentences = []
        for text, sim in attention.retrieved[:3]:
            if sim > 0.2 and text != base:
                extra_sentences.append(text.strip().rstrip(".") + ".")
        if extra_sentences:
            return base + " " + " ".join(extra_sentences[:2])
        return base

    def generate_comparison(self, a: str, b: str) -> str:
        """Generate a comparison between two entities."""
        base = self.agent.call_tool("compare", f"{a} and {b}")
        return base

    def generate_summary(self, text: str) -> str:
        """Generate a summary of a text passage."""
        # Extract key facts
        from .learn_from_text import extract_facts
        facts = extract_facts(text)
        if not facts:
            return text[:200] + "..."
        # Build summary from facts
        sentences = []
        for fact in facts[:5]:
            if fact.predicate == "capital_of":
                sentences.append(f"{fact.subject} is the capital of {fact.object}.")
            elif fact.predicate == "located_in":
                sentences.append(f"{fact.subject} is located in {fact.object}.")
            elif fact.predicate == "is_a":
                sentences.append(f"{fact.subject} is {fact.object}.")
            else:
                sentences.append(f"{fact.subject} {fact.predicate.replace('_',' ')} {fact.object}.")
        return " ".join(sentences)
