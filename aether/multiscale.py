"""
multiscale.py — Multi-scale encoding: char + word + phrase in parallel.

PROBLEM
-------
Single-scale encoding loses information. Char-level captures detail but
misses semantics. Word-level captures semantics but loses detail. Phrase-
level captures context but loses specifics.

SOLUTION
--------
Encode the input at THREE scales in parallel:

  - CHARACTER scale: char n-grams (detail, morphology)
  - WORD scale: word tokens (semantics)
  - PHRASE scale: phrase-level bundling (global context)

The three HD vectors are BUNDLED into a single multi-scale HD vector.
Retrieval can match at any scale — a query that's similar at the word
level will match even if the chars differ.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import logging

from .hd import HDVector, DIM, bundle, ngram_encode
from .semantic import SemanticEncoder, char_ngrams

log = logging.getLogger(__name__)


@dataclass
class MultiScaleVector:
    """A multi-scale HD encoding."""
    char_vec: HDVector    # character-level detail
    word_vec: HDVector    # word-level semantics
    phrase_vec: HDVector  # phrase-level context
    combined: HDVector    # bundled combination


class MultiScaleEncoder:
    """Encode text at multiple scales in parallel."""

    def __init__(self, agent):
        self.agent = agent
        self.dim = agent.dim
        self.semantic_encoder = SemanticEncoder(dim=self.dim)

    def encode(self, text: str) -> MultiScaleVector:
        """Encode text at char, word, and phrase scales."""
        # 1. CHARACTER scale: char n-grams
        char_vec = self._encode_char_scale(text)

        # 2. WORD scale: bundle of word HD vectors
        word_vec = self._encode_word_scale(text)

        # 3. PHRASE scale: encode the whole text as a phrase
        phrase_vec = self._encode_phrase_scale(text)

        # Combine: bundle all three
        combined = bundle([char_vec, word_vec, phrase_vec])

        return MultiScaleVector(
            char_vec=char_vec,
            word_vec=word_vec,
            phrase_vec=phrase_vec,
            combined=combined,
        )

    def _encode_char_scale(self, text: str) -> HDVector:
        """Encode at character level (morphology, detail)."""
        # Use char n-grams (2,3,4)
        grams = char_ngrams(text.lower(), ns=(2, 3, 4))
        if not grams:
            return HDVector.zero(self.dim)
        vecs = [self.semantic_encoder._get_ngram_vec(g) for g in grams[:50]]  # limit
        return bundle(vecs)

    def _encode_word_scale(self, text: str) -> HDVector:
        """Encode at word level (semantics)."""
        from .encoder import tokenize
        tokens = tokenize(text)
        if not tokens:
            return HDVector.zero(self.dim)
        vecs = [self.agent.assoc.get_symbol(t) for t in tokens]
        return bundle(vecs)

    def _encode_phrase_scale(self, text: str) -> HDVector:
        """Encode at phrase level (global context)."""
        # Use the agent's text encoder (n-gram superposition)
        return self.agent.encoder.encode_text(text)

    # ------------------------------------------------------------------ #
    # Similarity at different scales
    # ------------------------------------------------------------------ #
    def similarity_at_scale(self, text1: str, text2: str, scale: str = "combined") -> float:
        """Compute similarity at a specific scale.

        Args:
            scale: "char", "word", "phrase", or "combined"
        """
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        if scale == "char":
            return v1.char_vec.similarity(v2.char_vec)
        elif scale == "word":
            return v1.word_vec.similarity(v2.word_vec)
        elif scale == "phrase":
            return v1.phrase_vec.similarity(v2.phrase_vec)
        else:
            return v1.combined.similarity(v2.combined)

    def retrieve_at_scale(self, query: str, candidates: List[str],
                         scale: str = "combined", top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the most similar candidates at a specific scale."""
        q_vec = self.encode(query)
        results = []
        for cand in candidates:
            c_vec = self.encode(cand)
            if scale == "char":
                sim = q_vec.char_vec.similarity(c_vec.char_vec)
            elif scale == "word":
                sim = q_vec.word_vec.similarity(c_vec.word_vec)
            elif scale == "phrase":
                sim = q_vec.phrase_vec.similarity(c_vec.phrase_vec)
            else:
                sim = q_vec.combined.similarity(c_vec.combined)
            results.append((cand, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
