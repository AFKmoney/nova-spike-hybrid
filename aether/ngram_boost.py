"""
ngram_boost.py — N-gram boosted retrieval for coherent generation.

PROBLEM
-------
Single-token prediction has no long-term coherence. Each token is predicted
independently, so errors accumulate.

SOLUTION
--------
Predict MULTIPLE token lengths in parallel and let them vote:

  - 1-token prediction (current behavior)
  - 2-token (bigram) prediction
  - 3-token (trigram) prediction

Each returns its top candidates with confidence. We pick the prediction
with the highest confidence. If the bigram is more confident than the
1-token, we commit 2 tokens at once — coherence guaranteed.

This is like a human thinking in phrases, not letters.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class NGramPrediction:
    """A prediction from one of the n-gram models."""
    tokens: List[str]
    confidence: float
    n: int  # 1, 2, or 3
    source: str  # "unigram", "bigram", "trigram"


@dataclass
class BoostedPrediction:
    """The final voted prediction."""
    tokens: List[str]
    confidence: float
    winning_n: int
    candidates: List[NGramPrediction] = field(default_factory=list)


class NGramBoostedPredictor:
    """Predict next 1, 2, or 3 tokens in parallel and vote."""

    def __init__(self, agent):
        self.agent = agent
        # Storage for n-gram statistics
        self.unigram_counts: Dict[str, int] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.trigram_counts: Dict[Tuple[str, str, str], int] = {}
        self.total_unigrams: int = 0

    # ------------------------------------------------------------------ #
    # Training: count n-grams from sequences
    # ------------------------------------------------------------------ #
    def train_sequence(self, tokens: List[str]) -> None:
        """Update n-gram counts from a token sequence."""
        for i, tok in enumerate(tokens):
            self.unigram_counts[tok] = self.unigram_counts.get(tok, 0) + 1
            self.total_unigrams += 1
            if i < len(tokens) - 1:
                bigram = (tok, tokens[i + 1])
                self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
            if i < len(tokens) - 2:
                trigram = (tok, tokens[i + 1], tokens[i + 2])
                self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1

    def train_text(self, text: str) -> None:
        """Train on a text by tokenizing and counting n-grams."""
        from .encoder import tokenize
        tokens = tokenize(text)
        # Add BOS/EOS markers
        tokens = ["<s>"] + tokens + ["</s>"]
        self.train_sequence(tokens)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_unigram(self, context: List[str]) -> Optional[NGramPrediction]:
        """Predict the next single token."""
        # Use the last token of context for bigram-based unigram prediction
        if not context:
            # No context: most frequent unigram
            if not self.unigram_counts:
                return None
            best = max(self.unigram_counts.items(), key=lambda x: x[1])
            conf = best[1] / self.total_unigrams
            return NGramPrediction([best[0]], conf, 1, "unigram")
        last = context[-1]
        # Find all bigrams starting with `last`
        candidates = [(b[1], c) for b, c in self.bigram_counts.items() if b[0] == last]
        if not candidates:
            return None
        total = sum(c for _, c in candidates)
        best = max(candidates, key=lambda x: x[1])
        conf = best[1] / total
        # Skip special tokens
        if best[0] in ("</s>", "<s>", "<unk>", "<pad>"):
            return None
        return NGramPrediction([best[0]], conf, 1, "unigram")

    def predict_bigram(self, context: List[str]) -> Optional[NGramPrediction]:
        """Predict the next 2 tokens as a bigram."""
        if len(context) < 1:
            return None
        last = context[-1]
        # Find all trigrams starting with `last`
        candidates = [(b[1:], c) for b, c in self.trigram_counts.items() if b[0] == last]
        if not candidates:
            return None
        total = sum(c for _, c in candidates)
        best = max(candidates, key=lambda x: x[1])
        conf = best[1] / total
        # Skip if contains special tokens
        if any(t in ("</s>", "<s>", "<unk>", "<pad>") for t in best[0]):
            return None
        return NGramPrediction(list(best[0]), conf, 2, "trigram")

    def predict_trigram(self, context: List[str]) -> Optional[NGramPrediction]:
        """Predict the next 3 tokens as a trigram."""
        if len(context) < 2:
            return None
        last_two = tuple(context[-2:])
        # Find all 4-grams starting with `last_two`
        # We don't store 4-grams, so approximate: find trigrams starting with last_two
        candidates = [(b[2:], c) for b, c in self.trigram_counts.items()
                      if b[:2] == last_two]
        if not candidates or len(candidates[0][0]) < 1:
            return None
        # For 3-token prediction, we need the trigram + one more
        # Approximate: take the trigram's last token + most likely next
        total = sum(c for _, c in candidates)
        best = max(candidates, key=lambda x: x[1])
        if not best[0]:
            return None
        conf = best[1] / total
        # best[0] is a 1-tuple (the third token). Extend by predicting 4th.
        third = best[0][0] if isinstance(best[0], tuple) else best[0]
        if isinstance(third, str):
            # Predict 4th from bigram (third, ?)
            next_candidates = [(b[1], c) for b, c in self.bigram_counts.items() if b[0] == third]
            if next_candidates:
                next_best = max(next_candidates, key=lambda x: x[1])
                return NGramPrediction([third, next_best[0]], conf * 0.7, 2, "trigram")
            return NGramPrediction([third], conf, 1, "trigram")
        return None

    # ------------------------------------------------------------------ #
    # Boosted prediction (the vote)
    # ------------------------------------------------------------------ #
    def predict_next(self, context: List[str]) -> Optional[BoostedPrediction]:
        """Predict the next token(s) by voting across n-gram models.

        Returns the prediction with the highest confidence.
        If a bigram is more confident than the unigram, commit 2 tokens.
        """
        candidates: List[NGramPrediction] = []
        uni = self.predict_unigram(context)
        if uni: candidates.append(uni)
        bi = self.predict_bigram(context)
        if bi: candidates.append(bi)
        tri = self.predict_trigram(context)
        if tri: candidates.append(tri)

        if not candidates:
            return None

        # Vote: pick the prediction with highest confidence
        # Weighted: longer n-grams get a bonus for coherence
        weighted = []
        for pred in candidates:
            # Bonus for longer n-grams (coherence)
            bonus = 1.0 + 0.1 * (pred.n - 1)
            weighted.append((pred, pred.confidence * bonus))

        best_pred, best_score = max(weighted, key=lambda x: x[1])

        return BoostedPrediction(
            tokens=best_pred.tokens,
            confidence=best_score,
            winning_n=best_pred.n,
            candidates=candidates,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: List[str], max_tokens: int = 20,
                 stop_tokens: Optional[set] = None) -> List[str]:
        """Generate a continuation of the prompt using n-gram boosting."""
        if stop_tokens is None:
            stop_tokens = {"</s>", "<s>", "<unk>", "<pad>"}
        context = list(prompt)
        generated = []
        for _ in range(max_tokens):
            pred = self.predict_next(context)
            if not pred:
                break
            for tok in pred.tokens:
                if tok in stop_tokens:
                    return generated
                generated.append(tok)
                context.append(tok)
                if len(generated) >= max_tokens:
                    return generated
        return generated

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        return {
            "n_unigrams": len(self.unigram_counts),
            "n_bigrams": len(self.bigram_counts),
            "n_trigrams": len(self.trigram_counts),
            "total_unigrams": self.total_unigrams,
        }
