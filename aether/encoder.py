"""
encoder.py — Text <-> HD vector encoder.

Two layers of encoding:

  1. Token level: each token (word) gets a stable HD vector from a vocab
     table (deterministic seed). New tokens are added on the fly — this
     is "instant vocabulary learning", no training.

  2. Sequence level: encode a sequence of tokens using an n-gram superposition
     (bundle of position-bound token windows). This gives a fixed-size HD
     vector for any input length, capturing local word order.

Decoding: HD vector -> top-k most similar known tokens, or top-k most similar
stored episodes. We use this both for "speaking" (retrieving a stored reply)
and for "next-token prediction" via a small HD language model.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional

import numpy as np

from .hd import HDVector, DIM, ngram_encode, bundle, _sign
from .memory import AssociativeMemory


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+|[^\sA-Za-zÀ-ÿ0-9_]")


def tokenize(text: str) -> List[str]:
    """Light tokenizer: words (incl. accented) + single-char punctuation."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class TextEncoder:
    """Encode text into HD vectors and decode back, using a vocab table."""

    def __init__(self, dim: int = DIM, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
        self.assoc = AssociativeMemory(dim=dim)
        # Pre-seed common structural tokens
        for tok in ["<s>", "</s>", "<unk>", "<pad>", "?", ".", ",", "!"]:
            self.assoc.get_symbol(tok)

    # ----- tokenization + vocab -----
    def encode_tokens(self, text: str) -> List[HDVector]:
        toks = tokenize(text)
        return [self.assoc.get_symbol(t) for t in toks]

    def encode_text(self, text: str) -> HDVector:
        """Encode full text as an n-gram superposition HD vector."""
        toks = tokenize(text)
        if not toks:
            return HDVector.zero(self.dim)
        # Add start/end markers
        toks = ["<s>"] + toks + ["</s>"]
        vecs = [self.assoc.get_symbol(t) for t in toks]
        if len(vecs) <= self.ngram:
            return bundle(vecs)
        return ngram_encode(vecs, n=self.ngram)

    # ----- decoding -----
    def decode_to_tokens(self, vec: HDVector, top_k: int = 1) -> List[Tuple[str, float]]:
        """Find the top-k tokens most similar to `vec`."""
        if not self.assoc.vocab:
            return []
        sims = [(name, vec.similarity(v)) for name, v in self.assoc.vocab.items()]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def decode_to_text(self, vec: HDVector, top_k: int = 3) -> str:
        """Return the top-k closest tokens as a string (for inspection)."""
        toks = self.decode_to_tokens(vec, top_k=top_k)
        return " | ".join(f"{t}({s:.2f})" for t, s in toks)

    # ----- tiny HD language model -----
    def learn_sequence(self, text: str) -> None:
        """Store a sequence in the LM: for each position, store
        (context_vector -> next_token_vector) so we can generate later.
        Context = bundle of previous (ngram-1) tokens with positional binding.
        """
        toks = tokenize(text)
        toks = ["<s>"] + toks + ["</s>"]
        vecs = [self.assoc.get_symbol(t) for t in toks]
        ctx_n = max(1, self.ngram - 1)
        for i in range(len(vecs) - 1):
            ctx_vecs = vecs[max(0, i - ctx_n + 1):i + 1]
            ctx = self._encode_context(ctx_vecs)
            self.assoc.kb_store.write(ctx, vecs[i + 1])

    def _encode_context(self, ctx_vecs: List[HDVector]) -> HDVector:
        """Encode a context window with positional binding."""
        if not ctx_vecs:
            return HDVector.random(self.dim)
        role = HDVector.from_text_seed("ctx_role", self.dim)
        bound = []
        for j, v in enumerate(ctx_vecs):
            bound.append(HDVector(data=np.roll(v.data, j), dim=self.dim))
        return bundle(bound)

    def predict_next(self, context_tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Predict the next token given a context of token strings."""
        if not context_tokens:
            ctx_vecs = [self.assoc.get_symbol("<s>")]
        else:
            ctx_n = max(1, self.ngram - 1)
            recent = context_tokens[-ctx_n:]
            ctx_vecs = [self.assoc.get_symbol(t) for t in recent]
        ctx = self._encode_context(ctx_vecs)
        retrieved = self.assoc.kb_store.read(ctx)
        if retrieved is None:
            return None
        # Find best matching token
        best_name, best_sim = None, -1.0
        for name, vec in self.assoc.vocab.items():
            sim = retrieved.similarity(vec)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None or best_sim < 0.02:
            return None
        return best_name, best_sim

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Greedy generation using the HD language model."""
        toks = tokenize(prompt)
        output = list(toks)
        for _ in range(max_tokens):
            ctx_tokens = output[-3:] if len(output) >= 3 else output
            pred = self.predict_next(ctx_tokens)
            if pred is None:
                break
            nxt, _ = pred
            if nxt == "</s>":
                break
            if nxt == "<s>" or nxt == "<unk>":
                break
            output.append(nxt)
        return " ".join(output)
