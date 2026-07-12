"""
hd.py — Hyperdimensional Vector primitives (Vector Symbolic Architecture).

Bipolar high-dimensional vectors (±1) with three core operations:
  - bind  : element-wise multiplication (XOR-equivalent). Associations.
  - bundle: majority vote (sum then sign). Sets / superpositions.
  - permute: cyclic shift. Sequences / positional encoding.

These three operations are enough to build a Turing-complete algebra on
symbolic structures (Kanerva 1996, Plate 1995, Gayler 1998).

Complexity: O(dim) per operation. No matrices, no GPU.
"""

from __future__ import annotations
import numpy as np
from typing import List, Iterator, Tuple, Optional


# Default dimension. 4096 is a good tradeoff:
#   - Large enough that random vectors are ~orthogonal (similarity ≈ 0)
#   - Small enough that numpy ops are <1ms
#   - Powers of two are cache-friendly
DIM = 4096


class HDVector:
    """A bipolar hyperdimensional vector (entries in {-1, +1})."""

    __slots__ = ("data", "dim")

    def __init__(self, data: Optional[np.ndarray] = None, dim: int = DIM):
        self.dim = dim
        if data is None:
            self.data = np.random.choice([-1, 1], size=dim).astype(np.int8)
        else:
            self.data = np.asarray(data, dtype=np.int8)
            if self.data.shape != (dim,):
                # auto-resize: assume user knows what they're doing
                self.dim = self.data.shape[0]

    # ----- constructors -----
    @classmethod
    def random(cls, dim: int = DIM) -> "HDVector":
        return cls(dim=dim)

    @classmethod
    def zero(cls, dim: int = DIM) -> "HDVector":
        return cls(data=np.zeros(dim, dtype=np.int8), dim=dim)

    @classmethod
    def from_text_seed(cls, text: str, dim: int = DIM) -> "HDVector":
        """Deterministic vector from text — useful for stable vocab."""
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return cls(data=rng.choice([-1, 1], size=dim).astype(np.int8), dim=dim)

    # ----- core operations -----
    def bind(self, other: "HDVector") -> "HDVector":
        """Binding: element-wise product. Inverse is itself (self-inverse)."""
        return HDVector(data=self.data * other.data, dim=self.dim)

    def unbind(self, other: "HDVector") -> "HDVector":
        """Unbinding = binding for bipolar (self-inverse algebra)."""
        return self.bind(other)

    def bundle(self, other: "HDVector") -> "HDVector":
        """Bundling: sum then sign. Random tiebreak on zeros."""
        s = self.data.astype(np.int16) + other.data.astype(np.int16)
        return HDVector(data=_sign(s), dim=self.dim)

    def permute(self, n: int = 1) -> "HDVector":
        """Cyclic permutation by n positions. Role-tag for sequences."""
        return HDVector(data=np.roll(self.data, n), dim=self.dim)

    def inverse_permute(self, n: int = 1) -> "HDVector":
        return HDVector(data=np.roll(self.data, -n), dim=self.dim)

    # ----- similarity -----
    def similarity(self, other: "HDVector") -> float:
        """Cosine similarity. For bipolar: dot / dim, in [-1, +1].

        IMPORTANT: cast to int32 before dot — int8 dot overflows at dim>=256.
        """
        return float(np.dot(self.data.astype(np.int32), other.data.astype(np.int32)) / self.dim)

    def hamming(self, other: "HDVector") -> int:
        return int(np.count_nonzero(self.data != other.data))

    # ----- utilities -----
    def copy(self) -> "HDVector":
        return HDVector(data=self.data.copy(), dim=self.dim)

    def __repr__(self) -> str:
        return f"HDVector(dim={self.dim}, |1|={int(np.sum(self.data == 1))}, |-1|={int(np.sum(self.data == -1))})"


def _sign(arr: np.ndarray) -> np.ndarray:
    """Sign with random tiebreak on zero. Avoids bias in bundling."""
    result = np.where(arr > 0, 1, np.where(arr < 0, -1, 0))
    # Random tiebreak for zeros
    zero_mask = (result == 0)
    if zero_mask.any():
        n_zeros = int(zero_mask.sum())
        result[zero_mask] = np.random.choice([-1, 1], size=n_zeros).astype(np.int8)
    return result.astype(np.int8)


def bundle(vectors: List[HDVector], weights: Optional[List[float]] = None) -> HDVector:
    """Bundle a list of vectors with optional weights."""
    if not vectors:
        return HDVector.zero()
    dim = vectors[0].dim
    if weights is None:
        weights = [1.0] * len(vectors)
    acc = np.zeros(dim, dtype=np.float32)
    for v, w in zip(vectors, weights):
        acc += w * v.data.astype(np.float32)
    return HDVector(data=_sign(acc), dim=dim)


def bind_sequence(vectors: List[HDVector]) -> HDVector:
    """Encode an ordered sequence: bundle of bind(token_i, perm^i(role))."""
    if not vectors:
        return HDVector.zero()
    dim = vectors[0].dim
    role = HDVector.random(dim)
    acc = np.zeros(dim, dtype=np.int16)
    for i, v in enumerate(vectors):
        acc += np.roll(role.data * v.data, i).astype(np.int16)
    return HDVector(data=_sign(acc), dim=dim)


def ngram_encode(vectors: List[HDVector], n: int = 3) -> HDVector:
    """Encode a sequence as a superposition of n-grams.

    For each window of size n, bind tokens with permuted positional roles
    and bundle them all. This is the canonical VSA text encoder.
    """
    if not vectors:
        return HDVector.zero()
    dim = vectors[0].dim
    roles = [HDVector.random(dim) for _ in range(n)]
    acc = np.zeros(dim, dtype=np.int16)
    for i in range(len(vectors) - n + 1):
        window = vectors[i:i + n]
        gram = np.ones(dim, dtype=np.int8)
        for j, tok in enumerate(window):
            gram *= np.roll(tok.data, j)  # permute by position
        acc += gram.astype(np.int16)
    if acc.sum() == 0:
        # Fallback: single tokens
        for v in vectors:
            acc += v.data.astype(np.int16)
    return HDVector(data=_sign(acc), dim=dim)
