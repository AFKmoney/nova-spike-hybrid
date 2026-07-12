"""
memory.py — Memory subsystem for AETHER.

Two complementary stores:

  1. SparseDistributedMemory (Kanerva 1988):
     - N "hard locations" sampled uniformly in the HD space.
     - Write/read activate the k nearest locations (top-k by Hamming distance).
     - Counters accumulate ±1 per write; read returns majority vote.
     - O(N * dim) per op, no GPU. Sparse and noise-robust.

  2. AssociativeMemory:
     - Symbol table: name -> HDVector (deterministic from text seed).
     - Episodic store: list of (vector, payload) for explicit recall.
     - Concept KB: stores bindings like bind(subject, predicate) -> object.

Both learn in O(1) per item — true one-shot, no epochs.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from .hd import HDVector, DIM, _sign


class SparseDistributedMemory:
    """
    Kanerva's Sparse Distributed Memory.

    Hard locations are random bipolar vectors. A read/write addresses the
    k closest locations (by Hamming distance, or equivalently by inner
    product since bipolar). Writes accumulate ±1 counters; reads return
    the sign of the summed counters across activated locations.
    """

    def __init__(self, dim: int = DIM, n_locations: int = 5000, k: int = 15):
        self.dim = dim
        self.n_locations = n_locations
        self.k = k
        # Hard locations — random bipolar vectors
        rng = np.random.default_rng(42)
        self.locations = rng.choice([-1, 1], size=(n_locations, dim)).astype(np.int8)
        # Per-location per-bit counters (int8 to bound memory)
        self.counters = np.zeros((n_locations, dim), dtype=np.int8)
        self.write_count = np.zeros(n_locations, dtype=np.int32)

    def _activate(self, address: np.ndarray) -> np.ndarray:
        """Return indices of the k closest hard locations to `address`."""
        # For bipolar, dot product = (dim - 2*hamming). Higher dot = closer.
        dots = self.locations @ address.astype(np.int16)
        k = min(self.k, self.n_locations)
        if k >= self.n_locations:
            return np.arange(self.n_locations)
        # Top-k by dot product
        return np.argpartition(-dots, k - 1)[:k]

    def write(self, address: HDVector, data: HDVector) -> None:
        """Write `data` at `address`. Activates k nearest hard locations."""
        active = self._activate(address.data)
        # Each activated location: counter += data (with int8 saturation)
        # We use int16 accumulation to avoid overflow, then saturate to int8
        upd = self.counters[active].astype(np.int16) + data.data.astype(np.int16)
        self.counters[active] = np.clip(upd, -127, 127).astype(np.int8)
        self.write_count[active] += 1

    def read(self, address: HDVector) -> Optional[HDVector]:
        """Read from `address`. Returns None if no writes happened nearby."""
        active = self._activate(address.data)
        total_writes = self.write_count[active].sum()
        if total_writes == 0:
            return None
        # CRITICAL: cast to int32 BEFORE sum — int8 sum overflows for k>15
        summed = self.counters[active].astype(np.int32).sum(axis=0)
        if np.all(summed == 0):
            return None
        return HDVector(data=_sign(summed), dim=self.dim)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_locations": self.n_locations,
            "dim": self.dim,
            "k": self.k,
            "total_writes": int(self.write_count.sum()),
            "active_locations": int((self.write_count > 0).sum()),
        }


@dataclass
class _Episode:
    vector: HDVector
    payload: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssociativeMemory:
    """
    Symbol-grounded associative memory.

    - vocab: name -> HDVector (deterministic from text seed for stability)
    - episodes: list of (vector, text) for retrieval
    - bindings: stored as (subject, predicate) -> object HD vector
      This is the symbolic KB layer.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vocab: Dict[str, HDVector] = {}
        self.episodes: List[_Episode] = []
        # KB of triples: store as bind(bind(s_vec, p_vec), o_vec) in an SDM
        self.kb_store = SparseDistributedMemory(dim=dim, n_locations=5000, k=15)
        # Also keep an explicit triple list for inspection / debugging
        self.triples: List[Tuple[str, str, str]] = []

    # ----- vocabulary -----
    def get_symbol(self, name: str) -> HDVector:
        """Get or create a stable HD vector for a symbol."""
        name = name.lower().strip()
        if name not in self.vocab:
            self.vocab[name] = HDVector.from_text_seed(name, self.dim)
        return self.vocab[name]

    def has_symbol(self, name: str) -> bool:
        return name.lower().strip() in self.vocab

    # ----- episodic storage (raw text + vector) -----
    def add_episode(self, text: str, vector: HDVector, metadata: Optional[Dict] = None) -> None:
        self.episodes.append(_Episode(vector=vector, payload=text, metadata=metadata or {}))

    def retrieve_similar(self, query: HDVector, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the top-k most similar stored episodes."""
        if not self.episodes:
            return []
        sims = [(ep.payload, query.similarity(ep.vector)) for ep in self.episodes]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    # ----- semantic KB (subject, predicate, object) -----
    def learn_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Store a triple in BOTH directions so we can query either way.

        Convention (bind is commutative for bipolar):
          - bind(s, p) -> o   : answers 'what is S?' / 'where is S?'
          - bind(p, o) -> s   : answers 'what is the P of O?'

        Example: 'Paris is the capital of France'
          s=Paris, p=capital_of, o=France
          - bind(Paris, capital_of) -> France  (for 'Paris is the capital of what?')
          - bind(capital_of, France) -> Paris  (for 'What is the capital of France?')
        """
        s = self.get_symbol(subject)
        p = self.get_symbol(predicate)
        o = self.get_symbol(obj)
        # Subject-predicate -> object  (supports 'what is S?' queries)
        self.kb_store.write(s.bind(p), o)
        # Predicate-object -> subject  (supports 'what is the P of O?' queries)
        self.kb_store.write(p.bind(o), s)
        self.triples.append((subject.lower(), predicate.lower(), obj.lower()))

    def query_triple(self, subject: str, predicate: str) -> Optional[Tuple[str, float]]:
        """Query (subject, predicate) -> ?object. Returns (best_match, similarity).

        Structural tokens (<s>, </s>, <unk>, <pad>, single-char punctuation) are
        excluded — they should never be returned as factual answers.
        """
        s = self.get_symbol(subject)
        p = self.get_symbol(predicate)
        addr = s.bind(p)
        retrieved = self.kb_store.read(addr)
        if retrieved is None:
            return None
        # Tokens that should never be returned as a factual answer
        STRUCTURAL = {"<s>", "</s>", "<unk>", "<pad>", "?", ".", ",", "!", ":", ";", "|"}
        # Match against known symbols
        best_name, best_sim = None, -1.0
        for name, vec in self.vocab.items():
            if name in STRUCTURAL:
                continue
            sim = retrieved.similarity(vec)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None or best_sim < 0.10:
            return None
        return best_name, best_sim

    def list_triples(self) -> List[Tuple[str, str, str]]:
        return list(self.triples)

    # ----- persistence -----
    def stats(self) -> Dict[str, Any]:
        return {
            "vocab_size": len(self.vocab),
            "episodes": len(self.episodes),
            "triples": len(self.triples),
            "kb_store": self.kb_store.stats(),
        }
