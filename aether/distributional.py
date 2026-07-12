"""
distributional.py — Distributional reasoning: analogies via HD algebra.

PROBLEM
-------
blending.py's analogy() is basic — it just looks up (a, p, b) and finds
(c, p, ?) for the same predicate p. This only works for explicit relations.

SOLUTION
--------
True distributional reasoning uses HD algebra (Plate 1995):

  analogy(a, b, c) = ? such that a:b :: c:?

  Computed as: bind(subtract(b, a), c) → retrieve nearest token

Where subtract(b, a) = bundle([b, inverse(a)]) captures the "transformation"
from a to b. Binding that with c gives the analog of c under the same
transformation.

Example:
  a = "Paris", b = "France" → transformation = "is capital of"
  c = "Tokyo" → apply transformation → ? = "Japan"

This is true analogical reasoning — it works even when the relation is
not explicitly stored, as long as the HD vectors encode the relation.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import numpy as np

from .hd import HDVector, DIM, bundle, _sign

log = logging.getLogger(__name__)


@dataclass
class AnalogyResult:
    """Result of an analogy query."""
    a: str
    b: str
    c: str
    answer: Optional[str]
    confidence: float
    candidates: List[Tuple[str, float]]  # top-k candidates with scores


class DistributionalReasoner:
    """Distributional reasoning via HD algebra."""

    def __init__(self, agent):
        self.agent = agent

    # ------------------------------------------------------------------ #
    # Core HD operations
    # ------------------------------------------------------------------ #
    def subtract(self, a: HDVector, b: HDVector) -> HDVector:
        """HD subtraction: bundle([b, inverse(a)]).

        inverse(a) for bipolar = a itself (self-inverse under bind).
        So subtract(a, b) = bundle([b, a_negated])

        But for the analogy to work, we need a notion of "inverse" that's
        different from bind. The standard approach is:
          subtract(a, b) = bundle([b, negate(a)])
        where negate(a) flips all bits of a.
        """
        # Negate a (flip all bits)
        a_neg = HDVector(data=(-a.data).astype(np.int8), dim=a.dim)
        return bundle([b, a_neg])

    def add(self, a: HDVector, b: HDVector) -> HDVector:
        """HD addition = bundle."""
        return bundle([a, b])

    # ------------------------------------------------------------------ #
    # Analogy
    # ------------------------------------------------------------------ #
    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> AnalogyResult:
        """Solve A:B :: C:?

        Strategy:
          1. Get HD vectors for a, b, c
          2. Compute transformation = subtract(a, b)
          3. Apply to c: result = bind(c, transformation)
             (or add: result = add(c, transformation))
          4. Find the nearest token in the vocab to result
        """
        # Get vectors
        vec_a = self.agent.assoc.get_symbol(a.lower())
        vec_b = self.agent.assoc.get_symbol(b.lower())
        vec_c = self.agent.assoc.get_symbol(c.lower())

        # Compute the transformation (a -> b)
        transformation = self.subtract(vec_a, vec_b)
        # Apply to c: result ≈ c + (b - a)
        result_vec = self.add(vec_c, transformation)

        # Find nearest tokens (excluding a, b, c themselves)
        candidates = []
        exclude = {a.lower(), b.lower(), c.lower()}
        for name, vec in self.agent.assoc.vocab.items():
            if name in exclude: continue
            if name.startswith("<") or name in {"?", ".", ",", "!", ":", ";"}: continue
            sim = result_vec.similarity(vec)
            candidates.append((name, sim))
        candidates.sort(key=lambda x: -x[1])
        top_candidates = candidates[:top_k]

        if top_candidates:
            answer = top_candidates[0][0]
            confidence = top_candidates[0][1]
        else:
            answer = None
            confidence = 0.0

        return AnalogyResult(
            a=a, b=b, c=c, answer=answer,
            confidence=confidence, candidates=top_candidates,
        )

    # ------------------------------------------------------------------ #
    # Structural similarity
    # ------------------------------------------------------------------ #
    def structural_similarity(self, pair1: Tuple[str, str],
                              pair2: Tuple[str, str]) -> float:
        """How similar is the relation (a, b) to (c, d)?

        If a:b and c:d stand in the same relation, the transformations
        subtract(a, b) and subtract(c, d) should be similar.
        """
        a1, b1 = pair1
        a2, b2 = pair2
        vec_a1 = self.agent.assoc.get_symbol(a1.lower())
        vec_b1 = self.agent.assoc.get_symbol(b1.lower())
        vec_a2 = self.agent.assoc.get_symbol(a2.lower())
        vec_b2 = self.agent.assoc.get_symbol(b2.lower())
        trans1 = self.subtract(vec_a1, vec_b1)
        trans2 = self.subtract(vec_a2, vec_b2)
        return trans1.similarity(trans2)

    # ------------------------------------------------------------------ #
    # Find analogies
    # ------------------------------------------------------------------ #
    def find_analogies(self, entity: str, top_k: int = 5) -> List[Tuple[str, str, str, float]]:
        """Find entities that stand in the same relation to `entity` as
        some known pair (a, b).

        Returns list of (a, b, candidate, similarity) tuples where
        a:b :: entity:candidate.
        """
        results = []
        # Find all (a, b) pairs from the KB
        triples = self.agent.assoc.list_triples()
        for s, p, o in triples[:20]:  # limit to avoid combinatorial explosion
            if s.lower() == entity.lower() or o.lower() == entity.lower():
                continue
            # Try: a=s, b=o, c=entity, find ?
            analogy_result = self.analogy(s, o, entity, top_k=3)
            for candidate, sim in analogy_result.candidates:
                if candidate.lower() != entity.lower():
                    results.append((s, o, candidate, sim))
        # Sort by similarity and return top_k
        results.sort(key=lambda x: -x[3])
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # Concept distance
    # ------------------------------------------------------------------ #
    def concept_distance(self, a: str, b: str) -> float:
        """Semantic distance between two concepts (0 = same, 1 = unrelated)."""
        vec_a = self.agent.assoc.get_symbol(a.lower())
        vec_b = self.agent.assoc.get_symbol(b.lower())
        return 1.0 - vec_a.similarity(vec_b)

    def nearest_concepts(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the nearest concepts to a given concept."""
        vec = self.agent.assoc.get_symbol(concept.lower())
        candidates = []
        for name, v in self.agent.assoc.vocab.items():
            if name == concept.lower(): continue
            if name.startswith("<") or len(name) < 2: continue
            sim = vec.similarity(v)
            candidates.append((name, sim))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]
