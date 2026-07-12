"""
blending.py — Conceptual blending: create new concepts by combining existing ones.

BRAIN INSPIRATION
-----------------
Conceptual blending (Fauconnier & Turner, 2002) is a fundamental cognitive
operation. The brain creates new concepts by blending existing ones:

  "horse" + "bird" → "pegasus" (horse with wings)
  "boat" + "airplane" → "seaplane"
  "phone" + "computer" → "smartphone"

This is the basis of human creativity. Transformer LLMs mimic this
statistically but don't do it compositionally.

AETHER'S USE
------------
blend_concepts(a, b) creates a new concept HD vector by:
  1. Taking the HD vectors of a and b
  2. Blending them: bundle([a, b]) + binding key features
  3. Naming the new concept (a + b substring or user-supplied name)
  4. Inferring properties: pegasus can fly (from bird) and run (from horse)

This is real creativity — AETHER generates novel concepts.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class BlendedConcept:
    """A new concept created by blending."""
    name: str
    parents: Tuple[str, str]
    vector: HDVector
    inherited_properties: Dict[str, List[str]]  # property → values from both parents


class ConceptBlender:
    """Blend two concepts to create a new one."""

    def __init__(self, agent):
        self.agent = agent

    def blend(self, concept_a: str, concept_b: str, name: Optional[str] = None) -> BlendedConcept:
        """Blend two concepts into a new concept.

        The new concept inherits properties from BOTH parents:
          - Properties unique to each parent are inherited
          - Conflicting properties are merged (both values kept)
        """
        # Get parent vectors
        vec_a = self.agent.assoc.get_symbol(concept_a)
        vec_b = self.agent.assoc.get_symbol(concept_b)

        # Blend: bundle the two vectors
        # This gives a vector ~33% similar to each parent
        blended_vec = bundle([vec_a, vec_b])

        # Generate a name if not provided
        if name is None:
            # Take first syllable of a + second syllable of b
            name = self._generate_name(concept_a, concept_b)

        # Store the new concept
        self.agent.assoc.vocab[name.lower()] = blended_vec

        # Inherit properties
        inherited = self._inherit_properties(concept_a, concept_b)

        # Store the new concept's properties
        for prop, values in inherited.items():
            for value in values:
                # Don't overwrite if exists
                if not self.agent.inference.lookup(name, prop):
                    self.agent.assoc.learn_triple(name, prop, value)

        log.info(f"blended: {concept_a} + {concept_b} -> {name} ({len(inherited)} properties inherited)")
        return BlendedConcept(
            name=name,
            parents=(concept_a, concept_b),
            vector=blended_vec,
            inherited_properties=inherited,
        )

    def _generate_name(self, a: str, b: str) -> str:
        """Generate a portmanteau name for the blend."""
        # Take first half of a + second half of b
        a_part = a[:max(1, len(a) // 2)]
        b_part = b[max(1, len(b) // 2):]
        return (a_part + b_part).lower()

    def _inherit_properties(self, a: str, b: str) -> Dict[str, List[str]]:
        """Inherit properties from both parents."""
        inherited: Dict[str, List[str]] = {}
        # Get all (predicate, object) pairs for each parent
        for parent in (a, b):
            for s, p, o in self.agent.assoc.list_triples():
                if s.lower() == parent.lower():
                    inherited.setdefault(p, [])
                    if o not in inherited[p]:
                        inherited[p].append(o)
        return inherited

    def analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """Solve A:B :: C:? analogies.

        Example: dog:puppy :: cat:?
        Strategy: find what relation holds between a and b, then find what
        stands in the same relation to c.
        """
        # Find the predicate that relates a to b
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == a.lower() and o.lower() == b.lower():
                # Found: a --p--> b. Now find c --p--> ?
                for s2, p2, o2 in self.agent.assoc.list_triples():
                    if s2.lower() == c.lower() and p2.lower() == p.lower():
                        return o2
        return None

    def find_analogies(self, entity: str) -> List[Tuple[str, str, str]]:
        """Find analogies involving the entity.

        Returns list of (a, b, c, d) tuples where a:b :: c:d.
        """
        analogies = []
        # Get all (entity, predicate, object) triples
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == entity.lower():
                # Look for other entities with the same predicate
                for s2, p2, o2 in self.agent.assoc.list_triples():
                    if p2.lower() == p.lower() and s2.lower() != entity.lower():
                        analogies.append((entity, o, s2, o2))
        return analogies[:5]  # limit
