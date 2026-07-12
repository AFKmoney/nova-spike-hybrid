"""
causal.py — Causal model: cause → effect reasoning.

PROBLEM
-------
Transformer LLMs learn correlations, not causation. They can say "rain is
associated with wet ground" but can't reason "if it rains, the ground will
be wet" vs "if the ground is wet, it might have rained".

SOLUTION
--------
AETHER maintains an explicit causal model:
  - Causes are stored as (cause_event, effect_event) pairs
  - The causal relation is asymmetric (different from correlation)
  - AETHER can predict effects, abduce causes, and run interventions

Causal triples use the predicate "causes":
  ("rain", "causes", "wet ground")
  ("fire", "causes", "heat")
  ("sun", "causes", "light")

AETHER can answer:
  - "What happens if it rains?" → wet ground (forward prediction)
  - "Why is the ground wet?" → it rained (abduction)
  - "If I prevent rain, will the ground be wet?" → no (intervention)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import logging
import re

log = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """A chain of cause → effect."""
    events: List[str]
    confidence: float = 1.0

    def __str__(self) -> str:
        return " → ".join(self.events)


class CausalModel:
    """Explicit causal model over the KB."""

    CAUSAL_PREDICATES = {"causes", "leads_to", "produces", "results_in", "prevents"}

    def __init__(self, agent):
        self.agent = agent

    def learn_cause(self, cause: str, effect: str) -> str:
        """Teach a causal relation: cause → effect."""
        self.agent.assoc.learn_triple(cause, "causes", effect)
        return f"learned causal: {cause} → {effect}"

    def predict_effect(self, cause: str, _visited: Optional[set] = None) -> List[Tuple[str, float]]:
        """Predict the effects of an event (forward prediction).

        Returns list of (effect, confidence) pairs. Cycles are avoided
        via the _visited set.
        """
        if _visited is None:
            _visited = set()
        if cause.lower() in _visited:
            return []
        _visited.add(cause.lower())
        results = []
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == cause.lower() and p.lower() in self.CAUSAL_PREDICATES:
                results.append((o, 1.0))
                # Recursive: effects of effects
                for o2, conf in self.predict_effect(o, _visited):
                    results.append((o2, conf * 0.7))  # dampen
        return results

    def abduce_cause(self, effect: str, _visited: Optional[set] = None) -> List[Tuple[str, float]]:
        """Abduce the cause of an observed effect (backward reasoning).

        Returns list of (cause, confidence) pairs. Cycles are avoided.
        """
        if _visited is None:
            _visited = set()
        if effect.lower() in _visited:
            return []
        _visited.add(effect.lower())
        results = []
        for s, p, o in self.agent.assoc.list_triples():
            if o.lower() == effect.lower() and p.lower() in self.CAUSAL_PREDICATES:
                results.append((s, 1.0))
                for s2, conf in self.abduce_cause(s, _visited):
                    results.append((s2, conf * 0.7))
        return results

    def intervention(self, prevent: str) -> Dict[str, List[str]]:
        """Simulate an intervention: if we prevent `prevent`, what effects disappear?

        This is the do-operator from Pearl's causal calculus.
        """
        # Find all effects that depend on `prevent`
        disappeared: List[str] = []
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == prevent.lower() and p.lower() in self.CAUSAL_PREDICATES:
                disappeared.append(o)
                # Recursive: effects of effects also disappear
                for sub_o, _ in self.predict_effect(o):
                    if sub_o not in disappeared:
                        disappeared.append(sub_o)
        return {
            "prevented": prevent,
            "disappeared_effects": disappeared,
            "explanation": f"If we prevent {prevent}, the following effects would not occur: {', '.join(disappeared)}",
        }

    def find_causal_chain(self, start: str, end: str, max_depth: int = 4) -> Optional[CausalChain]:
        """Find a causal chain from start to end."""
        # BFS
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start.lower()}
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth: continue
            if current.lower() == end.lower() and len(path) > 1:
                return CausalChain(path, 1.0 / len(path))
            for s, p, o in self.agent.assoc.list_triples():
                if s.lower() == current.lower() and p.lower() in self.CAUSAL_PREDICATES:
                    if o.lower() not in visited:
                        visited.add(o.lower())
                        queue.append((o, path + [o]))
        return None

    def all_causal_relations(self) -> List[Tuple[str, str, str]]:
        """Return all causal relations in the KB."""
        return [(s, p, o) for s, p, o in self.agent.assoc.list_triples()
                if p.lower() in self.CAUSAL_PREDICATES]

    def summary(self) -> Dict[str, int]:
        rels = self.all_causal_relations()
        return {
            "n_causal_relations": len(rels),
            "n_unique_causes": len(set(s for s, _, _ in rels)),
            "n_unique_effects": len(set(o for _, _, o in rels)),
        }
