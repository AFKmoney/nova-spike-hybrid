"""
counterfactual.py — Counterfactual simulation: "what if" reasoning.

PROBLEM
-------
Transformer LLMs can't reliably reason about hypothetical scenarios. They
mix up reality with the hypothetical. A human can reason: "If Paris were
in Japan, then Paris would be in Asia."

SOLUTION
--------
CounterfactualSimulator creates an ALTERNATIVE KB (a "what if" world) by:
  1. Copying the current KB
  2. Making a hypothetical change (add/remove/modify a fact)
  3. Re-running inference in the alternative KB
  4. Comparing the results to the real KB

This is true counterfactual reasoning — AETHER can imagine alternative worlds.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual simulation."""
    hypothesis: str
    change_made: str
    real_answer: Optional[str]
    counterfactual_answer: Optional[str]
    differs: bool
    explanation: str


class CounterfactualSimulator:
    """Simulate 'what if' scenarios by creating alternative KBs."""

    def __init__(self, agent):
        self.agent = agent

    def what_if(self, hypothesis: str) -> CounterfactualResult:
        """Simulate a 'what if' scenario.

        Args:
            hypothesis: a hypothetical fact, e.g., "Paris is the capital of Japan"

        Returns:
            CounterfactualResult with the real vs counterfactual answers
        """
        # Parse the hypothesis
        from .generator import parse_triple
        triple = parse_triple(hypothesis)
        if not triple:
            return CounterfactualResult(
                hypothesis=hypothesis, change_made="(could not parse)",
                real_answer=None, counterfactual_answer=None, differs=False,
                explanation="Could not parse the hypothesis as a triple.",
            )
        s, p, o = triple

        # Find the real value (if any)
        real_result = self.agent.inference.lookup(s, p)
        real_answer = real_result[0] if real_result else None

        # Save the original triples
        original_triples = self.agent.assoc.list_triples()

        # Apply the counterfactual: temporarily overwrite (s, p, ?) with (s, p, o)
        # Remove any existing (s, p, ?) triples
        new_triples = [t for t in original_triples
                       if not (t[0].lower() == s.lower() and t[1].lower() == p.lower())]
        new_triples.append((s, p, o))

        # Temporarily replace the agent's triples
        # (we manipulate the in-memory list; the SDM still has the old data
        # but lookup() checks the explicit list first)
        self.agent.assoc.triples = new_triples

        # Now ask a question based on the counterfactual
        # E.g., if hypothesis was "Paris is the capital of Japan",
        # we ask "What is the capital of Japan?"
        if p == "capital_of":
            question = f"What is the capital of {o}?"
        elif p == "located_in":
            question = f"Where is {s} located?"
        elif p == "is_a":
            question = f"What is {s}?"
        else:
            question = f"What is the {p.replace('_', ' ')} of {s}?"

        # Get the counterfactual answer
        cf_result = self.agent.inference.lookup(s, p) if p != "capital_of" \
                    else self.agent.inference.lookup(o, "capital_of")
        cf_answer = cf_result[0] if cf_result else None

        # Restore the original triples
        self.agent.assoc.triples = original_triples

        differs = real_answer != cf_answer
        if real_answer and cf_answer and real_answer.lower() != cf_answer.lower():
            explanation = (f"In reality, {s} {p.replace('_',' ')} {real_answer}. "
                          f"But if {hypothesis}, then {s} {p.replace('_',' ')} {cf_answer}.")
        elif not real_answer and cf_answer:
            explanation = (f"In reality, I don't know what {s} {p.replace('_',' ')}. "
                          f"But if {hypothesis}, then {s} {p.replace('_',' ')} {cf_answer}.")
        else:
            explanation = f"Hypothesis: {hypothesis}. No change in inference."

        return CounterfactualResult(
            hypothesis=hypothesis,
            change_made=f"set ({s}, {p}, {o})",
            real_answer=real_answer,
            counterfactual_answer=cf_answer,
            differs=differs,
            explanation=explanation,
        )

    def imagine_alternative(self, entity: str) -> List[str]:
        """Generate alternative scenarios for an entity.

        Returns a list of "what if" hypotheses.
        """
        hypotheses = []
        # For each known predicate of the entity, generate alternatives
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == entity.lower():
                # Find other entities with the same predicate
                for s2, p2, o2 in self.agent.assoc.list_triples():
                    if p2.lower() == p.lower() and o2.lower() != o.lower():
                        hypotheses.append(f"{entity} {p2.replace('_',' ')} {o2}")
                        if len(hypotheses) >= 5: break
                if len(hypotheses) >= 5: break
        return hypotheses
