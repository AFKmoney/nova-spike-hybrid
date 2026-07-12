"""
commonsense.py — Built-in commonsense knowledge base.

PROBLEM
-------
Transformer LLMs learn commonsense implicitly from training data — they
often get it wrong because they don't have explicit rules. A human knows
"water is wet" without being taught explicitly.

SOLUTION
--------
Pre-load AETHER with a curated commonsense KB:
  - Physical: water is wet, fire is hot, ice is cold, rocks are hard
  - Spatial: above/below, inside/outside, near/far
  - Temporal: before/after, past/future
  - Biological: humans are alive, plants grow, animals eat
  - Social: people have names, friends help each other
  - Quantitative: bigger things weigh more, 1+1=2

This gives AETHER "world knowledge" without training.
"""

from __future__ import annotations
from typing import List, Tuple


# Curated commonsense facts (subject, predicate, object)
COMMONSENSE_FACTS: List[Tuple[str, str, str]] = [
    # Physical properties
    ("water", "is_a", "liquid"),
    ("water", "has_property", "wet"),
    ("water", "freezes_at", "0"),
    ("water", "boils_at", "100"),
    ("ice", "is_a", "solid"),
    ("ice", "has_property", "cold"),
    ("ice", "is_frozen", "water"),
    ("fire", "has_property", "hot"),
    ("fire", "produces", "heat"),
    ("fire", "produces", "light"),
    ("fire", "needs", "oxygen"),
    ("steam", "is_a", "gas"),
    ("steam", "is_heated", "water"),
    ("rock", "has_property", "hard"),
    ("rock", "has_property", "solid"),
    ("feather", "has_property", "light"),
    ("feather", "has_property", "soft"),
    ("gold", "has_property", "shiny"),
    ("gold", "has_property", "valuable"),
    ("gold", "is_a", "metal"),
    ("iron", "is_a", "metal"),
    ("iron", "has_property", "strong"),
    ("wood", "is_a", "material"),
    ("wood", "floats_in", "water"),
    ("wood", "burns_in", "fire"),
    ("glass", "has_property", "fragile"),
    ("glass", "is_a", "material"),
    ("glass", "is_transparent", "true"),

    # Spatial
    ("up", "is_opposite_of", "down"),
    ("left", "is_opposite_of", "right"),
    ("inside", "is_opposite_of", "outside"),
    ("above", "is_opposite_of", "below"),
    ("near", "is_opposite_of", "far"),
    ("front", "is_opposite_of", "back"),
    ("top", "is_opposite_of", "bottom"),

    # Temporal
    ("before", "is_opposite_of", "after"),
    ("past", "is_opposite_of", "future"),
    ("early", "is_opposite_of", "late"),
    ("fast", "is_opposite_of", "slow"),
    ("now", "is_a", "present"),
    ("yesterday", "is_a", "past"),
    ("tomorrow", "is_a", "future"),

    # Biological
    ("human", "is_a", "animal"),
    ("human", "has", "brain"),
    ("human", "has", "heart"),
    ("human", "needs", "water"),
    ("human", "needs", "food"),
    ("human", "needs", "sleep"),
    ("human", "needs", "oxygen"),
    ("animal", "is_a", "alive"),
    ("animal", "needs", "food"),
    ("plant", "is_a", "alive"),
    ("plant", "needs", "water"),
    ("plant", "needs", "light"),
    ("plant", "produces", "oxygen"),
    ("tree", "is_a", "plant"),
    ("tree", "has", "leaves"),
    ("tree", "has", "roots"),
    ("flower", "is_a", "plant"),
    ("fruit", "grows_on", "tree"),
    ("seed", "grows_into", "plant"),

    # Color
    ("red", "is_a", "color"),
    ("blue", "is_a", "color"),
    ("green", "is_a", "color"),
    ("yellow", "is_a", "color"),
    ("sky", "has_color", "blue"),
    ("grass", "has_color", "green"),
    ("sun", "has_color", "yellow"),
    ("blood", "has_color", "red"),

    # Weather
    ("rain", "is_a", "weather"),
    ("rain", "causes", "wet_ground"),
    ("snow", "is_a", "weather"),
    ("snow", "has_property", "cold"),
    ("snow", "has_color", "white"),
    ("wind", "is_a", "weather"),
    ("storm", "has", "wind"),
    ("storm", "has", "rain"),
    ("cloud", "produces", "rain"),
    ("sun", "produces", "light"),
    ("sun", "produces", "heat"),
    ("sun", "is_a", "star"),

    # Astronomy
    ("earth", "is_a", "planet"),
    ("earth", "orbits", "sun"),
    ("earth", "has", "moon"),
    ("moon", "orbits", "earth"),
    ("moon", "has_property", "reflects_light"),
    ("mars", "is_a", "planet"),
    ("mars", "has_color", "red"),
    ("jupiter", "is_a", "planet"),
    ("jupiter", "is_the", "largest_planet"),
    ("saturn", "is_a", "planet"),
    ("saturn", "has", "rings"),
    ("sun", "is_a", "star"),
    ("star", "produces", "light"),
    ("star", "produces", "heat"),

    # Quantitative
    ("one", "is_a", "number"),
    ("two", "is_a", "number"),
    ("three", "is_a", "number"),
    ("one", "plus", "one"),
    ("two", "plus", "two"),

    # Mathematics
    ("addition", "is_a", "math_operation"),
    ("subtraction", "is_a", "math_operation"),
    ("multiplication", "is_a", "math_operation"),
    ("division", "is_a", "math_operation"),

    # Social
    ("friend", "helps", "friend"),
    ("mother", "is_parent_of", "child"),
    ("father", "is_parent_of", "child"),
    ("parent", "cares_for", "child"),
    ("teacher", "teaches", "student"),
    ("doctor", "treats", "patient"),
    ("chef", "cooks", "food"),

    # Technology
    ("computer", "is_a", "machine"),
    ("computer", "needs", "electricity"),
    ("computer", "processes", "data"),
    ("phone", "is_a", "device"),
    ("phone", "needs", "electricity"),
    ("internet", "connects", "computers"),
    ("robot", "is_a", "machine"),
    ("robot", "is_built_by", "human"),

    # Causal (selected for common sense)
    ("fire", "causes", "heat"),
    ("rain", "causes", "wet_ground"),
    ("sun", "causes", "light"),
    ("cold", "causes", "freezing"),
    ("heat", "causes", "melting"),
    ("eating", "causes", "satiety"),
    ("exercise", "causes", "tiredness"),
    ("study", "causes", "learning"),
]


def load_commonsense(agent) -> int:
    """Load all commonsense facts into an AETHER agent.

    Returns the number of facts loaded.
    """
    count = 0
    for s, p, o in COMMONSENSE_FACTS:
        # Only add if not already known (avoid duplicates)
        if not agent.inference.lookup(s, p):
            agent.assoc.learn_triple(s, p, o)
            count += 1
    return count


def commonsense_stats() -> dict:
    """Return stats about the commonsense KB."""
    return {
        "n_facts": len(COMMONSENSE_FACTS),
        "categories": [
            "physical", "spatial", "temporal", "biological",
            "color", "weather", "astronomy", "quantitative",
            "mathematics", "social", "technology", "causal",
        ],
    }
