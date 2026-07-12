"""
mental_simulation.py — Mental simulation of scenes and scenarios.

BRAIN INSPIRATION
-----------------
Humans can simulate scenes mentally: "Imagine a red ball on a table."
The brain constructs an internal model of the scene and can reason about it:
"What happens if I push the ball?" → "It falls off the table."

Transformer LLMs can describe scenes but can't truly simulate them —
they generate likely next tokens, not internal world states.

AETHER'S USE
------------
MentalSimulator constructs an HD-based scene representation:
  - Entities (each as an HD vector)
  - Relations between entities (binding)
  - Spatial layout (positional encoding)
  - Temporal dynamics (simulation steps)

Scenes can be queried: "Where is the ball?" "What happens if I push it?"
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import logging
import numpy as np

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class SceneEntity:
    """An entity in a mental scene."""
    name: str
    properties: Dict[str, str] = field(default_factory=dict)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z


@dataclass
class MentalScene:
    """A mental scene: entities + relations + HD vector."""
    entities: Dict[str, SceneEntity] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (subj, rel, obj)
    vector: Optional[HDVector] = None
    description: str = ""


class MentalSimulator:
    """Construct and simulate mental scenes."""

    def __init__(self, agent):
        self.agent = agent
        self.current_scene: Optional[MentalScene] = None

    def construct_scene(self, description: str) -> MentalScene:
        """Construct a mental scene from a text description.

        Parses the description for entities and relations, then builds
        an HD vector representing the scene.
        """
        scene = MentalScene(description=description)
        # Simple entity extraction: capitalized words and common nouns
        import re
        words = re.findall(r'\b[A-Za-z]+\b', description.lower())
        # Filter stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "on", "in",
                     "at", "to", "of", "and", "or", "with", "by"}
        entities = [w for w in words if w not in stopwords and len(w) > 2]

        # Add unique entities
        for ent in set(entities):
            scene.entities[ent] = SceneEntity(name=ent)

        # Extract simple relations: "X is on Y"
        for m in re.finditer(r"(\w+)\s+is\s+on\s+(\w+)", description.lower()):
            scene.relations.append((m.group(1), "on", m.group(2)))
        for m in re.finditer(r"(\w+)\s+is\s+in\s+(\w+)", description.lower()):
            scene.relations.append((m.group(1), "in", m.group(2)))
        for m in re.finditer(r"(\w+)\s+is\s+next\s+to\s+(\w+)", description.lower()):
            scene.relations.append((m.group(1), "next_to", m.group(2)))

        # Build the HD vector: bundle of (entity_vec bound with position_vec)
        # + bundle of (relation_vec bound with bind(subj, obj))
        parts = []
        for name, ent in scene.entities.items():
            ent_vec = self.agent.assoc.get_symbol(name)
            parts.append(ent_vec)
        for subj, rel, obj in scene.relations:
            rel_vec = self.agent.assoc.get_symbol(rel)
            subj_vec = self.agent.assoc.get_symbol(subj)
            obj_vec = self.agent.assoc.get_symbol(obj)
            parts.append(rel_vec.bind(subj_vec.bind(obj_vec)))

        if parts:
            scene.vector = bundle(parts)
        else:
            scene.vector = HDVector.zero(self.agent.dim)

        self.current_scene = scene
        log.info(f"constructed scene: {len(scene.entities)} entities, {len(scene.relations)} relations")
        return scene

    def query_scene(self, query: str) -> Optional[str]:
        """Query the current mental scene.

        Args:
            query: a question like "Where is X?" or "What is on Y?"
        """
        if not self.current_scene:
            return "No scene loaded."
        scene = self.current_scene
        query = query.lower()

        # "Where is X?"
        import re
        m = re.match(r"where is (\w+)\??", query)
        if m:
            target = m.group(1)
            for subj, rel, obj in scene.relations:
                if subj == target:
                    return f"{target.capitalize()} is {rel} {obj}."
            return f"{target.capitalize()} is in the scene."

        # "What is on/in/next_to Y?"
        m = re.match(r"what is (\w+) (\w+)\??", query)
        if m:
            rel, target = m.group(1), m.group(2)
            for subj, r, obj in scene.relations:
                if r == rel and obj == target:
                    return f"{subj.capitalize()} is {rel} {target}."
            return f"Nothing is {rel} {target}."

        # "List entities"
        if "entities" in query or "what" in query:
            return f"Scene contains: {', '.join(scene.entities.keys())}"

        return f"Scene has {len(scene.entities)} entities."

    def simulate_action(self, action: str) -> str:
        """Simulate an action on the current scene.

        Args:
            action: e.g., "push the ball", "remove the table"
        """
        if not self.current_scene:
            return "No scene to simulate on."
        scene = self.current_scene
        action = action.lower()

        import re
        # "push X" — X moves
        m = re.match(r"push\s+(?:the\s+)?(\w+)", action)
        if m:
            target = m.group(1)
            if target in scene.entities:
                return f"If you push the {target}, it would move and possibly fall off."
            return f"There is no {target} in the scene."

        # "remove X" — X disappears
        m = re.match(r"remove\s+(?:the\s+)?(\w+)", action)
        if m:
            target = m.group(1)
            if target in scene.entities:
                # Check what depends on it
                dependents = [subj for subj, rel, obj in scene.relations if obj == target]
                if dependents:
                    return (f"If you remove the {target}, then {', '.join(dependents)} "
                           f"would fall (they were {target}-supported).")
                return f"If you remove the {target}, nothing else changes."

        # "add X" — X appears
        m = re.match(r"add\s+(?:a\s+|an\s+|the\s+)?(\w+)", action)
        if m:
            target = m.group(1)
            return f"If you add a {target}, the scene now contains {target}."

        return f"Cannot simulate action: {action}"

    def scene_similarity(self, other_description: str) -> float:
        """Compare the current scene to another described scene."""
        if not self.current_scene: return 0.0
        other_vec = self.agent.encoder.encode_text(other_description)
        return self.current_scene.vector.similarity(other_vec)
