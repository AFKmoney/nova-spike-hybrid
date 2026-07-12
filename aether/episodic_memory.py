"""
episodic_memory.py — Dual memory: episodic (decay) + semantic (persistent).

BRAIN INSPIRATION
-----------------
The brain has two distinct memory systems:

  EPISODIC MEMORY (hippocampus):
    - Specific events: "I talked about X with the user at 15h"
    - Rich context: who, when, where, what was said
    - DECAYS over time (memory consolidation: episodic → semantic)
    - Can fail to recall details but preserves the gist

  SEMANTIC MEMORY (neocortex):
    - General knowledge: "Paris is the capital of France"
    - No temporal context (the fact just IS)
    - PERSISTS (doesn't decay)
    - Extracted from many episodic memories

The consolidation process: episodic memories are replayed, their common
structure is extracted into semantic memories, and the episodic detail
decays. This is what dream.py simulates.

AETHER'S USE
------------
We maintain two separate stores:

  EPISODIC: every conversation turn, with timestamp + context
    - Decays: each episode's HD vector is "softened" (blended with noise)
    - After N turns, the episode is mostly noise (forgotten)

  SEMANTIC: triples (subject, predicate, object)
    - Persistent: stored once, never decays
    - Strengthened by repetition (multiple episodic instances → strong semantic)

The agent can say: "I remember you told me about X yesterday" (episodic)
AND "the fact is Y" (semantic).
"""

from __future__ import annotations
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class EpisodicEntry:
    """An episodic memory: a specific event with context."""
    id: int
    timestamp: float
    user_input: str
    agent_response: str
    context: str  # what was the conversation about
    vector: HDVector
    decay_level: float = 0.0  # 0 = fresh, 1 = fully decayed
    n_rehearsals: int = 0  # how many times this was replayed

    @property
    def is_forgotten(self) -> bool:
        return self.decay_level >= 0.8


@dataclass
class SemanticEntry:
    """A semantic memory: a persistent fact."""
    subject: str
    predicate: str
    object: str
    strength: float = 1.0  # increases with repetition
    learned_at: float = field(default_factory=time.time)
    n_episodes_source: int = 1  # how many episodic memories formed this


class DualMemorySystem:
    """Episodic + semantic memory with consolidation."""

    def __init__(self, agent):
        self.agent = agent
        # Episodic store
        self.episodic: List[EpisodicEntry] = []
        self._episodic_counter = 0
        # Semantic store (mirrors agent.assoc.triples but with strength)
        self.semantic: Dict[Tuple[str, str, str], SemanticEntry] = {}
        # Decay parameters
        self.decay_rate = 0.05  # per turn
        self.decay_threshold = 0.8  # above this = forgotten
        self.consolidation_threshold = 3  # rehearsals needed to consolidate

    # ------------------------------------------------------------------ #
    # Episodic memory: recording events
    # ------------------------------------------------------------------ #
    def record_episode(self, user_input: str, agent_response: str,
                       context: str = "") -> EpisodicEntry:
        """Record a new episodic memory."""
        self._episodic_counter += 1
        text = f"{user_input} -> {agent_response}"
        vec = self.agent.encoder.encode_text(text)
        entry = EpisodicEntry(
            id=self._episodic_counter,
            timestamp=time.time(),
            user_input=user_input,
            agent_response=agent_response,
            context=context,
            vector=vec,
        )
        self.episodic.append(entry)
        # Keep episodic store bounded
        if len(self.episodic) > 200:
            self.episodic = self.episodic[-200:]
        # Apply decay to all previous episodes
        self._apply_decay()
        return entry

    def _apply_decay(self) -> None:
        """Apply decay to all episodic memories."""
        for entry in self.episodic[:-1]:  # don't decay the just-added one
            entry.decay_level = min(1.0, entry.decay_level + self.decay_rate)

    # ------------------------------------------------------------------ #
    # Episodic recall
    # ------------------------------------------------------------------ #
    def recall_episode(self, query: str, top_k: int = 3) -> List[Tuple[EpisodicEntry, float]]:
        """Recall episodic memories matching a query.

        Returns list of (entry, similarity) sorted by similarity.
        Decayed memories have reduced similarity.
        """
        q_vec = self.agent.encoder.encode_text(query)
        results = []
        for entry in self.episodic:
            if entry.is_forgotten: continue
            sim = q_vec.similarity(entry.vector)
            # Decay reduces recall
            effective_sim = sim * (1.0 - entry.decay_level)
            results.append((entry, effective_sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def recall_conversation(self, n_turns: int = 5) -> List[EpisodicEntry]:
        """Recall the last n conversation turns."""
        recent = [e for e in self.episodic if not e.is_forgotten]
        return recent[-n_turns:]

    def remember_talking_about(self, topic: str) -> Optional[str]:
        """Did I talk about this topic with the user?"""
        results = self.recall_episode(topic, top_k=1)
        if results and results[0][1] > 0.3:
            entry = results[0][0]
            time_str = time.strftime("%H:%M", time.localtime(entry.timestamp))
            return (f"Yes, I remember discussing this with you at {time_str}. "
                   f"You said: {entry.user_input!r}, and I responded: {entry.agent_response!r}.")
        return None

    # ------------------------------------------------------------------ #
    # Semantic memory: persistent facts
    # ------------------------------------------------------------------ #
    def store_semantic(self, subject: str, predicate: str, object: str) -> SemanticEntry:
        """Store a semantic fact (or strengthen if it exists)."""
        key = (subject.lower(), predicate.lower(), object.lower())
        if key in self.semantic:
            entry = self.semantic[key]
            entry.strength = min(1.0, entry.strength + 0.1)
            entry.n_episodes_source += 1
        else:
            entry = SemanticEntry(subject, predicate, object)
            self.semantic[key] = entry
            # Also store in the agent's regular KB
            self.agent.assoc.learn_triple(subject, predicate, object)
        return entry

    def retrieve_semantic(self, subject: str, predicate: str) -> Optional[Tuple[str, float]]:
        """Retrieve a semantic fact."""
        key_subject = subject.lower()
        key_predicate = predicate.lower()
        for (s, p, o), entry in self.semantic.items():
            if s == key_subject and p == key_predicate:
                return (o, entry.strength)
        return None

    # ------------------------------------------------------------------ #
    # Consolidation: episodic → semantic
    # ------------------------------------------------------------------ #
    def consolidate(self) -> Dict[str, int]:
        """Consolidate episodic memories into semantic facts.

        For each episodic memory that's been rehearsed enough times,
        extract the semantic fact and store it persistently.
        """
        n_consolidated = 0
        n_forgotten = 0
        # Extract facts from episodic memories
        from .learn_from_text import extract_facts
        for entry in self.episodic:
            if entry.is_forgotten: continue
            facts = extract_facts(entry.user_input)
            for fact in facts:
                # Store as semantic
                self.store_semantic(fact.subject, fact.predicate, fact.object)
                entry.n_rehearsals += 1
                n_consolidated += 1
            # Mark strongly-rehearsed episodes for forgetting
            if entry.n_rehearsals >= self.consolidation_threshold:
                entry.decay_level = max(entry.decay_level, 0.7)  # accelerate forgetting
        # Count forgotten
        n_forgotten = sum(1 for e in self.episodic if e.is_forgotten)
        log.info(f"consolidation: {n_consolidated} facts extracted, {n_forgotten} forgotten")
        return {
            "n_consolidated": n_consolidated,
            "n_forgotten": n_forgotten,
            "n_episodic_active": sum(1 for e in self.episodic if not e.is_forgotten),
            "n_semantic": len(self.semantic),
        }

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "n_episodic_total": len(self.episodic),
            "n_episodic_active": sum(1 for e in self.episodic if not e.is_forgotten),
            "n_episodic_forgotten": sum(1 for e in self.episodic if e.is_forgotten),
            "n_semantic": len(self.semantic),
            "mean_decay": float(np.mean([e.decay_level for e in self.episodic])) if self.episodic else 0.0,
        }
