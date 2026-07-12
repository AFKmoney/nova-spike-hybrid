"""
hv_attention.py — HV-based attention: contextual retrieval during generation.

PROBLEM
-------
The current generator uses only the prompt window. It doesn't retrieve
relevant memories DURING generation — so it can't reference earlier
context or knowledge.

SOLUTION
--------
HV attention: at each generation step, retrieve the top-k most relevant
HD vectors from memory and bundle them into the working context.

This is attention without weight matrices — just similarity-gated bundling.

  1. Compute the current working memory HD vector
  2. Retrieve top-k similar episodes/triples from memory
  3. Bundle them with the working memory (weighted by similarity)
  4. Use the bundled vector for the next prediction

The result: generation that references relevant knowledge dynamically.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import logging

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Result of one HV attention step."""
    query_vec: HDVector
    retrieved: List[Tuple[str, float]]  # (text, similarity) pairs
    attended_vec: HDVector  # the bundled attention result


class HVAttention:
    """HV-based attention mechanism for contextual retrieval."""

    def __init__(self, agent, top_k: int = 3, similarity_threshold: float = 0.1,
                 attention_decay: float = 0.7):
        self.agent = agent
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.attention_decay = attention_decay
        # Persistent attention state (working memory)
        self.working_memory: Optional[HDVector] = None

    # ------------------------------------------------------------------ #
    # Attention step
    # ------------------------------------------------------------------ #
    def attend(self, query_vec: HDVector) -> AttentionResult:
        """Perform one attention step.

        1. Retrieve top-k similar episodes from memory
        2. Weight them by similarity
        3. Bundle with the query (and decayed working memory)
        """
        # Retrieve from episodic memory
        retrieved = self.agent.assoc.retrieve_similar(query_vec, top_k=self.top_k)
        # Filter by threshold
        retrieved = [(t, s) for t, s in retrieved if s >= self.similarity_threshold]

        # Build the attended vector
        vecs_to_bundle = [query_vec]
        weights = [1.0]
        for text, sim in retrieved:
            text_vec = self.agent.encoder.encode_text(text)
            vecs_to_bundle.append(text_vec)
            weights.append(sim)  # weight by similarity

        # Include decayed working memory
        if self.working_memory is not None:
            vecs_to_bundle.append(self.working_memory)
            weights.append(self.attention_decay)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Weighted bundle
        attended = self._weighted_bundle(vecs_to_bundle, weights)

        # Update working memory
        self.working_memory = attended

        return AttentionResult(
            query_vec=query_vec,
            retrieved=retrieved,
            attended_vec=attended,
        )

    def _weighted_bundle(self, vectors: List[HDVector], weights: List[float]) -> HDVector:
        """Bundle vectors with weights (weighted majority vote)."""
        if not vectors:
            return HDVector.zero(self.agent.dim)
        dim = vectors[0].dim
        acc = np.zeros(dim, dtype=np.float32)
        for v, w in zip(vectors, weights):
            acc += w * v.data.astype(np.float32)
        # Sign with random tiebreak
        from .hd import _sign
        return HDVector(data=_sign(acc), dim=dim)

    # ------------------------------------------------------------------ #
    # Attention during generation
    # ------------------------------------------------------------------ #
    def attend_to_text(self, query_text: str) -> AttentionResult:
        """Encode text and attend to it."""
        q_vec = self.agent.encoder.encode_text(query_text)
        return self.attend(q_vec)

    def attend_to_kb(self, subject: str, predicate: str) -> AttentionResult:
        """Attend to a KB query: retrieve relevant triples."""
        # Build the query vector
        s_vec = self.agent.assoc.get_symbol(subject)
        p_vec = self.agent.assoc.get_symbol(predicate)
        query_vec = s_vec.bind(p_vec)
        return self.attend(query_vec)

    # ------------------------------------------------------------------ #
    # Multi-head attention (different memory sources)
    # ------------------------------------------------------------------ #
    def multi_head_attend(self, query_vec: HDVector) -> Dict[str, AttentionResult]:
        """Attend to multiple memory sources in parallel (multi-head).

        Each "head" retrieves from a different memory:
          - "episodes": episodic memory
          - "triples": KB triples
          - "attractors": attractor network
        """
        results = {}
        # Head 1: episodic memory
        results["episodes"] = self.attend(query_vec)
        # Head 2: KB triples (retrieve by binding)
        kb_retrieved = self.agent.assoc.kb_store.read(query_vec)
        if kb_retrieved is not None:
            # Find the closest labeled memory
            best_text, best_sim = None, -1.0
            for label, vec in self.agent.attractor.labeled_memories.items():
                sim = kb_retrieved.similarity(vec)
                if sim > best_sim:
                    best_sim, best_text = sim, label
            if best_text and best_sim >= self.similarity_threshold:
                results["triples"] = AttentionResult(
                    query_vec=query_vec,
                    retrieved=[(best_text, best_sim)],
                    attended_vec=kb_retrieved,
                )
        return results

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset working memory."""
        self.working_memory = None

    def stats(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "working_memory_active": self.working_memory is not None,
        }
