"""
dream_v2.py — Dream consolidation v2: Hebbian co-activation + concept centroids.

IMPROVEMENTS OVER v1
--------------------
v1 dream was a simple replay that occasionally combined two episodes.
v2 implements proper Hebbian consolidation:

  1. HEBBIAN CO-ACTIVATION MATRIX
     - During waking, every pair of concepts that fire together gets
       its co-activation count incremented
     - During sleep, pairs with high co-activation but no explicit
       relation get linked (Hebbian learning)

  2. CONCEPT CENTROIDS
     - Cluster similar episodes into "concept centroids"
     - Each centroid is the bundle of all episodes in the cluster
     - Centroids become new abstract concepts in the KB
     - This is how AETHER "discovers" abstractions on its own

  3. CONSOLIDATION PHASES
     - NREM: replay episodes, strengthen attractors, hebbian linking
     - REM: random recombination, hypothesis generation, creative insight
     - Wake transition: prune weak/noisy connections

  4. IDLE DETECTION
     - When AETHER hasn't been asked anything for N seconds, it naps
     - Each nap runs a full consolidation cycle
"""

from __future__ import annotations
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass
class ConceptCentroid:
    """An abstract concept discovered by clustering episodes."""
    name: str
    vector: Any  # HDVector
    members: List[str] = field(default_factory=list)  # episode texts
    co_activation_count: int = 0


@dataclass
class DreamReportV2:
    """Comprehensive report of a v2 dream cycle."""
    phase: str  # "NREM" or "REM"
    cycles: int
    episodes_replayed: int
    hebbian_links_formed: List[Tuple[str, str, str]]  # (a, relation, b)
    centroids_created: List[ConceptCentroid]
    centroids_strengthened: int
    new_triples_discovered: List[Tuple[str, str, str]]
    hypotheses_generated: List[str]
    creative_insights: List[str]
    duration_ms: float = 0.0


class DreamConsolidatorV2:
    """Hebbian + centroid-based dream consolidation."""

    def __init__(self, agent):
        self.agent = agent
        self.rng = random.Random()
        # Hebbian co-activation matrix (sparse: dict of (a, b) → count)
        self.co_activations: Dict[Tuple[str, str], int] = defaultdict(int)
        # Concept centroids (clustered episodes)
        self.centroids: List[ConceptCentroid] = []
        self.last_dream: Optional[DreamReportV2] = None
        # Idle tracking
        self.last_activity: float = time.time()

    # ------------------------------------------------------------------ #
    # Wake-time co-activation tracking
    # ------------------------------------------------------------------ #
    def observe_concepts(self, concepts: List[str]) -> None:
        """Called when multiple concepts fire together (during waking).

        Increments pairwise co-activation counts — concepts that fire
        together wire together (Hebb 1949).
        """
        self.last_activity = time.time()
        unique = list(set(c.lower() for c in concepts if c))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                key = tuple(sorted([unique[i], unique[j]]))
                self.co_activations[key] += 1

    def is_idle(self, idle_threshold_s: float = 5.0) -> bool:
        """Has AETHER been idle for the threshold duration?"""
        return time.time() - self.last_activity > idle_threshold_s

    # ------------------------------------------------------------------ #
    # Main dream entry point
    # ------------------------------------------------------------------ #
    def dream(self, cycles: int = 30, phase: str = "mixed") -> DreamReportV2:
        """Run a dream session.

        Args:
            cycles: number of consolidation cycles
            phase: "NREM" (consolidation), "REM" (creative), or "mixed"
        """
        t0 = time.perf_counter()
        if phase == "mixed":
            # Split cycles between NREM and REM
            n_nrem = cycles // 2
            n_rem = cycles - n_nrem
            nrem = self._dream_nrem(n_nrem)
            rem = self._dream_rem(n_rem)
            report = DreamReportV2(
                phase="mixed", cycles=cycles,
                episodes_replayed=nrem.episodes_replayed + rem.episodes_replayed,
                hebbian_links_formed=nrem.hebbian_links_formed + rem.hebbian_links_formed,
                centroids_created=nrem.centroids_created + rem.centroids_created,
                centroids_strengthened=nrem.centroids_strengthened + rem.centroids_strengthened,
                new_triples_discovered=nrem.new_triples_discovered + rem.new_triples_discovered,
                hypotheses_generated=rem.hypotheses_generated,
                creative_insights=rem.creative_insights,
            )
        elif phase == "NREM":
            report = self._dream_nrem(cycles)
        else:
            report = self._dream_rem(cycles)
        report.duration_ms = (time.perf_counter() - t0) * 1000
        self.last_dream = report
        log.info(f"dream v2 ({phase}): {report.episodes_replayed} replays, "
                 f"{len(report.hebbian_links_formed)} hebbian links, "
                 f"{len(report.centroids_created)} new centroids, "
                 f"{len(report.new_triples_discovered)} new triples, "
                 f"{len(report.creative_insights)} insights")
        return report

    # ------------------------------------------------------------------ #
    # NREM: consolidation (replay + Hebbian + centroids)
    # ------------------------------------------------------------------ #
    def _dream_nrem(self, cycles: int) -> DreamReportV2:
        report = DreamReportV2(
            phase="NREM", cycles=cycles, episodes_replayed=0,
            hebbian_links_formed=[], centroids_created=[],
            centroids_strengthened=0, new_triples_discovered=[],
            hypotheses_generated=[], creative_insights=[],
        )
        episodes = self.agent.assoc.episodes
        if not episodes: return report

        for cycle in range(cycles):
            # 1. Pick a random episode
            ep = self.rng.choice(episodes)
            report.episodes_replayed += 1

            # 2. Find co-activating concepts in this episode
            from .learn_from_text import extract_facts
            facts = extract_facts(ep.payload)
            concepts_in_ep = []
            for f in facts:
                concepts_in_ep.append(f.subject)
                concepts_in_ep.append(f.object)

            # 3. Check Hebbian matrix for strong co-activations not yet in KB
            for i in range(len(concepts_in_ep)):
                for j in range(i + 1, len(concepts_in_ep)):
                    a, b = concepts_in_ep[i], concepts_in_ep[j]
                    if a.lower() == b.lower(): continue
                    key = tuple(sorted([a.lower(), b.lower()]))
                    count = self.co_activations.get(key, 0)
                    if count >= 3:  # threshold for Hebbian linking
                        # Check if a relation already exists
                        existing = self._any_relation(a, b)
                        if not existing:
                            # Form a Hebbian link: "associated_with"
                            self.agent.assoc.learn_triple(a, "associated_with", b)
                            report.hebbian_links_formed.append((a, "associated_with", b))
                            log.debug(f"Hebbian link: {a} <-> {b} (count={count})")

            # 4. Strengthen attractor
            if ep.payload in self.agent.attractor.labeled_memories:
                self.agent.attractor.store(
                    self.agent.attractor.labeled_memories[ep.payload],
                    label=ep.payload, n_reinforce=1,
                )
                report.centroids_strengthened += 1

            # 5. Every 10 cycles, try to find/create a centroid
            if cycle % 10 == 0:
                centroid = self._find_or_create_centroid(ep)
                if centroid and centroid not in report.centroids_created:
                    report.centroids_created.append(centroid)

            # 6. Every 5 cycles, try transitive closure
            if cycle % 5 == 0:
                new_triple = self._try_transitive_discovery()
                if new_triple:
                    report.new_triples_discovered.append(new_triple)

        return report

    def _any_relation(self, a: str, b: str) -> bool:
        """Check if any relation exists between a and b (either direction)."""
        for s, p, o in self.agent.assoc.list_triples():
            if (s.lower() == a.lower() and o.lower() == b.lower()) or \
               (s.lower() == b.lower() and o.lower() == a.lower()):
                return True
        return False

    def _find_or_create_centroid(self, ep) -> Optional[ConceptCentroid]:
        """Find a centroid this episode fits, or create a new one."""
        # Compare to existing centroids
        best_centroid = None
        best_sim = -1.0
        for c in self.centroids:
            sim = ep.vector.similarity(c.vector)
            if sim > best_sim:
                best_sim = sim
                best_centroid = c

        if best_centroid and best_sim > 0.6:
            # Add to existing centroid
            if ep.payload not in best_centroid.members:
                best_centroid.members.append(ep.payload)
                # Update centroid vector (bundle with new episode)
                from .hd import bundle
                best_centroid.vector = bundle([best_centroid.vector, ep.vector])
                best_centroid.co_activation_count += 1
            return None

        # Create new centroid if we have enough similar episodes
        # Find similar episodes
        similar = self.agent.assoc.retrieve_similar(ep.vector, top_k=3)
        similar = [t for t, s in similar if s > 0.5]
        if len(similar) >= 2:
            from .hd import bundle
            # Build centroid vector
            episode_vecs = [self.agent.encoder.encode_text(t) for t in similar]
            centroid_vec = bundle(episode_vecs)
            # Generate a name (combine key tokens)
            name = self._name_centroid(similar)
            centroid = ConceptCentroid(
                name=name, vector=centroid_vec, members=similar,
                co_activation_count=len(similar),
            )
            self.centroids.append(centroid)
            # Register as a new concept in vocab
            self.agent.assoc.vocab[name.lower()] = centroid_vec
            log.info(f"new centroid: {name} ({len(similar)} members)")
            return centroid
        return None

    def _name_centroid(self, members: List[str]) -> str:
        """Generate a name for a centroid based on its members."""
        # Extract common tokens
        from collections import Counter
        all_tokens = []
        for m in members:
            all_tokens.extend(m.lower().split())
        common = Counter(all_tokens).most_common(3)
        if common:
            return "_".join(t for t, _ in common if len(t) > 2)[:30]
        return f"concept_{len(self.centroids)}"

    def _try_transitive_discovery(self) -> Optional[Tuple[str, str, str]]:
        """Find new triples by transitive closure.

        If (A, R, B) and (B, R, C) exist but (A, R, C) doesn't, infer it.
        """
        triples = self.agent.assoc.list_triples()
        # Build a graph per predicate
        graph: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for s, p, o in triples:
            graph[(p.lower(), s.lower())].append(o.lower())

        # Try random pairs for transitivity
        if len(triples) < 2: return None
        s1, p1, o1 = self.rng.choice(triples)
        # Find (o1, p1, ?)
        nexts = graph.get((p1.lower(), o1.lower()), [])
        for o2 in nexts:
            if o2.lower() == s1.lower(): continue  # cycle
            # Check if (s1, p1, o2) already exists
            exists = any(ts.lower() == s1.lower() and tp.lower() == p1.lower() and to.lower() == o2.lower()
                        for ts, tp, to in triples)
            if not exists:
                # Check consistency
                consistent = not any(ts.lower() == s1.lower() and tp.lower() == p1.lower()
                                    and to.lower() != o2.lower()
                                    for ts, tp, to in triples)
                if consistent:
                    self.agent.assoc.learn_triple(s1, p1, o2)
                    return (s1, p1, o2)
        return None

    # ------------------------------------------------------------------ #
    # REM: creative recombination
    # ------------------------------------------------------------------ #
    def _dream_rem(self, cycles: int) -> DreamReportV2:
        report = DreamReportV2(
            phase="REM", cycles=cycles, episodes_replayed=0,
            hebbian_links_formed=[], centroids_created=[],
            centroids_strengthened=0, new_triples_discovered=[],
            hypotheses_generated=[], creative_insights=[],
        )
        triples = self.agent.assoc.list_triples()
        if len(triples) < 2: return report

        for cycle in range(cycles):
            # 1. Pick two random triples
            t1, t2 = self.rng.sample(triples, 2)
            s1, p1, o1 = t1
            s2, p2, o2 = t2

            # 2. Creative recombination: swap subjects, objects, predicates
            insight = None
            r = self.rng.random()
            if r < 0.33:
                # Subject swap: what if (s1, p2, o2)?
                if not self._triple_exists(s1, p2, o2):
                    insight = f"what if {s1} {p2.replace('_',' ')} {o2}?"
            elif r < 0.66:
                # Object swap: what if (s1, p1, o2)?
                if not self._triple_exists(s1, p1, o2):
                    insight = f"what if {s1} {p1.replace('_',' ')} {o2}?"
            else:
                # Cross product: what if (o1, p2, s2)?
                if not self._triple_exists(o1, p2, s2):
                    insight = f"what if {o1} {p2.replace('_',' ')} {s2}?"

            if insight and insight not in report.creative_insights:
                report.creative_insights.append(insight)
                # Test the insight (does it contradict existing KB?)
                if self._is_novel_and_consistent(insight):
                    report.hypotheses_generated.append(insight)

        return report

    def _triple_exists(self, s: str, p: str, o: str) -> bool:
        for ts, tp, to in self.agent.assoc.list_triples():
            if ts.lower() == s.lower() and tp.lower() == p.lower() and to.lower() == o.lower():
                return True
        return False

    def _is_novel_and_consistent(self, hypothesis: str) -> bool:
        """Check if a hypothesis is novel and doesn't contradict existing facts."""
        # Simple check: hypothesis should mention at least one known entity
        triples = self.agent.assoc.list_triples()
        known_entities = set()
        for s, p, o in triples:
            known_entities.add(s.lower())
            known_entities.add(o.lower())
        for ent in known_entities:
            if ent in hypothesis.lower():
                return True
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def idle_consolidate(self, idle_threshold_s: float = 5.0, cycles: int = 10) -> Optional[DreamReportV2]:
        """If idle for the threshold, run a brief consolidation."""
        if not self.is_idle(idle_threshold_s): return None
        return self.dream(cycles=cycles, phase="mixed")

    def summary(self) -> Dict[str, Any]:
        if self.last_dream is None:
            return {"dreams_run": 0, "centroids": len(self.centroids),
                    "co_activations": len(self.co_activations)}
        return {
            "dreams_run": 1,
            "phase": self.last_dream.phase,
            "cycles": self.last_dream.cycles,
            "episodes_replayed": self.last_dream.episodes_replayed,
            "hebbian_links": len(self.last_dream.hebbian_links_formed),
            "centroids_created": len(self.last_dream.centroids_created),
            "new_triples": len(self.last_dream.new_triples_discovered),
            "creative_insights": len(self.last_dream.creative_insights),
            "total_centroids": len(self.centroids),
            "total_co_activations": len(self.co_activations),
            "duration_ms": self.last_dream.duration_ms,
        }
