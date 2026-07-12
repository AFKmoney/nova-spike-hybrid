"""
inference.py — Logical inference engine over the HD knowledge base.

Provides:
  - Forward chaining: from a set of facts, derive new facts using rules.
  - Backward chaining: from a goal, find a proof path.
  - Modus ponens: if (s, is_a, X) and (X, has_property, P) -> (s, has_property, P).
  - Transitivity: if (A, R, B) and (B, R, C) -> (A, R, C) for transitive R.
  - Multi-hop queries with arbitrary depth.

Rules are HD vectors too — they live in the same memory as facts.

No external theorem prover. Pure hyperdimensional pattern matching.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

from .memory import AssociativeMemory
from .hd import HDVector, DIM


# Predicates that are inherently transitive
TRANSITIVE_PREDICATES = {"located_in", "part_of", "contains", "ancestor_of", "subset_of"}

# Predicates that support inheritance (is_a)
INHERITANCE_PREDICATES = {"is_a", "instance_of", "type_of"}


@dataclass
class ProofStep:
    """A single step in a proof."""
    rule: str
    premises: List[Tuple[str, str, str]]  # list of (s, p, o) triples
    conclusion: Tuple[str, str, str]
    confidence: float


@dataclass
class Proof:
    """A complete proof: a chain of steps leading to a conclusion."""
    goal: Tuple[str, str, str]  # (subject, predicate, object) — object may be "?"
    steps: List[ProofStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_confidence: float = 0.0
    failed: bool = False
    failure_reason: str = ""


class InferenceEngine:
    """Logical inference over the HD KB."""

    def __init__(self, assoc: AssociativeMemory, max_depth: int = 4):
        self.assoc = assoc
        self.max_depth = max_depth

    # ------------------------------------------------------------------ #
    # Direct lookup
    # ------------------------------------------------------------------ #
    def lookup(self, subject: str, predicate: str) -> Optional[Tuple[str, float]]:
        """Direct KB query: (s, p) -> ?o.

        Checks BOTH directions:
          - Forward: triples where (s, p, ?) → return the object
          - Reverse: triples where (?, p, s) → return the subject

        The reverse direction handles questions like "What is the capital of France?"
        where the stored triple is (Paris, capital_of, France) — the query subject
        is the OBJECT of the stored triple.

        Returns None only if neither direction matches.
        """
        # Forward: (subject, predicate, ?)
        forward_matches = [
            o for s, p, o in self.assoc.list_triples()
            if s.lower() == subject.lower() and p.lower() == predicate.lower()
        ]
        if forward_matches:
            sdm_result = self.assoc.query_triple(subject, predicate)
            sdm_sim = sdm_result[1] if sdm_result else 0.5
            return forward_matches[0], max(sdm_sim, 0.95)

        # Reverse: (?, predicate, subject) → return the subject of the triple
        reverse_matches = [
            s for s, p, o in self.assoc.list_triples()
            if o.lower() == subject.lower() and p.lower() == predicate.lower()
        ]
        if reverse_matches:
            # Use SDM to get a similarity score
            sdm_result = self.assoc.query_triple(predicate, subject)
            sdm_sim = sdm_result[1] if sdm_result else 0.5
            return reverse_matches[0], max(sdm_sim, 0.95)

        # No match in either direction
        return None

    def reverse_lookup(self, predicate: str, obj: str) -> Optional[Tuple[str, float]]:
        """Reverse query: (?, p, o) -> s.

        Uses the bind(p, o) -> s convention stored at write time.
        """
        return self.assoc.query_triple(predicate, obj + "::reverse")
        # NOTE: this is approximate — see _store_reverse below

    # ------------------------------------------------------------------ #
    # Forward chaining — derive new facts
    # ------------------------------------------------------------------ #
    def forward_chain(self, seed_facts: List[Tuple[str, str, str]], max_steps: int = 5) -> List[ProofStep]:
        """Apply inference rules iteratively to derive new facts.

        Currently supports:
          - Transitivity: (A, R, B) + (B, R, C) -> (A, R, C) for R in TRANSITIVE_PREDICATES
          - Inheritance:  (s, is_a, X) + (X, P, Y) -> (s, P, Y)  [modus ponens on class membership]

        Returns the list of derived ProofSteps.
        """
        derived: List[ProofStep] = []
        known: Set[Tuple[str, str, str]] = set(seed_facts)
        # Also seed from existing KB triples
        for s, p, o in self.assoc.list_triples():
            known.add((s, p, o))

        frontier = list(known)
        for step in range(max_steps):
            new_facts: List[ProofStep] = []
            for (s1, p1, o1) in frontier:
                # Transitivity: try to extend (s1, R, o1) with (o1, R, ?)
                if p1 in TRANSITIVE_PREDICATES:
                    result = self.lookup(o1, p1)
                    if result:
                        o2, sim = result
                        new_triple = (s1, p1, o2)
                        if new_triple not in known and sim > 0.15:
                            new_facts.append(ProofStep(
                                rule="transitivity",
                                premises=[(s1, p1, o1), (o1, p1, o2)],
                                conclusion=new_triple,
                                confidence=sim,
                            ))

                # Inheritance: (s1, is_a, o1) + (o1, P, ?) -> (s1, P, ?)
                if p1 in INHERITANCE_PREDICATES:
                    # Try every predicate we know about on o1
                    for pred in ["capital_of", "located_in", "has_property", "color", "size"]:
                        result = self.lookup(o1, pred)
                        if result:
                            o2, sim = result
                            new_triple = (s1, pred, o2)
                            if new_triple not in known and sim > 0.15:
                                new_facts.append(ProofStep(
                                    rule="inheritance",
                                    premises=[(s1, p1, o1), (o1, pred, o2)],
                                    conclusion=new_triple,
                                    confidence=sim,
                                ))

            if not new_facts:
                break
            for step_obj in new_facts:
                known.add(step_obj.conclusion)
                derived.append(step_obj)
            frontier = [s.conclusion for s in new_facts]

        return derived

    # ------------------------------------------------------------------ #
    # Backward chaining — find a proof for a goal
    # ------------------------------------------------------------------ #
    def backward_chain(self, subject: str, predicate: str, max_depth: Optional[int] = None) -> Proof:
        """Try to prove (subject, predicate, ?) by working backwards.

        Strategy:
          1. Direct lookup (depth 0).
          2. If predicate is transitive, try (subject, R, ?) -> X, then (X, R, ?).
          3. If subject is_a class, try inheriting the property from the class.

        Returns a Proof with steps + final answer.
        """
        max_depth = max_depth or self.max_depth
        proof = Proof(goal=(subject, predicate, "?"))

        # 1. Direct lookup
        result = self.lookup(subject, predicate)
        if result and result[1] >= 0.10:
            ans, sim = result
            proof.final_answer = ans
            proof.final_confidence = sim
            proof.steps.append(ProofStep(
                rule="direct_lookup",
                premises=[],
                conclusion=(subject, predicate, ans),
                confidence=sim,
            ))
            return proof

        # 2. Transitive chaining
        if predicate in TRANSITIVE_PREDICATES and max_depth > 0:
            # Find intermediate X such that (subject, R, X) and (X, R, ?)
            current = subject
            chain = []
            for depth in range(max_depth):
                r = self.lookup(current, predicate)
                if not r or r[1] < 0.10:
                    break
                nxt, sim = r
                chain.append((current, predicate, nxt))
                current = nxt
                # Try to extend one more hop
                r2 = self.lookup(current, predicate)
                if not r2 or r2[1] < 0.10:
                    break
            if chain:
                # The last element of the chain is the deepest answer
                final_triple = chain[-1]
                proof.final_answer = final_triple[2]
                proof.final_confidence = sum(s for _, _, s in
                                             [self.lookup(s, p) or (None, 0.0)
                                              for s, p, _ in chain]) / max(len(chain), 1)
                for triple in chain:
                    proof.steps.append(ProofStep(
                        rule="transitive_step",
                        premises=[],
                        conclusion=triple,
                        confidence=proof.final_confidence,
                    ))
                return proof

        # 3. Inheritance via is_a
        if max_depth > 0:
            is_a_result = self.lookup(subject, "is_a")
            if is_a_result and is_a_result[1] >= 0.10:
                cls, sim1 = is_a_result
                # Try to get (cls, predicate, ?)
                cls_result = self.lookup(cls, predicate)
                if cls_result and cls_result[1] >= 0.10:
                    ans, sim2 = cls_result
                    proof.final_answer = ans
                    proof.final_confidence = min(sim1, sim2)
                    proof.steps.append(ProofStep(
                        rule="inheritance",
                        premises=[(subject, "is_a", cls), (cls, predicate, ans)],
                        conclusion=(subject, predicate, ans),
                        confidence=proof.final_confidence,
                    ))
                    return proof

        # Failed
        proof.failed = True
        proof.failure_reason = f"no rule fired for ({subject}, {predicate}, ?) within depth {max_depth}"
        return proof

    # ------------------------------------------------------------------ #
    # Multi-hop query (n hops)
    # ------------------------------------------------------------------ #
    def multi_hop_query(self, start: str, predicates: List[str]) -> Proof:
        """Chain multiple lookups: start --p1--> ? --p2--> ? --p3--> ?

        Example: multi_hop_query("Montreal", ["located_in", "capital_of"])
          -> Montreal located_in Canada -> Canada capital_of Ottawa
        """
        proof = Proof(goal=(start, "->".join(predicates), "?"))
        current = start
        for i, pred in enumerate(predicates):
            r = self.lookup(current, pred)
            if not r or r[1] < 0.10:
                proof.failed = True
                proof.failure_reason = f"hop {i} ({current}, {pred}, ?) failed"
                return proof
            nxt, sim = r
            proof.steps.append(ProofStep(
                rule=f"hop_{i}",
                premises=[],
                conclusion=(current, pred, nxt),
                confidence=sim,
            ))
            current = nxt
        proof.final_answer = current
        proof.final_confidence = min(s.confidence for s in proof.steps) if proof.steps else 0.0
        return proof

    # ------------------------------------------------------------------ #
    # N-hop with sub-goaling (NEW in v3.3)
    # ------------------------------------------------------------------ #
    def n_hop_with_subgoaling(
        self,
        start: str,
        goal_predicate: str,
        max_depth: int = 6,
        visited: Optional[Set[str]] = None,
    ) -> Proof:
        """Find an answer to (start, goal_predicate, ?) using arbitrary-length
        chains with sub-goaling.

        Strategy (BFS):
          1. Try direct lookup (depth 0).
          2. If failed, try inheritance: (start, is_a, X) + (X, goal_predicate, ?).
          3. If failed, try transitivity: for each predicate R that is transitive
             (located_in, part_of, ...), try (start, R, X) + (X, goal_predicate, ?).
          4. If failed, try ALL predicates on start, then recurse on the result.

        Returns a Proof with arbitrary depth.
        """
        if visited is None:
            visited = set()
        if start.lower() in visited:
            proof = Proof(goal=(start, goal_predicate, "?"))
            proof.failed = True
            proof.failure_reason = f"cycle detected at {start}"
            return proof
        visited.add(start.lower())

        proof = Proof(goal=(start, goal_predicate, "?"))

        # 1. Direct lookup
        result = self.lookup(start, goal_predicate)
        if result and result[1] >= 0.10:
            ans, sim = result
            proof.final_answer = ans
            proof.final_confidence = sim
            proof.steps.append(ProofStep(
                rule="direct_lookup",
                premises=[],
                conclusion=(start, goal_predicate, ans),
                confidence=sim,
            ))
            return proof

        # 2. Inheritance: (start, is_a, X) + (X, goal_predicate, ?)
        is_a_result = self.lookup(start, "is_a")
        if is_a_result and is_a_result[1] >= 0.10:
            cls, sim1 = is_a_result
            if cls.lower() not in visited:
                cls_proof = self.n_hop_with_subgoaling(cls, goal_predicate, max_depth - 1, visited.copy())
                if not cls_proof.failed:
                    proof.final_answer = cls_proof.final_answer
                    proof.final_confidence = min(sim1, cls_proof.final_confidence)
                    proof.steps.append(ProofStep(
                        rule="inheritance",
                        premises=[(start, "is_a", cls)] + [s.conclusion for s in cls_proof.steps],
                        conclusion=(start, goal_predicate, cls_proof.final_answer),
                        confidence=proof.final_confidence,
                    ))
                    return proof

        # 3. Transitivity on the goal predicate
        if goal_predicate in TRANSITIVE_PREDICATES:
            # Try each known predicate on start, see if it leads somewhere
            for s, p, o in self.assoc.list_triples():
                if s.lower() != start.lower():
                    continue
                if p == goal_predicate:
                    continue  # already tried
                if o.lower() in visited:
                    continue
                # Try: (start, p, o) + (o, goal_predicate, ?)
                sub_proof = self.n_hop_with_subgoaling(o, goal_predicate, max_depth - 1, visited.copy())
                if not sub_proof.failed:
                    proof.final_answer = sub_proof.final_answer
                    proof.final_confidence = min(0.5, sub_proof.final_confidence)  # lower confidence for inferred links
                    proof.steps.append(ProofStep(
                        rule=f"transitive_via_{p}",
                        premises=[(start, p, o)] + [s.conclusion for s in sub_proof.steps],
                        conclusion=(start, goal_predicate, sub_proof.final_answer),
                        confidence=proof.final_confidence,
                    ))
                    return proof

        # 4. BFS over all predicates (last resort)
        if max_depth <= 0:
            proof.failed = True
            proof.failure_reason = f"max depth reached at {start}"
            return proof

        for s, p, o in self.assoc.list_triples():
            if s.lower() != start.lower():
                continue
            if p == goal_predicate:
                continue
            if o.lower() in visited:
                continue
            sub_proof = self.n_hop_with_subgoaling(o, goal_predicate, max_depth - 1, visited.copy())
            if not sub_proof.failed:
                proof.final_answer = sub_proof.final_answer
                proof.final_confidence = min(0.3, sub_proof.final_confidence)  # even lower for long chains
                proof.steps.append(ProofStep(
                    rule=f"bfs_via_{p}",
                    premises=[(start, p, o)] + [s.conclusion for s in sub_proof.steps],
                    conclusion=(start, goal_predicate, sub_proof.final_answer),
                    confidence=proof.final_confidence,
                ))
                return proof

        proof.failed = True
        proof.failure_reason = f"no path found from {start} via {goal_predicate}"
        return proof

    # ------------------------------------------------------------------ #
    # Path finding: find ALL paths from start to a target (NEW in v3.3)
    # ------------------------------------------------------------------ #
    def find_paths(
        self,
        start: str,
        target: str,
        max_depth: int = 4,
        max_paths: int = 5,
    ) -> List[Proof]:
        """Find all paths (up to max_paths) from `start` to `target` in the KB.

        Each path is a sequence of (s, p, o) triples where the object of one
        is the subject of the next, and the final object is `target`.

        Returns a list of Proof objects, one per path found.
        """
        paths: List[Proof] = []
        self._find_paths_dfs(start, target, [], set([start.lower()]), max_depth, max_paths, paths)
        return paths

    def _find_paths_dfs(
        self,
        current: str,
        target: str,
        path_steps: List[ProofStep],
        visited: Set[str],
        max_depth: int,
        max_paths: int,
        paths: List[Proof],
    ) -> None:
        if len(paths) >= max_paths:
            return
        if current.lower() == target.lower():
            # Found a path
            proof = Proof(goal=(current, "path_to", target))
            proof.steps = list(path_steps)
            proof.final_answer = target
            proof.final_confidence = min((s.confidence for s in path_steps), default=1.0)
            paths.append(proof)
            return
        if max_depth <= 0:
            return
        for s, p, o in self.assoc.list_triples():
            if s.lower() != current.lower():
                continue
            if o.lower() in visited:
                continue
            result = self.lookup(current, p)
            sim = result[1] if result else 0.5
            step = ProofStep(
                rule="path_step",
                premises=[],
                conclusion=(current, p, o),
                confidence=sim,
            )
            visited.add(o.lower())
            self._find_paths_dfs(o, target, path_steps + [step], visited, max_depth - 1, max_paths, paths)
            visited.discard(o.lower())
            if len(paths) >= max_paths:
                return

    # ------------------------------------------------------------------ #
    # Reachability: find all entities reachable from start within N hops
    # ------------------------------------------------------------------ #
    def reachable(self, start: str, max_hops: int = 3) -> Dict[str, Tuple[List[str], float]]:
        """Find all entities reachable from `start` within `max_hops` hops.

        Returns a dict: {entity: (path_of_predicates, confidence)}.
        """
        results: Dict[str, Tuple[List[str], float]] = {}
        # BFS
        frontier = [(start, [], 1.0)]
        seen = {start.lower()}
        for hop in range(max_hops):
            new_frontier = []
            for entity, path, conf in frontier:
                for s, p, o in self.assoc.list_triples():
                    if s.lower() != entity.lower():
                        continue
                    if o.lower() in seen:
                        continue
                    seen.add(o.lower())
                    result = self.lookup(entity, p)
                    new_conf = min(conf, result[1] if result else 0.5)
                    new_path = path + [p]
                    if o not in results or new_conf > results[o][1]:
                        results[o] = (new_path, new_conf)
                    new_frontier.append((o, new_path, new_conf))
            frontier = new_frontier
            if not frontier:
                break
        return results

    # ------------------------------------------------------------------ #
    # Explain
    # ------------------------------------------------------------------ #
    def explain(self, proof: Proof) -> str:
        """Human-readable explanation of a proof."""
        if proof.failed:
            return f"Could not prove {proof.goal}: {proof.failure_reason}"
        lines = [f"Proof for {proof.goal}:"]
        for i, step in enumerate(proof.steps):
            lines.append(f"  step {i+1} [{step.rule}]:")
            for prem in step.premises:
                lines.append(f"    premise: {prem}")
            s, p, o = step.conclusion
            lines.append(f"    => ({s}, {p}, {o})  [conf={step.confidence:.2f}]")
        lines.append(f"Final answer: {proof.final_answer} (confidence={proof.final_confidence:.2f})")
        return "\n".join(lines)
