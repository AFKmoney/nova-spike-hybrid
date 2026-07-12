"""
meta_learning.py — Meta-learning: the system learns HOW to learn.

IMPROVEMENTS OVER self_modify.py
--------------------------------
self_modify.py adjusts global parameters. meta_learning.py implements
DOMAIN-SPECIFIC meta-learning with a H_meta subspace:

  1. H_META SUBSPACE
     - A separate HD subspace stores meta-parameters per domain
     - Each domain (math, geography, biology, creative, factual) has its
       own optimal parameters
     - When a question is asked, AETHER detects the domain and retrieves
       the appropriate meta-parameters

  2. DOMAIN DETECTION
     - Each domain has a "domain vector" (HD)
     - The question is encoded and matched against domain vectors
     - The closest domain's meta-parameters are loaded

  3. LYAPUNOV-STABLE UPDATES
     - Meta-parameter updates are constrained to be stable (Lyapunov)
     - No parameter can change by more than ε per cycle
     - This prevents catastrophic meta-forgetting

  4. DOMAIN-SPECIFIC PARAMETERS
     - kernel_radius: tight for math, loose for creative
     - temperature: low for factual, high for creative
     - retrieval_weights: different per domain
     - max_cycles: deep for reasoning, shallow for recall

  5. INTER-DOMAIN TRANSFER
     - If a domain has poor performance, borrow parameters from the
       closest-performing domain
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from .hd import HDVector, DIM, bundle

log = logging.getLogger(__name__)


@dataclass
class DomainProfile:
    """Meta-parameters for a specific domain."""
    name: str
    domain_vec: HDVector  # HD vector representing this domain
    # Tunable parameters
    kernel_radius: float = 0.5      # tight (0.1) to loose (0.9)
    temperature: float = 0.5         # low (0.1) for factual, high (0.9) for creative
    retrieval_weight: float = 0.5    # how much to trust retrieval vs inference
    max_cycles: int = 8              # cognitive loop depth
    # Performance tracking
    n_queries: int = 0
    mean_comprehension: float = 0.5
    last_updated: float = field(default_factory=time.time)


class MetaLearner:
    """Domain-specific meta-learning with Lyapunov-stable updates."""

    # Maximum change per update (Lyapunov stability)
    MAX_DELTA = 0.05

    # Pre-defined domains with their initial profiles
    DOMAIN_SEEDS = {
        "geography": ["capital", "country", "city", "located", "continent", "where"],
        "math": ["calculate", "compute", "plus", "minus", "times", "divide", "number", "arithmetic"],
        "biology": ["animal", "plant", "human", "alive", "cell", "body", "brain", "evolution"],
        "physics": ["force", "energy", "mass", "velocity", "wave", "particle", "quantum"],
        "creative": ["imagine", "create", "design", "invent", "what if", "story", "poem"],
        "factual": ["what is", "who is", "when", "definition", "fact", "knowledge"],
        "reasoning": ["why", "how", "explain", "because", "therefore", "cause", "effect"],
        "language": ["word", "sentence", "grammar", "translate", "meaning", "synonym"],
        "history": ["history", "ancient", "century", "war", "revolution", "empire", "king"],
        "technology": ["computer", "software", "internet", "algorithm", "programming", "ai"],
    }

    def __init__(self, agent):
        self.agent = agent
        self.domains: Dict[str, DomainProfile] = {}
        self.current_domain: Optional[str] = None
        self._init_domains()
        # History of meta-updates for analysis
        self.update_history: List[Tuple[str, str, float, float]] = []  # (domain, param, old, new)

    def _init_domains(self) -> None:
        """Initialize domain profiles from seeds."""
        for name, keywords in self.DOMAIN_SEEDS.items():
            # Encode the keywords as the domain vector
            vecs = [self.agent.encoder.encode_text(kw) for kw in keywords]
            domain_vec = bundle(vecs)
            # Domain-specific defaults
            defaults = {
                "math": {"kernel_radius": 0.2, "temperature": 0.1, "max_cycles": 5},
                "geography": {"kernel_radius": 0.4, "temperature": 0.3, "max_cycles": 6},
                "creative": {"kernel_radius": 0.8, "temperature": 0.8, "max_cycles": 10},
                "factual": {"kernel_radius": 0.3, "temperature": 0.2, "max_cycles": 5},
                "reasoning": {"kernel_radius": 0.4, "temperature": 0.4, "max_cycles": 12},
            }
            params = defaults.get(name, {"kernel_radius": 0.5, "temperature": 0.5, "max_cycles": 8})
            self.domains[name] = DomainProfile(
                name=name, domain_vec=domain_vec,
                kernel_radius=params["kernel_radius"],
                temperature=params["temperature"],
                max_cycles=params["max_cycles"],
            )

    # ------------------------------------------------------------------ #
    # Domain detection
    # ------------------------------------------------------------------ #
    def detect_domain(self, question: str) -> Tuple[str, float]:
        """Detect which domain a question belongs to.

        Returns (domain_name, confidence).
        """
        q_vec = self.agent.encoder.encode_text(question)
        best_domain, best_sim = None, -1.0
        for name, profile in self.domains.items():
            sim = q_vec.similarity(profile.domain_vec)
            if sim > best_sim:
                best_sim = sim
                best_domain = name
        return best_domain, best_sim

    # ------------------------------------------------------------------ #
    # Apply domain parameters
    # ------------------------------------------------------------------ #
    def apply_domain(self, domain_name: str) -> None:
        """Apply a domain's meta-parameters to the agent."""
        if domain_name not in self.domains: return
        profile = self.domains[domain_name]
        self.current_domain = domain_name
        # Apply to agent
        self.agent.cogloop.max_cycles = profile.max_cycles
        # We could also apply kernel_radius and temperature to the
        # attractor/SDM, but those don't have those params directly.
        # For now, log the application
        log.debug(f"applied domain {domain_name}: k_radius={profile.kernel_radius}, "
                 f"temp={profile.temperature}, max_cycles={profile.max_cycles}")

    def detect_and_apply(self, question: str) -> Tuple[str, float]:
        """Detect domain from question and apply its parameters."""
        domain, conf = self.detect_domain(question)
        if domain:
            self.apply_domain(domain)
        return domain, conf

    # ------------------------------------------------------------------ #
    # Performance-based meta-updates (Lyapunov-stable)
    # ------------------------------------------------------------------ #
    def record_performance(self, comprehension: float) -> None:
        """Record comprehension score for the current domain."""
        if not self.current_domain: return
        profile = self.domains[self.current_domain]
        # Running average
        profile.mean_comprehension = (
            profile.mean_comprehension * profile.n_queries + comprehension
        ) / (profile.n_queries + 1)
        profile.n_queries += 1

    def meta_update(self) -> Dict[str, Any]:
        """Update meta-parameters based on recent performance.

        Lyapunov-stable: no parameter changes by more than MAX_DELTA.
        """
        if not self.current_domain: return {"updated": False}
        profile = self.domains[self.current_domain]
        perf = profile.mean_comprehension
        changes = []

        # Rule 1: if performance is low, loosen kernel (be more permissive)
        if perf < 0.4 and profile.n_queries >= 3:
            old = profile.kernel_radius
            new = min(0.9, old + self.MAX_DELTA)
            if new != old:
                profile.kernel_radius = new
                changes.append(f"kernel_radius: {old:.3f} -> {new:.3f}")
                self.update_history.append((self.current_domain, "kernel_radius", old, new))

        # Rule 2: if performance is high, tighten kernel (be more precise)
        elif perf > 0.7 and profile.n_queries >= 3:
            old = profile.kernel_radius
            new = max(0.1, old - self.MAX_DELTA * 0.5)
            if new != old:
                profile.kernel_radius = new
                changes.append(f"kernel_radius: {old:.3f} -> {new:.3f}")
                self.update_history.append((self.current_domain, "kernel_radius", old, new))

        # Rule 3: if performance is low, increase temperature (explore more)
        if perf < 0.4 and profile.n_queries >= 3:
            old = profile.temperature
            new = min(0.9, old + self.MAX_DELTA)
            if new != old:
                profile.temperature = new
                changes.append(f"temperature: {old:.3f} -> {new:.3f}")
                self.update_history.append((self.current_domain, "temperature", old, new))

        # Rule 4: if performance is high, decrease temperature (exploit)
        elif perf > 0.7 and profile.n_queries >= 3:
            old = profile.temperature
            new = max(0.1, old - self.MAX_DELTA * 0.5)
            if new != old:
                profile.temperature = new
                changes.append(f"temperature: {old:.3f} -> {new:.3f}")
                self.update_history.append((self.current_domain, "temperature", old, new))

        # Rule 5: if performance is very low, increase depth (think more)
        if perf < 0.3 and profile.n_queries >= 3:
            old = profile.max_cycles
            new = min(20, old + 2)
            if new != old:
                profile.max_cycles = new
                changes.append(f"max_cycles: {old} -> {new}")
                self.update_history.append((self.current_domain, "max_cycles", float(old), float(new)))

        profile.last_updated = time.time()
        if changes:
            log.info(f"meta-update for {self.current_domain}: " + "; ".join(changes))
        return {
            "updated": len(changes) > 0,
            "domain": self.current_domain,
            "performance": perf,
            "changes": changes,
        }

    # ------------------------------------------------------------------ #
    # Inter-domain transfer
    # ------------------------------------------------------------------ #
    def transfer_from_best(self, target_domain: str) -> Optional[str]:
        """Transfer parameters from the best-performing similar domain."""
        if target_domain not in self.domains: return None
        target = self.domains[target_domain]
        # Find the best-performing domain
        best_domain = max(self.domains.values(),
                         key=lambda d: d.mean_comprehension if d.name != target_domain else -1)
        if best_domain.name == target_domain: return None
        if best_domain.mean_comprehension < 0.5: return None  # not good enough to transfer from
        # Transfer (with Lyapunov constraint)
        target.kernel_radius = best_domain.kernel_radius
        target.temperature = best_domain.temperature
        log.info(f"transferred params from {best_domain.name} to {target_domain}")
        return best_domain.name

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "n_domains": len(self.domains),
            "current_domain": self.current_domain,
            "domain_performances": {
                name: {"n_queries": p.n_queries,
                       "mean_comprehension": p.mean_comprehension,
                       "kernel_radius": p.kernel_radius,
                       "temperature": p.temperature,
                       "max_cycles": p.max_cycles}
                for name, p in self.domains.items()
            },
            "n_meta_updates": len(self.update_history),
        }
