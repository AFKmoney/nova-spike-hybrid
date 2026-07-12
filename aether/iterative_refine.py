"""
iterative_refine.py — Generate → re-encode → correct (iterative refinement).

PROBLEM
-------
The current generator produces output in a single pass with no self-
correction. A human writes a draft, re-reads it, and corrects. AETHER
should do the same.

SOLUTION
--------
IterativeRefiner implements a multi-pass generation:

  Pass 1: Generate a draft using the standard pipeline
  Pass 2: Re-encode the draft as HD vector → retrieve similar memories
          → identify inconsistencies → generate corrections
  Pass 3: Apply corrections → produce final output

Each pass improves quality. 2-3 passes converge to a stable answer.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import re

log = logging.getLogger(__name__)


@dataclass
class RefinementPass:
    """One pass of iterative refinement."""
    pass_number: int
    draft: str
    issues_found: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    final_text: str = ""
    confidence: float = 0.0


@dataclass
class RefinementResult:
    """Result of iterative refinement."""
    passes: List[RefinementPass] = field(default_factory=list)
    final_text: str = ""
    final_confidence: float = 0.0
    n_passes: int = 0
    improved: bool = False


class IterativeRefiner:
    """Multi-pass generate → re-encode → correct."""

    def __init__(self, agent, max_passes: int = 3, confidence_threshold: float = 0.8):
        self.agent = agent
        self.max_passes = max_passes
        self.confidence_threshold = confidence_threshold

    def refine(self, question: str, initial_answer: str) -> RefinementResult:
        """Refine an initial answer through multiple passes."""
        result = RefinementResult()
        current_text = initial_answer
        current_confidence = 0.5  # start neutral

        for pass_num in range(1, self.max_passes + 1):
            # 1. Re-encode the current draft
            draft_vec = self.agent.encoder.encode_text(current_text)

            # 2. Retrieve similar memories
            retrieved = self.agent.assoc.retrieve_similar(draft_vec, top_k=3)

            # 3. Identify issues
            issues = self._find_issues(current_text, retrieved)

            # 4. Generate corrections
            corrections = self._generate_corrections(question, current_text, issues, retrieved)

            # 5. Apply corrections
            corrected = self._apply_corrections(current_text, corrections)

            # 6. Compute new confidence
            new_confidence = self._compute_confidence(corrected, retrieved)

            pass_result = RefinementPass(
                pass_number=pass_num,
                draft=current_text,
                issues_found=issues,
                corrections=corrections,
                final_text=corrected,
                confidence=new_confidence,
            )
            result.passes.append(pass_result)

            # Check convergence
            if new_confidence >= self.confidence_threshold:
                current_text = corrected
                current_confidence = new_confidence
                break
            if corrected == current_text:
                # No changes — converged
                break
            current_text = corrected
            current_confidence = new_confidence

        result.final_text = current_text
        result.final_confidence = current_confidence
        result.n_passes = len(result.passes)
        result.improved = result.final_confidence > 0.5 and result.final_text != initial_answer
        return result

    # ------------------------------------------------------------------ #
    # Issue detection
    # ------------------------------------------------------------------ #
    def _find_issues(self, text: str, retrieved: List[Tuple[str, float]]) -> List[str]:
        """Identify issues in the current draft."""
        issues = []
        # Check 1: empty or too short
        if len(text.strip()) < 3:
            issues.append("answer is too short")
        # Check 2: contains "I don't know" pattern
        if re.search(r"i don't know|couldn't find|no answer", text, re.I):
            issues.append("answer expresses uncertainty")
        # Check 3: contains error patterns
        if re.search(r"error|undefined|None|null", text, re.I):
            issues.append("answer contains error indicators")
        # Check 4: retrieved memories contradict the answer
        for memory_text, sim in retrieved:
            if sim > 0.5 and self._contradicts(text, memory_text):
                issues.append(f"contradicts retrieved memory: {memory_text[:50]}")
        return issues

    def _contradicts(self, answer: str, memory: str) -> bool:
        """Heuristic: does the answer contradict the memory?"""
        # Simple check: if memory says "X is Y" and answer says "X is Z"
        # (different Y and Z), it's a contradiction
        m = re.match(r"(.+?) is (.+)", memory.lower())
        a = re.match(r"(.+?) is (.+)", answer.lower())
        if m and a and m.group(1) == a.group(1):
            return m.group(2) != a.group(2)
        return False

    # ------------------------------------------------------------------ #
    # Correction generation
    # ------------------------------------------------------------------ #
    def _generate_corrections(self, question: str, draft: str,
                             issues: List[str], retrieved: List[Tuple[str, float]]) -> List[str]:
        """Generate corrections for the identified issues."""
        corrections = []
        for issue in issues:
            if "uncertainty" in issue or "too short" in issue:
                # Try to find a better answer in retrieved memories
                # Lower the threshold to find any relevant memory
                for memory_text, sim in retrieved:
                    if sim > 0.1 and "i don't know" not in memory_text.lower():
                        corrections.append(f"replace with: {memory_text}")
                        break
                # If no good retrieved memory, try asking the agent directly
                if not corrections:
                    # Re-ask the question to get a fresh answer
                    fresh = self.agent.ask(question)
                    if fresh and "i don't know" not in fresh.lower() and "couldn't find" not in fresh.lower():
                        corrections.append(f"replace with: {fresh}")
            elif "error" in issue:
                # Regenerate by asking the agent
                fresh = self.agent.ask(question)
                if fresh and "error" not in fresh.lower():
                    corrections.append(f"replace with: {fresh}")
            elif "contradicts" in issue:
                # Find the contradicting memory and use it
                for memory_text, sim in retrieved:
                    if sim > 0.3:
                        corrections.append(f"use retrieved fact: {memory_text}")
                        break
        return corrections

    # ------------------------------------------------------------------ #
    # Apply corrections
    # ------------------------------------------------------------------ #
    def _apply_corrections(self, text: str, corrections: List[str]) -> str:
        """Apply corrections to the text."""
        if not corrections:
            return text
        # Simple strategy: if a correction says "replace with X", use X
        for correction in corrections:
            if correction.startswith("replace with:"):
                return correction[len("replace with:"):].strip()
            if correction.startswith("use retrieved fact:"):
                return correction[len("use retrieved fact:"):].strip()
        return text

    # ------------------------------------------------------------------ #
    # Confidence computation
    # ------------------------------------------------------------------ #
    def _compute_confidence(self, text: str, retrieved: List[Tuple[str, float]]) -> float:
        """Compute confidence in the current text."""
        if not text or len(text) < 3:
            return 0.1
        # Confidence = max similarity to retrieved memories
        if not retrieved:
            return 0.3
        max_sim = max(s for _, s in retrieved)
        # Penalize uncertainty patterns
        if re.search(r"i don't know|couldn't find|no answer", text, re.I):
            max_sim *= 0.3
        # Penalize error patterns
        if re.search(r"error|undefined|None|null", text, re.I):
            max_sim *= 0.2
        return min(1.0, max_sim)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "max_passes": self.max_passes,
            "confidence_threshold": self.confidence_threshold,
        }
