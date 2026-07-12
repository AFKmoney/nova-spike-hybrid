"""
learn_from_text_v2.py — Enhanced text learning: coreference + entity linking.

IMPROVEMENTS OVER v1
--------------------
v1 extracted triples sentence-by-sentence without resolving pronouns.
"He invented the theory of relativity" extracted (He, invented, the)
which is useless.

v2 implements:

  1. COREFERENCE RESOLUTION
     - Track the last-mentioned entity as the "current subject"
     - Replace pronouns (he, she, it, they, this, that) with the
       current subject before extracting triples
     - "Einstein was born in 1879. He invented relativity." →
       (Einstein, invented, relativity)

  2. ENTITY LINKING
     - Detect multi-word entities ("Albert Einstein", "Mount Fuji")
     - Detect dates, numbers, and proper nouns
     - Link entities to existing KB entries when possible

  3. RICHER TRIPLE PATTERNS
     - Handle "X was born in Y", "X died in Y", "X wrote Y"
     - Handle "X, a Y, ..." apposition
     - Handle "X (Y) ..." parenthetical definitions

  4. SECTION AWARENESS
     - When given a multi-paragraph text, segment into sections
     - Tag triples with their source section
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

from .learn_from_text import ExtractedFact, split_sentences, extract_concepts


# Pronouns to resolve
PRONOUNS = {"he", "she", "it", "they", "this", "that", "these", "those", "his", "her", "its", "their"}


@dataclass
class ExtractedFactV2(ExtractedFact):
    """Enhanced fact with coreference info."""
    source_paragraph: int = 0
    resolved_pronoun: bool = False
    linked_entity: Optional[str] = None  # if entity was linked to existing KB


class TextLearnerV2:
    """Enhanced text learner with coreference resolution."""

    def __init__(self, agent):
        self.agent = agent
        self.current_subject: Optional[str] = None
        self.current_paragraph: int = 0

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def learn(self, text: str) -> Dict:
        """Learn from a text passage with coreference + entity linking."""
        # Reset state
        self.current_subject = None
        self.current_paragraph = 0

        # Split into paragraphs (double newlines)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        all_facts: List[ExtractedFactV2] = []
        all_concepts: Dict[str, List[str]] = {"proper_nouns": [], "numbers": [], "years": []}

        for para_idx, para in enumerate(paragraphs):
            self.current_paragraph = para_idx
            # Track the first proper noun as the current subject
            sentences = split_sentences(para)
            for sent in sentences:
                # Resolve pronouns
                resolved_sent, resolved = self._resolve_pronouns(sent)
                # Extract facts
                facts = self._extract_facts_v2(resolved_sent, sent, para_idx, resolved)
                all_facts.extend(facts)
                # Update current subject from this sentence
                new_subj = self._extract_subject(sent)
                if new_subj:
                    self.current_subject = new_subj
            # Aggregate concepts
            concepts = extract_concepts(para)
            for k in all_concepts:
                all_concepts[k].extend(concepts.get(k, []))

        # Deduplicate concepts
        for k in all_concepts:
            all_concepts[k] = list(set(all_concepts[k]))

        # Store all facts as triples
        learned_triples = []
        for fact in all_facts:
            teach_text = self._fact_to_teach_text(fact)
            if teach_text:
                self.agent.teach(teach_text, silent=True)
                learned_triples.append((fact.subject, fact.predicate, fact.object))

        # Also store the full text as an episode
        self.agent.assoc.add_episode(text, self.agent.encoder.encode_text(text))
        self.agent.encoder.learn_sequence(text)

        return {
            "n_paragraphs": len(paragraphs),
            "n_sentences": sum(len(split_sentences(p)) for p in paragraphs),
            "n_facts_extracted": len(all_facts),
            "n_pronouns_resolved": sum(1 for f in all_facts if f.resolved_pronoun),
            "n_linked_entities": sum(1 for f in all_facts if f.linked_entity),
            "triples_learned": learned_triples,
            "proper_nouns": all_concepts["proper_nouns"][:15],
            "numbers": all_concepts["numbers"][:10],
            "years": all_concepts["years"][:10],
            "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
        }

    # ------------------------------------------------------------------ #
    # Coreference resolution
    # ------------------------------------------------------------------ #
    def _resolve_pronouns(self, sentence: str) -> Tuple[str, bool]:
        """Replace pronouns with the current subject.

        Returns (resolved_sentence, was_resolved).
        """
        if not self.current_subject:
            return sentence, False
        words = sentence.split()
        resolved_any = False
        result = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if clean in PRONOUNS:
                # Preserve capitalization and punctuation
                punct = re.search(r'[^a-zA-Z]*$', w).group(0) if re.search(r'[^a-zA-Z]*$', w) else ""
                # Capitalize if pronoun was capitalized
                if w[0].isupper():
                    result.append(self.current_subject.split()[0].capitalize() + punct)
                else:
                    result.append(self.current_subject.lower() + punct)
                resolved_any = True
            else:
                result.append(w)
        return " ".join(result), resolved_any

    def _extract_subject(self, sentence: str) -> Optional[str]:
        """Extract the subject of a sentence (first proper noun or noun)."""
        # Try proper nouns first
        proper = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', sentence)
        # Filter out sentence-initial capitalized words that aren't proper nouns
        stopwords = {"The", "A", "An", "It", "He", "She", "They", "This", "That",
                    "These", "Those", "What", "Where", "When", "Who", "How", "Why",
                    "Is", "Was", "Are", "Were", "In", "On", "At", "Of", "To", "For"}
        for p in proper:
            if p not in stopwords and len(p) > 1:
                return p
        return None

    # ------------------------------------------------------------------ #
    # Fact extraction with coreference
    # ------------------------------------------------------------------ #
    def _extract_facts_v2(self, resolved_sent: str, original_sent: str,
                         para_idx: int, pronoun_resolved: bool) -> List[ExtractedFactV2]:
        """Extract facts from a (possibly resolved) sentence."""
        from .learn_from_text import EXTRACTION_PATTERNS
        s = resolved_sent.strip().rstrip('.')
        facts: List[ExtractedFactV2] = []
        for pattern, predicate, (subj_idx, obj_idx) in EXTRACTION_PATTERNS:
            m = re.match(pattern, s, re.IGNORECASE)
            if m:
                subject = m.group(subj_idx).strip()
                obj = m.group(obj_idx).strip()
                if len(subject) < 1 or len(obj) < 1: continue
                if subject.lower() in ("it", "this", "that", "there", "here"): continue
                if len(subject.split()) > 5 or len(obj.split()) > 5: continue
                obj = re.sub(r'[,;:]$', '', obj).strip()
                subject = re.sub(r'[,;:]$', '', subject).strip()
                # Try to link entity to existing KB
                linked = self._link_entity(subject)
                facts.append(ExtractedFactV2(
                    subject=subject, predicate=predicate, object=obj,
                    source_sentence=original_sent,
                    source_paragraph=para_idx,
                    resolved_pronoun=pronoun_resolved,
                    linked_entity=linked,
                ))
                break
        return facts

    def _link_entity(self, entity: str) -> Optional[str]:
        """Try to link an entity to an existing KB entry."""
        # Check if the entity already exists as a subject
        for s, p, o in self.agent.assoc.list_triples():
            if s.lower() == entity.lower():
                return s
        return None

    def _fact_to_teach_text(self, fact: ExtractedFactV2) -> str:
        """Convert a fact to a teach string."""
        if fact.predicate == "capital_of":
            return f"{fact.subject} is the capital of {fact.object}"
        elif fact.predicate == "located_in":
            return f"{fact.subject} is located in {fact.object}"
        elif fact.predicate == "is_a":
            return f"{fact.subject} is {fact.object}"
        else:
            return f"{fact.subject} | {fact.predicate} | {fact.object}"
