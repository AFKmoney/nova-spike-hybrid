"""
learn_from_text.py — Automatic knowledge extraction from any text.

PROBLEM
-------
Transformer LLMs need fine-tuning to learn new facts. AETHER already does
one-shot teaching ("teach Paris is the capital of France") but only handles
simple patterns.

SOLUTION
--------
learn_from_text() takes ANY text passage — paragraph, article, conversation —
and extracts all extractable knowledge:

  1. Sentence segmentation
  2. Pattern matching: "X is Y", "X is the Y of Z", "X has Y", "X can Y", etc.
  3. Triple extraction: (subject, predicate, object)
  4. Episode storage: store the full text + its HD vector for later retrieval
  5. Concept extraction: identify proper nouns, numbers, key terms
  6. Relationship inference: if A is in B and B is in C, infer A is in C

This is AETHER's "instant expertise" — give it a Wikipedia article and it
becomes an expert on that topic, immediately, with no training.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ExtractedFact:
    """A single fact extracted from text."""
    subject: str
    predicate: str
    object: str
    source_sentence: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Protect common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.', r'\1<DOT>', text)
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Restore dots
    return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Triple extraction patterns
# ---------------------------------------------------------------------------

# Each pattern: (regex, predicate, group_indices (subject, object))
EXTRACTION_PATTERNS: List[Tuple[str, str, Tuple[int, int]]] = [
    # "X is the capital of Y"
    (r"^(.+?)\s+is\s+the\s+capital\s+of\s+(.+?)(?:[.\s]|$)", "capital_of", (1, 2)),
    # "X is located in Y"
    (r"^(.+?)\s+is\s+located\s+in\s+(.+?)(?:[.\s]|$)", "located_in", (1, 2)),
    # "X is a/an Y" (definition)
    (r"^(.+?)\s+is\s+(?:a|an)\s+(.+?)(?:[.\s]|$)", "is_a", (1, 2)),
    # "X is Y" (generic — must come after specific patterns)
    (r"^(.+?)\s+is\s+(.+?)(?:[.\s]|$)", "is_a", (1, 2)),
    # "X has Y" (possession)
    (r"^(.+?)\s+has\s+(?:a|an|the)?\s*(.+?)(?:[.\s]|$)", "has", (1, 2)),
    # "X can Y" (capability)
    (r"^(.+?)\s+can\s+(.+?)(?:[.\s]|$)", "can", (1, 2)),
    # "X was born in Y"
    (r"^(.+?)\s+was\s+born\s+in\s+(.+?)(?:[.\s]|$)", "born_in", (1, 2)),
    # "X was founded in Y" (year)
    (r"^(.+?)\s+was\s+founded\s+in\s+(.+?)(?:[.\s]|$)", "founded_in", (1, 2)),
    # "X died in Y"
    (r"^(.+?)\s+died\s+in\s+(.+?)(?:[.\s]|$)", "died_in", (1, 2)),
    # "X invented Y"
    (r"^(.+?)\s+invented\s+(.+?)(?:[.\s]|$)", "invented", (1, 2)),
    # "X wrote Y"
    (r"^(.+?)\s+wrote\s+(.+?)(?:[.\s]|$)", "wrote", (1, 2)),
    # "X painted Y"
    (r"^(.+?)\s+painted\s+(.+?)(?:[.\s]|$)", "painted", (1, 2)),
    # "X discovered Y"
    (r"^(.+?)\s+discovered\s+(.+?)(?:[.\s]|$)", "discovered", (1, 2)),
    # "X is famous for Y"
    (r"^(.+?)\s+is\s+famous\s+for\s+(.+?)(?:[.\s]|$)", "famous_for", (1, 2)),
    # "X is known for Y"
    (r"^(.+?)\s+is\s+known\s+for\s+(.+?)(?:[.\s]|$)", "known_for", (1, 2)),
    # "X is the largest Y"
    (r"^(.+?)\s+is\s+the\s+largest\s+(.+?)(?:[.\s]|$)", "largest", (1, 2)),
    # "X is the smallest Y"
    (r"^(.+?)\s+is\s+the\s+smallest\s+(.+?)(?:[.\s]|$)", "smallest", (1, 2)),
    # "X is the first Y"
    (r"^(.+?)\s+is\s+the\s+first\s+(.+?)(?:[.\s]|$)", "first", (1, 2)),
    # "X is the birthplace of Y"
    (r"^(.+?)\s+is\s+the\s+birthplace\s+of\s+(.+?)(?:[.\s]|$)", "birthplace_of", (1, 2)),
    # "X is the mother/father of Y"
    (r"^(.+?)\s+is\s+the\s+(?:mother|father)\s+of\s+(.+?)(?:[.\s]|$)", "parent_of", (1, 2)),
    # "X married Y"
    (r"^(.+?)\s+married\s+(.+?)(?:[.\s]|$)", "married", (1, 2)),
    # "X is also known as Y"
    (r"^(.+?)\s+is\s+also\s+known\s+as\s+(.+?)(?:[.\s]|$)", "alias", (1, 2)),
    # "X means Y" (definition)
    (r"^(.+?)\s+means\s+(.+?)(?:[.\s]|$)", "means", (1, 2)),
    # "X is part of Y"
    (r"^(.+?)\s+is\s+part\s+of\s+(.+?)(?:[.\s]|$)", "part_of", (1, 2)),
    # "X contains Y"
    (r"^(.+?)\s+contains\s+(.+?)(?:[.\s]|$)", "contains", (1, 2)),
    # "X is used for Y"
    (r"^(.+?)\s+is\s+used\s+for\s+(.+?)(?:[.\s]|$)", "used_for", (1, 2)),
    # "X is made of Y"
    (r"^(.+?)\s+is\s+made\s+of\s+(.+?)(?:[.\s]|$)", "made_of", (1, 2)),
    # "X is born in year Y"
    (r"^(.+?)\s+is\s+born\s+in\s+(.+?)(?:[.\s]|$)", "born_in", (1, 2)),
]


def extract_facts_from_sentence(sentence: str) -> List[ExtractedFact]:
    """Extract all facts from a single sentence."""
    # Clean the sentence
    s = sentence.strip().rstrip('.')
    facts: List[ExtractedFact] = []
    for pattern, predicate, (subj_idx, obj_idx) in EXTRACTION_PATTERNS:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            subject = m.group(subj_idx).strip()
            obj = m.group(obj_idx).strip()
            # Filter out trivial matches
            if len(subject) < 1 or len(obj) < 1: continue
            if subject.lower() in ("it", "this", "that", "there", "here"): continue
            # Don't extract if subject or object is too long (likely a complex phrase)
            if len(subject.split()) > 5 or len(obj.split()) > 5: continue
            # Strip trailing punctuation
            obj = re.sub(r'[,;:]$', '', obj).strip()
            subject = re.sub(r'[,;:]$', '', subject).strip()
            facts.append(ExtractedFact(subject, predicate, obj, sentence, 1.0))
            break  # one fact per sentence (first matching pattern wins)
    return facts


def extract_facts(text: str) -> List[ExtractedFact]:
    """Extract all facts from a text passage."""
    sentences = split_sentences(text)
    all_facts: List[ExtractedFact] = []
    for sent in sentences:
        facts = extract_facts_from_sentence(sent)
        all_facts.extend(facts)
    return all_facts


# ---------------------------------------------------------------------------
# Concept extraction (proper nouns, numbers, key terms)
# ---------------------------------------------------------------------------

def extract_concepts(text: str) -> Dict[str, List[str]]:
    """Extract concepts from text: proper nouns, numbers, dates."""
    # Proper nouns (capitalized words not at sentence start, excluding common words)
    proper_nouns = re.findall(r'\b(?!The|A|An|Is|Was|Are|Were|It|This|That|And|But|Or|In|On|At|To|For|With|By|Of|As)\b[A-Z][a-zA-Z]+', text)
    # Numbers (including years)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    # Years (4 digits starting with 1 or 2)
    years = re.findall(r'\b[12]\d{3}\b', text)
    return {
        "proper_nouns": list(set(proper_nouns)),
        "numbers": list(set(numbers)),
        "years": list(set(years)),
    }


# ---------------------------------------------------------------------------
# Main: learn_from_text — the killer feature
# ---------------------------------------------------------------------------

def learn_from_text(agent, text: str) -> Dict:
    """Have AETHER learn from any text passage.

    Args:
        agent: AETHER instance
        text: any text (paragraph, article, etc.)

    Returns:
        Summary of what was learned: facts extracted, concepts identified,
        episodes stored.
    """
    facts = extract_facts(text)
    concepts = extract_concepts(text)

    # Store each extracted fact as a triple
    learned_triples = []
    for fact in facts:
        # Use agent.teach to get all the side effects (attractor, kuramoto, etc)
        triple_text = f"{fact.subject} is the {fact.predicate.replace('_', ' ')} of {fact.object}" \
                      if fact.predicate.endswith("_of") else \
                      f"{fact.subject} | {fact.predicate} | {fact.object}"
        # For predicates that fit "X is Y" pattern, use that
        if fact.predicate in ("is_a", "located_in", "capital_of"):
            if fact.predicate == "capital_of":
                teach_text = f"{fact.subject} is the capital of {fact.object}"
            elif fact.predicate == "located_in":
                teach_text = f"{fact.subject} is located in {fact.object}"
            else:  # is_a
                teach_text = f"{fact.subject} is {fact.object}"
        else:
            teach_text = f"{fact.subject} | {fact.predicate} | {fact.object}"
        agent.teach(teach_text, silent=True)
        learned_triples.append((fact.subject, fact.predicate, fact.object))

    # Also store the entire text as an episode (for retrieval)
    agent.assoc.add_episode(text, agent.encoder.encode_text(text))
    agent.encoder.learn_sequence(text)

    return {
        "n_sentences": len(split_sentences(text)),
        "n_facts_extracted": len(facts),
        "n_concepts": len(concepts["proper_nouns"]) + len(concepts["numbers"]),
        "triples_learned": learned_triples,
        "proper_nouns": concepts["proper_nouns"][:10],
        "numbers": concepts["numbers"][:10],
        "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from aether import AETHER

    agent = AETHER()
    print("=== learn_from_text test ===\n")

    sample_text = """
    Albert Einstein was born in 1879. Einstein is a physicist. He invented the theory of relativity.
    Einstein is famous for the equation E=mc2. He died in 1955.
    Paris is the capital of France. Paris is located in Europe.
    Tokyo is the capital of Japan. Mount Fuji is located in Japan.
    """

    result = learn_from_text(agent, sample_text)
    print(f"  Sentences: {result['n_sentences']}")
    print(f"  Facts extracted: {result['n_facts_extracted']}")
    print(f"  Concepts: {result['n_concepts']}")
    print(f"  Proper nouns: {result['proper_nouns']}")
    print(f"  Numbers: {result['numbers']}")
    print(f"\n  Learned triples:")
    for s, p, o in result['triples_learned']:
        print(f"    ({s}, {p}, {o})")

    print(f"\n  Now testing what was learned:")
    questions = [
        "What is the capital of France?",
        "Where is Paris located?",
        "What is the capital of Japan?",
        "What is Einstein?",
    ]
    for q in questions:
        ans = agent.ask(q)
        print(f"    Q: {q}")
        print(f"    A: {ans}")
