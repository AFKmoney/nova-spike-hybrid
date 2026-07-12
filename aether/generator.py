"""
generator.py — Natural language response generation.

The v1 AETHER returned either a bare token ("paris") or a verbatim episode
stored in memory. That's not a chatbot answer.

v2 generator takes a fact + question + context and produces a natural
language response by:
  1. Detecting the question type (capital-of, location, definition, calc, ...)
  2. Selecting an appropriate template
  3. Filling the slots from the retrieved fact(s)
  4. Variating slightly to avoid mechanical repetition

Templates are learned from the corpus the agent is taught (when you teach
"Paris is the capital of France", the generator reverse-engineers the
template "X is the capital of Y" and stores both slots).

No transformer, no LLM — just templated NLG over HD-retrieved facts.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
import random

from .hd import HDVector, DIM, bundle
from .memory import AssociativeMemory
from .encoder import TextEncoder, tokenize


# ---------------------------------------------------------------------------
# Question type classifier
# ---------------------------------------------------------------------------

# (regex, type, slots)
QUESTION_PATTERNS: List[Tuple[str, str, List[str]]] = [
    # Meta / conversational
    (r"^hello$|^hi$|^hey$|^bonjour$",                           "greeting",      []),
    (r"^bye$|^goodbye$|^quit$|^exit$",                          "farewell",      []),
    (r"^thank.*$|^thanks$|^merci$",                             "thanks",        []),
    (r"^help$|^\?$",                                            "help",          []),
    (r"^stats$|^status$",                                       "stats",         []),
    # Identity / capabilities / self-explanation
    (r"^what are you\??$|^who are you\??$",                     "identity",      []),
    (r"^what can you do\??$|^what do you do\??$|^help me\???$", "capabilities",  []),
    (r"^how do you work\??$|^how do you function\??$|^how are you different\??$|^explain yourself\??$", "self_explain", []),
    # KB queries — capital
    (r"^what is the capital of (.+)\??$",                       "capital_of",    ["country"]),
    (r"^what is capital of (.+)\??$",                           "capital_of",    ["country"]),
    (r"^capital of (.+)\???$",                                  "capital_of",    ["country"]),
    # KB queries — location
    (r"^where is (.+) located\???$",                            "located_in",    ["subject"]),
    (r"^where is (.+)\???$",                                    "located_in",    ["subject"]),
    # KB queries — generic predicate
    (r"^what is the (.+) of (.+)\???$",                         "predicate_of",  ["predicate", "subject"]),
    # KB queries — definition
    (r"^what is (.+)\???$",                                     "definition",    ["subject"]),
    (r"^who is (.+)\???$",                                      "definition",    ["subject"]),
    (r"^what does (.+) mean\???$",                              "definition",    ["subject"]),
    # Tools — calc (use a single combined pattern with named group)
    (r"^(?:calculate|calc|compute)\s+(.+)$",                    "calc",          ["expr"]),
    (r"^(\d[\d\s\+\-\*\/\(\)\.,%]+)$",                          "calc",          ["expr"]),
    # Tools — time
    (r"^what time.*$|^time$|^now$|^date$",                      "time",          []),
    # Tools — recall / remember
    (r"^recall (.+)$|^remember (.+)$",                          "recall",        ["query"]),
    # Tools — list KB
    (r"^list.*kb$|^list triples$|^show kb$|^list facts$",       "list_kb",       []),
    # Tools — explain
    (r"^explain (.+)$|^describe (.+)$|^tell me about (.+)$",    "explain",       ["subject"]),
    # Tools — compare
    (r"^(?:compare|difference between)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)$",
                                                                "compare",       ["x", "y"]),
    # Tools — summarize
    (r"^summarize\s*(\d*)$|^summary\s*(\d*)$",                  "summarize",     ["n"]),
    # Tools — count
    (r"^count\s+(\w+)$|^how many\s+(\w+)",                      "count",         ["what"]),
    # Tools — define
    (r"^define\s+(.+)$",                                        "define",        ["subject"]),
    # Teach (will be intercepted before reaching here, but include for safety)
    (r"^teach\s+(.+)$|^learn\s+(.+)$",                          "teach",         ["fact"]),
]


@dataclass
class QuestionAnalysis:
    qtype: str
    slots: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def is_question(self) -> bool:
        return self.qtype not in ("greeting", "farewell", "thanks", "help", "stats",
                                  "list_kb", "calc", "time", "recall", "identity",
                                  "capabilities", "self_explain")


def analyze_question(text: str) -> QuestionAnalysis:
    """Classify a user input into a question type + extract slots."""
    t = text.strip()
    lower = t.lower().rstrip("?.!")

    # SPECIAL CASE: multi-hop "capital of the country where X is located"
    # must be detected BEFORE the generic "capital_of" pattern.
    m = re.match(
        r"what is the capital of (?:the )?country where (.+?) is located\???",
        lower,
    )
    if m:
        return QuestionAnalysis(
            qtype="multi_hop_capital",
            slots={"place": m.group(1).strip()},
            raw=text,
        )

    # SPECIAL CASE: multi-hop "where is the capital of X located"
    m = re.match(
        r"where is the capital of (.+?) located\???",
        lower,
    )
    if m:
        return QuestionAnalysis(
            qtype="multi_hop_location",
            slots={"country": m.group(1).strip()},
            raw=text,
        )

    # SPECIAL CASE: comparison "compare X and Y" / "difference between X and Y"
    m = re.match(
        r"(?:compare|what is the difference between)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)\???$",
        lower,
    )
    if m:
        return QuestionAnalysis(
            qtype="compare",
            slots={"x": m.group(1).strip(), "y": m.group(2).strip()},
            raw=text,
        )

    for pattern, qtype, slots in QUESTION_PATTERNS:
        m = re.match(pattern, lower)
        if m:
            slot_values = {}
            # For patterns with alternations (multiple groups), pick the first non-None group
            groups = m.groups()
            non_none = [g for g in groups if g is not None]
            for i, slot_name in enumerate(slots):
                if i < len(non_none):
                    val = non_none[i]
                    if val:
                        slot_values[slot_name] = val.strip()
            return QuestionAnalysis(qtype=qtype, slots=slot_values, raw=text)
    return QuestionAnalysis(qtype="unknown", slots={}, raw=text)


# ---------------------------------------------------------------------------
# Templates for NLG
# ---------------------------------------------------------------------------

# qtype -> list of templates, each with {slot} placeholders
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "capital_of": [
        "The capital of {country} is {answer}.",
        "{answer} is the capital of {country}.",
        "It's {answer}.",
    ],
    "located_in": [
        "{subject} is located in {answer}.",
        "{subject} is in {answer}.",
        "{answer}.",
    ],
    "definition": [
        "{subject} is {answer}.",
        "{answer}.",
    ],
    "predicate_of": [
        "The {predicate} of {subject} is {answer}.",
        "{answer}.",
    ],
    "multi_hop_capital": [
        "The capital is {answer}.",
        "It's {answer}.",
        "{answer}.",
    ],
    "multi_hop_location": [
        "It's located in {answer}.",
        "{answer}.",
    ],
    "calc": [
        "{answer}",
        "The result is {answer}.",
    ],
    "time": [
        "It's currently {answer}.",
        "{answer}",
    ],
    "recall": [
        "Here's what I remember: {answer}",
        "I recall: {answer}",
    ],
    "list_kb": [
        "Here's what I know: {answer}",
    ],
    "identity": [
        "I am AETHER — a non-transformer, GPU-free, instant-learning cognitive agent built on hyperdimensional computing.",
        "I'm AETHER. I think with hyperdimensional vectors, remember with Kanerva SDM, and reason with a continuous cognitive loop. No transformer, no GPU, no external LLM.",
    ],
    "capabilities": [
        "I can answer questions from my knowledge base, learn new facts instantly (one-shot, no training), use tools like calc/time/recall/python, reason across multiple hops, and explain my thinking. Try: 'teach X is the capital of Y' then 'What is the capital of Y?'.",
    ],
    "self_explain": [
        "I work in three layers: (1) hyperdimensional vectors represent every concept, token, and rule; (2) a sparse distributed memory (Kanerva 1988) stores associations with O(1) one-shot writes — no epochs, no backprop; (3) a continuous cognitive loop iterates PERCEIVE→RETRIEVE→DELIBERATE→ACT until it converges on an answer. Tools are HD vectors too — when working memory matches a tool's trigger above a threshold, the tool fires. Everything runs in pure NumPy on CPU, in tens of milliseconds.",
    ],
    "greeting": [
        "Hello! I'm AETHER. Ask me anything, or teach me with 'X is the capital of Y'.",
        "Hi there! What would you like to know?",
    ],
    "farewell": [
        "Goodbye!",
        "See you later!",
    ],
    "thanks": [
        "You're welcome!",
        "Anytime.",
    ],
    "help": [
        "Try: 'teach X is the capital of Y' to learn a fact, then 'What is the capital of Y?' to query. Tools: calc, time, recall, list kb, python. Meta: stats, explain.",
    ],
    "stats": [
        "{answer}",
    ],
    "explain": [
        "{answer}",
    ],
    "compare": [
        "{answer}",
    ],
    "summarize": [
        "{answer}",
    ],
    "count": [
        "{answer}",
    ],
    "define": [
        "{answer}",
    ],
    "teach": [
        "{answer}",
    ],
    "unknown": [
        "I'm not sure I understood. Could you rephrase? You can teach me with 'X is Y' or ask 'What is the capital of Z?'.",
        "Hmm, I don't have an answer for that. Try teaching me the fact first.",
    ],
    "no_answer": [
        "I don't know the answer to that yet. Teach me with: 'X is the capital of Y'.",
        "I couldn't find that in my knowledge. You can teach me by saying 'X is the capital of Y'.",
    ],
}


# ---------------------------------------------------------------------------
# Response generator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """Generate natural language responses from retrieved facts."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate(
        self,
        question: str,
        answer: Optional[str] = None,
        analysis: Optional[QuestionAnalysis] = None,
        confidence: float = 1.0,
    ) -> str:
        """Generate a natural language response.

        Args:
            question: the raw user question
            answer: the bare fact retrieved (e.g., "paris") or None
            analysis: optional pre-computed question analysis
            confidence: retrieval confidence [0, 1]
        """
        if analysis is None:
            analysis = analyze_question(question)

        # If no answer retrieved, return a no_answer template
        if answer is None or confidence < 0.10:
            templates = RESPONSE_TEMPLATES.get("no_answer", ["I don't know."])
            return self.rng.choice(templates)

        # Pick a template for the question type
        templates = RESPONSE_TEMPLATES.get(analysis.qtype, ["{answer}."])
        template = self.rng.choice(templates)

        # Build the slot dict
        slots = dict(analysis.slots)
        slots["answer"] = self._beautify_answer(answer, analysis.qtype)

        # Fill the template, gracefully dropping missing slots
        try:
            response = template.format(**slots)
        except KeyError:
            # Fall back to bare answer
            response = answer

        # Capitalize first letter, ensure terminal punctuation
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        if response and response[-1] not in ".!?":
            response += "."

        return response

    def _beautify_answer(self, answer: str, qtype: str) -> str:
        """Clean up a retrieved answer for display."""
        # Strip start/end markers
        answer = answer.replace("<s>", "").replace("</s>", "").strip()
        # For single-token answers like "paris", capitalize proper nouns
        if qtype in ("capital_of", "located_in", "predicate_of") and " " not in answer:
            # Capitalize if it looks like a proper noun (single word, lowercase)
            answer = answer.capitalize()
        return answer


# ---------------------------------------------------------------------------
# Helper: extract (subject, predicate, object) from a "X is the Y of Z" sentence
# ---------------------------------------------------------------------------

def parse_triple(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse common natural-language triple patterns.

    Returns (subject, predicate, object) or None.
    """
    t = text.strip().rstrip(".")
    lower = t.lower()

    # "X is the capital of Y"
    m = re.match(r"(.+?)\s+is\s+the\s+capital\s+of\s+(.+)", lower)
    if m:
        return (m.group(1).strip(), "capital_of", m.group(2).strip())

    # "X is located in Y"
    m = re.match(r"(.+?)\s+is\s+located\s+in\s+(.+)", lower)
    if m:
        return (m.group(1).strip(), "located_in", m.group(2).strip())

    # "X is a Y" / "X is an Y"
    m = re.match(r"(.+?)\s+is\s+(?:a|an)\s+(.+)", lower)
    if m:
        return (m.group(1).strip(), "is_a", m.group(2).strip())

    # "X is Y" (generic)
    m = re.match(r"(.+?)\s+is\s+(.+)", lower)
    if m:
        return (m.group(1).strip(), "is_a", m.group(2).strip())

    # "X | predicate | Y"
    if "|" in t:
        parts = [p.strip() for p in t.split("|")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])

    return None
