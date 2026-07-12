"""
template_extract.py — Extract and reuse response templates.

PROBLEM
-------
Each response is generated from scratch. But many responses follow patterns:
  - "X is the capital of Y"
  - "X is a Y"
  - "X is located in Y"

Extracting these templates from learned data allows structured, coherent
responses instead of ad-hoc generation.

SOLUTION
--------
TemplateExtractor:
  1. Scan all stored triples and extract template patterns
  2. Store templates with their predicate
  3. When generating a response, find the best-matching template
  4. Fill in the slots with the retrieved facts

Example:
  Triple: (Paris, capital_of, France)
  Template: "{subject} is the capital of {object}"
  Filled: "Paris is the capital of France"
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class Template:
    """A response template."""
    predicate: str
    template: str  # e.g., "{subject} is the capital of {object}"
    n_uses: int = 0
    examples: List[Tuple[str, str, str]] = field(default_factory=list)  # (s, p, o)


# Default templates per predicate
DEFAULT_TEMPLATES: Dict[str, List[str]] = {
    "capital_of": [
        "{subject} is the capital of {object}.",
        "The capital of {object} is {subject}.",
        "It's {subject}.",
    ],
    "located_in": [
        "{subject} is located in {object}.",
        "{subject} is in {object}.",
        "It's in {object}.",
    ],
    "is_a": [
        "{subject} is {object}.",
        "{subject} is a {object}.",
        "It's a {object}.",
    ],
    "has": [
        "{subject} has {object}.",
    ],
    "can": [
        "{subject} can {object}.",
    ],
    "born_in": [
        "{subject} was born in {object}.",
    ],
    "died_in": [
        "{subject} died in {object}.",
    ],
    "founded_in": [
        "{subject} was founded in {object}.",
    ],
    "invented": [
        "{subject} invented {object}.",
    ],
    "famous_for": [
        "{subject} is famous for {object}.",
    ],
    "known_for": [
        "{subject} is known for {object}.",
    ],
    "discovered": [
        "{subject} discovered {object}.",
    ],
    "wrote": [
        "{subject} wrote {object}.",
    ],
    "married": [
        "{subject} married {object}.",
    ],
    "parent_of": [
        "{subject} is the parent of {object}.",
    ],
    "part_of": [
        "{subject} is part of {object}.",
    ],
    "contains": [
        "{subject} contains {object}.",
    ],
    "used_for": [
        "{subject} is used for {object}.",
    ],
    "made_of": [
        "{subject} is made of {object}.",
    ],
    "causes": [
        "{subject} causes {object}.",
    ],
    "associated_with": [
        "{subject} is associated with {object}.",
    ],
}


class TemplateExtractor:
    """Extract and reuse response templates."""

    def __init__(self, agent):
        self.agent = agent
        self.templates: Dict[str, List[Template]] = {}
        # Initialize with defaults
        for pred, templates in DEFAULT_TEMPLATES.items():
            self.templates[pred] = [
                Template(predicate=pred, template=t) for t in templates
            ]

    # ------------------------------------------------------------------ #
    # Template extraction from stored data
    # ------------------------------------------------------------------ #
    def extract_from_triples(self) -> int:
        """Scan all stored triples and extract template patterns.

        For each triple (s, p, o), we already know the predicate.
        The template is the sentence pattern that fits.
        """
        n_extracted = 0
        for s, p, o in self.agent.assoc.list_triples():
            if p in self.templates:
                # Pick the first template and add this as an example
                template = self.templates[p][0]
                if (s, p, o) not in template.examples:
                    template.examples.append((s, p, o))
                    n_extracted += 1
        log.info(f"extracted {n_extracted} template examples")
        return n_extracted

    # ------------------------------------------------------------------ #
    # Template selection and filling
    # ------------------------------------------------------------------ #
    def select_template(self, predicate: str, n_examples: int = 0) -> Optional[Template]:
        """Select the best template for a predicate.

        If n_examples > 0, prefer templates that have been used more.
        """
        if predicate not in self.templates or not self.templates[predicate]:
            return None
        # Sort by number of examples (most used first)
        templates = sorted(self.templates[predicate], key=lambda t: -len(t.examples))
        return templates[0]

    def fill_template(self, template: Template, subject: str, obj: str) -> str:
        """Fill a template with subject and object."""
        text = template.template.format(subject=subject, object=obj)
        template.n_uses += 1
        return text

    def generate_response(self, subject: str, predicate: str, obj: str) -> Optional[str]:
        """Generate a response using the best template."""
        template = self.select_template(predicate)
        if template is None:
            # Fallback: simple "X is Y"
            return f"{subject} is {obj}."
        return self.fill_template(template, subject, obj)

    # ------------------------------------------------------------------ #
    # Template discovery from text
    # ------------------------------------------------------------------ #
    def discover_templates(self, text: str) -> List[str]:
        """Discover potential templates from a text passage.

        Looks for patterns like "X is the Y of Z" and extracts the template.
        """
        discovered = []
        # Pattern: "X is the Y of Z"
        for m in re.finditer(r"(.+?) is the (\w+) of (.+?)[.\n]", text):
            predicate = f"{m.group(2)}_of"
            template = "{subject} is the {predicate_words} of {object}"
            discovered.append(template.format(predicate_words=m.group(2)))
            if predicate not in self.templates:
                self.templates[predicate] = [Template(predicate=predicate, template=template)]
        # Pattern: "X is a Y"
        for m in re.finditer(r"(.+?) is a (.+?)[.\n]", text):
            if "is_a" not in self.templates:
                self.templates["is_a"] = []
            discovered.append("{subject} is a {object}")
        return discovered

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        return {
            "n_predicates": len(self.templates),
            "n_templates": sum(len(ts) for ts in self.templates.values()),
            "n_examples": sum(len(t.examples for ts in self.templates.values() for t in ts)),
        }
