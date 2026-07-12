"""
creative_writing.py — Creative writing with narrative arc.

PROBLEM
-------
GPT-4 can write stories, poems, essays with narrative structure.
AETHER is template-bound and can't generate creative content.

SOLUTION
--------
CreativeWriter generates:
  1. Stories with narrative arc (setup → conflict → resolution)
  2. Poems with rhyme/meter
  3. Essays with introduction → body → conclusion
  4. Dialogues between characters
  5. Descriptions (person, place, object)

Uses HD-vector-based idea generation: combine random concepts from
the KB to create novel premises, then structure them with templates.
"""

from __future__ import annotations
import random
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class CreativeWork:
    """A piece of creative writing."""
    title: str
    text: str
    genre: str  # "story", "poem", "essay", "dialogue", "description"
    premise: str = ""
    word_count: int = 0


class CreativeWriter:
    """Generate creative writing with narrative structure."""

    # Story templates (narrative arc)
    STORY_TEMPLATES = [
        {
            "arc": ["setup", "inciting_incident", "rising_action", "climax", "resolution"],
            "templates": {
                "setup": "Once upon a time, there was {protagonist} who lived in {setting}. "
                        "{protagonist} was {trait}, and spent their days {activity}.",
                "inciting_incident": "One day, {event} changed everything. "
                                    "{protagonist} could no longer ignore {conflict}.",
                "rising_action": "{protagonist} set out to {goal}. Along the way, they encountered "
                                "{obstacle} and had to {action}.",
                "climax": "The moment of truth arrived. {protagonist} faced {antagonist} in a "
                         "final confrontation. Everything was at stake.",
                "resolution": "In the end, {protagonist} {outcome}. The experience taught them "
                             "that {theme}.",
            }
        }
    ]

    # Poem templates
    POEM_TEMPLATES = [
        "{topic}, {topic}, so {adjective} and {adjective},\n"
        "{topic}, in the {place} of {place},\n"
        "{verb} me, {verb} me, to the {noun} of {noun},\n"
        "where {topic} {verb} in the {adjective} {noun}.",

        "Roses are {color1},\n"
        "Violets are {color2},\n"
        "{topic} is {adjective},\n"
        "And so are you.",
    ]

    # Essay templates
    ESSAY_TEMPLATES = {
        "introduction": "In this essay, I will explore {topic}. {topic} is a fascinating subject "
                       "that touches on {aspect1}, {aspect2}, and {aspect3}. "
                       "Through examining {topic}, we can better understand {theme}.",
        "body_1": "First, let us consider {aspect1}. {topic} relates to {aspect1} because "
                 "{connection1}. This is evident when we examine {example1}.",
        "body_2": "Furthermore, {aspect2} plays a crucial role. The relationship between "
                 "{topic} and {aspect2} reveals {insight}. Consider {example2}.",
        "body_3": "Finally, {aspect3} cannot be overlooked. {topic} impacts {aspect3} in ways "
                 "that {explanation}. This is demonstrated by {example3}.",
        "conclusion": "In conclusion, {topic} is a multifaceted subject. Through examining "
                     "{aspect1}, {aspect2}, and {aspect3}, we have seen that {thesis}. "
                     "The study of {topic} reminds us that {final_thought}.",
    }

    def __init__(self, agent):
        self.agent = agent
        self.rng = random.Random()

    # ------------------------------------------------------------------ #
    # Story generation
    # ------------------------------------------------------------------ #
    def write_story(self, theme: Optional[str] = None, length: str = "short") -> CreativeWork:
        """Generate a story with a narrative arc."""
        # Generate or use provided theme
        if theme is None:
            theme = self._generate_random_theme()

        # Generate story elements
        elements = self._generate_story_elements(theme)

        # Use the story template
        template = self.STORY_TEMPLATES[0]
        sections = []
        for arc_stage in template["arc"]:
            template_str = template["templates"][arc_stage]
            try:
                section = template_str.format(**elements)
                sections.append(section)
            except KeyError:
                sections.append(f"[{arc_stage}]")

        # Combine into a story
        text = "\n\n".join(sections)
        title = self._generate_title(theme)

        return CreativeWork(
            title=title, text=text, genre="story",
            premise=theme, word_count=len(text.split()),
        )

    def _generate_random_theme(self) -> str:
        """Generate a random story theme from KB concepts."""
        # Get random entities from the KB
        entities = [s for s, p, o in self.agent.assoc.list_triples()]
        if not entities:
            return "adventure"
        # Pick a random entity as the theme
        return self.rng.choice(entities)

    def _generate_story_elements(self, theme: str) -> Dict[str, str]:
        """Generate elements for the story."""
        # Get related concepts from KB
        related = self.agent.assoc.retrieve_similar(
            self.agent.encoder.encode_text(theme), top_k=5
        )
        related_words = [theme] + [t.split()[0] for t, _ in related if t]

        # Story element pools
        traits = ["brave", "curious", "kind", "clever", "determined", "gentle", "bold"]
        settings = ["a small village", "a great city", "a distant kingdom", "a quiet forest",
                   "a mountain top", "an ancient temple", "a vast desert"]
        activities = ["reading ancient scrolls", "tending to their garden", "exploring the countryside",
                     "studying the stars", "helping their neighbors"]
        events = ["a mysterious stranger arrived", "a great storm came", "they found a map",
                 "a prophecy was revealed", "they received a letter"]
        conflicts = ["the growing darkness", "a terrible secret", "an ancient curse",
                    "a missing person", "a broken promise"]
        goals = ["find the truth", "save their home", "discover their destiny",
                "right a terrible wrong", "find a lost treasure"]
        obstacles = ["a fearsome dragon", "a treacherous river", "a cunning rival",
                    "a riddle that defied solving", "a mountain that could not be climbed"]
        actions = ["use their wits", "make a sacrifice", "find an unlikely ally",
                  "learn a forgotten skill", "face their deepest fear"]
        antagonists = ["the dark lord", "a rival adventurer", "their own doubt",
                       "an ancient evil", "a traitor among friends"]
        outcomes = ["triumphed against all odds", "found what they were seeking",
                   "learned the true meaning of courage", "saved the day",
                   "discovered that the journey was the real treasure"]
        themes = ["courage is not the absence of fear but action in spite of it",
                 "true strength comes from within",
                 "the greatest adventures begin with a single step",
                 "friendship is the most powerful magic of all",
                 "every ending is a new beginning"]

        protagonist = self.rng.choice(related_words) if related_words else "a hero"

        return {
            "protagonist": protagonist.capitalize(),
            "trait": self.rng.choice(traits),
            "setting": self.rng.choice(settings),
            "activity": self.rng.choice(activities),
            "event": self.rng.choice(events),
            "conflict": self.rng.choice(conflicts),
            "goal": self.rng.choice(goals),
            "obstacle": self.rng.choice(obstacles),
            "action": self.rng.choice(actions),
            "antagonist": self.rng.choice(antagonists),
            "outcome": self.rng.choice(outcomes),
            "theme": self.rng.choice(themes),
        }

    def _generate_title(self, theme: str) -> str:
        """Generate a title for a story."""
        title_patterns = [
            f"The {theme.capitalize()} of {self.rng.choice(['Destiny', 'Dreams', 'Shadows', 'Light', 'Tomorrow'])}",
            f"{theme.capitalize()} and the {self.rng.choice(['Dragon', 'Crown', 'Journey', 'Promise', 'Star'])}",
            f"The Last {theme.capitalize()}",
            f"A {theme.capitalize()} Tale",
        ]
        return self.rng.choice(title_patterns)

    # ------------------------------------------------------------------ #
    # Poem generation
    # ------------------------------------------------------------------ #
    def write_poem(self, topic: str) -> CreativeWork:
        """Generate a short poem about a topic."""
        template = self.rng.choice(self.POEM_TEMPLATES)

        # Generate fill words
        colors = ["red", "blue", "green", "gold", "violet", "silver"]
        adjectives = ["fair", "rare", "bright", "light", "pure", "true"]
        places = ["dreams", "streams", "hills", "fields", "skies", "shores"]
        verbs = ["guide", "find", "free", "hold", "bind", "lift"]
        nouns = ["heart", "soul", "mind", "light", "dawn", "song"]

        try:
            if "color1" in template:
                text = template.format(
                    topic=topic,
                    color1=self.rng.choice(colors),
                    color2=self.rng.choice(colors),
                    adjective=self.rng.choice(adjectives),
                    place=self.rng.choice(places),
                    verb=self.rng.choice(verbs),
                    noun=self.rng.choice(nouns),
                )
            else:
                text = template.format(
                    topic=topic,
                    adjective=self.rng.choice(adjectives),
                    place=self.rng.choice(places),
                    verb=self.rng.choice(verbs),
                    noun=self.rng.choice(nouns),
                )
        except KeyError:
            text = f"{topic}, {topic}, forever {topic}."

        title = f"Ode to {topic.capitalize()}"

        return CreativeWork(
            title=title, text=text, genre="poem",
            premise=topic, word_count=len(text.split()),
        )

    # ------------------------------------------------------------------ #
    # Essay generation
    # ------------------------------------------------------------------ #
    def write_essay(self, topic: str) -> CreativeWork:
        """Generate a structured essay."""
        aspects = ["history", "culture", "science", "philosophy", "society", "nature"]
        chosen_aspects = self.rng.sample(aspects, 3)
        examples = [f"the case of {topic} in {self.rng.choice(['ancient Greece', 'modern times', 'the Renaissance', 'the 20th century'])}"
                   for _ in range(3)]

        elements = {
            "topic": topic,
            "aspect1": chosen_aspects[0],
            "aspect2": chosen_aspects[1],
            "aspect3": chosen_aspects[2],
            "connection1": f"it embodies the essence of {chosen_aspects[0]}",
            "connection2": f"it shapes our understanding of {chosen_aspects[1]}",
            "connection3": f"it reflects the principles of {chosen_aspects[2]}",
            "example1": examples[0],
            "example2": examples[1],
            "example3": examples[2],
            "insight": f"the deeper nature of {topic}",
            "explanation": f"{topic} operates on multiple levels",
            "theme": f"the human condition",
            "thesis": f"{topic} is far more complex than it first appears",
            "final_thought": f"knowledge is a journey, not a destination",
        }

        sections = []
        for key in ["introduction", "body_1", "body_2", "body_3", "conclusion"]:
            try:
                section = self.ESSAY_TEMPLATES[key].format(**elements)
                sections.append(section)
            except KeyError:
                pass

        text = "\n\n".join(sections)
        title = f"An Essay on {topic.capitalize()}"

        return CreativeWork(
            title=title, text=text, genre="essay",
            premise=topic, word_count=len(text.split()),
        )

    # ------------------------------------------------------------------ #
    # Description generation
    # ------------------------------------------------------------------ #
    def write_description(self, subject: str) -> CreativeWork:
        """Generate a descriptive passage about a subject."""
        # Get known facts about the subject
        result = self.agent.inference.lookup(subject, "is_a")
        category = result[0] if result else "mysterious entity"

        description = (
            f"{subject.capitalize()} is a {category} that captures the imagination. "
            f"Its presence evokes a sense of wonder and curiosity. "
            f"To understand {subject} is to glimpse a facet of the world that is both "
            f"familiar and extraordinary. "
            f"Those who encounter {subject} often find themselves changed by the experience, "
            f"carrying with them a new appreciation for the complexity of existence."
        )

        return CreativeWork(
            title=f"On {subject.capitalize()}",
            text=description, genre="description",
            premise=subject, word_count=len(description.split()),
        )
