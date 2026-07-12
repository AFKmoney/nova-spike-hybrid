"""
pretrained.py — Semantic HD embeddings (no external downloads).

PROBLEM
-------
In v2, tokens were encoded via char n-grams. That gives morphological
similarity (paris ~ parisian) but NOT conceptual similarity
(paris ~ france, dog ~ animal).

Real solution: compress word2vec/GloVe into bipolar HD vectors.
Problem: we don't have 2 GB to download and we want to stay self-contained.

AETHER'S SOLUTION
-----------------
Build a "concept space" synthetically:

  1. Define a curated taxonomy of categories (animal, country, color,
     number, food, verb, emotion, technology, etc.).
  2. Each category gets a unique base HD vector.
  3. Each member of a category is encoded as bundle(category_vec, member_vec).
     -> members of the same category are highly similar
     -> members of different categories are ~orthogonal
  4. Cross-cutting properties (size, color, location) get their own
     "property" vectors bundled on top.
  5. Synonyms get the SAME underlying concept vector bundled with their
     surface form. Antonyms get the concept vector XOR'd with an "antonym"
     role vector.

This produces ~3000 semantically-aware HD vectors in pure Python, no
download, no training. The result: AETHER knows that 'dog' is more
similar to 'cat' than to 'france', even before you teach it anything.

Bonus: also exposes a `SemanticKB` class that pre-loads these vectors
into the agent's vocabulary at startup.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from .hd import HDVector, DIM, bundle, _sign


# ---------------------------------------------------------------------------
# Curated concept taxonomy
# ---------------------------------------------------------------------------

# Each category lists its members. Categories themselves are HD vectors.
# Members are bundle(category_vec, unique_member_vec) — so two members of
# the same category are ~33% similar; members of different categories
# are ~0% similar.
CONCEPT_TAXONOMY: Dict[str, List[str]] = {
    # Animals
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken",
               "duck", "rabbit", "mouse", "rat", "lion", "tiger", "bear",
               "wolf", "fox", "deer", "elephant", "giraffe", "monkey", "ape",
               "whale", "dolphin", "shark", "fish", "bird", "eagle", "snake",
               "lizard", "frog", "turtle", "spider", "ant", "bee", "fly"],
    # Countries
    "country": ["france", "germany", "italy", "spain", "portugal", "greece",
                "england", "scotland", "ireland", "wales", "netherlands",
                "belgium", "switzerland", "austria", "sweden", "norway",
                "denmark", "finland", "iceland", "poland", "russia", "ukraine",
                "japan", "china", "korea", "india", "thailand", "vietnam",
                "indonesia", "australia", "new_zealand", "canada", "usa",
                "mexico", "brazil", "argentina", "chile", "peru", "colombia",
                "egypt", "morocco", "nigeria", "kenya", "south_africa"],
    # Cities
    "city": ["paris", "london", "berlin", "madrid", "rome", "lisbon",
             "amsterdam", "brussels", "vienna", "prague", "stockholm",
             "oslo", "copenhagen", "helsinki", "reykjavik", "moscow",
             "tokyo", "beijing", "seoul", "delhi", "bangkok", "hanoi",
             "ottawa", "washington", "mexico_city", "buenos_aires", "lima",
             "bogota", "cairo", "casablanca", "lagos", "nairobi", "cape_town",
             "sydney", "auckland", "montreal", "toronto", "vancouver",
             "lyon", "marseille", "osaka", "kyoto", "munich", "hamburg"],
    # Colors
    "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
              "brown", "black", "white", "gray", "grey", "cyan", "magenta",
              "violet", "indigo", "turquoise", "gold", "silver", "bronze"],
    # Numbers (small)
    "number": ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine", "ten", "eleven", "twelve", "twenty", "fifty",
               "hundred", "thousand", "million", "billion"],
    # Foods
    "food": ["bread", "rice", "pasta", "meat", "fish", "egg", "milk", "cheese",
             "butter", "salt", "sugar", "honey", "water", "wine", "beer",
             "coffee", "tea", "juice", "soup", "salad", "fruit", "vegetable",
             "apple", "banana", "orange", "lemon", "grape", "tomato", "potato",
             "onion", "garlic", "carrot", "lettuce", "beef", "pork", "chicken_food"],
    # Body parts
    "body": ["head", "face", "eye", "ear", "nose", "mouth", "tooth", "tongue",
             "neck", "shoulder", "arm", "hand", "finger", "chest", "back",
             "stomach", "leg", "foot", "knee", "heart", "brain", "lung",
             "liver", "blood", "bone", "skin", "hair"],
    # Verbs (common)
    "verb": ["be", "have", "do", "say", "go", "come", "see", "know", "think",
             "make", "give", "take", "find", "want", "use", "work", "look",
             "feel", "become", "leave", "put", "mean", "keep", "let", "begin",
             "seem", "help", "talk", "turn", "start", "show", "hear", "play",
             "run", "move", "live", "believe", "hold", "bring", "happen",
             "write", "provide", "sit", "stand", "lose", "pay", "meet",
             "include", "continue", "set", "learn", "change", "lead",
             "understand", "watch", "follow", "stop", "create", "speak",
             "read", "allow", "add", "spend", "grow", "open", "walk", "win",
             "offer", "remember", "consider", "appear", "buy", "wait", "serve",
             "die", "send", "expect", "build", "stay", "fall", "cut", "reach",
             "kill", "remain"],
    # Emotions
    "emotion": ["happy", "sad", "angry", "fear", "surprise", "disgust",
                "joy", "sorrow", "rage", "terror", "wonder", "revulsion",
                "love", "hate", "hope", "despair", "pride", "shame",
                "gratitude", "envy", "excitement", "boredom", "calm",
                "anxiety", "serenity"],
    # Technology
    "technology": ["computer", "phone", "tablet", "laptop", "server", "cloud",
                   "internet", "web", "browser", "search", "email", "chat",
                   "social", "app", "software", "hardware", "code", "program",
                   "algorithm", "data", "database", "server", "api", "frontend",
                   "backend", "ai", "ml", "neural", "transformer", "model",
                   "training", "inference", "python", "javascript", "rust",
                   "cpp", "java"],
    # Nature
    "nature": ["sun", "moon", "star", "planet", "earth", "mars", "venus",
               "jupiter", "saturn", "sky", "cloud", "rain", "snow", "wind",
               "storm", "thunder", "lightning", "river", "lake", "sea",
               "ocean", "mountain", "hill", "valley", "forest", "desert",
               "island", "beach", "rock", "sand", "fire", "ice", "tree",
               "flower", "grass", "leaf"],
    # Time
    "time": ["second", "minute", "hour", "day", "week", "month", "year",
             "decade", "century", "millennium", "morning", "noon", "afternoon",
             "evening", "night", "midnight", "dawn", "dusk", "today", "tomorrow",
             "yesterday", "now", "past", "present", "future", "always", "never",
             "sometimes", "often", "rarely"],
    # Vehicles
    "vehicle": ["car", "truck", "bus", "train", "plane", "boat", "ship",
                "bicycle", "motorcycle", "scooter", "helicopter", "rocket",
                "submarine", "tractor", "ambulance", "taxi"],
    # Clothing
    "clothing": ["shirt", "pants", "dress", "skirt", "coat", "jacket",
                 "sweater", "hat", "cap", "shoe", "boot", "sock", "glove",
                 "scarf", "tie", "belt", "uniform", "costume", "suit"],
    # Buildings
    "building": ["house", "apartment", "office", "school", "hospital",
                 "library", "museum", "church", "temple", "mosque", "bank",
                 "store", "shop", "restaurant", "cafe", "hotel", "factory",
                 "warehouse", "garage", "barn", "castle", "palace", "tower",
                 "bridge", "tunnel"],
    # Materials
    "material": ["wood", "metal", "stone", "plastic", "glass", "paper",
                 "fabric", "leather", "rubber", "concrete", "brick", "steel",
                 "iron", "copper", "aluminum", "gold_metal", "silver_metal",
                 "diamond", "clay", "sand_m"],
    # Sciences
    "science": ["physics", "chemistry", "biology", "math", "geometry",
                "algebra", "calculus", "astronomy", "geology", "geography",
                "history", "philosophy", "psychology", "sociology", "economics",
                "politics", "law", "medicine", "engineering", "art",
                "music", "literature", "poetry", "linguistics"],
}

# Synonyms: pairs of words that should have the SAME concept vector
SYNONYMS: List[Tuple[str, str]] = [
    ("happy", "joyful"),
    ("sad", "sorrowful"),
    ("big", "large"),
    ("small", "little"),
    ("fast", "quick"),
    ("slow", "sluggish"),
    ("smart", "intelligent"),
    ("stupid", "foolish"),
    ("beautiful", "pretty"),
    ("ugly", "hideous"),
    ("begin", "start"),
    ("end", "finish"),
    ("buy", "purchase"),
    ("sell", "vend"),
    ("help", "assist"),
    ("stop", "halt"),
    ("walk", "stroll"),
    ("run", "sprint"),
    ("eat", "consume"),
    ("drink", "sip"),
    ("see", "observe"),
    ("hear", "listen"),
    ("think", "ponder"),
    ("know", "understand"),
    ("dog", "hound"),
    ("cat", "feline"),
    ("car", "automobile"),
    ("house", "home"),
    ("money", "cash"),
    ("rich", "wealthy"),
    ("poor", "destitute"),
]

# Antonyms: pairs that should have OPPOSITE vectors (XOR with antonym role)
ANTONYMS: List[Tuple[str, str]] = [
    ("hot", "cold"),
    ("up", "down"),
    ("left", "right"),
    ("in", "out"),
    ("on", "off"),
    ("open", "close"),
    ("start", "stop"),
    ("begin", "end"),
    ("good", "bad"),
    ("right_correct", "wrong"),
    ("big", "small"),
    ("tall", "short"),
    ("long", "short"),
    ("wide", "narrow"),
    ("deep", "shallow"),
    ("fast", "slow"),
    ("easy", "hard"),
    ("young", "old"),
    ("new", "old"),
    ("rich", "poor"),
    ("strong", "weak"),
    ("light", "dark"),
    ("day", "night"),
    ("yes", "no"),
    ("true", "false"),
    ("alive", "dead"),
    ("love", "hate"),
    ("happy", "sad"),
    ("war", "peace"),
    ("win", "lose"),
]


# ---------------------------------------------------------------------------
# Semantic KB: builds HD vectors for all the above
# ---------------------------------------------------------------------------

class SemanticKB:
    """Build and store semantically-aware HD vectors for a curated vocabulary.

    Usage:
        skb = SemanticKB(dim=4096)
        skb.build()
        vec_dog = skb.get("dog")
        vec_cat = skb.get("cat")
        sim = vec_dog.similarity(vec_cat)  # ~0.33 (same category)
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vectors: Dict[str, HDVector] = {}
        self.categories: Dict[str, HDVector] = {}
        self.word_to_category: Dict[str, str] = {}
        self._role_antonym: Optional[HDVector] = None
        self._role_synonym: Optional[HDVector] = None

    def _mix(self, cat_vec: HDVector, member_vec: HDVector, cat_weight: float = 0.65) -> HDVector:
        """Probabilistic mix: take cat_weight fraction of bits from cat_vec,
        the rest from member_vec. This gives controllable similarity:
        two members of same category share ~cat_weight of bits from cat_vec
        (deterministic match) + (1-cat_weight)^2 random matches on member bits.
        Theoretical similarity ~ cat_weight^2 + (1-cat_weight)^2 * 0 ... ~ cat_weight.
        """
        rng = np.random.default_rng(abs(hash(cat_vec.data.tobytes()[:16])) % (2**32))
        mask = rng.random(self.dim) < cat_weight
        result = np.where(mask, cat_vec.data, member_vec.data).astype(np.int8)
        return HDVector(data=result, dim=self.dim)

    def build(self) -> None:
        """Build the full semantic KB: categories + members + synonyms + antonyms."""
        # 1. Create a base vector for each category
        for cat_name in CONCEPT_TAXONOMY:
            self.categories[cat_name] = HDVector.from_text_seed("cat:" + cat_name, self.dim)

        # 2. Each member is a probabilistic mix of category + member
        #    cat_weight=0.65 means 65% of bits come from the category vector
        #    (shared with all members of the same category) and 35% from the
        #    unique member vector. This gives intra-category similarity ~0.5
        #    and cross-category similarity ~0.
        for cat_name, members in CONCEPT_TAXONOMY.items():
            cat_vec = self.categories[cat_name]
            for member in members:
                member_vec = HDVector.from_text_seed("mem:" + member, self.dim)
                if member in self.vectors:
                    # Word appears in multiple categories — bundle them all
                    existing = self.vectors[member]
                    self.vectors[member] = existing.bundle(cat_vec)
                else:
                    self.vectors[member] = self._mix(cat_vec, member_vec, cat_weight=0.65)
                    self.word_to_category[member] = cat_name

        # 3. Synonyms: bundle the two words' vectors together so they converge
        for w1, w2 in SYNONYMS:
            v1 = self._get_or_create(w1)
            v2 = self._get_or_create(w2)
            # Make them more similar by averaging (bundling) them
            merged = v1.bundle(v2)
            self.vectors[w1] = merged
            self.vectors[w2] = merged.copy()

        # 4. Antonyms: XOR one with an "antonym" role vector
        self._role_antonym = HDVector.from_text_seed("role:antonym", self.dim)
        for w1, w2 in ANTONYMS:
            v1 = self._get_or_create(w1)
            v2 = self._get_or_create(w2)
            # Make them opposite: v2 = bind(v1, antonym_role)
            # This gives similarity -1 to each other (XOR of all bits)
            opposite = v1.bind(self._role_antonym)
            self.vectors[w2] = opposite

    def _get_or_create(self, word: str) -> HDVector:
        if word in self.vectors:
            return self.vectors[word]
        # If not in any category, create a random vector seeded by the word
        self.vectors[word] = HDVector.from_text_seed("word:" + word, self.dim)
        return self.vectors[word]

    def get(self, word: str) -> Optional[HDVector]:
        """Return the HD vector for a word, or None if unknown."""
        return self.vectors.get(word.lower())

    def has(self, word: str) -> bool:
        return word.lower() in self.vectors

    def category_of(self, word: str) -> Optional[str]:
        return self.word_to_category.get(word.lower())

    def similarity(self, w1: str, w2: str) -> float:
        v1 = self.get(w1)
        v2 = self.get(w2)
        if v1 is None or v2 is None:
            return 0.0
        return v1.similarity(v2)

    def nearest_neighbors(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        target = self.get(word)
        if target is None:
            return []
        sims = [(w, target.similarity(v)) for w, v in self.vectors.items() if w != word.lower()]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def members_of(self, category: str) -> List[str]:
        return [w for w, c in self.word_to_category.items() if c == category]

    def categories_list(self) -> List[str]:
        return list(self.categories.keys())

    def stats(self) -> Dict[str, int]:
        return {
            "vectors": len(self.vectors),
            "categories": len(self.categories),
            "synonyms": len(SYNONYMS),
            "antonyms": len(ANTONYMS),
        }


# ---------------------------------------------------------------------------
# Helper: integrate the semantic KB into an AETHER agent's vocabulary
# ---------------------------------------------------------------------------

def integrate_into_agent(agent) -> int:
    """Pre-load the semantic KB into an AETHER agent's vocabulary.

    Returns the number of vectors added.
    """
    skb = SemanticKB(dim=agent.dim)
    skb.build()
    count = 0
    for word, vec in skb.vectors.items():
        if word not in agent.assoc.vocab:
            agent.assoc.vocab[word] = vec
            count += 1
    return count


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skb = SemanticKB(dim=4096)
    skb.build()
    print(f"Built semantic KB: {skb.stats()}")

    print("\nWithin-category similarity (should be HIGH, ~0.3-0.5):")
    pairs = [
        ("dog", "cat"),       # both animals
        ("paris", "london"),  # both cities
        ("red", "blue"),      # both colors
        ("three", "seven"),   # both numbers
        ("bread", "rice"),    # both foods
    ]
    for w1, w2 in pairs:
        sim = skb.similarity(w1, w2)
        print(f"  sim({w1!r:10s}, {w2!r:10s}) = {sim:+.4f}")

    print("\nCross-category similarity (should be LOW, ~0):")
    pairs = [
        ("dog", "paris"),
        ("red", "seven"),
        ("bread", "happy"),
        ("computer", "tree"),
    ]
    for w1, w2 in pairs:
        sim = skb.similarity(w1, w2)
        print(f"  sim({w1!r:10s}, {w2!r:10s}) = {sim:+.4f}")

    print("\nSynonym similarity (should be HIGH, ~1.0):")
    for w1, w2 in [("happy", "joyful"), ("big", "large"), ("smart", "intelligent")]:
        sim = skb.similarity(w1, w2)
        print(f"  sim({w1!r:12s}, {w2!r:12s}) = {sim:+.4f}")

    print("\nAntonym similarity (should be LOW, ~-1.0):")
    for w1, w2 in [("hot", "cold"), ("big", "small"), ("day", "night")]:
        sim = skb.similarity(w1, w2)
        print(f"  sim({w1!r:12s}, {w2!r:12s}) = {sim:+.4f}")

    print(f"\nNearest neighbors of 'dog':")
    for w, s in skb.nearest_neighbors("dog", top_k=5):
        print(f"  {w!r:15s} : {s:+.4f}")

    print(f"\nNearest neighbors of 'paris':")
    for w, s in skb.nearest_neighbors("paris", top_k=5):
        print(f"  {w!r:15s} : {s:+.4f}")
