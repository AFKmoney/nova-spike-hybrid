"""
semantic.py — Semantic HD embeddings via char n-grams.

PROBLEM
-------
In v1, every token got a RANDOM HD vector. That meant 'cat' and 'cats' had
~0 similarity, and 'dog' and 'puppy' had ~0 similarity. The agent couldn't
recognize morphological or lexical relationships.

SOLUTION
--------
Encode each token as a superposition of HD vectors for its character n-grams
(with positional binding). Two tokens that share many character n-grams will
naturally have similar HD vectors.

  'Paris'    -> ngrams: 'par', 'ari', 'ris', 'paris', 'pa', 'ar', 'ri', 'is', ...
  'parisian' -> ngrams: 'par', 'ari', 'ris', 'isa', 'sai', 'aia', 'ian', ...

  -> 'Paris' and 'parisian' will have HIGH similarity (they share 'par/ari/ris').

This is the same trick used by fastText, but in hyperdimensional space.
Combined with a concept-tag overlay (numbers, punctuation, names detected
heuristically), this gives AETHER a meaningful lexical embedding layer
WITHOUT any pretrained model, without any training.

Bonus: also exposes similarity(word1, word2) and nearest_neighbors(word).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np

from .hd import HDVector, DIM, bundle, _sign


# ---------------------------------------------------------------------------
# Char n-gram extraction
# ---------------------------------------------------------------------------

def char_ngrams(token: str, ns: Iterable[int] = (2, 3, 4)) -> List[str]:
    """Extract all character n-grams from a token.

    Pads the token with leading/trailing '#' to capture prefixes/suffixes.
    """
    padded = "#" + token + "#"
    grams: List[str] = []
    for n in ns:
        if len(padded) >= n:
            for i in range(len(padded) - n + 1):
                grams.append(padded[i:i + n])
    # Also include the raw token itself (1-gram)
    if token:
        grams.append(token)
    return grams


# ---------------------------------------------------------------------------
# Semantic encoder
# ---------------------------------------------------------------------------

class SemanticEncoder:
    """Encode tokens as superpositions of char n-gram HD vectors.

    The n-gram vectors are seeded deterministically from the n-gram string,
    so equivalent n-grams in different tokens map to the same HD vector
    (this is what creates cross-token similarity).
    """

    def __init__(self, dim: int = DIM, ns: Tuple[int, ...] = (2, 3, 4)):
        self.dim = dim
        self.ns = ns
        self._ngram_cache: Dict[str, HDVector] = {}
        # Concept-tag vectors — also deterministic from tag name
        self._tag_cache: Dict[str, HDVector] = {}

    def _get_ngram_vec(self, ngram: str) -> HDVector:
        if ngram not in self._ngram_cache:
            self._ngram_cache[ngram] = HDVector.from_text_seed("ng:" + ngram, self.dim)
        return self._ngram_cache[ngram]

    def _get_tag_vec(self, tag: str) -> HDVector:
        if tag not in self._tag_cache:
            self._tag_cache[tag] = HDVector.from_text_seed("tag:" + tag, self.dim)
        return self._tag_vec(tag)

    def encode_token(self, token: str) -> HDVector:
        """Encode a single token as a superposition of char n-grams."""
        token = token.lower()
        if not token:
            return HDVector.zero(self.dim)
        grams = char_ngrams(token, ns=self.ns)
        if not grams:
            return HDVector.from_text_seed(token, self.dim)
        # Bundle (with light positional role to distinguish prefix/suffix)
        # Strategy: bundle n-gram vectors + a positional permutation by index mod 4
        vecs = []
        for i, g in enumerate(grams):
            v = self._get_ngram_vec(g)
            if i % 4 != 0:
                # Slight positional binding to avoid total order-insensitivity
                v = HDVector(data=np.roll(v.data, i % 8), dim=self.dim)
            vecs.append(v)
        return bundle(vecs)

    # ----- similarity helpers -----
    def similarity(self, w1: str, w2: str) -> float:
        return self.encode_token(w1).similarity(self.encode_token(w2))

    def nearest_neighbors(self, word: str, vocab: Iterable[str], top_k: int = 5) -> List[Tuple[str, float]]:
        target = self.encode_token(word)
        sims = [(w, target.similarity(self.encode_token(w))) for w in vocab if w != word]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]


# ---------------------------------------------------------------------------
# Concept tagging (heuristic, no training)
# ---------------------------------------------------------------------------

# Common stop words — tagged so the agent can identify "structure" vs "content"
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "not", "if", "then", "so", "because", "while",
    "what", "who", "where", "when", "why", "how", "which", "that", "this",
    "these", "those", "it", "they", "he", "she", "we", "you", "i",
    "do", "does", "did", "done", "have", "has", "had", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must",
}

# Question words — tagged so we can identify question patterns
QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}

# Common verbs — tagged so we can identify predicates
COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "go", "goes", "went", "gone", "make", "makes", "made",
    "see", "sees", "saw", "seen", "know", "knows", "knew", "known",
    "think", "thinks", "thought", "say", "says", "said",
    "find", "finds", "found", "give", "gives", "gave", "given",
    "take", "takes", "took", "taken", "get", "gets", "got", "gotten",
    "come", "comes", "came", "become", "becomes", "became",
    "located", "live", "lives", "lived", "exists", "exist", "existed",
    "calculate", "compute", "eval", "find", "search",
}

# Capitalized tokens are likely proper nouns (names, places)
def is_proper_noun(token: str) -> bool:
    return len(token) > 0 and token[0].isupper() and token.lower() not in STOP_WORDS

# All-digit tokens are numbers
def is_number(token: str) -> bool:
    if not token:
        return False
    try:
        float(token.replace(",", "").replace(".", ""))
        return True
    except ValueError:
        return False

# Punctuation
def is_punct(token: str) -> bool:
    return len(token) == 1 and not token.isalnum()


def tag_token(token: str) -> List[str]:
    """Return a list of concept tags for a token.

    Tags: STOP, QUESTION, VERB, PROPER, NUMBER, PUNCT, WORD
    """
    lower = token.lower()
    tags = []
    if lower in STOP_WORDS:
        tags.append("STOP")
    if lower in QUESTION_WORDS:
        tags.append("QUESTION")
    if lower in COMMON_VERBS:
        tags.append("VERB")
    if is_proper_noun(token) and lower not in STOP_WORDS:
        tags.append("PROPER")
    if is_number(token):
        tags.append("NUMBER")
    if is_punct(token):
        tags.append("PUNCT")
    if not tags:
        tags.append("WORD")
    return tags


def tag_encode(token: str, dim: int = DIM) -> HDVector:
    """Encode a token's concept tags as a bundle of tag HD vectors."""
    tags = tag_token(token)
    vecs = [HDVector.from_text_seed("tag:" + t, dim) for t in tags]
    return bundle(vecs)
