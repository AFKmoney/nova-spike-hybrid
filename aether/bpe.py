"""
bpe.py — Byte-Pair Encoding tokenizer for AETHER.

PROBLEM
-------
Char-level tokenization means "hello" = 5 tokens = 5 separate predictions.
Each has ~5% error → after 20 chars, 65% chance of drift. A BPE tokenizer
makes "hello" = 1 token = 1 prediction = zero drift.

SOLUTION
--------
Standard BPE (Sennrich et al. 2016):
  1. Start with char-level tokens
  2. Count adjacent pairs, merge most frequent
  3. Repeat until vocab_size reached

This reduces token count by ~5x, dramatically improving generation coherence.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import logging

log = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    Usage:
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.train(corpus_text)
        tokens = tokenizer.encode("hello world")
        text = tokenizer.decode(tokens)
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []  # ordered list of merges
        self.vocab: Dict[str, int] = {}  # token → id
        self.id_to_token: Dict[int, str] = {}  # id → token
        self._trained = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, corpus: str, verbose: bool = False) -> None:
        """Train BPE on a corpus.

        Args:
            corpus: text to train on
            verbose: print progress
        """
        # Initialize vocabulary with characters + special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        chars = sorted(set(corpus.lower()))
        self.vocab = {tok: i for i, tok in enumerate(special_tokens + chars)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        # Split corpus into words (preserve spaces with a marker)
        words = re.findall(r'\S+|\s+', corpus.lower())
        # Represent each word as a tuple of chars
        word_tuples: List[Tuple[str, ...]] = [tuple(w) for w in words]
        # Count word frequencies
        word_counts: Counter = Counter(word_tuples)

        # Convert to mutable lists
        word_lists = [(list(w), c) for w, c in word_counts.items()]

        n_merges = self.vocab_size - len(self.vocab)
        for merge_idx in range(n_merges):
            # Count adjacent pairs
            pair_counts: Counter = Counter()
            for word, count in word_lists:
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += count

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # no more useful merges

            # Merge in all words
            new_token = best_pair[0] + best_pair[1]
            new_word_lists = []
            for word, count in word_lists:
                new_word = self._merge_word(word, best_pair, new_token)
                new_word_lists.append((new_word, count))
            word_lists = new_word_lists

            # Add to vocab
            if new_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[new_token] = new_id
                self.id_to_token[new_id] = new_token
            self.merges.append(best_pair)

            if verbose and (merge_idx + 1) % 100 == 0:
                log.info(f"BPE merge {merge_idx+1}/{n_merges}: {best_pair} → {new_token!r} (count={best_count})")

        self._trained = True
        log.info(f"BPE trained: {len(self.vocab)} tokens, {len(self.merges)} merges")

    def _merge_word(self, word: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
        """Merge a pair in a word."""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return result

    # ------------------------------------------------------------------ #
    # Encoding / Decoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[str]:
        """Encode text to a list of BPE tokens."""
        if not self._trained:
            # Fallback: char-level
            return list(text.lower())

        words = re.findall(r'\S+|\s+', text.lower())
        tokens = []
        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        return tokens

    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word using BPE merges."""
        chars = list(word)
        # Apply merges in order
        for pair, new_token in [(m, m[0] + m[1]) for m in self.merges]:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars

    def decode(self, tokens: List[str]) -> str:
        """Decode a list of BPE tokens back to text."""
        return "".join(tokens)

    def encode_to_ids(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        tokens = self.encode(text)
        return [self.vocab.get(t, self.vocab[self.UNK_TOKEN]) for t in tokens]

    def decode_from_ids(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text."""
        tokens = [self.id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        return self.decode(tokens)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        return {
            "vocab_size": len(self.vocab),
            "n_merges": len(self.merges),
            "trained": self._trained,
        }

    def compression_ratio(self, text: str) -> float:
        """How much does BPE compress the text? (chars / tokens)"""
        chars = len(text)
        tokens = len(self.encode(text))
        return chars / max(tokens, 1)


def train_default_bpe(corpus: Optional[str] = None) -> BPETokenizer:
    """Train a BPE tokenizer on a default corpus."""
    if corpus is None:
        # Default: a diverse sample corpus
        corpus = """
        The capital of France is Paris. The capital of Japan is Tokyo.
        Albert Einstein was a physicist. He discovered relativity.
        Water is a liquid. Fire is hot. Ice is cold.
        Python is a programming language. Computers process data.
        The brain is a complex organ. Humans are animals.
        Tokyo is located in Japan. Paris is located in France.
        Dogs are animals. Cats are animals. Birds can fly.
        The sun is a star. The moon orbits the earth.
        Mathematics is the study of numbers. Physics studies energy.
        History is the study of the past. Science seeks truth.
        """ * 20  # repeat for more merge opportunities
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(corpus)
    return tokenizer


if __name__ == "__main__":
    tok = train_default_bpe()
    print(f"BPE stats: {tok.stats()}")
    test_texts = [
        "hello world",
        "The capital of France is Paris.",
        "Albert Einstein was a physicist.",
    ]
    for t in test_texts:
        tokens = tok.encode(t)
        ratio = tok.compression_ratio(t)
        print(f"  {t!r}")
        print(f"    {len(t)} chars → {len(tokens)} tokens (ratio={ratio:.2f}x)")
        print(f"    tokens: {tokens}")
        print(f"    decoded: {tok.decode(tokens)!r}")
