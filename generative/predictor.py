"""
GenerativePredictor — fast n-gram predictor with BPE and orders 1-7.

This is a standalone predictor that does NOT depend on AETHER's cognitive
loop. It's optimized for:
  - Fast pre-training (just count n-grams, no Kuramoto/attractor/GWT)
  - High-order n-grams (1 to 7) with smart backoff
  - BPE tokenization for subword coverage
  - Temperature sampling for creative generation

Usage:
    pred = GenerativePredictor(orders=(1,2,3,4,5,6,7), use_bpe=True)
    pred.train_text("Hello world. Hello again.")
    tokens = pred.generate(["hello"], max_tokens=20, temperature=0.8)
    text = pred.decode(tokens)
"""

from __future__ import annotations
import re
import math
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional


# Word-level tokenizer
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def word_tokenize(text: str) -> list[str]:
    """Word-level tokenization (lowercase)."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------- #
# Minimal BPE (self-contained, no external dependency)
# ---------------------------------------------------------------------- #

class MinimalBPE:
    """Minimal BPE tokenizer — trains in seconds, encodes/decodes fast."""

    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._trained = False

    def train(self, corpus: str, verbose: bool = False) -> None:
        """Train BPE on a corpus string."""
        # Init vocab with characters
        chars = sorted(set(corpus.lower()))
        special = ["<pad>", "<unk>", "<s>", "</s>"]
        self.vocab = {t: i for i, t in enumerate(special + chars)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

        # Split corpus into words (with end-of-word marker)
        words = re.findall(r"\S+", corpus.lower())
        word_counts = Counter(words)

        # Represent each word as a tuple of chars + </w>
        word_tuples = {}
        for w, c in word_counts.items():
            word_tuples[w] = (tuple(list(w) + ["</w>"]), c)

        n_merges = self.vocab_size - len(self.vocab)
        for merge_idx in range(n_merges):
            # Count adjacent pairs
            pair_counts = Counter()
            for w, (chars_t, c) in word_tuples.items():
                for i in range(len(chars_t) - 1):
                    pair_counts[(chars_t[i], chars_t[i + 1])] += c

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            # New token
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.id_to_token[len(self.id_to_token)] = new_token
            self.merges.append(best_pair)

            # Apply merge to all words
            new_word_tuples = {}
            for w, (chars_t, c) in word_tuples.items():
                new_chars = []
                i = 0
                while i < len(chars_t):
                    if i < len(chars_t) - 1 and (chars_t[i], chars_t[i + 1]) == best_pair:
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars_t[i])
                        i += 1
                new_word_tuples[w] = (tuple(new_chars), c)
            word_tuples = new_word_tuples

            if verbose and merge_idx < 10:
                print(f"  merge {merge_idx}: {best_pair} -> {new_token!r}")

        self._trained = True

    def encode(self, text: str) -> list[str]:
        """Encode text to list of BPE token strings."""
        if not self._trained:
            return word_tokenize(text)

        words = re.findall(r"\S+", text.lower())
        tokens = []
        for word in words:
            # Apply merges in order
            chars = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                new_chars = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new_chars.append(a + b)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            tokens.extend(chars)
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Decode BPE tokens back to text."""
        text = ""
        for t in tokens:
            if t in ("<pad>", "<unk>", "<s>"):
                continue
            if t == "</s>":
                text += " "
            elif t.endswith("</w>"):
                text += t[:-4] + " "
            else:
                text += t
        return text.strip()


# ---------------------------------------------------------------------- #
# GenerativePredictor
# ---------------------------------------------------------------------- #

@dataclass
class GenerativePredictor:
    """
    Fast n-gram predictor with orders 1-7 and BPE support.

    Args:
        orders: tuple of n-gram orders to use (default 1-7)
        use_bpe: if True, use BPE tokenization; else word-level
        bpe_vocab_size: BPE vocabulary size
    """
    orders: tuple = (1, 2, 3, 4, 5, 6, 7)
    use_bpe: bool = True
    bpe_vocab_size: int = 2000

    # Storage: n-gram counts
    # ngram_counts[order] = Counter of tuples
    ngram_counts: dict = field(init=False, default_factory=dict)
    # context_counts[order] = Counter of context tuples (for normalization)
    context_counts: dict = field(init=False, default_factory=dict)
    # Unigram counts (special case)
    unigram_counts: Counter = field(init=False, default_factory=Counter)
    total_unigrams: int = field(init=False, default=0)
    # Vocab
    vocab: set = field(init=False, default_factory=set)

    # BPE
    bpe: Optional[MinimalBPE] = field(init=False, default=None)

    def __post_init__(self):
        for n in self.orders:
            self.ngram_counts[n] = Counter()
            self.context_counts[n] = Counter()
        if self.use_bpe:
            self.bpe = MinimalBPE(vocab_size=self.bpe_vocab_size)

    # ---------------------------------------------------------------- #
    # Tokenization
    # ---------------------------------------------------------------- #
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using BPE or word-level."""
        if self.use_bpe and self.bpe and self.bpe._trained:
            return self.bpe.encode(text)
        return word_tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        if self.use_bpe and self.bpe and self.bpe._trained:
            return self.bpe.decode(tokens)
        return " ".join(tokens).replace(" </w>", " ").replace("</w>", " ").strip()

    # ---------------------------------------------------------------- #
    # Training (fast — just counting)
    # ---------------------------------------------------------------- #
    def train_text(self, text: str) -> None:
        """Train on a text. Splits into sentences and counts n-grams."""
        # Add BOS/EOS markers around the text
        tokens = ["<s>"] + self.tokenize(text) + ["</s>"]
        self._train_sequence(tokens)

    def train_corpus(self, corpus: str) -> dict:
        """Train on a full corpus string. Returns stats."""
        # First, train BPE if needed
        if self.use_bpe and self.bpe and not self.bpe._trained:
            self.bpe.train(corpus)

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", corpus)
        n = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 3:
                continue
            self.train_text(sent)
            n += 1

        return {
            "sentences_trained": n,
            "vocab_size": len(self.vocab),
            "total_unigrams": self.total_unigrams,
            "ngram_counts": {n: len(c) for n, c in self.ngram_counts.items()},
        }

    def _train_sequence(self, tokens: list[str]) -> None:
        """Count all n-grams in a token sequence."""
        # Add to vocab
        self.vocab.update(tokens)

        # Unigram
        for t in tokens:
            self.unigram_counts[t] += 1
            self.total_unigrams += 1

        # Higher-order n-grams
        for n in self.orders:
            if n < 2:
                continue
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                context = ngram[:-1]
                self.ngram_counts[n][ngram] += 1
                self.context_counts[n][context] += 1

    # ---------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------- #
    def predict_next(self, context: list[str]) -> Optional[tuple[str, float, int]]:
        """
        Predict the next token given a context.
        Uses backoff: tries highest order first, falls back to lower orders.

        Returns (token, probability, order_used) or None.
        """
        if not context:
            context = ["<s>"]

        # Try each order from highest to lowest
        for n in reversed(self.orders):
            if n < 2:
                continue
            if len(context) < n - 1:
                continue

            ctx = tuple(context[-(n - 1):]) if n > 1 else ()
            ngrams_with_ctx = {ng: c for ng, c in self.ngram_counts[n].items()
                                if ng[:-1] == ctx}

            if not ngrams_with_ctx:
                continue

            ctx_total = self.context_counts[n].get(ctx, 0)
            if ctx_total == 0:
                continue

            # Find best token
            best_token = None
            best_count = 0
            for ng, c in ngrams_with_ctx.items():
                if c > best_count:
                    best_count = c
                    best_token = ng[-1]

            if best_token is not None:
                prob = best_count / ctx_total
                return best_token, prob, n

        # Backoff to unigram
        if self.total_unigrams > 0:
            # Pick most common unigram (excluding markers)
            best_tok, best_count = None, 0
            for tok, cnt in self.unigram_counts.items():
                if tok in ("<s>", "</s>", "<pad>", "<unk>"):
                    continue
                if cnt > best_count:
                    best_count = cnt
                    best_tok = tok
            if best_tok:
                return best_tok, best_count / self.total_unigrams, 1

        return None

    def predict_distribution(self, context: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """Get top-k next tokens with probabilities. Uses backoff."""
        if not context:
            context = ["<s>"]

        # Find the highest order with data
        for n in reversed(self.orders):
            if n < 2:
                continue
            if len(context) < n - 1:
                continue

            ctx = tuple(context[-(n - 1):])
            ngrams_with_ctx = {ng: c for ng, c in self.ngram_counts[n].items()
                                if ng[:-1] == ctx}
            if not ngrams_with_ctx:
                continue

            ctx_total = self.context_counts[n].get(ctx, 0)
            if ctx_total == 0:
                continue

            # Compute probabilities
            dist = [(ng[-1], c / ctx_total) for ng, c in ngrams_with_ctx.items()]
            dist.sort(key=lambda x: -x[1])
            return dist[:top_k]

        # Backoff to unigram
        if self.total_unigrams > 0:
            dist = [(tok, cnt / self.total_unigrams)
                    for tok, cnt in self.unigram_counts.items()
                    if tok not in ("<s>", "</s>", "<pad>", "<unk>")]
            dist.sort(key=lambda x: -x[1])
            return dist[:top_k]

        return []

    # ---------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------- #
    def generate(self, prompt_tokens: list[str], max_tokens: int = 20,
                 temperature: float = 0.8, top_k: int = 5,
                 stop_tokens: Optional[set] = None,
                 rng: Optional[np.random.Generator] = None) -> list[str]:
        """
        Generate a continuation of the prompt.

        Args:
            prompt_tokens: starting tokens
            max_tokens: max tokens to generate
            temperature: 0=greedy, 1=sample by prob, >1=flatter, <1=sharper
            top_k: only sample from top-k candidates
            stop_tokens: tokens that stop generation
            rng: random generator

        Returns:
            list of generated tokens (prompt excluded)
        """
        if rng is None:
            rng = np.random.default_rng()
        if stop_tokens is None:
            stop_tokens = {"</s>", "<s>", "<pad>", "<unk>"}

        context = list(prompt_tokens)
        generated = []

        for _ in range(max_tokens):
            dist = self.predict_distribution(context, top_k=max(top_k, 5))
            if not dist:
                break

            candidates = dist[:top_k]

            if temperature <= 0.01:
                # Greedy
                next_token = candidates[0][0]
            else:
                # Temperature sampling
                probs = np.array([p for _, p in candidates], dtype=np.float64)
                # Apply temperature
                scaled = np.log(np.maximum(probs, 1e-10)) / temperature
                # Softmax
                scaled = scaled - scaled.max()
                exp = np.exp(scaled)
                sample_probs = exp / exp.sum()

                idx = rng.choice(len(candidates), p=sample_probs)
                next_token = candidates[idx][0]

            if next_token in stop_tokens:
                break

            generated.append(next_token)
            context.append(next_token)

            # Avoid infinite loops: if last 3 tokens are identical, stop
            if len(generated) >= 3 and len(set(generated[-3:])) == 1:
                break

        return generated

    def generate_text(self, prompt: str, max_tokens: int = 20,
                      temperature: float = 0.8, top_k: int = 5,
                      rng: Optional[np.random.Generator] = None) -> str:
        """Generate text from a string prompt. Returns full text (prompt + generated)."""
        prompt_tokens = self.tokenize(prompt)
        generated = self.generate(prompt_tokens, max_tokens=max_tokens,
                                    temperature=temperature, top_k=top_k, rng=rng)
        all_tokens = prompt_tokens + generated
        return self.detokenize(all_tokens)

    # ---------------------------------------------------------------- #
    # Stats
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        return {
            "orders": list(self.orders),
            "use_bpe": self.use_bpe,
            "bpe_vocab": len(self.bpe.vocab) if self.bpe and self.bpe._trained else 0,
            "vocab_size": len(self.vocab),
            "total_unigrams": self.total_unigrams,
            "ngram_counts": {n: len(c) for n, c in self.ngram_counts.items()},
        }


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    corpus = """
    The cat is a small domesticated carnivorous mammal. Cats are known for their
    agility and hunting skills. A cat sleeps on average twelve to sixteen hours per day.
    The dog is a domesticated descendant of the wolf. Dogs are known for their loyalty
    and intelligence. A dog can learn hundreds of commands.
    Paris is the capital of France. Paris is located on the Seine river.
    Paris is known for the Eiffel Tower and the Louvre museum.
    The sun is a star at the center of the solar system. The sun provides energy
    to Earth through sunlight. The sun is a nearly perfect ball of hot plasma.
    Water is a chemical compound with the formula H2O. Water covers about seventy
    one percent of the Earth surface. Water is essential for all known forms of life.
    """

    print("=== Test GenerativePredictor with BPE ===")
    t0 = time.time()
    pred = GenerativePredictor(orders=(1, 2, 3, 4, 5, 6, 7), use_bpe=True)
    stats = pred.train_corpus(corpus)
    t1 = time.time()
    print(f"Training: {t1-t0:.2f}s")
    print(f"Stats: {stats}")

    print("\n=== Generation tests ===")
    for prompt in ["The cat", "Paris is", "The sun", "Water is"]:
        text = pred.generate_text(prompt, max_tokens=12, temperature=0.7)
        print(f"  {prompt!r} → {text!r}")

    print("\n=== Without BPE (word-level) ===")
    pred2 = GenerativePredictor(orders=(1, 2, 3, 4, 5, 6, 7), use_bpe=False)
    pred2.train_corpus(corpus)
    for prompt in ["The cat", "Paris is"]:
        text = pred2.generate_text(prompt, max_tokens=12, temperature=0.7)
        print(f"  {prompt!r} → {text!r}")
