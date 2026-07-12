"""
language_model.py — HD n-gram language model with backoff + temperature sampling.

Features:
  - Multi-order n-gram storage (n=2,3,4,5) in the SAME SDM (one address space)
  - Backoff: try n=5, then n=4, then n=3, then n=2, then unigram
  - Temperature sampling: convert similarities to probabilities via softmax(1/T)
  - Beam search: keep top-k partial sequences, expand by sampling top-k next tokens
  - Sequence completion: given a prompt, generate up to N tokens

All in hyperdimensional space. No transformer. No GPU.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from .hd import HDVector, DIM, bundle, _sign
from .memory import AssociativeMemory


# ---------------------------------------------------------------------------
# N-gram address encoder
# ---------------------------------------------------------------------------

def encode_ngram_context(token_vecs: List[HDVector], n: int, dim: int = DIM) -> HDVector:
    """Encode the last n tokens as an HD context vector with positional binding.

    Uses permutation by index for positional encoding:
        context = bundle(perm^0(t_{-n+1}), perm^1(t_{-n+2}), ..., perm^{n-1}(t_0))

    This is the classic VSA sequence encoding.
    """
    if not token_vecs:
        return HDVector.random(dim)
    # Take the last n tokens
    ctx_vecs = token_vecs[-n:] if len(token_vecs) >= n else token_vecs
    bound = []
    for j, v in enumerate(ctx_vecs):
        # Permute by index to mark position
        bound.append(HDVector(data=np.roll(v.data, j), dim=dim))
    return bundle(bound)


# ---------------------------------------------------------------------------
# HD Language Model
# ---------------------------------------------------------------------------

class HDLanguageModel:
    """N-gram HD language model with backoff and temperature sampling.

    Storage strategy: for each n-gram order (n=2..5), maintain a separate
    "address space" by binding with an "order role" vector. This prevents
    different n-gram orders from colliding in the SDM.

      addr_n(context) = bind(order_role_n, encode_ngram_context(ctx, n))

    Write: when we see a sequence t_1...t_k, for each n in {2,3,4,5} and each
           position i >= n-1, write (addr_n(t_{i-n+1}..t_i) -> t_{i+1}).

    Read: predict next token by querying addr_n(last n tokens), backoff to n-1
          if no confident match.
    """

    def __init__(self, assoc: AssociativeMemory, dim: int = DIM,
                 orders: Tuple[int, ...] = (2, 3, 4, 5)):
        self.assoc = assoc
        self.dim = dim
        self.orders = orders
        # Order-role vectors: bind context with these to namespace by order
        self._order_roles: Dict[int, HDVector] = {
            n: HDVector.from_text_seed(f"role:order:{n}", dim) for n in orders
        }
        # Unigram counter (raw counts, not HD)
        self._unigram: Dict[str, int] = {}
        self._unigram_total = 0

    # ------------------------------------------------------------------ #
    # Training (one-shot, no epochs)
    # ------------------------------------------------------------------ #
    def learn_sequence(self, tokens: List[str]) -> None:
        """Learn a sequence of tokens. Adds n-gram contexts for all orders."""
        if len(tokens) < 2:
            # Just add unigrams
            for tok in tokens:
                self._add_unigram(tok)
            return

        # Add start/end markers
        marked = ["<s>"] + list(tokens) + ["</s>"]
        vecs = [self.assoc.get_symbol(t) for t in marked]

        # Unigram counts
        for tok in marked:
            self._add_unigram(tok)

        # For each order n, write (context_n -> next_token) pairs
        for n in self.orders:
            role = self._order_roles[n]
            for i in range(len(vecs) - n):
                ctx_vecs = vecs[i:i + n]
                target = vecs[i + n]
                ctx = encode_ngram_context(ctx_vecs, n, dim=self.dim)
                addr = ctx.bind(role)
                self.assoc.kb_store.write(addr, target)

    def _add_unigram(self, token: str) -> None:
        self._unigram[token] = self._unigram.get(token, 0) + 1
        self._unigram_total += 1

    # ------------------------------------------------------------------ #
    # Prediction (with backoff)
    # ------------------------------------------------------------------ #
    def predict_next(self, context_tokens: List[str]) -> Optional[Tuple[str, float, int]]:
        """Predict the next token given a context. Uses backoff.

        Returns (token, similarity, order_used) or None.
        """
        # Add start marker if context is empty
        if not context_tokens:
            context_tokens = ["<s>"]
        else:
            context_tokens = list(context_tokens)

        # Try each order from highest to lowest
        for n in reversed(self.orders):
            if len(context_tokens) < n:
                continue
            ctx_vecs = [self.assoc.get_symbol(t) for t in context_tokens[-n:]]
            ctx = encode_ngram_context(ctx_vecs, n, dim=self.dim)
            role = self._order_roles[n]
            addr = ctx.bind(role)
            retrieved = self.assoc.kb_store.read(addr)
            if retrieved is None:
                continue
            # Find best matching token
            best_name, best_sim = None, -1.0
            for name, vec in self.assoc.vocab.items():
                sim = retrieved.similarity(vec)
                if sim > best_sim:
                    best_sim, best_name = sim, name
            if best_name is not None and best_sim >= 0.10:
                return best_name, best_sim, n

        # Backoff to unigram (most frequent token)
        if self._unigram_total > 0:
            # Pick the most common unigram (excluding markers)
            best_tok, best_count = None, 0
            for tok, cnt in self._unigram.items():
                if tok in ("<s>", "</s>", "<unk>", "<pad>"):
                    continue
                if cnt > best_count:
                    best_count, best_tok = cnt, tok
            if best_tok is not None:
                # Confidence = frequency
                return best_tok, best_count / self._unigram_total, 1

        return None

    def predict_distribution(self, context_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Get a probability distribution over the top-k next tokens.

        Returns list of (token, probability) sorted by descending probability.
        """
        if not context_tokens:
            context_tokens = ["<s>"]
        else:
            context_tokens = list(context_tokens)

        # Find the highest order with a confident match
        best_retrieved = None
        best_order = 0
        for n in reversed(self.orders):
            if len(context_tokens) < n:
                continue
            ctx_vecs = [self.assoc.get_symbol(t) for t in context_tokens[-n:]]
            ctx = encode_ngram_context(ctx_vecs, n, dim=self.dim)
            role = self._order_roles[n]
            addr = ctx.bind(role)
            retrieved = self.assoc.kb_store.read(addr)
            if retrieved is not None:
                best_retrieved = retrieved
                best_order = n
                break

        if best_retrieved is None:
            # Fall back to unigram distribution
            if self._unigram_total == 0:
                return []
            dist = []
            for tok, cnt in self._unigram.items():
                if tok in ("<s>", "</s>", "<unk>", "<pad>"):
                    continue
                dist.append((tok, cnt / self._unigram_total))
            dist.sort(key=lambda x: -x[1])
            return dist[:top_k]

        # Compute similarities to all vocab tokens
        sims = []
        for name, vec in self.assoc.vocab.items():
            if name in ("<s>", "</s>", "<unk>", "<pad>"):
                continue
            sim = best_retrieved.similarity(vec)
            sims.append((name, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    # ------------------------------------------------------------------ #
    # Generation with temperature sampling
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt_tokens: List[str],
        max_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 5,
        stop_tokens: Optional[set] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Generate a continuation of the prompt.

        Args:
            prompt_tokens: the starting tokens
            max_tokens: max tokens to generate
            temperature: 0 = greedy, 1 = sample by similarity, >1 = flatter, <1 = sharper
            top_k: only sample from the top-k candidates
            stop_tokens: tokens that stop generation (e.g., {".", "</s>"})
            rng: optional numpy random generator for reproducibility

        Returns:
            list of generated tokens (not including the prompt)
        """
        if rng is None:
            rng = np.random.default_rng()
        if stop_tokens is None:
            stop_tokens = {"</s>", "<s>", "<unk>", "<pad>"}

        context = list(prompt_tokens)
        generated = []

        for _ in range(max_tokens):
            dist = self.predict_distribution(context, top_k=max(top_k, 5))
            if not dist:
                break

            # Filter to top_k
            candidates = dist[:top_k]

            # Temperature sampling
            if temperature <= 0.01:
                # Greedy
                next_token = candidates[0][0]
            else:
                # Softmax with temperature
                sims = np.array([s for _, s in candidates], dtype=np.float64)
                # Scale by temperature: high T -> flatter, low T -> sharper
                scaled = sims / max(temperature, 0.01)
                # Shift to avoid overflow
                scaled = scaled - scaled.max()
                exp_s = np.exp(scaled * 5.0)  # scale up similarities for sharper distribution
                probs = exp_s / exp_s.sum()
                # Sample
                idx = rng.choice(len(candidates), p=probs)
                next_token = candidates[idx][0]

            if next_token in stop_tokens:
                break

            generated.append(next_token)
            context.append(next_token)

            # Stop if context gets too long (keep last 5 tokens for memory)
            if len(context) > 10:
                context = context[-5:]

        return generated

    # ------------------------------------------------------------------ #
    # Beam search
    # ------------------------------------------------------------------ #
    def beam_search(
        self,
        prompt_tokens: List[str],
        max_tokens: int = 15,
        beam_width: int = 4,
        length_penalty: float = 0.0,
    ) -> List[Tuple[List[str], float]]:
        """Beam search generation.

        Returns list of (sequence, score) sorted by descending score.
        Score is the average log-similarity of the chosen tokens.
        """
        import math

        # Each beam: (tokens_so_far, context_so_far, cumulative_log_score)
        beams: List[Tuple[List[str], List[str], float]] = [([], list(prompt_tokens), 0.0)]
        completed: List[Tuple[List[str], float]] = []

        for step in range(max_tokens):
            new_beams = []
            for tokens, context, score in beams:
                dist = self.predict_distribution(context, top_k=beam_width)
                if not dist:
                    completed.append((tokens, score / max(len(tokens), 1)))
                    continue
                for cand_token, cand_sim in dist[:beam_width]:
                    new_tokens = tokens + [cand_token]
                    new_context = (context + [cand_token])[-5:]
                    # Log-score: similarity can be negative, so use sign-aware
                    log_s = math.log(max(cand_sim, 0.01))
                    new_score = score + log_s
                    if cand_token in ("</s>", "<s>", "<unk>", "<pad>", "."):
                        completed.append((new_tokens, new_score / max(len(new_tokens), 1)))
                    else:
                        new_beams.append((new_tokens, new_context, new_score))

            if not new_beams:
                break

            # Apply length penalty: prefer longer sequences slightly
            new_beams.sort(key=lambda x: x[2] / (len(x[0]) ** length_penalty + 1e-6), reverse=True)
            beams = new_beams[:beam_width]

        # Add remaining beams to completed
        for tokens, _, score in beams:
            completed.append((tokens, score / max(len(tokens), 1)))

        # Sort by score descending
        completed.sort(key=lambda x: -x[1])
        return completed[:beam_width]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        return {
            "orders": list(self.orders),
            "vocab_size": len(self.assoc.vocab),
            "unigram_types": len(self._unigram),
            "unigram_tokens": self._unigram_total,
            "kb_writes": int(self.assoc.kb_store.write_count.sum()),
        }
