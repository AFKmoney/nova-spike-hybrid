"""
RL fine-tuning — reinforcement learning for the GenerativePredictor.

Implements a simple policy gradient approach:
  1. Generate multiple responses to a prompt
  2. Score each response with a reward function
  3. Update n-gram counts to favor high-reward responses

The reward function can be:
  - Keyword match (does the response contain expected keywords?)
  - Length (penalize too short / too long)
  - Coherence (repetition penalty)
  - Custom (user-provided)

This is NOT backprop — it's direct n-gram count adjustment based on rewards.
"""

from __future__ import annotations
import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Callable, Optional
from dataclasses import dataclass, field


@dataclass
class RLEpisode:
    """A single RL episode: prompt → response → reward."""
    prompt: str
    response: str
    reward: float
    generated_tokens: list[str] = field(default_factory=list)


class RLFineTuner:
    """
    Reinforcement learning fine-tuner for GenerativePredictor.

    Uses a simple policy gradient:
      - Generate N responses per prompt
      - Score each with a reward function
      - For each token in high-reward responses: increase its n-gram count
      - For each token in low-reward responses: decrease its n-gram count

    This is a form of REINFORCE adapted for n-gram models.
    """

    def __init__(self, predictor, learning_rate: float = 0.1):
        """
        Args:
            predictor: GenerativePredictor instance to fine-tune
            learning_rate: how much to adjust counts per episode
        """
        self.predictor = predictor
        self.learning_rate = learning_rate
        self.episodes: list[RLEpisode] = []
        self.reward_history: list[float] = []

    # ---------------------------------------------------------------- #
    # Reward functions
    # ---------------------------------------------------------------- #
    @staticmethod
    def reward_keyword_match(response: str, keywords: list[str]) -> float:
        """Reward based on how many expected keywords are in the response."""
        response_lower = response.lower()
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        return matches / max(1, len(keywords))

    @staticmethod
    def reward_length(response: str, ideal_min: int = 5, ideal_max: int = 30) -> float:
        """Reward based on response length — penalize too short or too long."""
        n_words = len(response.split())
        if n_words < ideal_min:
            return n_words / ideal_min * 0.5
        elif n_words > ideal_max:
            return max(0, 1.0 - (n_words - ideal_max) / ideal_max)
        else:
            return 1.0

    @staticmethod
    def reward_coherence(response: str) -> float:
        """Reward based on coherence — penalize repetition."""
        words = response.lower().split()
        if len(words) < 2:
            return 0.0
        unique = len(set(words))
        return unique / len(words)

    @staticmethod
    def reward_combined(response: str, keywords: list[str] = None,
                         ideal_min: int = 5, ideal_max: int = 30) -> float:
        """Combined reward: keyword match + length + coherence."""
        r_kw = RLFineTuner.reward_keyword_match(response, keywords or [])
        r_len = RLFineTuner.reward_length(response, ideal_min, ideal_max)
        r_coh = RLFineTuner.reward_coherence(response)
        return 0.5 * r_kw + 0.3 * r_len + 0.2 * r_coh

    # ---------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------- #
    def train_episode(self, prompt: str, reward_fn: Callable[[str], float],
                      max_tokens: int = 20, temperature: float = 0.8,
                      n_samples: int = 3) -> RLEpisode:
        """
        Run one RL episode:
          1. Generate n_samples responses
          2. Score each with reward_fn
          3. Use the best response and adjust counts

        Returns the best episode.
        """
        prompt_tokens = self.predictor.tokenize(prompt)

        # Generate n_samples responses
        responses = []
        for _ in range(n_samples):
            tokens = self.predictor.generate(
                prompt_tokens, max_tokens=max_tokens,
                temperature=temperature, top_p=0.9,
                repetition_penalty=1.2,
            )
            text = self.predictor.detokenize(prompt_tokens + tokens)
            responses.append((tokens, text))

        # Score each response
        scored = []
        for tokens, text in responses:
            reward = reward_fn(text)
            scored.append((tokens, text, reward))

        # Find best and worst
        scored.sort(key=lambda x: x[2], reverse=True)
        best_tokens, best_text, best_reward = scored[0]
        worst_reward = scored[-1][2]

        # Compute advantage (relative reward)
        mean_reward = np.mean([s[2] for s in scored])
        advantage = best_reward - mean_reward

        # Update n-gram counts based on advantage
        # If advantage > 0: increase counts for tokens in best response
        # If advantage < 0: decrease counts
        if abs(advantage) > 0.01:
            self._update_counts(prompt_tokens, best_tokens, advantage)

        # Record episode
        episode = RLEpisode(
            prompt=prompt,
            response=best_text,
            reward=best_reward,
            generated_tokens=best_tokens,
        )
        self.episodes.append(episode)
        self.reward_history.append(best_reward)

        return episode

    def _update_counts(self, prompt_tokens: list[str],
                        generated_tokens: list[str], advantage: float) -> None:
        """
        Adjust n-gram counts based on advantage.
        Positive advantage → increase counts (reinforce)
        Negative advantage → decrease counts (discourage)
        """
        all_tokens = ["<s>"] + prompt_tokens + generated_tokens + ["</s>"]
        adjustment = self.learning_rate * advantage

        # Update for each order
        for n in self.predictor.orders:
            if n < 2:
                continue
            for i in range(len(all_tokens) - n + 1):
                ngram = tuple(all_tokens[i:i + n])
                # Increase or decrease count
                current = self.predictor.ngram_counts[n].get(ngram, 0)
                new_count = max(0, current + adjustment)
                self.predictor.ngram_counts[n][ngram] = new_count

                # Also update context counts
                context = ngram[:-1]
                ctx_current = self.predictor.context_counts[n].get(context, 0)
                self.predictor.context_counts[n][context] = max(0, ctx_current + adjustment)

        # Clear cache since counts changed
        self.predictor._dist_cache.clear()

    # ---------------------------------------------------------------- #
    # Batch training
    # ---------------------------------------------------------------- #
    def train_batch(self, prompts: list[str],
                     reward_fn: Callable[[str], float],
                     max_tokens: int = 20,
                     n_samples: int = 3,
                     verbose: bool = False) -> dict:
        """Train on a batch of prompts."""
        rewards = []
        for prompt in prompts:
            episode = self.train_episode(
                prompt, reward_fn,
                max_tokens=max_tokens,
                n_samples=n_samples,
            )
            rewards.append(episode.reward)
            if verbose:
                print(f"  [{prompt[:30]}...] reward={episode.reward:.3f}")
                print(f"    response: {episode.response[:60]}...")

        return {
            "n_episodes": len(prompts),
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
            "reward_trend": rewards,
        }

    # ---------------------------------------------------------------- #
    # Stats
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        return {
            "n_episodes": len(self.episodes),
            "mean_reward": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "learning_rate": self.learning_rate,
            "reward_history_length": len(self.reward_history),
            "recent_rewards": self.reward_history[-10:] if self.reward_history else [],
        }


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from generative.predictor import GenerativePredictor

    # Train predictor on a small corpus
    corpus = """
    The cat is a small domesticated carnivorous mammal. Cats are known for their agility.
    Paris is the capital of France. Paris is located on the Seine river.
    The sun is a star at the center of the solar system. The sun provides energy.
    Water is a chemical compound. Water is essential for life.
    Einstein developed the theory of relativity. Einstein was a physicist.
    """

    pred = GenerativePredictor(orders=(1, 2, 3, 4, 5), use_bpe=False)
    pred.train_corpus(corpus)

    # Create RL fine-tuner
    tuner = RLFineTuner(pred, learning_rate=0.5)

    # Define a reward function: prefer responses containing "paris"
    def reward_paris(response: str) -> float:
        return RLFineTuner.reward_combined(response, keywords=["paris", "capital"])

    # Train on a batch
    print("=== RL Fine-tuning ===")
    prompts = [
        "What is the capital of France?",
        "Tell me about Paris",
        "What is Paris?",
    ]

    print("\nBefore training:")
    for p in prompts:
        r = pred.generate_text(p, max_tokens=10)
        print(f"  {p!r} → {r!r}")

    result = tuner.train_batch(prompts, reward_paris, n_samples=3, verbose=True)

    print(f"\nTraining result: mean_reward={result['mean_reward']:.3f}")

    print("\nAfter training:")
    for p in prompts:
        r = pred.generate_text(p, max_tokens=10)
        print(f"  {p!r} → {r!r}")

    print(f"\nTuner stats: {tuner.stats()}")
