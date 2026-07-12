"""
neuromodulators.py — Neuromodulation system.

BRAIN INSPIRATION
-----------------
The brain has specialized neuromodulator systems that don't transmit
specific content but instead MODULATE the global state of the brain:

  - DOPAMINE (VTA, SNc): reward prediction error. Drives learning,
    motivation, and goal-directed behavior. Phasic bursts = "better
    than expected"; dips = "worse than expected".

  - SEROTONIN (raphe nuclei): mood, patience, long-term planning.
    High serotonin = willing to wait for bigger reward; low = impulsive.

  - ACETYLCHOLINE (basal forebrain): attention, learning rate.
    High ACh = focused attention, faster learning, sharper cortical
    responses; low = broad attention, slower learning.

  - NOREPINEPHRINE (locus coeruleus): arousal, exploration vs exploitation.
    Tonic high = stressed, vigilant; phasic = focused exploitation; low =
    drowsy; intermediate = optimal exploration.

AETHER'S USE
------------
We implement a NeuromodulatorSystem with 4 levels (one per modulator).
Each modulator is a scalar [0, 1] that GATES the behavior of other modules:

  - DOPAMINE gates learning rate and reward-driven updates
  - SEROTONIN gates patience (how long to deliberate before acting)
  - ACETYLCHOLINE gates attention focus (sharp vs broad)
  - NOREPINEPHRINE gates exploration vs exploitation

The agent's behavior SHIFTS based on these levels — like a human's.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Neuromodulator levels
# ---------------------------------------------------------------------------

@dataclass
class NeuromodulatorLevels:
    """Current levels of all four neuromodulators, all in [0, 1]."""
    dopamine: float = 0.5       # reward / learning drive
    serotonin: float = 0.5      # patience / long-term planning
    acetylcholine: float = 0.5  # attention / learning rate
    norepinephrine: float = 0.5 # arousal / exploration

    def copy(self) -> "NeuromodulatorLevels":
        return NeuromodulatorLevels(
            self.dopamine, self.serotonin, self.acetylcholine, self.norepinepholine()
            if False else self.norepinephrine  # fix typo guard
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
        }


# ---------------------------------------------------------------------------
# Reward signal
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    """A reward prediction error signal (RPE)."""
    predicted_reward: float
    actual_reward: float
    rpe: float  # actual - predicted
    timestamp: int = 0


# ---------------------------------------------------------------------------
# Neuromodulator system
# ---------------------------------------------------------------------------

class NeuromodulatorSystem:
    """Tracks and updates neuromodulator levels based on agent experience.

    The system receives:
      - Reward signals (drives dopamine)
      - Surprise signals (drives acetylcholine, norepinephrine)
      - Time elapsed without reward (drives serotonin down)
      - Task success/failure (drives global modulation)

    It outputs current modulator levels, which other modules read to
    adjust their behavior.
    """

    def __init__(
        self,
        initial_levels: Optional[NeuromodulatorLevels] = None,
        decay_rate: float = 0.95,        # exponential decay toward baseline
        baseline: float = 0.5,
        dopamine_boost: float = 0.4,     # how much RPE boosts dopamine
        ach_boost: float = 0.3,          # how much surprise boosts ACh
        ne_boost: float = 0.3,           # how much surprise boosts NE
        ser_drop: float = 0.05,          # how much time without reward drops serotonin
    ):
        self.levels = initial_levels or NeuromodulatorLevels()
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.dopamine_boost = dopamine_boost
        self.ach_boost = ach_boost
        self.ne_boost = ne_boost
        self.ser_drop = ser_drop

        # History for analysis
        self.history: List[NeuromodulatorLevels] = [self._copy_levels()]
        self.reward_history: List[RewardSignal] = []
        self.cycle: int = 0
        self.cycles_since_reward: int = 0
        self.predicted_reward: float = 0.0

    def _copy_levels(self) -> NeuromodulatorLevels:
        return NeuromodulatorLevels(
            dopamine=self.levels.dopamine,
            serotonin=self.levels.serotonin,
            acetylcholine=self.levels.acetylcholine,
            norepinephrine=self.levels.norepinephrine,
        )

    def _clip(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    # ------------------------------------------------------------------ #
    # Update loop
    # ------------------------------------------------------------------ #
    def update(
        self,
        reward: float = 0.0,
        surprise: float = 0.0,
        success: Optional[bool] = None,
    ) -> NeuromodulatorLevels:
        """Update modulator levels based on this cycle's experience.

        Args:
            reward: actual reward received this cycle (any real number)
            surprise: surprise signal [0, 1] from the predictive model
            success: optional explicit success/failure flag
        """
        self.cycle += 1

        # 1. Dopamine: reward prediction error
        rpe = reward - self.predicted_reward
        reward_signal = RewardSignal(
            predicted_reward=self.predicted_reward,
            actual_reward=reward,
            rpe=rpe,
            timestamp=self.cycle,
        )
        self.reward_history.append(reward_signal)
        if len(self.reward_history) > 100:
            self.reward_history = self.reward_history[-100:]

        # Dopamine responds to RPE (phasic) — bursts on positive, dips on negative
        dopamine_response = rpe * self.dopamine_boost
        self.levels.dopamine = self._clip(self.levels.dopamine * self.decay_rate + dopamine_response + 0.05)

        # Update predicted reward (running average of actual)
        self.predicted_reward = 0.9 * self.predicted_reward + 0.1 * reward

        # 2. Acetylcholine: surprise boosts ACh (focus attention)
        ach_response = surprise * self.ach_boost
        self.levels.acetylcholine = self._clip(self.levels.acetylcholine * self.decay_rate + ach_response + 0.05)

        # 3. Norepinephrine: surprise boosts NE (arousal)
        ne_response = surprise * self.ne_boost
        self.levels.norepinephrine = self._clip(self.levels.norepinephrine * self.decay_rate + ne_response + 0.05)

        # 4. Serotonin: drops with time without reward, boosts with success
        if reward > 0 or success is True:
            self.cycles_since_reward = 0
            self.levels.serotonin = self._clip(self.levels.serotonin + 0.1)
        else:
            self.cycles_since_reward += 1
            self.levels.serotonin = self._clip(self.levels.serotonin - self.ser_drop)

        # If explicit failure, drop serotonin more
        if success is False:
            self.levels.serotonin = self._clip(self.levels.serotonin - 0.15)

        # Record history
        self.history.append(self._copy_levels())
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return self._copy_levels()

    # ------------------------------------------------------------------ #
    # Behavior gating
    # ------------------------------------------------------------------ #
    def learning_rate(self, base_rate: float = 0.1) -> float:
        """Compute the effective learning rate (modulated by dopamine and ACh)."""
        # Both dopamine and ACh boost learning
        return base_rate * (1.0 + self.levels.dopamine + self.levels.acetylcholine)

    def patience(self, base_cycles: int = 5) -> int:
        """Compute the effective patience (cycles to deliberate before acting)."""
        # Higher serotonin = more patient
        return max(1, int(base_cycles * (0.5 + self.levels.serotonin)))

    def attention_focus(self) -> float:
        """How focused is the attention? [0, 1]. High ACh = focused."""
        return self.levels.acetylcholine

    def exploration_rate(self) -> float:
        """Exploration vs exploitation rate [0, 1]. High NE = explore."""
        # Intermediate NE is optimal for exploration
        ne = self.levels.norepinephrine
        # Bell curve centered at 0.5
        return 1.0 - abs(ne - 0.5) * 2.0

    def exploitation_rate(self) -> float:
        """Exploitation rate [0, 1] (inverse of exploration)."""
        return 1.0 - self.exploration_rate()

    def mood(self) -> str:
        """Categorical mood label based on modulator levels."""
        d, s, a, n = (self.levels.dopamine, self.levels.serotonin,
                      self.levels.acetylcholine, self.levels.norepinephrine)
        if d > 0.7 and s > 0.5:
            return "motivated"
        if d < 0.3 and s < 0.3:
            return "depressed"
        if n > 0.7:
            return "alert"
        if n < 0.3:
            return "drowsy"
        if a > 0.7:
            return "focused"
        if s > 0.7:
            return "patient"
        return "neutral"

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, any]:
        return {
            "cycle": self.cycle,
            "levels": self.levels.as_dict(),
            "mood": self.mood(),
            "learning_rate": self.learning_rate(),
            "patience": self.patience(),
            "attention_focus": self.attention_focus(),
            "exploration_rate": self.exploration_rate(),
            "cycles_since_reward": self.cycles_since_reward,
            "predicted_reward": self.predicted_reward,
        }

    def history_array(self) -> np.ndarray:
        """Return history as a (T, 4) numpy array."""
        return np.array([[l.dopamine, l.serotonin, l.acetylcholine, l.norepinephrine]
                         for l in self.history])

    def reset(self) -> None:
        self.levels = NeuromodulatorLevels()
        self.history = [self._copy_levels()]
        self.reward_history.clear()
        self.cycle = 0
        self.cycles_since_reward = 0
        self.predicted_reward = 0.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Neuromodulator System Test ===\n")

    sys = NeuromodulatorSystem()

    print(f"  Initial state: {sys.stats()}")
    print()

    # Simulate a sequence of events
    events = [
        # (reward, surprise, success, description)
        (0.0, 0.5, None, "Mild surprise, no reward"),
        (1.0, 0.0, True, "Success! Reward received"),
        (0.0, 0.0, None, "Nothing happens"),
        (0.0, 0.0, None, "Nothing happens"),
        (0.0, 0.0, None, "Nothing happens"),
        (-0.5, 0.7, False, "Failure! Surprise + negative reward"),
        (0.0, 0.0, None, "Recovery"),
        (2.0, 0.0, True, "Big success!"),
    ]

    print("  Event sequence:")
    for reward, surprise, success, desc in events:
        sys.update(reward=reward, surprise=surprise, success=success)
        mood = sys.mood()
        lr = sys.learning_rate()
        pat = sys.patience()
        expl = sys.exploration_rate()
        print(f"    [{desc}]")
        print(f"      DA={sys.levels.dopamine:.2f}  5HT={sys.levels.serotonin:.2f}  "
              f"ACh={sys.levels.acetylcholine:.2f}  NE={sys.levels.norepinephrine:.2f}")
        print(f"      mood={mood!r}  learning_rate={lr:.3f}  patience={pat}  exploration={expl:.3f}")
        print()

    print(f"  Final stats: {sys.stats()}")
