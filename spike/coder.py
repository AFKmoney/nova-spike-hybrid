"""
Codeur texte ↔ spikes.

Encodage:
  - Texte → tokens (word-level)
  - Token → index dans le vocabulaire → sous-population de neurones sensoriels
  - Pour chaque token actif, on génère un train de spikes Poisson sur la
    sous-population correspondante (rate coding).

Stratégie d'assignation:
  - On découpe la population sensorielle (n_sensory) en n_tokens_max slots.
  - Slot t = neurones [t*K, (t+1)*K) où K = n_sensory / n_tokens_max.
  - Quand on veut "dire" le token t, on injecte du courant dans son slot.

Décodage (population):
  - La population motrice (n_motor) est découpée en slots (1 par token).
  - Compte des spikes sur une fenêtre temporelle.
  - Argmax → token prédit.
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field

# Regex de tokenisation (mots + ponctuation)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SpikeCoder:
    """
    Codeur texte ↔ spikes.

    Maintient un vocabulaire dynamique avec slot sensoriel et slot moteur
    pour chaque token.

    Args:
        n_sensory: nombre total de neurones sensoriels
        n_motor: nombre total de neurones moteurs
        neurons_per_token: taille d'un slot (sensory et motor)
        max_tokens: taille max du vocabulaire (dérivé)
    """
    n_sensory: int
    n_motor: int
    neurons_per_token: int = 10

    token2id: dict[str, int] = field(init=False, default_factory=dict)
    id2token: list[str] = field(init=False, default_factory=list)
    # Tokens spéciaux
    special_tokens: list[str] = field(init=False,
        default_factory=lambda: ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    _rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self):
        self._rng = np.random.default_rng()
        # Calcule max_tokens AVANT d'ajouter les tokens spéciaux
        max_tokens_sensory = self.n_sensory // self.neurons_per_token
        max_tokens_motor = self.n_motor // self.neurons_per_token
        self.max_tokens = min(max_tokens_sensory, max_tokens_motor)
        # Ajoute les tokens spéciaux
        for tok in self.special_tokens:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token in self.token2id:
            return self.token2id[token]
        if len(self.id2token) >= self.max_tokens:
            # On refuse d'ajouter — vocab plein
            return self.token2id.get("<UNK>", 1)
        idx = len(self.id2token)
        self.token2id[token] = idx
        self.id2token.append(token)
        return idx

    # ---------------------------------------------------------------- #
    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    # ---------------------------------------------------------------- #
    # Encodage
    # ---------------------------------------------------------------- #
    def get_sensory_slot(self, token_id: int) -> slice:
        """Slice des neurones sensoriels pour le token_id."""
        start = token_id * self.neurons_per_token
        end = start + self.neurons_per_token
        if end > self.n_sensory:
            # Wrap-around pour ne pas déborder
            start = start % self.n_sensory
            end = start + self.neurons_per_token
        return slice(start, end)

    def get_motor_slot(self, token_id: int) -> slice:
        """Slice des neurones moteurs pour le token_id."""
        start = token_id * self.neurons_per_token
        end = start + self.neurons_per_token
        if end > self.n_motor:
            start = start % self.n_motor
            end = start + self.neurons_per_token
        return slice(start, end)

    def encode_text_to_current(self, text: str, gain: float = 2.0) -> np.ndarray:
        """
        Convertit un texte en courant sensoriel (injection).

        Pour chaque token du texte, on injecte `gain` sur les neurones
        du slot correspondant.

        Args:
            text: texte à encoder
            gain: amplitude du courant injecté (typiquement > V_thresh)

        Return:
            I_sensory (n_sensory,)
        """
        I = np.zeros(self.n_sensory, dtype=np.float32)
        tokens = tokenize(text)
        for tok in tokens:
            tid = self._add_token(tok)
            slot = self.get_sensory_slot(tid)
            I[slot] += gain / len(tokens)  # normalise par nb de tokens
        return I

    def encode_token_to_current(self, token_id: int, gain: float = 2.0) -> np.ndarray:
        """Encode un seul token en courant."""
        I = np.zeros(self.n_sensory, dtype=np.float32)
        slot = self.get_sensory_slot(token_id)
        I[slot] = gain
        return I

    def encode_poisson(self, text: str, n_ticks: int, rate: float = 0.3,
                       gain: float = 2.0) -> list[np.ndarray]:
        """
        Encode un texte en train de spikes Poisson sur n_ticks.

        Pour chaque tick, on tire des spikes aléatoires dans les slots actifs
        selon une probabilité `rate` par neurone.

        Return:
            Liste de n_ticks arrays (n_sensory,) — courant à injecter.
        """
        I_static = self.encode_text_to_current(text, gain=gain)
        currents = []
        for _ in range(n_ticks):
            # Mask Poisson: pour chaque neurone actif, on tire avec proba rate
            mask = (I_static > 0).astype(np.float32)
            spike_mask = (self._rng.random(self.n_sensory) < rate).astype(np.float32)
            I_tick = I_static * spike_mask
            currents.append(I_tick)
        return currents

    # ---------------------------------------------------------------- #
    # Décodage (population)
    # ---------------------------------------------------------------- #
    def decode_motor_counts(self, spike_counts: np.ndarray) -> list[tuple[str, int]]:
        """
        Décode le compte de spikes moteur en tokens classés par activité.

        Pour chaque token du vocabulaire, on somme les spikes de son slot.
        Renvoie la liste triée (token, count) en ordre décroissant.
        """
        results = []
        for tid in range(self.vocab_size):
            slot = self.get_motor_slot(tid)
            count = int(spike_counts[slot].sum())
            results.append((self.id2token[tid], count))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decode_top_token(self, spike_counts: np.ndarray,
                         exclude_specials: bool = True) -> str | None:
        """
        Renvoie le token avec le plus de spikes moteurs (au-dessus d'un seuil).
        """
        results = self.decode_motor_counts(spike_counts)
        specials = set(self.special_tokens) if exclude_specials else set()
        for tok, count in results:
            if tok in specials:
                continue
            if count > 0:
                return tok
        return None

    def decode_top_k(self, spike_counts: np.ndarray, k: int = 5,
                     exclude_specials: bool = True,
                     min_count: int = 1) -> list[tuple[str, int]]:
        """Top-k tokens par activité motrice."""
        results = self.decode_motor_counts(spike_counts)
        specials = set(self.special_tokens) if exclude_specials else set()
        out = []
        for tok, count in results:
            if tok in specials:
                continue
            if count < min_count:
                continue
            out.append((tok, count))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------- #
# Décoder une séquence de ticks en tokens (génération)
# ---------------------------------------------------------------------- #

@dataclass
class PopulationDecoder:
    """
    Décodeur de séquences: lit l'activité motrice tick par tick et
    reconstitue une séquence de tokens.
    """
    coder: SpikeCoder
    min_count: int = 2  # seuil minimum pour considérer un token actif
    cooldown: int = 3   # ticks de silence après un token (anti-répétition)

    last_emit_tick: int = field(init=False, default=-1000)

    def decode_tick(self, motor_spikes: np.ndarray, tick: int) -> str | None:
        """
        Décodage tick-par-tick. Renvoie un token si l'activité motrice
        dépasse le seuil, sinon None.
        """
        # Cooldown: ne pas émettre juste après un token
        if tick - self.last_emit_tick < self.cooldown:
            return None

        # Compteur de spikes par slot
        top = self.coder.decode_top_k(motor_spikes, k=1, min_count=self.min_count,
                                       exclude_specials=True)
        if not top:
            return None
        tok, count = top[0]
        self.last_emit_tick = tick
        return tok

    def reset(self) -> None:
        self.last_emit_tick = -1000

    def decode_sequence(self, motor_spike_log: list[np.ndarray]) -> list[str]:
        """
        Décode une liste de spikes moteurs (un par tick) en liste de tokens.
        """
        self.reset()
        tokens = []
        for tick, spikes in enumerate(motor_spike_log):
            # spikes est un masque booléen (n_motor,)
            # On veut des comptes — convertit
            counts = spikes.astype(np.int32)
            tok = self.decode_tick(counts, tick)
            if tok:
                tokens.append(tok)
        return tokens


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    coder = SpikeCoder(n_sensory=500, n_motor=100, neurons_per_token=5)
    print(f"Vocab initial: {coder.vocab_size} tokens (spéciaux)")
    print(f"Max tokens: {coder.max_tokens}")

    I = coder.encode_text_to_current("bonjour le monde", gain=2.0)
    print(f"\nTexte 'bonjour le monde' → courant sensoriel:")
    print(f"  Shape: {I.shape}, somme: {I.sum():.2f}")
    print(f"  Neurones actifs: {(I > 0).sum()}")
    print(f"  Vocab après: {coder.vocab_size}")

    # Poisson encoding
    currents = coder.encode_poisson("bonjour", n_ticks=10, rate=0.5)
    print(f"\n10 ticks Poisson pour 'bonjour':")
    for i, c in enumerate(currents):
        print(f"  tick {i}: {(c > 0).sum()} neurones actifs")
