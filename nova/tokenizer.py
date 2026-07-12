"""
Tokenizer simple — niveau mot, dynamique.

Pas de BPE, pas de SentencePiece. On veut:
  - Vocab dynamique (apprend au fur et à mesure)
  - Conservation de la ponctuation (utile pour le raisonnement)
  - Minuscules + split whitespace + ponctuation séparée

Le tokenizer maintient:
  - token2id et id2token
  - hd_vectors : vecteur HD aléatoire par token (item memory)
  - hd_pos     : vecteur HD aléatoire par position (role memory)

C'est une "item memory" au sens HDC.
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field

from .hd import HDVector, hd_random


# Regex: mots OU ponctuation
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenise: minuscule, split mot/ponctuation."""
    text = text.lower()
    return _TOKEN_RE.findall(text)


@dataclass
class SimpleTokenizer:
    """
    Tokenizer + item memory HD.

    Chaque token voit son vecteur HD alloué paresseusement à la première
    rencontre. Les vecteurs sont aléatoires bipolaires — c'est ce qui
    garantit la décorrélation dans l'espace HD.
    """
    D: int = 10000
    max_position: int = 256  # au-delà, on wrap (modulo)

    token2id: dict[str, int] = field(init=False, default_factory=dict)
    id2token: list[str] = field(init=False, default_factory=list)
    token_vectors: list[HDVector] = field(init=False, default_factory=list)
    position_vectors: list[HDVector] = field(init=False, default_factory=list)

    _rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self):
        self._rng = np.random.default_rng()
        # Tokens spéciaux
        for tok in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<QUERY>", "<ANS>"]:
            self._add_token(tok)
        # Vecteurs de position pré-alloués
        self.position_vectors = [hd_random(self.D, self._rng)
                                  for _ in range(self.max_position)]

    def _add_token(self, token: str) -> int:
        if token in self.token2id:
            return self.token2id[token]
        idx = len(self.id2token)
        self.token2id[token] = idx
        self.id2token.append(token)
        self.token_vectors.append(hd_random(self.D, self._rng))
        return idx

    # ---------------------------------------------------------------- #
    # API publique
    # ---------------------------------------------------------------- #
    def encode(self, text: str) -> list[int]:
        """Texte -> liste d'IDs. Ajoute les nouveaux tokens au vocab."""
        ids = []
        for tok in tokenize(text):
            ids.append(self._add_token(tok))
        return ids

    def decode(self, ids: list[int]) -> str:
        """IDs -> texte. <PAD> et <UNK> sautés."""
        out = []
        for i in ids:
            if i < len(self.id2token):
                tok = self.id2token[i]
                if tok in ("<PAD>",):
                    continue
                out.append(tok)
        # Jointure simple: espace entre mots, ponctuation collée
        text = ""
        for tok in out:
            if re.match(r"^\w+$", tok) and text and not text.endswith((" ", "(", "[")):
                text += " "
            text += tok
        return text.strip()

    def get_token_vector(self, token: str) -> HDVector:
        """Renvoie le vecteur HD d'un token (l'alloue si nouveau)."""
        idx = self._add_token(token)
        return self.token_vectors[idx]

    def get_id_vector(self, idx: int) -> HDVector:
        return self.token_vectors[idx]

    def get_position_vector(self, pos: int) -> HDVector:
        return self.position_vectors[pos % self.max_position]

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    # ---------------------------------------------------------------- #
    # Persistance
    # ---------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Sauvegarde le vocab et les vecteurs en .npz."""
        token_mat = np.stack([v.vec for v in self.token_vectors]).astype(np.int8)
        pos_mat = np.stack([v.vec for v in self.position_vectors]).astype(np.int8)
        np.savez(path,
                 D=self.D,
                 max_position=self.max_position,
                 id2token=np.array(self.id2token, dtype=object),
                 token_vectors=token_mat,
                 position_vectors=pos_mat)

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.D = int(data["D"])
        self.max_position = int(data["max_position"])
        self.id2token = list(data["id2token"])
        self.token2id = {tok: i for i, tok in enumerate(self.id2token)}
        tv = data["token_vectors"]
        self.token_vectors = [HDVector(tv[i]) for i in range(len(self.id2token))]
        pv = data["position_vectors"]
        self.position_vectors = [HDVector(pv[i]) for i in range(len(pv))]


if __name__ == "__main__":
    tok = SimpleTokenizer(D=1000)
    text = "Bonjour le monde! Je suis NOVA. Calcule 2+2."
    ids = tok.encode(text)
    print(f"Texte: {text}")
    print(f"IDs:   {ids}")
    print(f"Tokens: {[tok.id2token[i] for i in ids]}")
    print(f"Reconstruit: '{tok.decode(ids)}'")
    print(f"Vocab: {tok.vocab_size} tokens")
