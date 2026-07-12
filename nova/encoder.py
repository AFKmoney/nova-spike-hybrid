"""
Encodeur texte -> vecteurs hyperdimensionnels.

Stratégie d'encodage (TPR — Tensor Product Representation light):

  Pour une séquence de tokens [t0, t1, ..., tn-1]:
    seq_vec = bundle( bind(token_vec(ti), position_vec(i)) for i in range(n) )

  Chaque token est "lié" à sa position (binding), puis on superpose
  le tout (bundle). Le résultat est un vecteur unique D-dim qui
  représente toute la séquence — taille invariante.

Pour les paires clé/valeur (mémoire associative), on utilise:
    pair_vec = bind(key_vec, value_vec)

Pour les relations sémantiques (sujet/objet/verbe), on peut définir
des "roles" et faire:
    relation_vec = bundle( bind(role_r, filler_f) for r, f in pairs )
"""

from __future__ import annotations
import numpy as np

from .hd import (
    HDVector,
    hd_random,
    hd_bind,
    hd_bundle,
    hd_permute,
)
from .tokenizer import SimpleTokenizer


# Roles sémantiques prédéfinis (vecteurs HD aléatoires)
ROLES = ["subject", "verb", "object", "predicate", "fact", "query",
         "answer", "context", "tool", "result", "self", "other",
         "before", "after", "cause", "effect", "agent", "patient"]


class HDEncoder:
    """Encodeur texte -> HD."""

    def __init__(self, tokenizer: SimpleTokenizer):
        self.tok = tokenizer
        self.D = tokenizer.D
        self._rng = np.random.default_rng(42)
        self.roles = {r: hd_random(self.D, self._rng) for r in ROLES}

    # ---------------------------------------------------------------- #
    # Encodage de séquences
    # ---------------------------------------------------------------- #
    def encode_sequence(self, tokens: list[str] | list[int],
                        permute_positions: bool = True) -> HDVector:
        """
        Encode une séquence de tokens en un vecteur HD unique.

        Args:
            tokens: liste de strings ou d'IDs
            permute_positions: si True, utilise la permutation circulaire
                (plus robuste). Sinon, utilise des vecteurs de position
                aléatoires indépendants.

        Return:
            Un vecteur HD bipolaire D-dim.
        """
        if not tokens:
            return HDVector(np.zeros(self.D, dtype=np.int8))

        bound_vectors: list[HDVector] = []
        for i, t in enumerate(tokens):
            if isinstance(t, str):
                tv = self.tok.get_token_vector(t)
            else:
                tv = self.tok.get_id_vector(t)
            if permute_positions:
                # Permutation circulaire de i crans — séquence de Markov
                pv = hd_permute(tv, i)
            else:
                pos_vec = self.tok.get_position_vector(i)
                pv = hd_bind(tv, pos_vec)
            bound_vectors.append(pv)

        return hd_bundle(*bound_vectors)

    def encode_text(self, text: str, permute_positions: bool = True) -> HDVector:
        """Shortcut: texte -> HD."""
        tokens = self.tok.encode(text)
        return self.encode_sequence(tokens, permute_positions)

    # ---------------------------------------------------------------- #
    # Encodage de paires / relations
    # ---------------------------------------------------------------- #
    def encode_pair(self, key: HDVector, value: HDVector) -> HDVector:
        """Paire clé-valeur = binding."""
        return hd_bind(key, value)

    def encode_relation(self, role: str, filler: HDVector) -> HDVector:
        """Relation role-filler = binding."""
        assert role in self.roles, f"role inconnu: {role}"
        return hd_bind(self.roles[role], filler)

    def encode_facts(self, facts: list[tuple[str, HDVector]]) -> HDVector:
        """
        Encode un ensemble de (role, filler) en un seul vecteur HD.
        C'est l'encodage canonique d'une "pensée" structurée.
        """
        parts = [self.encode_relation(r, f) for r, f in facts]
        return hd_bundle(*parts)

    # ---------------------------------------------------------------- #
    # Rôles
    # ---------------------------------------------------------------- #
    def get_role(self, name: str) -> HDVector:
        return self.roles[name]

    def add_role(self, name: str) -> HDVector:
        if name not in self.roles:
            self.roles[name] = hd_random(self.D, self._rng)
        return self.roles[name]


if __name__ == "__main__":
    tok = SimpleTokenizer(D=2000)
    enc = HDEncoder(tok)

    v1 = enc.encode_text("le chat dort")
    v2 = enc.encode_text("le chat dort")
    v3 = enc.encode_text("le chien dort")
    v4 = enc.encode_text("dort le chat")  # ordre différent

    from .hd import hd_similarity
    print(f"sim('le chat dort', 'le chat dort')     = {hd_similarity(v1, v2):+.4f}")
    print(f"sim('le chat dort', 'le chien dort')    = {hd_similarity(v1, v3):+.4f}")
    print(f"sim('le chat dort', 'dort le chat')     = {hd_similarity(v1, v4):+.4f}")
