"""
Décodeur HD -> texte (tokens).

Le décodage repose sur la propriété d'auto-réversibilité du binding
et sur la "cleanup memory" (mémoire de nettoyage).

Pour décoder une séquence encodée par encode_sequence:
  1. Pour chaque position i, on "un-permute" de -i crans
  2. On teste la similarité avec chaque vecteur de token connu
  3. Le token le plus similaire (au-dessus d'un seuil) est retenu
  4. On s'arrête quand la similarité maximale tombe sous le seuil

Pour décoder une paire clé-valeur, on unbind puis on cleanup.
"""

from __future__ import annotations
import numpy as np

from .hd import (
    HDVector,
    hd_bind,
    hd_permute,
    hd_similarity,
)
from .tokenizer import SimpleTokenizer
from .encoder import HDEncoder


class HDDecoder:
    """Décodeur HD -> tokens / texte."""

    def __init__(self, tokenizer: SimpleTokenizer, encoder: HDEncoder,
                 sim_threshold: float = 0.15):
        self.tok = tokenizer
        self.enc = encoder
        self.sim_threshold = sim_threshold

    # ---------------------------------------------------------------- #
    # Décodage de séquence
    # ---------------------------------------------------------------- #
    def decode_sequence(self, vec: HDVector, max_len: int = 32,
                        permute_positions: bool = True) -> list[int]:
        """
        Décode un vecteur HD en liste d'IDs de tokens.
        Retourne [] si la similarité maximale est trop faible partout.
        """
        ids: list[int] = []
        # Matrice des vecteurs de tokens (N, D) — on la construit une fois
        if not self.tok.token_vectors:
            return ids
        token_mat = np.stack([v.vec for v in self.tok.token_vectors]).astype(np.int8)

        for i in range(max_len):
            if permute_positions:
                # Un-permute de -i crans
                candidate = hd_permute(vec, -i)
            else:
                pos_vec = self.tok.get_position_vector(i)
                candidate = hd_bind(vec, pos_vec)

            # Similarité avec chaque token
            sims = (token_mat.astype(np.float32)
                    @ candidate.vec.astype(np.float32)) / self.tok.D

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            # On s'arrête si on tombe sur <PAD> ou si la sim est trop faible
            if best_sim < self.sim_threshold:
                break
            tok = self.tok.id2token[best_idx]
            if tok in ("<PAD>", "<EOS>"):
                break
            ids.append(best_idx)

        return ids

    def decode_text(self, vec: HDVector, max_len: int = 32,
                    permute_positions: bool = True) -> str:
        """Shortcut: HD -> texte."""
        ids = self.decode_sequence(vec, max_len, permute_positions)
        return self.tok.decode(ids)

    # ---------------------------------------------------------------- #
    # Décodage de paire (unbind + cleanup)
    # ---------------------------------------------------------------- #
    def unbind_and_cleanup(self, pair_vec: HDVector, key_vec: HDVector,
                           references: dict[str, HDVector],
                           threshold: float = 0.0) -> tuple[str | None, float]:
        """
        Récupère la valeur d'une paire (key, value) encodée par binding.
        Étapes:
          1. unbound = bind(pair_vec, key_vec)  ≈ value_vec (bruité)
          2. cleanup: trouve le vecteur de référence le plus proche
        """
        unbound = hd_bind(pair_vec, key_vec)
        best_key = None
        best_sim = -2.0
        for k, ref in references.items():
            s = hd_similarity(unbound, ref)
            if s > best_sim:
                best_sim = s
                best_key = k
        if best_sim < threshold:
            return None, best_sim
        return best_key, best_sim

    # ---------------------------------------------------------------- #
    # Génération symbolique (recherche gloutonne)
    # ---------------------------------------------------------------- #
    def generate_greedy(self, context_vec: HDVector, memory_read_fn,
                        max_tokens: int = 32) -> list[int]:
        """
        Génère une séquence de tokens à partir d'un contexte.

        Pour chaque position:
          1. On lit dans la SDM à l'adresse context_vec
          2. On un-permute pour récupérer le prochain token
          3. On bind le token prédit à sa position et on l'ajoute au contexte
          4. On continue

        Args:
            context_vec: vecteur HD du contexte initial
            memory_read_fn: callable(HDVector) -> HDVector (rappel SDM)
            max_tokens: longueur max générée
        """
        ids: list[int] = []
        current = context_vec
        token_mat = (np.stack([v.vec for v in self.tok.token_vectors]).astype(np.int8)
                     if self.tok.token_vectors else None)
        if token_mat is None:
            return ids

        for i in range(max_tokens):
            # Récupère de la mémoire
            retrieved = memory_read_fn(current)

            # Un-permute à la position courante
            candidate = hd_permute(retrieved, -i)

            # Argmax sur le vocabulaire
            sims = (token_mat.astype(np.float32)
                    @ candidate.vec.astype(np.float32)) / self.tok.D
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            tok = self.tok.id2token[best_idx]
            if tok in ("<PAD>", "<EOS>") or best_sim < self.sim_threshold:
                break
            ids.append(best_idx)

            # Met à jour le contexte en ajoutant le token prédit
            # nouveau_contexte = bundle(contexte, permute(token_vec, i))
            tv = self.tok.get_id_vector(best_idx)
            permuted = hd_permute(tv, i)
            # Pour que le bundle reste prédominant pour le dernier token,
            # on pondère (2 * nouveau + ancien).
            from .hd import hd_bundle_weighted
            current = hd_bundle_weighted([(permuted, 2.0), (current, 1.0)])

        return ids


if __name__ == "__main__":
    tok = SimpleTokenizer(D=2000)
    enc = HDEncoder(tok)
    dec = HDDecoder(tok, enc)

    text = "le chat dort sur le tapis"
    v = enc.encode_text(text)
    decoded = dec.decode_text(v)
    print(f"Original:  {text}")
    print(f"Encodé:    {v.vec[:10]}...")
    print(f"Décodé:    {decoded}")
