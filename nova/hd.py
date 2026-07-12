"""
Noyau hyperdimensional de NOVA.

Vecteurs bipolaires {-1, +1}^D.
Opérations:
  - bind(a, b)       : produit élément-wise (réversible, commute pas)
  - bundle(*vs)      : somme + signe (superposition, lossy)
  - permute(v, k)    : rotation circulaire (marqueur de position)
  - similarity(a, b) : produit scalaire normalisé (cosinus)
  - distance(a, b)   : distance de Hamming
  - cleanup(v, ref)  : retrouve le vecteur de référence le plus proche

Toutes les opérations sont O(D) et tournent sur CPU. Avec D=10000 et numpy
on est largement sub-millisecond.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class HDVector:
    """Un vecteur hyperdimensionnel bipolaire."""
    vec: np.ndarray  # dtype=int8, shape=(D,)

    @property
    def dim(self) -> int:
        return self.vec.shape[0]

    def __len__(self) -> int:
        return self.dim


# ---------------------------------------------------------------------- #
# Génération
# ---------------------------------------------------------------------- #
def hd_random(D: int, rng: np.random.Generator | None = None) -> HDVector:
    """Vecteur aléatoire uniforme bipolaire."""
    rng = rng or np.random.default_rng()
    return HDVector(rng.choice([-1, 1], size=D).astype(np.int8))


def hd_random_dense(D: int, rng: np.random.Generator | None = None) -> HDVector:
    """Variante dense (float) pour états continus — réservé au résonateur."""
    rng = rng or np.random.default_rng()
    return HDVector(rng.standard_normal(D).astype(np.float32))


# ---------------------------------------------------------------------- #
# Opérations algébriques
# ---------------------------------------------------------------------- #
def hd_bind(a: HDVector, b: HDVector) -> HDVector:
    """
    Binding : produit élément-wise.
    Propriété clé : bind(a, bind(b, c)) = bind(bind(a,b), c) (associatif)
                    bind(a, b) ⊥ a, b en moyenne (décorrelation)
                    self-inverse : bind(bind(a,b), b) = a
    """
    assert a.dim == b.dim, f"dim mismatch {a.dim} vs {b.dim}"
    return HDVector(a.vec * b.vec)


def hd_bundle(*vectors: HDVector) -> HDVector:
    """
    Superposition : somme puis signe.
    Avec n vecteurs aléatoires indépendants, le bundle reste similaire à
    chacun (similarité ~ 1/sqrt(n)) tout en les contenant tous.
    Tie-breaking aléatoire pour les sommes nulles.
    """
    if not vectors:
        raise ValueError("bundle needs at least 1 vector")
    D = vectors[0].dim
    acc = np.zeros(D, dtype=np.int32)
    for v in vectors:
        assert v.dim == D
        acc += v.vec.astype(np.int32)
    # sign function: +1 if >0, -1 if <0, random if ==0
    out = np.sign(acc).astype(np.int8)
    zeros = out == 0
    if zeros.any():
        rng = np.random.default_rng(int(acc.sum()) & 0xFFFF)
        out[zeros] = rng.choice([-1, 1], size=int(zeros.sum())).astype(np.int8)
    return HDVector(out)


def hd_bundle_weighted(pairs: list[tuple[HDVector, float]]) -> HDVector:
    """Bundle pondéré — utile pour donner plus de poids à certains vecteurs."""
    if not pairs:
        raise ValueError("need at least 1 pair")
    D = pairs[0][0].dim
    acc = np.zeros(D, dtype=np.float32)
    for v, w in pairs:
        assert v.dim == D
        acc += w * v.vec.astype(np.float32)
    out = np.sign(acc).astype(np.int8)
    zeros = out == 0
    if zeros.any():
        rng = np.random.default_rng(int(acc.sum() * 1000) & 0xFFFF)
        out[zeros] = rng.choice([-1, 1], size=int(zeros.sum())).astype(np.int8)
    return HDVector(out)


def hd_permute(v: HDVector, k: int) -> HDVector:
    """
    Permutation (rotation circulaire) — utilisée pour marquer la position
    dans une séquence. Permute(v, 0) = v, Permute(v, k) = Permute(Permute(v, k-1), 1).
    """
    k = k % v.dim
    if k == 0:
        return HDVector(v.vec.copy())
    return HDVector(np.concatenate([v.vec[-k:], v.vec[:-k]]).astype(np.int8))


# ---------------------------------------------------------------------- #
# Mesures de (dis)similarité
# ---------------------------------------------------------------------- #
def hd_similarity(a: HDVector, b: HDVector) -> float:
    """
    Similarité cosinus dans [-1, 1]. Pour des vecteurs bipolaires,
    équivalent à <a,b>/D.

    IMPORTANT: int8 * int8 = int8 (overflow) — on convertit en int32 avant.
    """
    assert a.dim == b.dim
    return float(np.dot(a.vec.astype(np.int32), b.vec.astype(np.int32)) / a.dim)


def hd_distance(a: HDVector, b: HDVector) -> int:
    """Distance de Hamming — nombre de positions qui diffèrent."""
    assert a.dim == b.dim
    return int(np.count_nonzero(a.vec != b.vec))


# ---------------------------------------------------------------------- #
# Cleanup — mémoire associative
# ---------------------------------------------------------------------- #
def hd_cleanup(
    query: HDVector,
    references: dict[str, HDVector],
    threshold: float = 0.0,
) -> tuple[str | None, float]:
    """
    Retrouve le vecteur de référence le plus similaire à `query`.
    Renvoie (clé, similarité) ou (None, sim) si en-dessous du seuil.
    C'est la primitive de base du rappel associatif.
    """
    best_key = None
    best_sim = -2.0
    for key, ref in references.items():
        s = hd_similarity(query, ref)
        if s > best_sim:
            best_sim = s
            best_key = key
    if best_sim < threshold:
        return None, best_sim
    return best_key, best_sim


# ---------------------------------------------------------------------- #
# Démo / smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    D = 10000

    a = hd_random(D, rng)
    b = hd_random(D, rng)
    c = hd_random(D, rng)

    print(f"sim(a, b) = {hd_similarity(a, b):+.4f}  (attendu ~ 0)")
    print(f"sim(a, a) = {hd_similarity(a, a):+.4f}  (attendu = 1)")

    bound = hd_bind(a, b)
    print(f"sim(bind(a,b), a) = {hd_similarity(bound, a):+.4f}  (attendu ~ 0)")

    # self-inverse
    recovered = hd_bind(bound, b)
    print(f"sim(bind(bind(a,b), b), a) = {hd_similarity(recovered, a):+.4f}  (attendu = 1)")

    # bundle
    bun = hd_bundle(a, b, c)
    print(f"sim(bundle, a) = {hd_similarity(bun, a):+.4f}  (attendu ~ 1/sqrt(3) = {1/3**0.5:.4f})")

    # permutation
    p1 = hd_permute(a, 1)
    print(f"sim(perm(a,1), a) = {hd_similarity(p1, a):+.4f}  (attendu ~ 0)")
    print(f"sim(perm(perm(a,1),-1), a) = {hd_similarity(hd_permute(p1, -1), a):+.4f}  (attendu = 1)")

    # cleanup
    refs = {"chat": a, "chien": b, "oiseau": c}
    key, sim = hd_cleanup(bun, refs)
    print(f"cleanup(bundle) -> {key} (sim={sim:+.4f})")
