"""
Mémoire Distribuée Sparse (SDM) — Kanerva 1988.

Principe:
  - On tire N "hard locations" aléatoires dans l'espace HD (D-dim).
  - Écrire(address, data) : pour chaque location dans un rayon de `address`,
    on ajoute `data` (compteurs entiers) à son contenu.
  - Lire(address) : on moyenne les contenus de toutes les locations
    dans le rayon, on prend le signe → vecteur reconstruit.

Propriétés:
  - Content-addressable : pas besoin de clé, on cherche par similarité.
  - One-shot : une seule écriture suffit.
  - Graceful degradation : oublie proprement quand saturée.
  - Robustesse au bruit : lit même si l'address est bruitée (jusqu'à ~D/4).

Complexité:
  - Écriture: O(N_active * D) où N_active << N (seulement les locations proches)
  - Lecture: idem
  - Stockage: O(N * D) entiers — mais on peut utiliser int8 pour les compteurs
    saturés.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from .hd import HDVector, hd_random, hd_similarity


@dataclass
class SparseDistributedMemory:
    """
    Mémoire distribuée sparse de Kanerva.

    Attributes:
        D: dimension des vecteurs HD
        n_locations: nombre de hard locations (typiquement 1000-100000)
        k_active: nombre de locations activées en top-k (au lieu d'un
            seuil absolu — plus robuste aux distributions empiriques).
        activation_radius: rayon d'activation en similarité (0-1).
            Si > 0, on garde seulement les locations au-dessus du seuil
            parmi le top-k. 0 = on garde toutes les top-k.
        max_count: saturation des compteurs (évite l'overflow, agit comme
            un forget factor naturel).
    """
    D: int
    n_locations: int = 10000
    k_active: int = 32
    activation_radius: float = 0.0
    max_count: int = 127  # int8 signed

    # Etat interne
    locations: np.ndarray = field(init=False)   # (N, D) int8 — adresses
    locations_f32: np.ndarray = field(init=False)  # (N, D) float32 — cache
    contents: np.ndarray = field(init=False)    # (N, D) int8 — compteurs
    writes: int = field(init=False, default=0)

    def __post_init__(self):
        rng = np.random.default_rng()
        # Adresses aléatoires bipolaires
        self.locations = rng.choice([-1, 1],
                                    size=(self.n_locations, self.D)).astype(np.int8)
        # Cache float32 pour les similarités (pré-converti une fois)
        self.locations_f32 = self.locations.astype(np.float32)
        # Compteurs à zéro
        self.contents = np.zeros((self.n_locations, self.D), dtype=np.int8)
        self.writes = 0

    # ---------------------------------------------------------------- #
    # Activation — top-k par similarité
    # ---------------------------------------------------------------- #
    def _activate(self, address: np.ndarray) -> np.ndarray:
        """
        Renvoie les indices des k_active locations les plus similaires
        à `address`. Si activation_radius > 0, filtre par seuil ensuite.
        """
        # (N,) = (N,D) @ (D,) — float32 partout
        sims = self.locations_f32 @ address.astype(np.float32) / self.D
        # Top-k
        k = min(self.k_active, self.n_locations)
        if k >= self.n_locations:
            active_idx = np.arange(self.n_locations)
        else:
            # argpartition pour O(N) au lieu de O(N log N)
            active_idx = np.argpartition(-sims, k - 1)[:k]
        # Filtre optionnel par seuil
        if self.activation_radius > 0:
            mask = sims[active_idx] > self.activation_radius
            active_idx = active_idx[mask]
        return active_idx

    # ---------------------------------------------------------------- #
    # Écriture — apprentissage one-shot
    # ---------------------------------------------------------------- #
    def write(self, address: HDVector, data: HDVector) -> None:
        """
        Écrit `data` à l'adresse `address`. Distribue sur les locations
        proches. Incrémente/décrémente les compteurs.
        """
        active = self._activate(address.vec)
        if len(active) == 0:
            return
        # contents[active] += data  (saturation int8)
        delta = data.vec.astype(np.int16)
        cur = self.contents[active].astype(np.int16)
        new = np.clip(cur + delta, -self.max_count, self.max_count)
        self.contents[active] = new.astype(np.int8)
        self.writes += 1

    def write_batch(self, addresses: list[HDVector], datas: list[HDVector]) -> None:
        """Écrit plusieurs paires — version vectorisée."""
        for a, d in zip(addresses, datas):
            self.write(a, d)

    # ---------------------------------------------------------------- #
    # Lecture — rappel associatif
    # ---------------------------------------------------------------- #
    def read(self, address: HDVector) -> HDVector:
        """
        Lit à l'adresse `address`. Moyenne les contenus des locations
        actives, prend le signe. Renvoie un vecteur bipolaire.
        """
        active = self._activate(address.vec)
        if len(active) == 0:
            # Aucune location active — retourne un vecteur nul
            return HDVector(np.zeros(self.D, dtype=np.int8))
        summed = self.contents[active].sum(axis=0)
        out = np.sign(summed).astype(np.int8)
        # Tie-break aléatoire
        zeros = out == 0
        if zeros.any():
            rng = np.random.default_rng(int(summed.sum()) & 0xFFFF)
            out[zeros] = rng.choice([-1, 1], size=int(zeros.sum())).astype(np.int8)
        return HDVector(out)

    def read_with_confidence(self, address: HDVector) -> tuple[HDVector, float]:
        """Renvoie (vecteur, confiance) — confiance = similarité moyenne
        des locations activées."""
        active = self._activate(address.vec)
        if len(active) == 0:
            return HDVector(np.zeros(self.D, dtype=np.int8)), 0.0
        sims = (self.locations[active].astype(np.float32)
                @ address.vec.astype(np.float32)) / self.D
        confidence = float(sims.mean())
        summed = self.contents[active].sum(axis=0)
        out = np.sign(summed).astype(np.int8)
        zeros = out == 0
        if zeros.any():
            rng = np.random.default_rng(int(summed.sum()) & 0xFFFF)
            out[zeros] = rng.choice([-1, 1], size=int(zeros.sum())).astype(np.int8)
        return HDVector(out), confidence

    # ---------------------------------------------------------------- #
    # Diagnostics
    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        nonzero = np.count_nonzero(self.contents.any(axis=1))
        return {
            "n_locations": self.n_locations,
            "dim": self.D,
            "writes": self.writes,
            "active_locations": int(nonzero),
            "fill_ratio": float(nonzero / self.n_locations),
            "k_active": self.k_active,
            "activation_radius": self.activation_radius,
        }


# ---------------------------------------------------------------------- #
# Démo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    rng = np.random.default_rng(42)
    D = 10000
    sdm = SparseDistributedMemory(D=D, n_locations=5000, k_active=32)

    # On imprime 20 faits
    facts = []
    for i in range(20):
        key = hd_random(D, rng)
        val = hd_random(D, rng)
        facts.append((key, val))
        sdm.write(key, val)

    # On teste le rappel
    correct = 0
    t0 = time.time()
    for key, val in facts:
        recalled = sdm.read(key)
        sim = hd_similarity(recalled, val)
        if sim > 0.5:
            correct += 1
    t1 = time.time()

    print(f"SDM: {correct}/20 rappels corrects (sim>0.5)")
    print(f"Temps: {(t1-t0)*1000:.2f} ms pour 20 lectures")
    print(f"Stats: {sdm.stats()}")

    # Robustesse au bruit
    key, val = facts[0]
    # Pour le test on flippe 10% des bits
    flip = rng.random(D) < 0.1
    noisy = key.vec.copy()
    noisy[flip] *= -1
    recalled = sdm.read(HDVector(noisy))
    print(f"Rappel avec 10% de bruit: sim = {hd_similarity(recalled, val):+.4f}")
