"""
Résonateur — champ d'état dynamique qui évolue dans le temps.

C'est LE cœur du raisonnement continu. Au lieu d'une passe forward
figée (comme dans un transformer), on a un champ d'état D-dim qui
évolue selon une équation différentielle:

    dx/dt = -x/τ + W·x + I(t) + σ(x)

où:
  - x  : état D-dim (vecteur dense float, PAS bipolaire)
  - τ  : constantes de temps hétérogènes par dimension (liquid state)
  - W  : matrice de connectivité sparse (1% de non-zéros)
  - I(t): input courant (vecteur HD encodé)
  - σ  : non-linéarité (tanh)

Propriétés émergentes:
  - Attracteurs: l'état converge vers des bassins correspondant aux
    patterns appris
  - Résonance: si l'input correspond à un pattern connu, l'état
    amplifie cette composante (sélection attentive)
  - Mémoire de travail: τ hétérogènes = mémoire à plusieurs échelles
  - Pas de backprop: W est fixe (ou Hebbian local)

CPU-friendly:
  - W sparse: scipy.sparse, opérations O(nnz)
  - Float32 partout
  - Éuler avec dt=0.1, ~10-50 steps suffit pour converger
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field

from .hd import HDVector


@dataclass
class ResonatorConfig:
    """Configuration du résonateur."""
    D: int = 10000
    sparsity: float = 0.01         # fraction de poids non-nuls dans W
    spectral_radius: float = 0.90  # rayon spectral de W (stabilité)
    tau_min: float = 1.0
    tau_max: float = 8.0
    dt: float = 0.1
    n_steps: int = 30              # steps par appel à `reason()`
    input_gain: float = 0.3        # gain d'injection (évite saturation)
    input_persistence: float = 0.95  # decay du buffer d'input par step
    noise_std: float = 0.005       # bruit d'exploration
    leak: float = 0.1              # terme de fuite additif
    clamp_input: bool = True       # si True, réinjecte l'input à chaque step


class Resonator:
    """
    Champ d'état dynamique pour le raisonnement continu.

    Usage:
        res = Resonator(D=10000)
        res.inject(input_vec)       # perception
        trajectory = res.reason()   # évolution libre sur n_steps
        state = res.get_state()     # lecture
        res.reset()                 # reset mémoire de travail
    """

    def __init__(self, cfg: ResonatorConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or ResonatorConfig()
        self.rng = rng or np.random.default_rng()
        D = self.cfg.D

        # Matrice sparse W (D, D) avec sparsity% de non-zéros
        n_nonzero = int(D * D * self.cfg.sparsity)
        # Évite trop de mémoire pour D très grand
        if n_nonzero > 5_000_000:
            n_nonzero = 5_000_000
        # Construction sparse aléatoire
        rows = self.rng.integers(0, D, size=n_nonzero)
        cols = self.rng.integers(0, D, size=n_nonzero)
        vals = self.rng.standard_normal(n_nonzero).astype(np.float32)
        self.W = sp.csr_matrix((vals, (rows, cols)), shape=(D, D))

        # Normalisation: scaling pour atteindre le rayon spectral cible
        # Approximation: la plus grande valeur singulière ~ sqrt(D*sparsity)*sigma
        # On scale pour que le rayon spectral effectif ~ spectral_radius
        scale = self.cfg.spectral_radius / np.sqrt(np.maximum(1, D * self.cfg.sparsity))
        self.W *= scale

        # Constantes de temps hétérogènes (liquid state machine)
        self.tau = self.rng.uniform(self.cfg.tau_min, self.cfg.tau_max,
                                     size=D).astype(np.float32)

        # État courant (float dense)
        self.state = np.zeros(D, dtype=np.float32)
        # Tampon d'input (injecté, decay exponentiel)
        self.input_buffer = np.zeros(D, dtype=np.float32)

        # Historique récent (pour diagnostic)
        self.trajectory: list[np.ndarray] = []
        self.max_trajectory = 100

    # ---------------------------------------------------------------- #
    # API principale
    # ---------------------------------------------------------------- #
    def reset(self) -> None:
        """Reset l'état (mémoire de travail). Garde W et τ."""
        self.state = np.zeros(self.cfg.D, dtype=np.float32)
        self.input_buffer = np.zeros(self.cfg.D, dtype=np.float32)
        self.trajectory = []

    def soft_reset(self, decay: float = 0.5) -> None:
        """Reset partiel — conserve une trace de l'état précédent."""
        self.state *= decay
        self.input_buffer *= decay

    def inject(self, input_vec: HDVector | np.ndarray,
               gain: float | None = None) -> None:
        """
        Injecte un input dans le champ. L'input est converti en float
        et accumulé dans le buffer d'input (qui décroît exponentiellement).
        """
        g = gain if gain is not None else self.cfg.input_gain
        if isinstance(input_vec, HDVector):
            vec = input_vec.vec.astype(np.float32)
        else:
            vec = input_vec.astype(np.float32)
        self.input_buffer += g * vec

    # ---------------------------------------------------------------- #
    # Équation d'évolution
    # ---------------------------------------------------------------- #
    def _step(self) -> np.ndarray:
        """
        Un pas d'Euler de l'équation:
            dx/dt = -x/τ + W·x + I(t) + σ(x) + noise

        Si clamp_input=True, I(t) reste constant (input persistant).
        Sinon, I(t) décroît exponentiellement.
        """
        dt = self.cfg.dt
        # Terme de fuite: -x/τ
        leak = -self.state / self.tau
        # Récurrence: W @ x  (sparse @ dense = dense, efficace)
        recurrent = self.W @ self.state
        # Input
        if self.cfg.clamp_input:
            recurrent_input = self.input_buffer.copy()
        else:
            # Decay exponentiel
            recurrent_input = self.input_buffer.copy()
            self.input_buffer *= self.cfg.input_persistence
        # Non-linéarité (faible, stabilise)
        nonlin = np.tanh(self.state * 0.5) * 0.1
        # Bruit d'exploration
        noise = self.rng.standard_normal(self.cfg.D).astype(np.float32) * self.cfg.noise_std

        # Euler
        dx = (leak + recurrent + recurrent_input + nonlin + noise) * dt
        self.state = self.state + dx
        # Saturation douce
        self.state = np.tanh(self.state)

        return self.state

    def reason(self, n_steps: int | None = None,
               record: bool = False) -> np.ndarray:
        """
        Fait évoluer le champ pendant n_steps. Si record=True, garde
        l'historique dans self.trajectory.
        """
        n = n_steps or self.cfg.n_steps
        for _ in range(n):
            s = self._step()
            if record:
                self.trajectory.append(s.copy())
                if len(self.trajectory) > self.max_trajectory:
                    self.trajectory.pop(0)
        return self.state

    # ---------------------------------------------------------------- #
    # Lecture
    # ---------------------------------------------------------------- #
    def get_state(self) -> np.ndarray:
        """Renvoie l'état courant (float dense)."""
        return self.state.copy()

    def get_state_bipolar(self) -> HDVector:
        """Renvoie l'état binarisé en vecteur bipolaire (pour SDM/cleanup)."""
        out = np.sign(self.state).astype(np.int8)
        zeros = out == 0
        if zeros.any():
            out[zeros] = self.rng.choice([-1, 1], size=int(zeros.sum())).astype(np.int8)
        return HDVector(out)

    def energy(self) -> float:
        """
        "Énergie" du champ — utile pour détecter la convergence.
        Énergie basse = stable / attracteur.
        """
        s = self.state
        return float(-0.5 * s @ (self.W @ s) + 0.5 * np.sum(s*s))

    # ---------------------------------------------------------------- #
    # Persistance
    # ---------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Sauvegarde W (sparse), tau et la config."""
        sp.save_npz(path.replace('.npz', '_W.npz'), self.W)
        np.savez(path,
                 tau=self.tau,
                 D=self.cfg.D,
                 sparsity=self.cfg.sparsity,
                 spectral_radius=self.cfg.spectral_radius,
                 tau_min=self.cfg.tau_min,
                 tau_max=self.cfg.tau_max,
                 dt=self.cfg.dt,
                 n_steps=self.cfg.n_steps,
                 input_gain=self.cfg.input_gain,
                 noise_std=self.cfg.noise_std,
                 leak=self.cfg.leak)

    @classmethod
    def load(cls, path: str) -> "Resonator":
        data = np.load(path, allow_pickle=True)
        cfg = ResonatorConfig(
            D=int(data["D"]),
            sparsity=float(data["sparsity"]),
            spectral_radius=float(data["spectral_radius"]),
            tau_min=float(data["tau_min"]),
            tau_max=float(data["tau_max"]),
            dt=float(data["dt"]),
            n_steps=int(data["n_steps"]),
            input_gain=float(data["input_gain"]),
            noise_std=float(data["noise_std"]),
            leak=float(data["leak"]),
        )
        r = cls(cfg)
        r.W = sp.load_npz(path.replace('.npz', '_W.npz'))
        r.tau = data["tau"]
        return r


if __name__ == "__main__":
    import time
    print("Test du résonateur...")

    cfg = ResonatorConfig(D=5000, n_steps=50)
    res = Resonator(cfg)

    # Injecte un input aléatoire
    from .hd import hd_random
    inp = hd_random(cfg.D)
    res.inject(inp)

    # Évolution
    t0 = time.time()
    res.reason(record=True)
    t1 = time.time()
    print(f"50 steps en {(t1-t0)*1000:.2f} ms (D={cfg.D})")
    print(f"Énergie initiale: ???, énergie finale: {res.energy():.4f}")
    print(f"Trajectoire: {len(res.trajectory)} états enregistrés")
    print(f"Norme de l'état: {np.linalg.norm(res.state):.4f}")
    print(f"État bipolarisé similaire à l'input: "
          f"{float(np.dot(res.get_state_bipolar().vec, inp.vec) / cfg.D):+.4f}")
