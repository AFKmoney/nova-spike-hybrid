"""
STDP — Spike-Timing-Dependent Plasticity.

Règle d'apprentissage biologique, locale, asynchrone.

Principe:
  Pour une synapse pre→post:
    - Si pre spike juste AVANT post spike (Δt = t_post - t_pre > 0, petit):
      LTP (renforcement) — A+ * exp(-Δt / τ_+)
    - Si pre spike APRÈS post spike (Δt < 0):
      LTD (affaiblissement) — A- * exp(-Δt / τ_-)

Implémentation par "traces":
  Chaque neurone pré maintient une trace x_pre (décroissance exp).
  Chaque neurone post maintient une trace x_post.
  À chaque spike:
    - Updates des poids:
        Si pre spiké: pour toutes ses synapses post, Δw = -A- * x_post  (LTD)
        Si post spiké: pour toutes ses synapses pré, Δw = +A+ * x_pre   (LTP)
    - Update des traces:
        spike pre:  x_pre ← x_pre + 1
        spike post: x_post ← x_post + 1
  Décroissance par tick: x *= exp(-dt / τ_trace)

C'est O(nnz) par tick — on ne touche que les synapses des neurones actifs.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field

from .network import SynapseGroup


@dataclass
class STDPConfig:
    """Paramètres STDP."""
    A_plus: float = 0.1       # amplitude LTP
    A_minus: float = 0.1      # amplitude LTD (souvent A_plus * 1.05 pour stability)
    tau_plus: float = 20.0    # ms, constante LTP
    tau_minus: float = 20.0   # ms, constante LTD
    w_min: float = 0.0        # poids min (pour excitateur)
    w_max: float = 5.0        # poids max
    dt: float = 1.0           # pas de simulation


@dataclass
class STDPTracker:
    """
    Tracker STDP pour un groupe de synapses.

    Maintient:
      - x_pre (n_pre,): trace des spikes pré
      - x_post (n_post,): trace des spikes post
      - W: matrice CSR des poids (référence partagée avec le SynapseGroup)
    """
    syn: SynapseGroup
    cfg: STDPConfig = field(default_factory=STDPConfig)
    x_pre: np.ndarray = field(init=False)
    x_post: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x_pre = np.zeros(self.syn.n_pre, dtype=np.float32)
        self.x_post = np.zeros(self.syn.n_post, dtype=np.float32)

    # ---------------------------------------------------------------- #
    def update(self, spikes_pre: np.ndarray, spikes_post: np.ndarray) -> None:
        """
        Applique une étape STDP.

        Règle:
          1. Si pre spiké: pour chaque post qui a une trace non nulle,
             Δw = -A_minus * x_post[post]  (LTD)
          2. Si post spiké: pour chaque pre qui a une trace non nulle,
             Δw = +A_plus * x_pre[pre]    (LTP)
          3. Update traces (decay + ajout des spikes)
          4. Clamp W à [w_min, w_max]

        Args:
            spikes_pre: masque booléen (n_pre,)
            spikes_post: masque booléen (n_post,)
        """
        if not self.syn.plastic:
            # Decay quand même les traces pour la prochaine fois
            self._decay_traces()
            return

        cfg = self.cfg
        dt = cfg.dt
        decay_pre = np.exp(-dt / cfg.tau_plus).astype(np.float32)
        decay_post = np.exp(-dt / cfg.tau_minus).astype(np.float32)

        # ---- 1. LTD: pre spiké → affaiblir les synapses vers les post
        # qui ont spiké récemment (trace x_post > 0) ----
        if spikes_pre.any() and self.x_post.max() > 1e-6:
            pre_idx = np.where(spikes_pre)[0]
            # Pour chaque pre qui spiké, on soustrait A- * x_post[j] sur la ligne
            # du poids W[i, :]
            # W[pre_idx, :] -= A_minus * x_post  (broadcast)
            # Mais on veut une matrice sparse qui reste sparse, donc on update
            # seulement les poids déjà existants (les autres restent à 0).
            # Méthode: W[pre_idx] = W[pre_idx] - A_minus * x_post (broadcast sur les nnz)
            # Pour les poids déjà existants:
            existing_rows = self.syn.W[pre_idx]
            # existing_rows est sparse (k, n_post). Pour chaque ligne, on soustrait
            # A_minus * x_post[j] sur les colonnes où il y a un poids non nul.
            # En CSR, data[i] correspond à la colonne col_ind[i].
            # On veut: data[i] -= A_minus * x_post[col_ind[i]]
            new_data = existing_rows.data - cfg.A_minus * self.x_post[existing_rows.indices]
            # Clamp
            new_data = np.clip(new_data, cfg.w_min, cfg.w_max)
            existing_rows.data = new_data
            # On remet dans W
            self.syn.W[pre_idx] = existing_rows

        # ---- 2. LTP: post spiké → renforcer les synapses depuis les pre
        # qui ont spiké récemment (trace x_pre > 0) ----
        if spikes_post.any() and self.x_pre.max() > 1e-6:
            post_idx = np.where(spikes_post)[0]
            # Pour chaque post qui spiké, on ajoute A+ * x_pre[i] sur la colonne
            # W[:, post] += A_plus * x_pre
            # En sparse, on itère sur les colonnes.
            # Astuce: W.T[post_idx] donne les lignes transposées (sparse)
            existing_cols = self.syn.W.T[post_idx]
            new_data = existing_cols.data + cfg.A_plus * self.x_pre[existing_cols.indices]
            new_data = np.clip(new_data, cfg.w_min, cfg.w_max)
            existing_cols.data = new_data
            self.syn.W.T[post_idx] = existing_cols
            # Forcer la mise à jour CSR après modification via .T (qui est CSC)
            self.syn.W = self.syn.W.tocsr()

        # ---- 3. Update traces ----
        self.x_pre *= decay_pre
        self.x_post *= decay_post
        self.x_pre[spikes_pre] += 1.0
        self.x_post[spikes_post] += 1.0

    def _decay_traces(self) -> None:
        """Decay sans update de poids (pour synapses non plastiques)."""
        cfg = self.cfg
        dt = cfg.dt
        self.x_pre *= np.exp(-dt / cfg.tau_plus).astype(np.float32)
        self.x_post *= np.exp(-dt / cfg.tau_minus).astype(np.float32)

    # ---------------------------------------------------------------- #
    def reset(self) -> None:
        """Reset traces (mais PAS les poids)."""
        self.x_pre[:] = 0
        self.x_post[:] = 0

    def stats(self) -> dict:
        return {
            "syn_name": self.syn.name,
            "w_mean": float(self.syn.W.data.mean()) if self.syn.W.nnz > 0 else 0.0,
            "w_max": float(self.syn.W.data.max()) if self.syn.W.nnz > 0 else 0.0,
            "w_min": float(self.syn.W.data.min()) if self.syn.W.nnz > 0 else 0.0,
            "n_synapses": self.syn.W.nnz,
            "x_pre_max": float(self.x_pre.max()),
            "x_post_max": float(self.x_post.max()),
        }


# ---------------------------------------------------------------------- #
# Helper: one-shot imprint (pour apprentissage explicite)
# ---------------------------------------------------------------------- #

def imprint_path(syn: SynapseGroup, pre_indices: list[int],
                 post_indices: list[int], weight: float = 2.0) -> None:
    """
    One-shot: force les poids d'un chemin pre→post.
    Utilisé pour l'apprentissage explicite ("apprends que X est Y").

    Pour chaque (i, j) dans zip(pre_indices, post_indices):
        W[i, j] = weight
    """
    if not pre_indices or not post_indices:
        return
    # On met le même poids pour toutes les combinaisons pré x post
    rows = np.repeat(pre_indices, len(post_indices))
    cols = np.tile(post_indices, len(pre_indices))
    vals = np.full(len(rows), weight, dtype=np.float32)
    add = sp.csr_matrix((vals, (rows, cols)),
                         shape=syn.W.shape, dtype=np.float32)
    syn.W = syn.W + add
    # Clamp
    syn.W.data = np.clip(syn.W.data, 0, 5.0)


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time as _time
    from .network import SpikingNetwork, PopulationType

    print("Test STDP sur un petit réseau...")
    net = SpikingNetwork(n_sensory=50, n_associative=200, n_motor=20)

    # Tracker STDP sur sens→assoc
    tracker = STDPTracker(net.syn_sens_to_assoc, STDPConfig())

    # Simule 100 ticks avec input sur neurones sensory 0-9
    rng = np.random.default_rng()
    t0 = _time.time()
    for _ in range(100):
        I = np.zeros(50, dtype=np.float32)
        I[0:10] = 2.0
        # Tick
        net.tick(I)
        # STDP
        spikes_pre = net.last_spikes["sensory"]
        spikes_post = net.last_spikes["associative"]
        tracker.update(spikes_pre, spikes_post)
    t1 = _time.time()

    print(f"100 ticks + STDP en {(t1-t0)*1000:.2f} ms")
    print(f"Tracker stats: {tracker.stats()}")
    print(f"Synapse sens→assoc: {net.syn_sens_to_assoc.stats()}")

    # Vérifie que les poids ont changé
    print(f"Poids moyen avant STDP aurait été: 0.7")
    print(f"Poids moyen après STDP: {tracker.stats()['w_mean']:.4f}")
