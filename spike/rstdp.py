"""
R-STDP — Reward-modulated STDP.

Principe (biologique: dopamine):
  - La STDP "classique" calcule un Δw potentiel à chaque paire pre/post spike.
  - Mais Δw n'est APPLIQUÉ que si un signal de récompense (dopamine) arrive.
  - Entre les récompenses, les "traces d'éligibilité" e(i,j) accumulent les
    coïncidences pre/post.
  - Quand reward arrive: w += η · reward · e(i,j), puis e decay.

Avantage clé: apprentissage par renforcement local, sans backprop, sans
stocker l'état global. Chaque synapse "sait" si elle a participé à
l'activité récente (eligibility) et reçoit un signal global (reward).

Usage:
    tracker = RSTDPTracker(syn, cfg)
    tracker.update(spikes_pre, spikes_post)  # à chaque tick (calcule eligibility)
    tracker.apply_reward(reward=+1.0)         # quand on veut renforcer
    tracker.apply_reward(reward=-1.0)         # pour punir
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field

from .network import SynapseGroup
from .stdp import STDPConfig, STDPTracker


@dataclass
class RSTDPConfig(STDPConfig):
    """Config R-STDP — étend STDP avec un facteur d'éligibilité."""
    tau_eligibility: float = 200.0  # ms — décroissance de l'éligibilité (lent)
    reward_rate: float = 0.05       # η — learning rate du signal reward
    eligibility_init: float = 0.0


class RSTDPTracker(STDPTracker):
    """
    Tracker R-STDP.

    Maintient en plus de STDPTracker:
      - eligibility: matrice sparse des traces d'éligibilité (n_pre, n_post)
        Même sparsity pattern que W.
    """

    def __init__(self, syn: SynapseGroup, cfg: RSTDPConfig | None = None):
        super().__init__(syn, cfg or RSTDPConfig())
        self.rstdp_cfg: RSTDPConfig = self.cfg  # alias typé
        # Eligibility: sparse même pattern que W
        self.eligibility = sp.csr_matrix(syn.W.shape, dtype=np.float32)
        # Dernier reward appliqué (pour stats)
        self.last_reward: float = 0.0
        self.n_rewards: int = 0

    # ---------------------------------------------------------------- #
    def update(self, spikes_pre: np.ndarray, spikes_post: np.ndarray) -> None:
        """
        Update à chaque tick — calcule l'eligibility (pas de modif de W).
        L'eligibilité accumule les coïncidences pre/post selon STDP.

        Implémentation simple et robuste:
          - Pour chaque pre qui spiké: eligibility[pre, post_with_trace] -= A- * x_post
          - Pour chaque post qui spiké: eligibility[pre_with_trace, post] += A+ * x_pre
        On utilise seulement les synapses EXISTANTES pour rester sparse.
        """
        dt = self.cfg.dt
        decay_pre = np.exp(-dt / self.cfg.tau_plus).astype(np.float32)
        decay_post = np.exp(-dt / self.cfg.tau_minus).astype(np.float32)
        decay_elig = np.exp(-dt / self.rstdp_cfg.tau_eligibility).astype(np.float32)

        # 1. Decay de l'eligibility
        if self.eligibility.nnz > 0:
            self.eligibility.data *= decay_elig

        # 2. Sync eligibility pattern avec W (au cas où W a évolué)
        # Pour rester simple, on rebuild l'eligibility à partir du pattern de W
        # si elle est vide ou si les patterns divergent
        if self.eligibility.nnz == 0 and self.syn.W.nnz > 0:
            # Initialise eligibility avec le même pattern que W
            W_coo = self.syn.W.tocoo()
            self.eligibility = sp.csr_matrix(
                (np.zeros_like(W_coo.data), (W_coo.row, W_coo.col)),
                shape=self.syn.W.shape, dtype=np.float32
            )

        # 3. Update STDP traces (decay)
        self.x_pre *= decay_pre
        self.x_post *= decay_post

        # 4. Calcule le delta d'eligibilité pour les synapses EXISTANTES
        # On travaille en format COO pour itérer
        if self.eligibility.nnz > 0:
            elig_coo = self.eligibility.tocoo()
            # Pour chaque synapse (i, j):
            #   delta_e = 0
            #   if i in spikes_pre: delta_e -= A- * x_post[j]
            #   if j in spikes_post: delta_e += A+ * x_pre[i]
            pre_set = set(np.where(spikes_pre)[0].tolist())
            post_set = set(np.where(spikes_post)[0].tolist())

            if pre_set or post_set:
                delta_e = np.zeros_like(elig_coo.data)
                for k in range(len(elig_coo.data)):
                    i, j = elig_coo.row[k], elig_coo.col[k]
                    de = 0.0
                    if i in pre_set:
                        de -= self.cfg.A_minus * float(self.x_post[j])
                    if j in post_set:
                        de += self.cfg.A_plus * float(self.x_pre[i])
                    delta_e[k] = de

                if delta_e.any():
                    new_data = elig_coo.data + delta_e
                    self.eligibility = sp.csr_matrix(
                        (new_data, (elig_coo.row, elig_coo.col)),
                        shape=self.syn.W.shape, dtype=np.float32
                    )

        # 5. Update traces (après calcul d'eligibility)
        self.x_pre[spikes_pre] += 1.0
        self.x_post[spikes_post] += 1.0

    # ---------------------------------------------------------------- #
    def apply_reward(self, reward: float) -> None:
        """
        Applique le signal de récompense:
            W += η · reward · eligibility

        Version robuste qui assume que eligibility a maintenant le même
        pattern que W (mis à jour dans update()).
        """
        if not self.syn.plastic:
            return
        # Sync pattern si nécessaire
        if self.eligibility.nnz != self.syn.W.nnz:
            sync_eligibility_pattern(self)

        if self.eligibility.nnz == 0:
            self.last_reward = reward
            self.n_rewards += 1
            return

        eta = self.rstdp_cfg.reward_rate
        # Si W et eligibility ont le même pattern, on vectorise
        W_coo = self.syn.W.tocoo()
        E_csr = self.eligibility.tocsr()

        # Pour chaque entrée de W, on récupère l'eligibility correspondante
        # (lookup vectorisé)
        elig_data = np.zeros(len(W_coo.data), dtype=np.float32)
        for k in range(len(W_coo.data)):
            elig_data[k] = E_csr[W_coo.row[k], W_coo.col[k]]

        delta_w = eta * reward * elig_data
        new_data = np.clip(W_coo.data + delta_w,
                            self.cfg.w_min, self.cfg.w_max).astype(np.float32)
        self.syn.W = sp.csr_matrix(
            (new_data, (W_coo.row, W_coo.col)),
            shape=self.syn.W.shape, dtype=np.float32
        )

        self.last_reward = reward
        self.n_rewards += 1

    # ---------------------------------------------------------------- #
    def apply_reward_fast(self, reward: float) -> None:
        """
        Version vectorisée — suppose que eligibility a le même pattern que W.
        C'est le cas si on appelle update() avant apply_reward().
        """
        if not self.syn.plastic:
            return
        eta = self.rstdp_cfg.reward_rate
        # Si W et eligibility ont le même pattern (normalement oui):
        W = self.syn.W.tocoo()
        E = self.eligibility.tocoo()
        if W.nnz == E.nnz and np.array_equal(W.row, E.row) and np.array_equal(W.col, E.col):
            # Même pattern — on peut vectoriser
            delta_w = eta * reward * E.data
            new_data = np.clip(W.data + delta_w,
                                self.cfg.w_min, self.cfg.w_max).astype(np.float32)
            self.syn.W = sp.csr_matrix(
                (new_data, (W.row, W.col)),
                shape=self.syn.W.shape, dtype=np.float32
            )
        else:
            # Pattern différent — fallback sur apply_reward
            self.apply_reward(reward)
            return

        self.last_reward = reward
        self.n_rewards += 1

    # ---------------------------------------------------------------- #
    def stats(self) -> dict:
        base = super().stats()
        base.update({
            "last_reward": self.last_reward,
            "n_rewards": self.n_rewards,
            "eligibility_nnz": self.eligibility.nnz,
            "eligibility_max": float(self.eligibility.data.max()) if self.eligibility.nnz > 0 else 0.0,
        })
        return base


# ---------------------------------------------------------------------- #
# Helper: reshape eligibility pour qu'elle matche W
# ---------------------------------------------------------------------- #

def sync_eligibility_pattern(tracker: RSTDPTracker) -> None:
    """
    Force le pattern d'eligibility à matcher celui de W.
    À appeler après imprint (qui ajoute des synapses) pour s'assurer
    que l'eligibilité couvre les nouvelles synapses.
    """
    W_coo = tracker.syn.W.tocoo()
    # Si eligibility n'a pas ces entrées, on les ajoute à zéro
    new_elig = tracker.eligibility.tocoo()
    # Build dict (r,c) -> data
    elig_dict = {}
    for r, c, d in zip(new_elig.row, new_elig.col, new_elig.data):
        elig_dict[(int(r), int(c))] = float(d)
    # Add entries from W that are missing
    for r, c in zip(W_coo.row, W_coo.col):
        if (int(r), int(c)) not in elig_dict:
            elig_dict[(int(r), int(c))] = 0.0
    # Rebuild
    if elig_dict:
        rows = np.array([k[0] for k in elig_dict.keys()], dtype=np.int32)
        cols = np.array([k[1] for k in elig_dict.keys()], dtype=np.int32)
        data = np.array(list(elig_dict.values()), dtype=np.float32)
        tracker.eligibility = sp.csr_matrix(
            (data, (rows, cols)),
            shape=tracker.syn.W.shape, dtype=np.float32
        )
    else:
        tracker.eligibility = sp.csr_matrix(tracker.syn.W.shape, dtype=np.float32)


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time as _time
    from .network import SpikingNetwork

    print("Test R-STDP...")
    net = SpikingNetwork(n_sensory=50, n_associative=200, n_motor=20)
    # Augmente les poids initiaux pour que l'activité se propage
    net.syn_sens_to_assoc.random_connect(density=0.20, w_mean=2.0, w_std=0.3, seed=1)
    tracker = RSTDPTracker(net.syn_sens_to_assoc, RSTDPConfig())

    # Simule 100 ticks avec input fort sur sensory 0-9
    rng = np.random.default_rng()
    total_assoc_spikes = 0
    for _ in range(100):
        I = np.zeros(50, dtype=np.float32)
        I[0:10] = 3.0
        net.tick(I)
        tracker.update(net.last_spikes["sensory"], net.last_spikes["associative"])
        total_assoc_spikes += int(net.last_spikes["associative"].sum())

    print(f"Total assoc spikes pendant la sim: {total_assoc_spikes}")
    print(f"Avant reward: W mean = {net.syn_sens_to_assoc.W.data.mean():.4f}, "
          f"eligibility nnz = {tracker.eligibility.nnz}, "
          f"eligibility max = {tracker.eligibility.data.max() if tracker.eligibility.nnz > 0 else 0:.4f}")

    # Applique une récompense positive
    w_before = net.syn_sens_to_assoc.W.data.mean()
    tracker.apply_reward(+1.0)
    w_after = net.syn_sens_to_assoc.W.data.mean()
    print(f"Après reward +1: W mean = {w_after:.4f} (delta = {w_after - w_before:+.4f})")

    # Punition
    w_before = net.syn_sens_to_assoc.W.data.mean()
    tracker.apply_reward(-0.5)
    w_after = net.syn_sens_to_assoc.W.data.mean()
    print(f"Après reward -0.5: W mean = {w_after:.4f} (delta = {w_after - w_before:+.4f})")
    print(f"Stats: {tracker.stats()}")
