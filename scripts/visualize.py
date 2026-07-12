"""
Visualisations matplotlib pour SPIKE.

Génère des plots statiques:
  1. Spike raster — un point par spike, axe X=tick, axe Y=neurone
  2. Weight heatmap — matrice des poids synaptiques
  3. Motor activity — bar chart des comptes par slot
  4. STDP trace — évolution des poids pendant l'apprentissage
  5. Population dynamics — activité cumulée par population
  6. Energy landscape — évolution de l'énergie du résonateur NOVA

Usage:
    python scripts/visualize.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

# Setup fonts pour matplotlib
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#0a0e1a'
plt.rcParams['axes.facecolor'] = '#151b2e'
plt.rcParams['axes.edgecolor'] = '#2d3548'
plt.rcParams['axes.labelcolor'] = '#c8d6e5'
plt.rcParams['xtick.color'] = '#c8d6e5'
plt.rcParams['ytick.color'] = '#c8d6e5'
plt.rcParams['text.color'] = '#c8d6e5'
plt.rcParams['axes.titlecolor'] = '#00d2ff'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import SpikeBrain, SpikeConfig
from nova import Nova, NovaConfig


def save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches=None, facecolor='#0a0e1a')
    plt.close(fig)
    print(f"  ✓ {path}")


# ---------------------------------------------------------------- #
# 1. Spike raster
# ---------------------------------------------------------------- #

def plot_raster(brain: SpikeBrain, n_ticks: int = 50, save_path: str = None):
    """Raster plot des spikes par tick."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    # Simule avec input
    brain.net.reset(soft=False)
    I_static = brain.coder.encode_text_to_current("le chat dort", gain=2.5)

    sensory_log = []
    assoc_log = []
    motor_log = []

    for tick in range(n_ticks):
        mask = (brain.rng.random(brain.cfg.n_sensory) < 0.6).astype(np.float32)
        I_tick = I_static * mask
        brain.net.tick(I_tick)
        if brain.cfg.stdp_enabled:
            brain._apply_stdp()
        sensory_log.append(brain.net.last_spikes["sensory"].copy())
        assoc_log.append(brain.net.last_spikes["associative"].copy())
        motor_log.append(brain.net.last_spikes["motor"].copy())

    # Plot par population
    populations = [
        ("Sensory", sensory_log, "#00d2ff"),
        ("Associative", assoc_log, "#feca57"),
        ("Motor", motor_log, "#ff6b6b"),
    ]
    for ax, (name, log, color) in zip(axes, populations):
        # Raster: chaque spike est un point
        for tick, spikes in enumerate(log):
            for n, s in enumerate(spikes):
                if s:
                    ax.plot(tick, n, '.', color=color, markersize=1)
        ax.set_ylabel(f"{name}\n(neurone)")
        ax.set_xlim(0, n_ticks)
        ax.set_ylim(0, len(log[0]))
        ax.set_title(f"Population {name} — {int(sum(s.sum() for s in log))} spikes totaux")
        ax.grid(True, alpha=0.1)

    axes[-1].set_xlabel("Tick")
    fig.suptitle("SPIKE — Spike raster (input: 'le chat dort')", color='#00d2ff', fontsize=14)
    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# 2. Weight heatmap
# ---------------------------------------------------------------- #

def plot_weights(brain: SpikeBrain, save_path: str = None):
    """Heatmap des poids synaptiques."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    synapses = [
        ("sens→assoc", brain.net.syn_sens_to_assoc.W, axes[0]),
        ("assoc→motor", brain.net.syn_assoc_to_motor.W, axes[1]),
    ]
    if brain.syn_sens_to_motor is not None:
        synapses.append(("sens→motor (direct)", brain.syn_sens_to_motor.W, axes[2]))

    for name, W, ax in synapses:
        # Sample 100x100 pour la visualisation
        n_pre = min(100, W.shape[0])
        n_post = min(100, W.shape[1])
        W_dense = W[:n_pre, :n_post].toarray()
        im = ax.imshow(W_dense, aspect='auto', cmap='viridis',
                        interpolation='nearest')
        ax.set_title(f"{name} ({W.nnz} synapses)")
        ax.set_xlabel("Neurone post")
        ax.set_ylabel("Neurone pre")
        plt.colorbar(im, ax=ax, label='Poids')

    fig.suptitle("SPIKE — Poids synaptiques (CSR sparse)", color='#00d2ff', fontsize=14)
    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# 3. Motor activity bar chart
# ---------------------------------------------------------------- #

def plot_motor_activity(brain: SpikeBrain, save_path: str = None):
    """Bar chart de l'activité motrice par slot (token)."""
    # Apprend quelques faits
    facts = [
        ("le chat", "un animal"),
        ("Paris", "la capitale"),
        ("la terre", "une planète"),
    ]
    for k, v in facts:
        brain.learn(k, v)

    # Rappel de chacun
    fig, axes = plt.subplots(1, len(facts), figsize=(5 * len(facts), 4),
                              constrained_layout=True)
    if len(facts) == 1:
        axes = [axes]

    for ax, (query, _) in zip(axes, facts):
        r = brain.recall(query)
        # Top 10 tokens par activité
        top = r["top_motor_tokens"][:10]
        if top:
            tokens, counts = zip(*top)
            # Tronque les tokens longs
            tokens = [t[:12] for t in tokens]
            bars = ax.barh(range(len(tokens)), counts, color='#00d2ff')
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.invert_yaxis()
            ax.set_xlabel("Spikes")
            ax.set_title(f"Query: '{query}'\n(score={r['score']:.1f})")
            ax.grid(True, alpha=0.1, axis='x')
        else:
            ax.text(0.5, 0.5, "Pas d'activité", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f"Query: '{query}'")

    fig.suptitle("SPIKE — Activité motrice par token (rappel)", color='#00d2ff', fontsize=14)
    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# 4. STDP evolution
# ---------------------------------------------------------------- #

def plot_stdp_evolution(brain: SpikeBrain, save_path: str = None):
    """Évolution des poids moyens pendant la simulation."""
    n_ticks = 100
    brain.net.reset(soft=False)

    history_sa = []
    history_am = []
    history_sm = []
    history_aa = []

    I_static = brain.coder.encode_text_to_current("bonjour le monde", gain=2.5)

    for tick in range(n_ticks):
        mask = (brain.rng.random(brain.cfg.n_sensory) < 0.6).astype(np.float32)
        brain.net.tick(I_static * mask)
        if brain.cfg.stdp_enabled:
            brain._apply_stdp()
        if tick % 5 == 0:
            history_sa.append(float(brain.net.syn_sens_to_assoc.W.data.mean())
                              if brain.net.syn_sens_to_assoc.W.nnz > 0 else 0)
            history_am.append(float(brain.net.syn_assoc_to_motor.W.data.mean())
                              if brain.net.syn_assoc_to_motor.W.nnz > 0 else 0)
            history_aa.append(float(brain.net.syn_assoc_to_assoc.W.data.mean())
                              if brain.net.syn_assoc_to_assoc.W.nnz > 0 else 0)
            if brain.syn_sens_to_motor is not None:
                history_sm.append(float(brain.syn_sens_to_motor.W.data.mean())
                                  if brain.syn_sens_to_motor.W.nnz > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ticks = np.arange(0, n_ticks, 5)
    ax.plot(ticks, history_sa, label='sens→assoc', color='#00d2ff', linewidth=2)
    ax.plot(ticks, history_am, label='assoc→motor', color='#feca57', linewidth=2)
    ax.plot(ticks, history_aa, label='assoc→assoc', color='#54a0ff', linewidth=2)
    if history_sm:
        ax.plot(ticks, history_sm, label='sens→motor (direct)', color='#ff6b6b', linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Poids moyen")
    ax.set_title("STDP — Évolution des poids pendant la simulation")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# 5. Population dynamics
# ---------------------------------------------------------------- #

def plot_population_dynamics(brain: SpikeBrain, save_path: str = None):
    """Activité cumulée par population au fil du temps."""
    n_ticks = 80
    brain.net.reset(soft=False)

    sensory_counts = []
    assoc_counts = []
    motor_counts = []

    # Phase 1: input actif (ticks 0-30)
    # Phase 2: input coupé (ticks 30-80)
    I_static = brain.coder.encode_text_to_current("test", gain=2.5)

    for tick in range(n_ticks):
        if tick < 30:
            mask = (brain.rng.random(brain.cfg.n_sensory) < 0.6).astype(np.float32)
            I_tick = I_static * mask
        else:
            I_tick = np.zeros(brain.cfg.n_sensory, dtype=np.float32)
        brain.net.tick(I_tick)
        if brain.cfg.stdp_enabled:
            brain._apply_stdp()
        sensory_counts.append(int(brain.net.last_spikes["sensory"].sum()))
        assoc_counts.append(int(brain.net.last_spikes["associative"].sum()))
        motor_counts.append(int(brain.net.last_spikes["motor"].sum()))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ticks = np.arange(n_ticks)
    ax.fill_between(ticks, 0, sensory_counts, alpha=0.6, label='Sensory', color='#00d2ff')
    ax.fill_between(ticks, 0, assoc_counts, alpha=0.6, label='Associative', color='#feca57')
    ax.fill_between(ticks, 0, motor_counts, alpha=0.6, label='Motor', color='#ff6b6b')
    ax.axvline(x=30, color='#576574', linestyle='--', alpha=0.5, label='Input coupé')
    ax.set_xlabel("Tick")
    ax.set_ylabel("Spikes par tick")
    ax.set_title("SPIKE — Dynamique des populations (input puis silence)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# 6. NOVA energy landscape
# ---------------------------------------------------------------- #

def plot_nova_energy(nova: Nova, save_path: str = None):
    """Évolution de l'énergie du résonateur NOVA."""
    nova.resonator.reset()
    from nova.hd import hd_random
    inp = hd_random(nova.cfg.D)
    nova.resonator.inject(inp, gain=1.0)

    energies = []
    states_norm = []
    for _ in range(50):
        nova.resonator.reason()
        energies.append(nova.resonator.energy())
        states_norm.append(float(np.linalg.norm(nova.resonator.state)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(energies, color='#00d2ff', linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Énergie")
    axes[0].set_title("NOVA — Énergie du résonateur (convergence vers attracteur)")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(states_norm, color='#feca57', linewidth=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("||état||")
    axes[1].set_title("Norme de l'état (stabilisation)")
    axes[1].grid(True, alpha=0.2)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def main():
    output_dir = "/home/z/my-project/download/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  SPIKE — Génération des visualisations")
    print("=" * 60)

    # Init brains
    print("\nInitialisation SPIKE...")
    spike = SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=40,
    ))
    print("Initialisation NOVA...")
    nova = Nova(NovaConfig(D=5000, sdm_locations=10000))

    # Genère les plots
    print("\nGénération des plots:")
    plot_raster(spike, n_ticks=50,
                save_path=os.path.join(output_dir, "01_raster.png"))
    plot_weights(spike,
                  save_path=os.path.join(output_dir, "02_weights.png"))
    plot_motor_activity(spike,
                         save_path=os.path.join(output_dir, "03_motor_activity.png"))
    plot_stdp_evolution(spike,
                         save_path=os.path.join(output_dir, "04_stdp_evolution.png"))
    plot_population_dynamics(spike,
                              save_path=os.path.join(output_dir, "05_population_dynamics.png"))
    plot_nova_energy(nova,
                      save_path=os.path.join(output_dir, "06_nova_energy.png"))

    print(f"\n✓ Tous les plots sont dans {output_dir}/")
    print("\nFichiers générés:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size // 1024} Ko)")


if __name__ == "__main__":
    main()
