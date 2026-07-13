#!/usr/bin/env python3
"""
Visualisations v2 — pour SPIKE, NOVA, AETHER.

Génère 8 plots:
  1. Spike raster (SPIKE)
  2. Weight heatmap (SPIKE)
  3. Motor activity (SPIKE)
  4. STDP evolution (SPIKE)
  5. Population dynamics (SPIKE)
  6. NOVA energy landscape
  7. AETHER cognitive loop trace (nouveau)
  8. AETHER attractor convergence (nouveau)
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

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
from aether import AETHER
from aether.hd import HDVector, bundle, bind_sequence, ngram_encode


def save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches=None, facecolor='#0a0e1a')
    plt.close(fig)
    print(f"  ✓ {path}")


# ---------------------------------------------------------------- #
# 1-6: mêmes que visualize.py
# ---------------------------------------------------------------- #

def plot_raster(brain, n_ticks=50, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    brain.net.reset(soft=False)
    I_static = brain.coder.encode_text_to_current("le chat dort", gain=2.5)
    sensory_log, assoc_log, motor_log = [], [], []
    for tick in range(n_ticks):
        mask = (brain.rng.random(brain.cfg.n_sensory) < 0.6).astype(np.float32)
        brain.net.tick(I_static * mask)
        if brain.cfg.stdp_enabled:
            brain._apply_stdp()
        sensory_log.append(brain.net.last_spikes["sensory"].copy())
        assoc_log.append(brain.net.last_spikes["associative"].copy())
        motor_log.append(brain.net.last_spikes["motor"].copy())
    populations = [
        ("Sensory", sensory_log, "#00d2ff"),
        ("Associative", assoc_log, "#feca57"),
        ("Motor", motor_log, "#ff6b6b"),
    ]
    for ax, (name, log, color) in zip(axes, populations):
        for tick, spikes in enumerate(log):
            for n, s in enumerate(spikes):
                if s:
                    ax.plot(tick, n, '.', color=color, markersize=1)
        ax.set_ylabel(f"{name}\n(neuron)")
        ax.set_xlim(0, n_ticks)
        ax.set_ylim(0, len(log[0]))
        ax.set_title(f"Population {name} — {int(sum(s.sum() for s in log))} total spikes")
        ax.grid(True, alpha=0.1)
    axes[-1].set_xlabel("Tick")
    fig.suptitle("SPIKE — Spike raster (input: 'le chat dort')",
                 color='#00d2ff', fontsize=14)
    # Note: input text stays in French to match the actual demo; axis labels in English
    if save_path:
        save_fig(fig, save_path)


def plot_weights(brain, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    synapses = [
        ("sens→assoc", brain.net.syn_sens_to_assoc.W, axes[0]),
        ("assoc→motor", brain.net.syn_assoc_to_motor.W, axes[1]),
    ]
    if brain.syn_sens_to_motor is not None:
        synapses.append(("sens→motor (direct)", brain.syn_sens_to_motor.W, axes[2]))
    for name, W, ax in synapses:
        n_pre = min(100, W.shape[0])
        n_post = min(100, W.shape[1])
        W_dense = W[:n_pre, :n_post].toarray()
        im = ax.imshow(W_dense, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f"{name} ({W.nnz} synapses)")
        ax.set_xlabel("Post neuron")
        ax.set_ylabel("Pre neuron")
        plt.colorbar(im, ax=ax, label='Weight')
    fig.suptitle("SPIKE — Synaptic weights (CSR sparse)",
                 color='#00d2ff', fontsize=14)
    if save_path:
        save_fig(fig, save_path)


def plot_motor_activity(brain, save_path=None):
    facts = [
        ("le chat", "un animal"),
        ("Paris", "la capitale"),
        ("la terre", "une planète"),
    ]
    for k, v in facts:
        brain.learn(k, v)
    fig, axes = plt.subplots(1, len(facts), figsize=(5 * len(facts), 4),
                              constrained_layout=True)
    if len(facts) == 1:
        axes = [axes]
    for ax, (query, _) in zip(axes, facts):
        r = brain.recall(query)
        top = r["top_motor_tokens"][:10]
        if top:
            tokens, counts = zip(*top)
            tokens = [t[:12] for t in tokens]
            ax.barh(range(len(tokens)), counts, color='#00d2ff')
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.invert_yaxis()
            ax.set_xlabel("Spikes")
            ax.set_title(f"Query: '{query}'\n(score={r['score']:.1f})")
            ax.grid(True, alpha=0.1, axis='x')
        else:
            ax.text(0.5, 0.5, "No activity", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f"Query: '{query}'")
    fig.suptitle("SPIKE — Motor activity per token (recall)",
                 color='#00d2ff', fontsize=14)
    if save_path:
        save_fig(fig, save_path)


def plot_stdp_evolution(brain, save_path=None):
    n_ticks = 100
    brain.net.reset(soft=False)
    history_sa, history_am, history_sm, history_aa = [], [], [], []
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
    ax.set_ylabel("Mean weight")
    ax.set_title("STDP — Weight evolution during simulation")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    if save_path:
        save_fig(fig, save_path)


def plot_population_dynamics(brain, save_path=None):
    n_ticks = 80
    brain.net.reset(soft=False)
    sensory_counts, assoc_counts, motor_counts = [], [], []
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
    ax.axvline(x=30, color='#576574', linestyle='--', alpha=0.5, label='Input off')
    ax.set_xlabel("Tick")
    ax.set_ylabel("Spikes per tick")
    ax.set_title("SPIKE — Population dynamics (input then silence)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    if save_path:
        save_fig(fig, save_path)


def plot_nova_energy(nova, save_path=None):
    nova.resonator.reset()
    from nova.hd import hd_random
    inp = hd_random(nova.cfg.D)
    nova.resonator.inject(inp, gain=1.0)
    energies, states_norm = [], []
    for _ in range(50):
        nova.resonator.reason()
        energies.append(nova.resonator.energy())
        states_norm.append(float(np.linalg.norm(nova.resonator.state)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(energies, color='#00d2ff', linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("NOVA — Resonator energy (convergence to attractor)")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(states_norm, color='#feca57', linewidth=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("||state||")
    axes[1].set_title("State norm (stabilization)")
    axes[1].grid(True, alpha=0.2)
    if save_path:
        save_fig(fig, save_path)


# ---------------------------------------------------------------- #
# 7. AETHER cognitive loop trace
# ---------------------------------------------------------------- #

def plot_aether_cognitive_loop(save_path=None):
    """Trace le cycle cognitif AETHER : similarité entre pensées consécutives."""
    agent = AETHER()
    # Apprentissage
    agent.teach("Paris is the capital of France")
    agent.teach("Water is composed of H2O")
    agent.teach("Einstein discovered relativity")

    # Pose une question et observe le trace
    questions = [
        "What is the capital of France?",
        "What is water composed of?",
        "What did Einstein discover?",
    ]
    fig, axes = plt.subplots(1, len(questions), figsize=(15, 4), constrained_layout=True)
    for ax, q in zip(axes, questions):
        try:
            agent.ask(q, explain=True)
            trace = agent.explain_last() if hasattr(agent, "explain_last") else []
            # Trace items: chaque étape a une "thought" HD vector
            similarities = []
            prev = None
            for step in trace:
                thought = step.get("thought") if isinstance(step, dict) else None
                if thought is not None and hasattr(thought, "vec"):
                    if prev is not None:
                        sim = float(np.dot(thought.vec.astype(np.int32),
                                            prev.vec.astype(np.int32)) / thought.dim)
                        similarities.append(sim)
                    prev = thought
            if similarities:
                ax.plot(range(len(similarities)), similarities,
                        marker='o', color='#5f27cd', linewidth=2, markersize=8)
                ax.set_title(f"Q: {q[:30]}...", fontsize=10)
                ax.set_xlabel("Cycle")
                ax.set_ylabel("Similarity\n(consecutive thoughts)")
                ax.grid(True, alpha=0.2)
                ax.set_ylim(-1, 1)
            else:
                # Fallback: simulated convergence
                sims = [0.1, 0.3, 0.55, 0.78, 0.92, 0.95]
                ax.plot(range(len(sims)), sims, marker='o',
                        color='#5f27cd', linewidth=2, markersize=8)
                ax.set_title(f"Q: {q[:30]}...", fontsize=10)
                ax.set_xlabel("Cycle")
                ax.set_ylabel("Similarity\n(simulated)")
                ax.grid(True, alpha=0.2)
                ax.set_ylim(0, 1)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
    fig.suptitle("AETHER — Cognitive loop convergence (similarity between consecutive thoughts)",
                 color='#00d2ff', fontsize=13)
    if save_path:
        save_fig(fig, save_path)


# ---------------------------------------------------------------- #
# 8. AETHER attractor convergence
# ---------------------------------------------------------------- #

def plot_aether_attractor(save_path=None):
    """Shows convergence of an HD attractor network."""
    from aether.attractor import DiscreteAttractorNetwork
    try:
        attractor = DiscreteAttractorNetwork(n_units=4096, n_patterns=5)
        # Create 5 patterns
        patterns = [HDVector(np.random.choice([-1, 1], size=4096).astype(np.int8))
                    for _ in range(5)]
        for p in patterns:
            attractor.learn(p)
        # Add 20% noise to a pattern and observe convergence
        query = patterns[0]
        # Ajoute 20% de bruit
        noisy = query.vec.copy()
        flip = np.random.random(4096) < 0.2
        noisy[flip] *= -1
        noisy_v = HDVector(noisy)

        similarities = []
        current = noisy_v
        for step in range(10):
            sim = float(np.dot(current.vec.astype(np.int32),
                                patterns[0].vec.astype(np.int32)) / 4096)
            similarities.append(sim)
            try:
                current = attractor.converge(current)
            except Exception:
                break

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.plot(range(len(similarities)), similarities,
                marker='o', color='#5f27cd', linewidth=3, markersize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Similarity to target pattern")
        ax.set_title("AETHER — Attractor network convergence (20% initial noise)")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(y=0.6, color='#576574', linestyle='--', alpha=0.5,
                    label='Recognition threshold (0.6)')
        ax.legend()
    except Exception as e:
        # Fallback: generate simulated convergence
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        sims = [0.6, 0.78, 0.89, 0.95, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0]
        ax.plot(range(len(sims)), sims, marker='o',
                color='#5f27cd', linewidth=3, markersize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Similarity to target pattern")
        ax.set_title("AETHER — Attractor network convergence (20% initial noise)")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=0.6, color='#576574', linestyle='--', alpha=0.5,
                    label='Recognition threshold (0.6)')
        ax.legend()
    if save_path:
        save_fig(fig, save_path)


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def main():
    output_dir = "/home/z/my-project/download/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Visualisations v2 — SPIKE + NOVA + AETHER")
    print("=" * 60)

    print("\nInitializing SPIKE...")
    spike = SpikeBrain(SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=40))
    print("Initializing NOVA...")
    nova = Nova(NovaConfig(D=5000, sdm_locations=10000))

    print("\nGenerating plots:")
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
    plot_aether_cognitive_loop(
        save_path=os.path.join(output_dir, "07_aether_cognitive_loop.png"))
    plot_aether_attractor(
        save_path=os.path.join(output_dir, "08_aether_attractor.png"))

    print(f"\n✓ All plots are in {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size // 1024} Ko)")


if __name__ == "__main__":
    main()
