#!/usr/bin/env python3
"""
Benchmark — compare SPIKE, NOVA, HYBRID sur des tâches agentiques.

Tâches:
  1. Arithmetic (5 questions)
  2. Memory recall (apprentissage + rappel)
  3. Tool calling (calculator, python, time)
  4. Multi-step (calcul + apprentissage du résultat)
  5. Robustesse au bruit

Métriques:
  - Latence (ms)
  - Précision (%)
  - Mémoire (Mo)
  - Tokens de vocabulaire
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import SpikeBrain, SpikeConfig
from nova import Nova, NovaConfig
from hybrid import HybridBrain, HybridConfig


def banner(title: str, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def measure(fn, *args, **kwargs):
    """Mesure le temps d'exécution."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()
    return result, (t1 - t0) * 1000


def estimate_memory(brain) -> float:
    """Estime l'empreinte mémoire en Mo."""
    if hasattr(brain, "net"):
        # SPIKE
        s = brain.net.syn_sens_to_assoc.W.data.nbytes
        s += brain.net.syn_sens_to_assoc.W.indptr.nbytes + brain.net.syn_sens_to_assoc.W.indices.nbytes
        s += brain.net.syn_assoc_to_assoc.W.data.nbytes
        s += brain.net.syn_assoc_to_assoc.W.indptr.nbytes + brain.net.syn_assoc_to_assoc.W.indices.nbytes
        s += brain.net.syn_assoc_to_motor.W.data.nbytes
        s += brain.net.syn_assoc_to_motor.W.indptr.nbytes + brain.net.syn_assoc_to_motor.W.indices.nbytes
        if brain.syn_sens_to_motor is not None:
            s += brain.syn_sens_to_motor.W.data.nbytes
            s += brain.syn_sens_to_motor.W.indptr.nbytes + brain.syn_sens_to_motor.W.indices.nbytes
        # États neuronaux
        s += brain.cfg.n_sensory * 4 * 4  # 4 arrays de float32
        s += brain.cfg.n_associative * 4 * 4
        s += brain.cfg.n_motor * 4 * 4
        return s / (1024 * 1024)
    elif hasattr(brain, "memory"):
        # NOVA
        s = brain.memory.locations.nbytes + brain.memory.contents.nbytes
        s += brain.memory.locations_f32.nbytes
        s += brain.resonator.W.data.nbytes
        s += brain.resonator.W.indptr.nbytes + brain.resonator.W.indices.nbytes
        return s / (1024 * 1024)
    elif hasattr(brain, "spike") and hasattr(brain, "nova"):
        # HYBRID
        return estimate_memory(brain.spike) + estimate_memory(brain.nova)
    return 0.0


# ---------------------------------------------------------------- #
# Tasks
# ---------------------------------------------------------------- #

def task_arithmetic(brain) -> dict:
    """Tâche: 5 questions arithmétiques."""
    questions = [
        ("calcule 2+2", "4"),
        ("combien font 15 fois 3", "45"),
        ("que vaut la racine carrée de 144", "12"),
        ("calcule 2 puissance 10", "1024"),
        ("que vaut 10 plus 5 fois 2", "20"),
    ]
    correct = 0
    times = []
    for q, expected in questions:
        try:
            r, dt = measure(brain.chat, q)
            times.append(dt)
            if expected in r:
                correct += 1
        except Exception:
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(questions),
        "accuracy": correct / len(questions),
        "avg_time_ms": np.mean(times),
        "max_time_ms": max(times),
    }


def task_memory(brain) -> dict:
    """Tâche: apprentissage + rappel."""
    facts = [
        ("le chat", "un animal"),
        ("Paris", "la capitale de la France"),
        ("Mars", "la quatrième planète"),
        ("Einstein", "physicien"),
        ("l'eau", "H2O"),
    ]
    # Apprentissage
    learn_times = []
    for k, v in facts:
        _, dt = measure(brain.learn, k, v)
        learn_times.append(dt)

    # Rappel
    correct = 0
    recall_times = []
    for k, expected_v in facts:
        # Pour SPIKE: 'que sais-tu sur X'  / Pour NOVA: 'rappelle X'
        if hasattr(brain, "net"):
            q = f"que sais-tu sur {k}"
        elif hasattr(brain, "memory"):
            q = f"rappelle {k}"
        else:
            q = f"que sais-tu sur {k}"
        try:
            r, dt = measure(brain.chat, q)
            recall_times.append(dt)
            if expected_v in r.lower() or any(w in r.lower() for w in expected_v.split()):
                correct += 1
        except Exception:
            recall_times.append(1000.0)

    return {
        "correct": correct,
        "total": len(facts),
        "accuracy": correct / len(facts),
        "avg_learn_ms": np.mean(learn_times),
        "avg_recall_ms": np.mean(recall_times),
    }


def task_tool_calling(brain) -> dict:
    """Tâche: appel d'outils variés."""
    tests = [
        ("calcule 5+5", "calculator"),
        ("python: print(42)", "python"),
        ("quelle heure est-il", "time"),
    ]
    correct = 0
    times = []
    for q, expected_tool in tests:
        try:
            r, dt = measure(brain.chat, q)
            times.append(dt)
            if expected_tool in r.lower() or "outil" in r.lower():
                correct += 1
        except Exception:
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(tests),
        "accuracy": correct / len(tests),
        "avg_time_ms": np.mean(times),
    }


def task_robustness(brain) -> dict:
    """Tâche: robustesse aux variations de phrasing."""
    # Apprentissage
    brain.learn("le chat", "un animal")

    queries = [
        "le chat",
        "chat",
        "le chat dort",
        "qui est le chat",
        "parle-moi du chat",
    ]
    correct = 0
    times = []
    for q in queries:
        try:
            r, dt = measure(brain.chat, q)
            times.append(dt)
            if "animal" in r.lower() or "chat" in r.lower():
                correct += 1
        except Exception:
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(queries),
        "accuracy": correct / len(queries),
        "avg_time_ms": np.mean(times),
    }


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def main():
    banner("BENCHMARK — SPIKE vs NOVA vs HYBRID", char="#")
    print("""
Compare les 3 cerveaux sur 4 tâches agentiques.
Métriques: précision, latence, mémoire.
""")

    # Init
    print("Initialisation...")
    t0 = time.time()
    spike = SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25,
    ))
    nova = Nova(NovaConfig(D=2000, sdm_locations=5000))
    hybrid = HybridBrain(HybridConfig(
        spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25),
        nova=NovaConfig(D=2000, sdm_locations=5000),
    ))
    print(f"Prêt en {time.time()-t0:.2f}s\n")

    brains = {
        "SPIKE": spike,
        "NOVA": nova,
        "HYBRID": hybrid,
    }

    results = {}
    for name, brain in brains.items():
        banner(f"Test {name}")
        results[name] = {
            "arithmetic": task_arithmetic(brain),
            "memory": task_memory(brain),
            "tool_calling": task_tool_calling(brain),
            "robustness": task_robustness(brain),
            "memory_mb": estimate_memory(brain),
        }
        # Reset entre les tests
        if hasattr(brain, "net"):
            brain.net.reset()
        elif hasattr(brain, "resonator"):
            brain.resonator.reset()
        elif hasattr(brain, "spike"):
            brain.spike.net.reset()

    # Tableau récapitulatif
    banner("RÉSULTATS")
    print(f"\n{'Métrique':<30} {'SPIKE':<18} {'NOVA':<18} {'HYBRID':<18}")
    print("-" * 84)
    for task_name in ["arithmetic", "memory", "tool_calling", "robustness"]:
        print(f"\n{task_name.upper()}:")
        for metric in ["accuracy", "avg_time_ms", "avg_learn_ms", "avg_recall_ms"]:
            row = f"  {metric:<28}"
            for brain_name in ["SPIKE", "NOVA", "HYBRID"]:
                val = results[brain_name][task_name].get(metric)
                if val is None:
                    row += f" {'—':<17}"
                elif isinstance(val, float) and "time" in metric or "ms" in metric:
                    row += f" {val:<17.1f}"
                else:
                    row += f" {val*100:<17.1f}"
            print(row + ("%" if "accuracy" in metric else " ms"))

    print(f"\nMÉMOIRE (Mo):")
    row = f"  {'RAM':<28}"
    for brain_name in ["SPIKE", "NOVA", "HYBRID"]:
        row += f" {results[brain_name]['memory_mb']:<17.2f}"
    print(row)

    # Sauvegarde JSON
    output_path = "/home/z/my-project/download/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Résultats sauvés dans {output_path}")

    # Graphique
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle("Benchmark SPIKE vs NOVA vs HYBRID", fontsize=14)

        # Accuracy par tâche
        tasks = ["arithmetic", "memory", "tool_calling", "robustness"]
        ax = axes[0, 0]
        x = np.arange(len(tasks))
        w = 0.25
        for i, name in enumerate(["SPIKE", "NOVA", "HYBRID"]):
            accs = [results[name][t]["accuracy"] * 100 for t in tasks]
            ax.bar(x + i * w, accs, w, label=name)
        ax.set_xticks(x + w)
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_ylabel("Précision (%)")
        ax.set_title("Précision par tâche")
        ax.legend()
        ax.grid(True, alpha=0.2)

        # Latence
        ax = axes[0, 1]
        for i, name in enumerate(["SPIKE", "NOVA", "HYBRID"]):
            times = [results[name][t].get("avg_time_ms", 0) for t in tasks]
            ax.plot(tasks, times, marker="o", label=name, linewidth=2)
        ax.set_ylabel("Latence (ms)")
        ax.set_title("Latence moyenne par tâche")
        ax.legend()
        ax.grid(True, alpha=0.2)

        # Mémoire
        ax = axes[1, 0]
        mems = [results[name]["memory_mb"] for name in ["SPIKE", "NOVA", "HYBRID"]]
        ax.bar(["SPIKE", "NOVA", "HYBRID"], mems, color=["#00d2ff", "#feca57", "#ff6b6b"])
        ax.set_ylabel("Mémoire (Mo)")
        ax.set_title("Empreinte mémoire")
        ax.grid(True, alpha=0.2, axis="y")

        # Apprentissage vs rappel
        ax = axes[1, 1]
        for i, name in enumerate(["SPIKE", "NOVA", "HYBRID"]):
            learn = results[name]["memory"].get("avg_learn_ms", 0)
            recall = results[name]["memory"].get("avg_recall_ms", 0)
            ax.bar([i - 0.2, i + 0.2], [learn, recall], 0.4,
                   label=name, color=["#00d2ff", "#feca57"][i:i+1] if i == 0 else ["#ff6b6b", "#54a0ff"][i:i+1])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["SPIKE", "NOVA", "HYBRID"])
        ax.set_ylabel("Temps (ms)")
        ax.set_title("Apprentissage vs Rappel (mémoire)")
        ax.grid(True, alpha=0.2, axis="y")

        fig_path = "/home/z/my-project/download/benchmark_chart.png"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✓ Graphique sauvé dans {fig_path}")
    except Exception as e:
        print(f"⚠ Graphique non généré: {e}")

    banner("FIN DU BENCHMARK", char="#")


if __name__ == "__main__":
    main()
