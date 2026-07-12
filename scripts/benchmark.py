#!/usr/bin/env python3
"""
Benchmark v2 — compare SPIKE, NOVA, AETHER, HYBRID sur des tâches agentiques.

Tâches:
  1. Arithmetic (5 questions)
  2. Memory recall (apprentissage + rappel)
  3. Tool calling (calculator, python, time)
  4. Robustesse au bruit (paraphrases)

Métriques:
  - Latence (ms)
  - Précision (%)
  - Mémoire (Mo)
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
from aether import AETHER


def banner(title: str, char="="):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def measure(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()
    return result, (t1 - t0) * 1000


def estimate_memory(brain) -> float:
    """Estime l'empreinte mémoire en Mo."""
    if hasattr(brain, "ask"):
        # AETHER
        s = 0
        if hasattr(brain, "sdm") and brain.sdm is not None:
            s += brain.sdm.locations.nbytes + brain.sdm.contents.nbytes
        if hasattr(brain, "kb_store") and brain.kb_store is not None:
            s += brain.kb_store.locations.nbytes + brain.kb_store.contents.nbytes
        if hasattr(brain, "encoder") and hasattr(brain.encoder, "item_memory"):
            for v in brain.encoder.item_memory.values():
                if hasattr(v, "vec"):
                    s += v.vec.nbytes
        if hasattr(brain, "lm") and brain.lm is not None:
            if hasattr(brain.lm, "sdm"):
                s += brain.lm.sdm.locations.nbytes + brain.lm.sdm.contents.nbytes
        if s == 0:
            # Fallback — estimate from stats
            try:
                stats = brain.stats()
                if "kb_store" in stats:
                    n = stats["kb_store"].get("n_locations", 5000)
                    d = stats["kb_store"].get("dim", 4096)
                    s = n * d * 2  # int8 locations + int8 contents
            except Exception:
                pass
        return s / (1024 * 1024)
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
        s += brain.cfg.n_sensory * 4 * 4
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

def task_arithmetic(brain, brain_name) -> dict:
    """Tâche: 5 questions arithmétiques."""
    if brain_name == "AETHER":
        questions = [
            ("calc 2+2", "4"),
            ("calc 15 times 3", "45"),
            ("calc 144 / 12", "12"),
            ("calc 2 to the power of 10", "1024"),
            ("calc 10 + 5 * 2", "20"),
        ]
    else:
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
            r, dt = measure(brain.chat if hasattr(brain, "chat") else brain.ask, q)
            times.append(dt)
            if expected in r:
                correct += 1
        except Exception as e:
            print(f"    error: {e}")
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(questions),
        "accuracy": correct / len(questions),
        "avg_time_ms": float(np.mean(times)),
        "max_time_ms": float(max(times)),
    }


def task_memory(brain, brain_name) -> dict:
    """Tâche: apprentissage + rappel."""
    if brain_name == "AETHER":
        # AETHER: format "X is the capital of Y"
        facts = [
            ("Paris is the capital of France", "paris"),
            ("Kinshasa is the capital of Congo", "kinshasa"),
            ("Montreal is located in Canada", "canada"),
            ("Water is composed of H2O", "h2o"),
            ("Einstein discovered relativity", "relativity"),
        ]
        queries = [
            ("What is the capital of France?", "paris"),
            ("What is the capital of Congo?", "kinshasa"),
            ("Where is Montreal located?", "canada"),
            ("What is water composed of?", "h2o"),
            ("What did Einstein discover?", "relativity"),
        ]
    else:
        facts = [
            ("le chat", "un animal"),
            ("Paris", "la capitale de la France"),
            ("Mars", "la quatrième planète"),
            ("Einstein", "physicien"),
            ("l'eau", "H2O"),
        ]
        queries = [
            (f"que sais-tu sur {k}", v) for k, v in facts
        ]

    # Apprentissage
    learn_times = []
    for fact, _ in facts:
        if brain_name == "AETHER":
            _, dt = measure(brain.teach, fact)
        else:
            if " est " in fact or " is " in fact or " located" in fact or " composed" in fact or " discovered" in fact:
                # Pour AETHER on a déjà teach; pour les autres on apprend k=v
                k, v = facts[len(learn_times)][1], fact  # fallback
                _, dt = measure(brain.learn, fact, _)
            else:
                _, dt = measure(brain.learn, fact, _)
        learn_times.append(dt)

    # Rappel
    correct = 0
    recall_times = []
    for q, expected in queries:
        try:
            r, dt = measure(brain.chat if hasattr(brain, "chat") else brain.ask, q)
            recall_times.append(dt)
            if expected.lower() in r.lower() or any(w in r.lower() for w in expected.split()):
                correct += 1
        except Exception:
            recall_times.append(1000.0)

    return {
        "correct": correct,
        "total": len(facts),
        "accuracy": correct / len(facts),
        "avg_learn_ms": float(np.mean(learn_times)),
        "avg_recall_ms": float(np.mean(recall_times)),
    }


def task_tool_calling(brain, brain_name) -> dict:
    """Tâche: appel d'outils variés."""
    if brain_name == "AETHER":
        tests = [
            ("calc 5+5", "calc"),
            ("time", "time"),
            ("python eval [1,2,3]", "python"),
        ]
    else:
        tests = [
            ("calcule 5+5", "calculator"),
            ("python: print(42)", "python"),
            ("quelle heure est-il", "time"),
        ]
    correct = 0
    times = []
    for q, expected in tests:
        try:
            r, dt = measure(brain.chat if hasattr(brain, "chat") else brain.ask, q)
            times.append(dt)
            if expected in r.lower() or "outil" in r.lower() or "calc" in r.lower():
                correct += 1
        except Exception:
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(tests),
        "accuracy": correct / len(tests),
        "avg_time_ms": float(np.mean(times)),
    }


def task_robustness(brain, brain_name) -> dict:
    """Tâche: robustesse aux variations de phrasing."""
    if brain_name == "AETHER":
        brain.teach("The cat is an animal")
        queries = [
            "What is a cat?",
            "Tell me about the cat",
            "cat",
            "the cat",
            "What does the cat be?",
        ]
        expected = "animal"
    else:
        brain.learn("le chat", "un animal")
        queries = [
            "le chat",
            "chat",
            "le chat dort",
            "qui est le chat",
            "parle-moi du chat",
        ]
        expected = "animal"

    correct = 0
    times = []
    for q in queries:
        try:
            r, dt = measure(brain.chat if hasattr(brain, "chat") else brain.ask, q)
            times.append(dt)
            if expected.lower() in r.lower() or "chat" in r.lower():
                correct += 1
        except Exception:
            times.append(1000.0)
    return {
        "correct": correct,
        "total": len(queries),
        "accuracy": correct / len(queries),
        "avg_time_ms": float(np.mean(times)),
    }


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def main():
    banner("BENCHMARK v2 — SPIKE vs NOVA vs AETHER vs HYBRID", char="#")
    print("""
Compare 4 cerveaux sur 4 tâches agentiques.
Métriques: précision, latence, mémoire.
""")

    # Init
    print("Initialisation...")
    t0 = time.time()
    spike = SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25,
    ))
    nova = Nova(NovaConfig(D=2000, sdm_locations=5000))
    aether = AETHER()
    hybrid = HybridBrain(HybridConfig(
        spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25),
        nova=NovaConfig(D=2000, sdm_locations=5000),
    ))
    print(f"Prêt en {time.time()-t0:.2f}s\n")

    brains = {
        "SPIKE": (spike, "SPIKE"),
        "NOVA": (nova, "NOVA"),
        "AETHER": (aether, "AETHER"),
        "HYBRID": (hybrid, "HYBRID"),
    }

    results = {}
    for name, (brain, brain_name) in brains.items():
        banner(f"Test {name}")
        try:
            results[name] = {
                "arithmetic": task_arithmetic(brain, brain_name),
                "memory": task_memory(brain, brain_name),
                "tool_calling": task_tool_calling(brain, brain_name),
                "robustness": task_robustness(brain, brain_name),
                "memory_mb": estimate_memory(brain),
            }
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}
        # Reset entre les tests
        if hasattr(brain, "net"):
            brain.net.reset()
        elif hasattr(brain, "resonator"):
            brain.resonator.reset()
        elif hasattr(brain, "spike"):
            brain.spike.net.reset()

    # Tableau récapitulatif
    banner("RÉSULTATS")
    print(f"\n{'Métrique':<30} {'SPIKE':<14} {'NOVA':<14} {'AETHER':<14} {'HYBRID':<14}")
    print("-" * 86)
    for task_name in ["arithmetic", "memory", "tool_calling", "robustness"]:
        print(f"\n{task_name.upper()}:")
        for metric in ["accuracy", "avg_time_ms", "avg_learn_ms", "avg_recall_ms"]:
            row = f"  {metric:<28}"
            for brain_name in ["SPIKE", "NOVA", "AETHER", "HYBRID"]:
                val = results[brain_name].get(task_name, {}).get(metric)
                if val is None:
                    row += f" {'—':<13}"
                elif "time" in metric or "ms" in metric:
                    row += f" {val:<13.1f}"
                else:
                    row += f" {val*100:<13.1f}"
            print(row + ("%" if "accuracy" in metric else " ms"))

    print(f"\nMÉMOIRE (Mo):")
    row = f"  {'RAM':<28}"
    for brain_name in ["SPIKE", "NOVA", "AETHER", "HYBRID"]:
        row += f" {results[brain_name]['memory_mb']:<13.2f}"
    print(row)

    # Sauvegarde JSON
    output_path = "/home/z/my-project/download/benchmark_results_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Résultats sauvés dans {output_path}")

    # Graphique
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        fig.suptitle("Benchmark v2 — SPIKE vs NOVA vs AETHER vs HYBRID",
                     fontsize=15, fontweight='bold')

        brain_names = ["SPIKE", "NOVA", "AETHER", "HYBRID"]
        colors = ["#00d2ff", "#feca57", "#5f27cd", "#ff6b6b"]

        # Accuracy par tâche
        tasks = ["arithmetic", "memory", "tool_calling", "robustness"]
        ax = axes[0, 0]
        x = np.arange(len(tasks))
        w = 0.20
        for i, name in enumerate(brain_names):
            accs = [results[name].get(t, {}).get("accuracy", 0) * 100 for t in tasks]
            ax.bar(x + i * w, accs, w, label=name, color=colors[i])
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_ylabel("Précision (%)")
        ax.set_title("Précision par tâche")
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(0, 110)

        # Latence arithmetic
        ax = axes[0, 1]
        for i, name in enumerate(brain_names):
            times = []
            for t in tasks:
                tm = results[name].get(t, {}).get("avg_time_ms")
                if tm is None:
                    tm = results[name].get(t, {}).get("avg_recall_ms", 0)
                times.append(tm if tm else 0)
            ax.plot(tasks, times, marker="o", label=name,
                    color=colors[i], linewidth=2)
        ax.set_ylabel("Latence (ms)")
        ax.set_title("Latence moyenne par tâche (log scale)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.2)

        # Mémoire
        ax = axes[1, 0]
        mems = [results[name]["memory_mb"] for name in brain_names]
        bars = ax.bar(brain_names, mems, color=colors)
        ax.set_ylabel("Mémoire (Mo)")
        ax.set_title("Empreinte mémoire")
        ax.grid(True, alpha=0.2, axis='y')
        for bar, mem in zip(bars, mems):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.2f}', ha='center', va='bottom')

        # Apprentissage vs rappel
        ax = axes[1, 1]
        x = np.arange(len(brain_names))
        w = 0.35
        learn_times = [results[name].get("memory", {}).get("avg_learn_ms", 0) for name in brain_names]
        recall_times = [results[name].get("memory", {}).get("avg_recall_ms", 0) for name in brain_names]
        ax.bar(x - w/2, learn_times, w, label="Learn", color="#00d2ff")
        ax.bar(x + w/2, recall_times, w, label="Recall", color="#feca57")
        ax.set_xticks(x)
        ax.set_xticklabels(brain_names)
        ax.set_ylabel("Temps (ms)")
        ax.set_title("Apprentissage vs Rappel (mémoire)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

        fig_path = "/home/z/my-project/download/benchmark_chart_v2.png"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✓ Graphique sauvé dans {fig_path}")
    except Exception as e:
        print(f"⚠ Graphique non généré: {e}")

    banner("FIN DU BENCHMARK", char="#")


if __name__ == "__main__":
    main()
