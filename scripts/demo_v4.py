"""
demo_v4.py — AETHER v4 brain-inspired demonstrations.

Showcases all 8 brain-inspired cognitive modules and how they integrate:
  1. Kuramoto oscillators — concepts bound by phase synchrony
  2. Attractor networks — thoughts as stable fixed points
  3. Global workspace — ignition + conscious broadcast
  4. Predictive coding — learning by minimizing surprise
  5. Predictive hierarchy — 4-level cortex (sensory→abstract)
  6. Neuromodulators — dopamine, serotonin, ACh, NE drive behavior
  7. Comprehension integrator — measurable "real understanding"
  8. Consciousness module — self-model + metacognition + narrative
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import (
    AETHER, HDVector,
    KuramotoNetwork, DiscreteAttractorNetwork, PatternCompleter,
    RingAttractor, SheetAttractor,
    GlobalWorkspace, PredictiveModel, SequencePredictor,
    PredictiveHierarchy, NeuromodulatorSystem,
    ComprehensionIntegrator, ConsciousnessModule,
)


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# 1. Kuramoto
# --------------------------------------------------------------------------- #
def demo_kuramoto():
    banner("1. Kuramoto Oscillators — concepts bound by phase synchrony")

    print("  BRAIN: distant neural populations synchronize (gamma ~40Hz) when")
    print("         representing bound features (Singer, Gray, Engel 1990s).")
    print()
    print("  AETHER: each concept = oscillators. Synchronized clusters = bound ideas.")
    print()

    net = KuramotoNetwork(n_oscillators=64, coupling=0.6)
    concepts = [
        ("Paris",   HDVector.from_text_seed("Paris",   4096)),
        ("France",  HDVector.from_text_seed("France",  4096)),
        ("Tokyo",   HDVector.from_text_seed("Tokyo",   4096)),
        ("Japan",   HDVector.from_text_seed("Japan",   4096)),
    ]
    for label, vec in concepts:
        net.add_concept(label, vec, n_osc=8)

    print(f"  Added {len(concepts)} concepts ({net.N} oscillators)")
    print(f"  Initial order parameter r = {net._compute_state().order_parameter:.4f}")
    print()
    print("  Running 200 Kuramoto steps...")
    states = net.run(200)

    # Show trajectory
    print("\n  Order parameter trajectory (every 20 steps):")
    for i in range(0, len(states), 20):
        r = states[i].order_parameter
        bar = "#" * int(r * 40)
        print(f"    t={i:3d}: r={r:.3f} |{bar}")

    print(f"\n  Final r = {states[-1].order_parameter:.4f}")
    print(f"  Comprehension score: {net.comprehension_score():.4f}")
    print(f"  Converged: {net.has_converged()}")

    print("\n  Concept synchrony matrix:")
    sync = net.concept_synchrony()
    labels = [c[0] for c in concepts]
    print(f"    {'':10s}", end="")
    for l in labels:
        print(f"{l:>10s}", end="")
    print()
    for a in labels:
        print(f"    {a:10s}", end="")
        for b in labels:
            score = sync.get((a, b), 0.0)
            print(f"{score:>10.3f}", end="")
        print()


# --------------------------------------------------------------------------- #
# 2. Attractor networks
# --------------------------------------------------------------------------- #
def demo_attractors():
    banner("2. Attractor Networks — thoughts as stable fixed points")

    print("  BRAIN: persistent firing in prefrontal cortex = working memory.")
    print("         Activity relaxes into attractor basins (Hopfield 1982).")
    print()
    print("  AETHER: discrete attractor (pattern completion) + continuous (ring, sheet).")
    print()

    # Discrete
    print("  === Discrete Attractor (pattern completion) ===")
    net = DiscreteAttractorNetwork(dim=4096, n_locations=3000, k=15)
    memories = [
        ("Paris",  HDVector.from_text_seed("Paris",  4096)),
        ("Tokyo",  HDVector.from_text_seed("Tokyo",  4096)),
        ("Python", HDVector.from_text_seed("Python", 4096)),
        ("Dog",    HDVector.from_text_seed("Dog",    4096)),
        ("Red",    HDVector.from_text_seed("Red",    4096)),
    ]
    for label, vec in memories:
        net.store_labeled(label, vec)

    completer = PatternCompleter(net)
    for noise in [0.1, 0.2]:
        print(f"\n    Noise = {noise*100:.0f}%:")
        correct = 0
        for label, vec in memories:
            completed, state = completer.complete(vec, noise_level=noise)
            settled = state.settled_at or "(none)"
            ok = settled == label
            if ok:
                correct += 1
            print(f"      {label!r:8s} -> settled={settled!r:12s} {'OK' if ok else 'FAIL'}")
        print(f"    Accuracy: {correct}/{len(memories)}")

    # Ring
    print("\n  === Ring Attractor (head direction / circular variable) ===")
    ring = RingAttractor(n_units=32, sigma=2.0)
    for target_deg in [0, 90, 180, 270]:
        ring.reset(0.0)
        target = np.radians(target_deg)
        external = ring._bump_at(target) * 2.0
        ring.relax(external_input=external, n_steps=30)
        actual = np.degrees(ring.current_angle())
        print(f"    Target: {target_deg:3d}° -> After relaxation: {actual:.1f}°")

    # Sheet
    print("\n  === Sheet Attractor (2D spatial) ===")
    sheet = SheetAttractor(width=16, height=16, sigma=2.0)
    for target in [(4, 4), (12, 12), (8, 14)]:
        sheet.reset(8, 8)
        external_2d = sheet._bump_at(*target) * 2.0
        sheet.relax(external_input=external_2d, n_steps=30)
        pos = sheet.current_position()
        print(f"    Target: {target} -> After relaxation: ({pos[0]:.1f}, {pos[1]:.1f})")


# --------------------------------------------------------------------------- #
# 3. Global workspace
# --------------------------------------------------------------------------- #
def demo_global_workspace():
    banner("3. Global Workspace — Baars theory + ignition")

    print("  BRAIN: specialists (vision, language, memory, motor) compete for")
    print("         broadcast. Winner's content becomes conscious (Baars 1988,")
    print("         Dehaene 2014). Ignition = sudden cortical synchronization.")
    print()
    print("  AETHER: 4 specialists (language, memory, tool, inference) compete.")
    print("          Above ignition threshold → broadcast to all specialists.")
    print()

    agent = AETHER()
    gw = agent.global_workspace

    print(f"  Specialists: {[s.name for s in gw.specialists]}")
    print(f"  Ignition threshold: {gw.ignition_threshold}")
    print()

    inputs = [
        "What is the capital of France?",
        "calc 2+2",
        "Hello",
        "Where is Tokyo located?",
    ]
    for inp in inputs:
        gw.reset()
        for _ in range(3):
            state = gw.cycle_step(inp)
        print(f"  Input: {inp!r}")
        print(f"    winner={state.winner!r} conf={state.winning_confidence:.3f} ignition={state.ignition}")
        for o in state.all_outputs:
            print(f"      {o.specialist_name:12s}: conf={o.confidence:.3f}")
        print(f"    => is_conscious: {gw.is_conscious()}")
        print()


# --------------------------------------------------------------------------- #
# 4. Predictive coding
# --------------------------------------------------------------------------- #
def demo_predictive():
    banner("4. Predictive Coding — Friston free energy")

    print("  BRAIN: the brain is a prediction machine. Higher levels predict")
    print("         lower levels. Only prediction errors propagate up.")
    print("         Surprise = free energy = learning signal.")
    print()
    print("  AETHER: PredictiveModel + SequencePredictor minimize surprise.")
    print()

    # Pattern learning
    print("  === Pattern Learning ===")
    model = PredictiveModel(dim=4096)
    base = HDVector.from_text_seed("base", 4096)
    pattern = []
    for i in range(5):
        noise = HDVector.from_text_seed(f"n_{i}", 4096)
        from aether.hd import bundle
        pattern.append(bundle([base, noise], weights=[0.8, 0.2]))

    for trial in range(4):
        for v in pattern:
            model.observe(v)
        print(f"    Trial {trial+1}: mean_surprise={model.mean_surprise():.3f}")

    # Surprise detection
    print(f"\n  Surprise detection:")
    surprise_vec = HDVector.from_text_seed("SURPRISE", 4096)
    err = model.observe(surprise_vec)
    print(f"    Novel input: surprise={err.surprise:.3f} is_surprised={model.is_surprised()}")
    err = model.observe(pattern[0])
    print(f"    Expected input: surprise={err.surprise:.3f} is_surprised={model.is_surprised()}")

    # Sequence prediction
    print(f"\n  === Sequence Prediction (A->B->C repeating) ===")
    sp = SequencePredictor(dim=4096, context_size=2)
    seq = [HDVector.from_text_seed(f"item_{i%3}", 4096) for i in range(15)]
    for i, item in enumerate(seq):
        err = sp.observe(item)
        if i >= 2:
            marker = "OK" if err.surprise < 0.1 else "surprise"
            print(f"    item {i+1}: surprise={err.surprise:.3f} [{marker}]")


# --------------------------------------------------------------------------- #
# 5. Predictive hierarchy
# --------------------------------------------------------------------------- #
def demo_hierarchy():
    banner("5. Predictive Hierarchy — 4-level cortex")

    print("  BRAIN: cortex has 6 layers, hierarchical organization.")
    print("         L0 sensory → L1 features → L2 concepts → L3 abstract")
    print("         Bottom-up: prediction errors. Top-down: predictions.")
    print()
    print("  AETHER: 4-level hierarchy with predictive coding at each level.")
    print()

    agent = AETHER()
    hierarchy = agent.hierarchy

    inputs = [
        "Paris is the capital of France",
        "Paris is the capital of France",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
        "Tokyo is the capital of Japan",
        "Tokyo is the capital of Japan",
        "Water is a liquid",  # very novel
        "Water is a liquid",  # repeat
        "Water is a liquid",  # repeat
    ]
    print(f"  Processing sequence (with repeats):")
    for i, inp in enumerate(inputs):
        states = hierarchy.process(inp)
        global_surprise = np.mean([s.surprise for s in states])
        level_surprises = [f"{s.surprise:.3f}" for s in states]
        marker = "NOVEL" if global_surprise > 0.4 else "expected"
        print(f"    cycle {i+1}: {inp!r:42s} surprise={global_surprise:.3f} [{marker}]")
        print(f"             levels: {level_surprises}")

    print(f"\n  Final: mean_surprise={hierarchy.mean_surprise():.3f} "
          f"trend={hierarchy.surprise_trend():.3f} "
          f"is_surprised={hierarchy.is_surprised()}")


# --------------------------------------------------------------------------- #
# 6. Neuromodulators
# --------------------------------------------------------------------------- #
def demo_neuromodulators():
    banner("6. Neuromodulators — DA, 5HT, ACh, NE drive behavior")

    print("  BRAIN: four neuromodulator systems gate global brain state:")
    print("    DOPAMINE (VTA):       reward prediction error → learning")
    print("    SEROTONIN (raphe):    mood, patience, long-term planning")
    print("    ACETYLCHOLINE (BF):   attention, learning rate, focus")
    print("    NOREPINEPHRINE (LC):  arousal, exploration vs exploitation")
    print()
    print("  AETHER: these 4 levels gate learning rate, patience, attention, exploration.")
    print()

    agent = AETHER()
    nm = agent.neuromodulators

    print(f"  Initial: DA={nm.levels.dopamine:.2f} 5HT={nm.levels.serotonin:.2f} "
          f"ACh={nm.levels.acetylcholine:.2f} NE={nm.levels.norepinephrine:.2f}")
    print(f"  Mood: {nm.mood()}, Learning rate: {nm.learning_rate():.3f}, "
          f"Patience: {nm.patience()}, Exploration: {nm.exploration_rate():.3f}")
    print()

    events = [
        (1.0, 0.0, True,   "Success! Reward received"),
        (1.0, 0.0, True,   "Another success"),
        (0.0, 0.5, None,   "Mild surprise"),
        (0.0, 0.0, None,   "Idle"),
        (0.0, 0.0, None,   "Idle"),
        (0.0, 0.0, None,   "Idle"),
        (-0.5, 0.8, False, "Failure! Surprise + punishment"),
        (2.0, 0.0, True,   "Big success!"),
    ]
    print(f"  Event sequence:")
    print(f"    {'Event':40s} | {'DA':>5s} {'5HT':>5s} {'ACh':>5s} {'NE':>5s} | {'mood':>10s} {'LR':>5s} {'Pat':>3s} {'Expl':>5s}")
    print(f"    {'-'*40}-+-{'-'*23}-+-{'-'*26}")
    for reward, surprise, success, desc in events:
        nm.update(reward=reward, surprise=surprise, success=success)
        print(f"    {desc:40s} | {nm.levels.dopamine:5.2f} {nm.levels.serotonin:5.2f} "
              f"{nm.levels.acetylcholine:5.2f} {nm.levels.norepinephrine:5.2f} | "
              f"{nm.mood():>10s} {nm.learning_rate():5.3f} {nm.patience():3d} {nm.exploration_rate():5.3f}")


# --------------------------------------------------------------------------- #
# 7. Comprehension
# --------------------------------------------------------------------------- #
def demo_comprehension():
    banner("7. Comprehension Integrator — measurable 'real understanding'")

    print("  PROBLEM: transformer LLMs have no test of comprehension — they")
    print("           just pattern-match. We don't know if they 'understand'.")
    print()
    print("  AETHER: comprehension = multi-indicator score combining:")
    print("    - attractor stability (Hopfield convergence)")
    print("    - prediction match (low free energy)")
    print("    - broadcast active (global workspace ignition)")
    print("    - oscillator sync (Kuramoto order parameter)")
    print("    - hierarchy calm (low multi-level surprise)")
    print("    - neuromodulator balance (homeostasis)")
    print()

    agent = AETHER()
    questions = [
        "Hello",
        "What is the capital of France?",
        "teach Lisbon is the capital of Portugal",
        "What is the capital of Portugal?",
        "What is the capital of the country where Osaka is located?",
        "compare Paris and Tokyo",
        "calc 2+2*5",
    ]
    print(f"  {'Question':50s} | Score  Conf   Comp?  Notes")
    print(f"  {'-'*50}-+-{'-'*40}")
    for q in questions:
        agent.ask(q)
        c = agent.last_comprehension
        comp_str = "YES" if c.is_comprehending else "no"
        print(f"  {q:50s} | {c.comprehension_score:.3f}  {c.confidence:.3f}  {comp_str:5s}  {c.notes}")


# --------------------------------------------------------------------------- #
# 8. Consciousness
# --------------------------------------------------------------------------- #
def demo_consciousness():
    banner("8. Consciousness Module — self-model + metacognition + narrative")

    print("  BRAIN: consciousness (functional) = self-model + metacognition +")
    print("         attention + narrative stream. Subjective experience is the")
    print("         'hard problem' (Chalmers) — we model the functional parts.")
    print()
    print("  AETHER: ConsciousnessModule wraps the cognitive system with:")
    print("    - SelfModel: HD vector representing 'me, right now'")
    print("    - Metacognition: monitors comprehension, surprise, mood")
    print("    - AttentionDirector: boosts/suppresses specialists")
    print("    - ForwardModel: predicts effects of own actions")
    print("    - NarrativeBuffer: stream of conscious states")
    print()

    agent = AETHER()
    print(f"  Identity: AETHER")
    print(f"  Initial self-awareness: {agent.consciousness.self_model.self_awareness_score():.3f}")
    print()

    conversation = [
        "Hello",
        "What are you?",
        "What is the capital of France?",
        "calc 1234 * 5678",
        "teach Reykjavik is the capital of Iceland",
        "What is the capital of Iceland?",
        "compare Paris and Tokyo",
        "Where is Montreal located?",
        "What is the capital of the country where Osaka is located?",
        "thank you",
    ]
    print(f"  Conversation:")
    print(f"    {'Turn':5s} | {'Input':50s} | {'Mood':>10s} {'Comp':>5s} {'Action':>20s}")
    print(f"    {'-'*5}-+-{'-'*50}-+-{'-'*40}")
    for q in conversation:
        agent.ask(q)
        meta = agent.consciousness.metacognition.read() if agent.consciousness.metacognition else None
        if meta:
            print(f"    {agent.consciousness.cycle:5d} | {q:50s} | {meta.mood:>10s} "
                  f"{meta.comprehension_score:5.2f} {agent.metacognitive_action():>20s}")

    # Final introspection
    print(f"\n  Final introspection:")
    intro = agent.introspect()
    for k, v in intro.items():
        if k != "narrative_summary":
            print(f"    {k}: {v}")
    print(f"\n  Recent narrative:")
    for line in intro["narrative_summary"]:
        print(f"    {line}")


# --------------------------------------------------------------------------- #
# 9. Full brain trace
# --------------------------------------------------------------------------- #
def demo_brain_trace():
    banner("9. Full Brain Trace — see AETHER think")

    print("  Watch the full cognitive process for a single question.")
    print()

    agent = AETHER()
    agent.verbose = True

    print("\n  Question: 'What is the capital of the country where Osaka is located?'")
    print("  " + "-" * 74)
    ans = agent.ask("What is the capital of the country where Osaka is located?", explain=True)
    print(f"\n  Final answer: {ans}")


# --------------------------------------------------------------------------- #
# 10. Performance
# --------------------------------------------------------------------------- #
def demo_performance():
    banner("10. Performance — still CPU-only, still fast, now with brain dynamics")

    agent = AETHER()
    print(f"  Numpy: {np.__version__}")
    print(f"  Vector dim: {agent.dim}")
    print(f"  Vocab: {len(agent.assoc.vocab)} tokens")
    print(f"  KB: {len(agent.assoc.triples)} triples")
    print(f"  Tools: {len(agent.list_tools())}")
    print(f"  Attractor memories: {len(agent.attractor.labeled_memories)}")
    print(f"  Kuramoto oscillators: {agent.kuramoto.N}")
    print(f"  Specialists: {len(agent.global_workspace.specialists)}")
    print(f"  Hierarchy levels: {agent.hierarchy.n_levels}")
    print()

    # Time ask() with full brain processing
    t0 = time.perf_counter()
    for _ in range(20):
        agent.ask("What is the capital of France?")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  20 ask() calls (full brain): {elapsed:.0f}ms total = {elapsed/20:.1f}ms/call")

    t0 = time.perf_counter()
    for _ in range(10):
        agent.ask("What is the capital of the country where Osaka is located?")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  10 multi-hop ask() calls: {elapsed:.0f}ms total = {elapsed/10:.1f}ms/call")

    t0 = time.perf_counter()
    for _ in range(10):
        agent.teach("Foo is the capital of Bar", silent=True)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  10 teach() calls: {elapsed:.0f}ms total = {elapsed/10:.1f}ms/call")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    demos = [
        demo_kuramoto,
        demo_attractors,
        demo_global_workspace,
        demo_predictive,
        demo_hierarchy,
        demo_neuromodulators,
        demo_comprehension,
        demo_consciousness,
        demo_brain_trace,
        demo_performance,
    ]
    for d in demos:
        try:
            d()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {d.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All v4 brain-inspired demos complete.")
    print("  AETHER v4.0.0 — brain-inspired edition")
    print("=" * 76)


if __name__ == "__main__":
    main()
