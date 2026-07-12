"""
test_aether_v4.py — Test suite for AETHER v4 (brain-inspired edition).

Tests all 8 new v4 brain-inspired modules:
  1. Kuramoto network (oscillator binding)
  2. Attractor networks (discrete + ring + sheet)
  3. Global workspace (Baars theory)
  4. Predictive coding (Friston free energy)
  5. Predictive hierarchy (4 levels)
  6. Neuromodulator system (DA, 5HT, ACh, NE)
  7. Comprehension integrator (multi-indicator score)
  8. Consciousness module (self-model, metacognition, narrative)
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether import (
    AETHER,
    KuramotoNetwork, hd_to_phase,
    DiscreteAttractorNetwork, RingAttractor, SheetAttractor, PatternCompleter,
    GlobalWorkspace, make_language_specialist, make_memory_specialist,
    make_tool_specialist, make_inference_specialist,
    PredictiveModel, SequencePredictor, compute_prediction_error,
    PredictiveHierarchy,
    NeuromodulatorSystem,
    ComprehensionIntegrator,
    ConsciousnessModule,
    HDVector,
)


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# 1. Kuramoto
# --------------------------------------------------------------------------- #
def test_kuramoto():
    banner("1. Kuramoto Network — oscillator binding")

    net = KuramotoNetwork(n_oscillators=64, coupling=0.6)
    concepts = [
        ("Paris",  HDVector.from_text_seed("Paris",  4096)),
        ("France", HDVector.from_text_seed("France", 4096)),
        ("Tokyo",  HDVector.from_text_seed("Tokyo",  4096)),
        ("Japan",  HDVector.from_text_seed("Japan",  4096)),
    ]
    for label, vec in concepts:
        net.add_concept(label, vec, n_osc=8)

    print(f"  Initial order parameter r = {net._compute_state().order_parameter:.4f}")
    states = net.run(200)
    print(f"  Final order parameter r = {states[-1].order_parameter:.4f}")
    print(f"  Number of clusters: {len(states[-1].clusters)}")
    print(f"  Comprehension score: {net.comprehension_score():.4f}")
    print(f"  Converged: {net.has_converged()}")

    print("\n  Concept synchrony (top pairs):")
    sync = net.concept_synchrony()
    for (a, b), score in sorted(sync.items(), key=lambda x: -x[1])[:5]:
        if a != b:
            print(f"    sync({a!r:8s}, {b!r:8s}) = {score:.4f}")


# --------------------------------------------------------------------------- #
# 2. Attractor networks
# --------------------------------------------------------------------------- #
def test_attractors():
    banner("2. Attractor Networks — stable thoughts")

    # Discrete attractor
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
    print("  Pattern completion at 10% noise:")
    for label, vec in memories:
        completed, state = completer.complete(vec, noise_level=0.1)
        settled = state.settled_at or "(none)"
        print(f"    {label!r:8s} -> settled={settled!r:12s} sim={vec.similarity(completed):.3f}")

    # Ring attractor
    print("\n  === Ring Attractor (circular variable) ===")
    ring = RingAttractor(n_units=32, sigma=2.0)
    ring.reset(0.0)
    target = np.pi / 2  # 90°
    external = ring._bump_at(target) * 2.0
    ring.relax(external_input=external, n_steps=30)
    print(f"    Target: 90°, After relaxation: {np.degrees(ring.current_angle()):.1f}°")

    # Sheet attractor
    print("\n  === Sheet Attractor (2D spatial) ===")
    sheet = SheetAttractor(width=16, height=16, sigma=2.0)
    sheet.reset(8, 8)
    target_2d = (12, 14)
    external_2d = sheet._bump_at(*target_2d) * 2.0
    sheet.relax(external_input=external_2d, n_steps=30)
    pos = sheet.current_position()
    print(f"    Target: {target_2d}, After relaxation: ({pos[0]:.1f}, {pos[1]:.1f})")


# --------------------------------------------------------------------------- #
# 3. Global workspace
# --------------------------------------------------------------------------- #
def test_global_workspace():
    banner("3. Global Workspace — Baars theory + ignition")

    agent = AETHER()
    gw = agent.global_workspace

    print(f"  Specialists: {[s.name for s in gw.specialists]}")
    print(f"  Initial: is_conscious={gw.is_conscious()}")

    inputs = [
        "What is the capital of France?",
        "calc 2+2",
        "Hello",
    ]
    for inp in inputs:
        for _ in range(3):
            state = gw.cycle_step(inp)
        print(f"\n  Input: {inp!r}")
        print(f"    winner={state.winner!r} conf={state.winning_confidence:.3f} ignition={state.ignition}")
        print(f"    All specialists:")
        for o in state.all_outputs:
            print(f"      {o.specialist_name:12s}: conf={o.confidence:.3f}")
        print(f"    Ignition rate: {gw.ignition_rate():.3f}")
        print(f"    Dominant: {gw.dominant_specialist()}")


# --------------------------------------------------------------------------- #
# 4. Predictive coding
# --------------------------------------------------------------------------- #
def test_predictive():
    banner("4. Predictive Coding — Friston free energy")

    model = PredictiveModel(dim=4096)
    base = HDVector.from_text_seed("base", 4096)
    pattern = []
    for i in range(5):
        noise = HDVector.from_text_seed(f"n_{i}", 4096)
        pattern.append(bundle_v(base, noise))

    print("  Training on 5-item pattern (3 trials):")
    for trial in range(3):
        for v in pattern:
            model.observe(v)
        print(f"    Trial {trial+1}: mean_surprise={model.mean_surprise():.3f}")

    # Surprising input
    surprise_vec = HDVector.from_text_seed("SURPRISE", 4096)
    err = model.observe(surprise_vec)
    print(f"\n  Surprising input: surprise={err.surprise:.3f} is_surprised={model.is_surprised()}")

    # Sequence predictor
    print("\n  === Sequence Predictor ===")
    sp = SequencePredictor(dim=4096, context_size=2)
    seq = [HDVector.from_text_seed(f"item_{i%3}", 4096) for i in range(15)]
    for i, item in enumerate(seq):
        err = sp.observe(item)
        if i >= 2:
            print(f"    item {i+1}: surprise={err.surprise:.3f}")


def bundle_v(a, b):
    from aether.hd import bundle
    return bundle([a, b], weights=[0.8, 0.2])


# --------------------------------------------------------------------------- #
# 5. Predictive hierarchy
# --------------------------------------------------------------------------- #
def test_hierarchy():
    banner("5. Predictive Hierarchy — 4-level cortex")

    agent = AETHER()
    hierarchy = agent.hierarchy

    inputs = [
        "Paris is the capital of France",
        "Paris is the capital of France",
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
        "Tokyo is the capital of Japan",
        "Water is a liquid",
    ]
    print(f"  Hierarchy levels: {[l.name for l in hierarchy.levels]}")
    print()
    for i, inp in enumerate(inputs):
        states = hierarchy.process(inp)
        global_surprise = np.mean([s.surprise for s in states])
        print(f"    cycle {i+1}: {inp!r:45s} surprise={global_surprise:.3f}")
    print(f"\n  Mean surprise: {hierarchy.mean_surprise():.3f}")
    print(f"  Trend: {hierarchy.surprise_trend():.3f}  (negative = improving)")
    print(f"  Is surprised: {hierarchy.is_surprised()}")


# --------------------------------------------------------------------------- #
# 6. Neuromodulators
# --------------------------------------------------------------------------- #
def test_neuromodulators():
    banner("6. Neuromodulator System — DA, 5HT, ACh, NE")

    agent = AETHER()
    nm = agent.neuromodulators

    print(f"  Initial: {nm.levels.as_dict()}")
    print(f"  Mood: {nm.mood()}")
    print(f"  Learning rate: {nm.learning_rate():.3f}")
    print(f"  Patience: {nm.patience()}")
    print(f"  Exploration: {nm.exploration_rate():.3f}")
    print()

    events = [
        (1.0, 0.0, True,   "Success!"),
        (0.0, 0.5, None,   "Mild surprise"),
        (0.0, 0.0, None,   "Idle"),
        (0.0, 0.0, None,   "Idle"),
        (-0.5, 0.8, False, "Failure!"),
        (2.0, 0.0, True,   "Big success!"),
    ]
    for reward, surprise, success, desc in events:
        nm.update(reward=reward, surprise=surprise, success=success)
        print(f"    [{desc:20s}] DA={nm.levels.dopamine:.2f} 5HT={nm.levels.serotonin:.2f} "
              f"ACh={nm.levels.acetylcholine:.2f} NE={nm.levels.norepinephrine:.2f} mood={nm.mood()}")


# --------------------------------------------------------------------------- #
# 7. Comprehension
# --------------------------------------------------------------------------- #
def test_comprehension():
    banner("7. Comprehension Integrator — multi-indicator score")

    agent = AETHER()
    print(f"  Initial comprehension: {agent.comprehension.stats()}")

    # Ask a few questions to drive the system
    questions = [
        "Hello",
        "What is the capital of France?",
        "teach Lisbon is the capital of Portugal",
        "What is the capital of Portugal?",
        "What is the capital of the country where Osaka is located?",
    ]
    for q in questions:
        agent.ask(q)
        comp = agent.last_comprehension
        print(f"\n  Q: {q}")
        print(f"    attractor_stability: {comp.attractor_stability:.3f}")
        print(f"    prediction_match:    {comp.prediction_match:.3f}")
        print(f"    broadcast_active:    {comp.broadcast_active:.3f}")
        print(f"    oscillator_sync:     {comp.oscillator_sync:.3f}")
        print(f"    hierarchy_calm:      {comp.hierarchy_calm:.3f}")
        print(f"    nm_balance:          {comp.neuromodulator_balance:.3f}")
        print(f"    SCORE: {comp.comprehension_score:.3f}  comprehending={comp.is_comprehending}")
        print(f"    notes: {comp.notes}")


# --------------------------------------------------------------------------- #
# 8. Consciousness
# --------------------------------------------------------------------------- #
def test_consciousness():
    banner("8. Consciousness Module — self-model + metacognition + narrative")

    agent = AETHER()
    print(f"  Identity: AETHER")
    print(f"  Self-awareness: {agent.consciousness.self_model.self_awareness_score():.3f}")

    # Run some cycles
    questions = [
        "Hello",
        "What are you?",
        "What is the capital of France?",
        "calc 2+2",
        "teach Reykjavik is the capital of Iceland",
        "What is the capital of Iceland?",
    ]
    for q in questions:
        agent.ask(q)

    # Introspect
    intro = agent.introspect()
    print(f"\n  After {intro['cycle']} cycles:")
    print(f"    self_awareness: {intro['self_awareness']:.3f}")
    print(f"    current_mood: {intro['current_mood']}")
    print(f"    comprehension: {intro['comprehension']:.3f}")
    print(f"    confidence: {intro['confidence']:.3f}")
    print(f"    is_confused: {intro['is_confused']}")
    print(f"    is_confident: {intro['is_confident']}")
    print(f"    is_stuck: {intro['is_stuck']}")
    print(f"\n  Recent narrative:")
    for line in intro["narrative_summary"]:
        print(f"    {line}")


# --------------------------------------------------------------------------- #
# 9. End-to-end with full brain trace
# --------------------------------------------------------------------------- #
def test_end_to_end():
    banner("9. End-to-End with Full Brain Trace")

    agent = AETHER()
    agent.verbose = True

    print("\n  Asking 'What is the capital of France?' with verbose trace:")
    ans = agent.ask("What is the capital of France?", explain=True)
    print(f"  Final answer: {ans}")

    print("\n  Final stats:")
    s = agent.stats()
    for k in ["version", "vocab_size", "comprehension_score", "mood",
              "metacognitive_action", "neuromodulators", "kuramoto_concepts",
              "attractor_memories", "consciousness_cycle"]:
        print(f"    {k}: {s.get(k)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_kuramoto,
        test_attractors,
        test_global_workspace,
        test_predictive,
        test_hierarchy,
        test_neuromodulators,
        test_comprehension,
        test_consciousness,
        test_end_to_end,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n  [FAILED: {t.__name__}: {e}]")
            traceback.print_exc()
    print("\n" + "=" * 76)
    print("  All v4 tests complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
