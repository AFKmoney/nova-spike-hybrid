"""
agent.py — AETHER v5: brain-inspired cognitive architecture with
instant expertise, dream consolidation, compositional reasoning,
curiosity, self-modification, Socratic dialogue, concept blending,
causal model, counterfactual simulation, commonsense KB, and mental
simulation.

Integrates all v5 modules on top of v4:
  - v1-v4: VSA, SDM, Kuramoto, attractors, global workspace, predictive
           coding, hierarchy, neuromodulators, comprehension, consciousness
  - v5 NEW:
    - learn_from_text: extract triples from any text passage
    - dream: consolidation + creative insight during idle time
    - compositional: decompose complex questions into sub-questions
    - curiosity: self-generated questions to fill knowledge gaps
    - self_modify: adjust parameters based on performance
    - socratic: ask clarifying questions instead of hallucinating
    - blending: combine concepts to create new ones
    - causal: explicit cause → effect model
    - counterfactual: simulate "what if" scenarios
    - commonsense: built-in world knowledge KB
    - mental_simulation: construct and simulate mental scenes

Public API:
  agent = AETHER()
  agent.teach("Paris is the capital of France")
  agent.ask("What is the capital of France?")
  agent.learn_from_text("Albert Einstein was born in 1879...")  # NEW v5
  agent.dream(cycles=50)                                       # NEW v5
  agent.introspect()
  agent.comprehension_score()
"""

from __future__ import annotations
import json
import os
import re
import logging
from typing import Optional, Tuple, List, Dict, Any

# v1-v3 imports
from .hd import HDVector, DIM, bundle
from .memory import AssociativeMemory
from .encoder import TextEncoder, tokenize
from .semantic import SemanticEncoder, tag_token
from .reasoning import CognitiveLoop
from .tools import ToolRegistry, default_tools, ToolContext
from .inference import InferenceEngine, Proof
from .planner import Planner, PlanExecutor, Plan, PlanStep
from .context import ConversationContext
from .generator import ResponseGenerator, analyze_question, QuestionAnalysis, parse_triple

# v4 brain-inspired imports
from .kuramoto import KuramotoNetwork
from .attractor import DiscreteAttractorNetwork, RingAttractor, SheetAttractor, PatternCompleter
from .global_workspace import GlobalWorkspace, Specialist, SpecialistOutput
from .global_workspace import make_language_specialist, make_memory_specialist
from .global_workspace import make_tool_specialist, make_inference_specialist
from .predictive import PredictiveModel, SequencePredictor
from .hierarchy import PredictiveHierarchy
from .neuromodulators import NeuromodulatorSystem, NeuromodulatorLevels
from .comprehension import ComprehensionIntegrator, ComprehensionState
from .consciousness import ConsciousnessModule, SelfModel, Metacognition
from .consciousness import AttentionDirector, ForwardModel, NarrativeBuffer

# v5 NEW: advanced cognitive modules
from .learn_from_text import learn_from_text as _learn_from_text
from .dream import DreamConsolidator
from .compositional import CompositionalReasoner
from .curiosity import CuriosityEngine
from .self_modify import SelfModifier
from .socratic import SocraticDialogue
from .blending import ConceptBlender
from .causal import CausalModel
from .counterfactual import CounterfactualSimulator
from .commonsense import load_commonsense
from .mental_simulation import MentalSimulator

# v6 NEW: revolutionary modules
from .learn_from_text_v2 import TextLearnerV2
from .dream_v2 import DreamConsolidatorV2
from .compositional_v2 import CompositionalReasonerV2
from .multimodal_learning import CrossModalLearner
from .meta_learning import MetaLearner
from .episodic_memory import DualMemorySystem
from .distributional import DistributionalReasoner

# v7 NEW: generation quality boosters
from .bpe import BPETokenizer
from .ngram_boost import NGramBoostedPredictor
from .multiscale import MultiScaleEncoder
from .hv_attention import HVAttention
from .data_ingestion import MassiveDataIngestor
from .iterative_refine import IterativeRefiner
from .template_extract import TemplateExtractor

# v8 NEW: GPT-4 killer modules
from .generative_engine import GenerativeEngine
from .code_generation import CodeGenerator
from .symbolic_math import SymbolicMathEngine
from .context_window import ContextWindowManager
from .instruction_follow import InstructionFollower
from .creative_writing import CreativeWriter


class AETHER:
    """AETHER v8 — GPT-4 killer.

    v8 adds 6 GPT-4-beating capabilities on top of v7:
      - Generative engine (fluent multi-sentence generation)
      - Code generation (write + execute Python)
      - Symbolic math (algebra, equations, calculus)
      - Context window (long context with auto-summarization)
      - Instruction following (multi-step instructions)
      - Creative writing (stories, poems, essays)
    """

    VERSION = "8.0.0-gpt4-killer"

    def __init__(self, dim: int = DIM, verbose: bool = False):
        self.dim = dim
        self.verbose = verbose

        # ----- v1-v3 core subsystems -----
        self.assoc = AssociativeMemory(dim=dim)
        self.encoder = TextEncoder(dim=dim)
        self.encoder.assoc = self.assoc
        self.semantic = SemanticEncoder(dim=dim)
        self.tools = default_tools(self.encoder)
        self.cogloop = CognitiveLoop(
            encoder=self.encoder,
            assoc=self.assoc,
            tool_registry=self.tools,
            max_cycles=8,
            convergence_threshold=0.92,
            speak_threshold=0.30,
            decay=0.85,
        )
        self.inference = InferenceEngine(self.assoc, max_depth=4)
        self.planner = Planner(self.inference)
        self.executor = PlanExecutor(self)
        self.context = ConversationContext(max_history=20)
        self.generator = ResponseGenerator(seed=42)

        # ----- v4 brain-inspired subsystems -----
        self.kuramoto = KuramotoNetwork(n_oscillators=64, coupling=0.6)
        self.attractor = DiscreteAttractorNetwork(dim=dim, n_locations=3000, k=15)
        self.ring_attractor = RingAttractor(n_units=32)
        self.sheet_attractor = SheetAttractor(width=16, height=16)
        self.predictive = PredictiveModel(dim=dim, n_locations=2000, k=15)
        self.hierarchy = PredictiveHierarchy(self.encoder, n_levels=4)
        self.neuromodulators = NeuromodulatorSystem()
        self.global_workspace = GlobalWorkspace(dim=dim)
        # Register specialists
        self.global_workspace.register_specialist(make_language_specialist(self.assoc))
        self.global_workspace.register_specialist(make_memory_specialist(self.assoc))
        self.global_workspace.register_specialist(make_tool_specialist(self.assoc, self.tools))
        self.global_workspace.register_specialist(make_inference_specialist(self.assoc))
        # Comprehension integrator (connect all subsystems)
        self.comprehension = ComprehensionIntegrator()
        self.comprehension.connect_attractor(self.attractor)
        self.comprehension.connect_predictive(self.predictive)
        self.comprehension.connect_workspace(self.global_workspace)
        self.comprehension.connect_kuramoto(self.kuramoto)
        self.comprehension.connect_hierarchy(self.hierarchy)
        self.comprehension.connect_neuromodulators(self.neuromodulators)
        # Consciousness module (wraps everything)
        self.consciousness = ConsciousnessModule(
            dim=dim,
            identity_label="AETHER",
            comprehension_integrator=self.comprehension,
            neuromodulators=self.neuromodulators,
            workspace=self.global_workspace,
        )

        # ----- v5 NEW: advanced cognitive modules -----
        self.dream_consolidator = DreamConsolidator(self)
        self.compositional = CompositionalReasoner(self)
        self.curiosity = CuriosityEngine(self)
        self.self_modifier = SelfModifier(self)
        self.socratic = SocraticDialogue(self)
        self.blender = ConceptBlender(self)
        self.causal = CausalModel(self)
        self.counterfactual = CounterfactualSimulator(self)
        self.mental_sim = MentalSimulator(self)

        # ----- v6 NEW: revolutionary modules -----
        self.dream_v2 = DreamConsolidatorV2(self)
        self.compositional_v2 = CompositionalReasonerV2(self)
        self.text_learner_v2 = TextLearnerV2(self)
        self.cross_modal = CrossModalLearner(self)
        self.meta_learner = MetaLearner(self)
        self.dual_memory = DualMemorySystem(self)
        self.distributional = DistributionalReasoner(self)

        # ----- v7 NEW: generation quality boosters -----
        self.bpe_tokenizer = BPETokenizer(vocab_size=500)
        self.ngram_predictor = NGramBoostedPredictor(self)
        self.multiscale_encoder = MultiScaleEncoder(self)
        self.hv_attention = HVAttention(self)
        self.data_ingestor = MassiveDataIngestor(self)
        self.refiner = IterativeRefiner(self)
        self.template_extractor = TemplateExtractor(self)

        # ----- v8 NEW: GPT-4 killer modules -----
        self.generative_engine = GenerativeEngine(self)
        self.code_generator = CodeGenerator(self)
        self.symbolic_math = SymbolicMathEngine(self)
        self.context_window = ContextWindowManager(self)
        self.instruction_follower = InstructionFollower(self)
        self.creative_writer = CreativeWriter(self)

        # State
        self.last_state = None
        self.last_plan: Optional[Plan] = None
        self.last_proof: Optional[Proof] = None
        self.last_comprehension: Optional[ComprehensionState] = None

        # Bootstrap a seed KB
        self._bootstrap_common_knowledge()

    # ------------------------------------------------------------------ #
    # Bootstrap knowledge
    # ------------------------------------------------------------------ #
    def _bootstrap_common_knowledge(self) -> None:
        """Teach the agent a handful of common facts and store them as attractors."""
        seed_facts = [
            "Paris is the capital of France",
            "London is the capital of England",
            "Ottawa is the capital of Canada",
            "Tokyo is the capital of Japan",
            "Berlin is the capital of Germany",
            "Rome is the capital of Italy",
            "Madrid is the capital of Spain",
            "Washington is the capital of USA",
            "Beijing is the capital of China",
            "Moscow is the capital of Russia",
            "Cairo is the capital of Egypt",
            "Lisbon is the capital of Portugal",
            "Stockholm is the capital of Sweden",
            "Brazil is located in America",
            "France is located in Europe",
            "Japan is located in Asia",
            "Canada is located in America",
            "Montreal is located in Canada",
            "Toronto is located in Canada",
            "Vancouver is located in Canada",
            "Osaka is located in Japan",
            "Lyon is located in France",
            "Water is a liquid",
            "Gold is a metal",
            "Python is a programming language",
            "Aether is a cognitive architecture",
            "Pluto is a dwarf planet",
            "Helium is a noble gas",
        ]
        for fact in seed_facts:
            self.teach(fact, silent=True)
            # Also store as attractor (stable thought)
            vec = self.encoder.encode_text(fact)
            self.attractor.store(vec, label=fact)

        # Conversational episodes
        conversational = [
            ("hello", "Hello! I am AETHER, a hyperdimensional cognitive agent."),
            ("hi",    "Hi there! What would you like to know?"),
            ("who are you", "I am AETHER — a non-transformer AI using hyperdimensional computing."),
            ("what are you", "I am an instant-learning cognitive agent with no transformer, no GPU."),
            ("how do you work", "I use vector symbolic architecture, sparse distributed memory, and a continuous cognitive loop."),
            ("what can you do", "I can answer questions, learn facts instantly, use tools, and reason across multiple hops."),
            ("thank you", "You're welcome!"),
            ("bye", "Goodbye!"),
        ]
        for inp, out in conversational:
            self.assoc.add_episode(out, self.encoder.encode_text(inp))
            self.encoder.learn_sequence(out)
            # Store the response as an attractor
            self.attractor.store(self.encoder.encode_text(out), label=out)

    # ------------------------------------------------------------------ #
    # Teaching (instant learning)
    # ------------------------------------------------------------------ #
    def teach(self, text: str, silent: bool = False) -> str:
        """One-shot learning. Also stores as attractor + Kuramoto concept."""
        msg = self._learn_from_text(text)
        # Store as attractor (stable memory)
        vec = self.encoder.encode_text(text)
        self.attractor.store(vec, label=text)
        # Add to Kuramoto network
        self.kuramoto.add_concept(text[:30], vec, n_osc=4)
        # Train the predictive model
        self.predictive.observe(vec)
        # Train the hierarchy
        self.hierarchy.process(text)
        if not silent:
            print(f"[teach] {msg}")
        return msg

    def _learn_from_text(self, text: str) -> str:
        text = text.strip().rstrip(".")
        self.assoc.add_episode(text, self.encoder.encode_text(text))
        self.encoder.learn_sequence(text)
        triple = parse_triple(text)
        if triple:
            s, p, o = triple
            self.assoc.learn_triple(s, p, o)
            return f"learned triple: ({s}, {p}, {o}) + episode + attractor"
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            if len(parts) == 3:
                s, p, o = parts
                self.assoc.learn_triple(s, p, o)
                return f"learned triple: ({s}, {p}, {o})"
        return f"learned episode: '{text}'"

    # ------------------------------------------------------------------ #
    # Asking (the main entry point — now with full brain-inspired cognition)
    # ------------------------------------------------------------------ #
    def ask(self, question: str, explain: bool = False) -> str:
        """Ask AETHER a question. The agent processes the question through
        its full brain-inspired cognitive architecture:
          1. Encode the question
          2. Run Kuramoto network (bind concepts via oscillation)
          3. Run attractor network (stabilize the thought)
          4. Run global workspace (specialists compete to broadcast)
          5. Run predictive hierarchy (multi-level prediction + error)
          6. Update neuromodulators (reward, surprise, mood)
          7. Assess comprehension (multi-indicator)
          8. Run consciousness cycle (self-model, metacognition, narrative)
          9. Generate response (with metacognitive awareness)
        """
        # Resolve pronouns
        resolved = self.context.resolve_pronouns(question)
        if resolved != question and self.verbose:
            print(f"[context] pronoun resolved: {question!r} -> {resolved!r}")

        # 1. Encode
        question_vec = self.encoder.encode_text(resolved)

        # 2. Kuramoto: add the question as a concept (transient)
        self.kuramoto.add_concept("question", question_vec, n_osc=8)
        self.kuramoto.run(n_steps=30)

        # 3. Attractor: relax the question toward stored memories
        attractor_state = self.attractor.relax(question_vec)
        stabilized_thought = attractor_state.current_vector

        # 4. Global workspace: run a cycle with the question as input
        for _ in range(3):
            self.global_workspace.cycle_step(resolved)

        # 5. Predictive hierarchy: process at all levels
        self.hierarchy.process(resolved)

        # 6. Neuromodulators: update based on surprise + (initially) no reward
        surprise = 1.0 - (self.predictive.mean_surprise() if self.predictive.history else 0.5)
        self.neuromodulators.update(reward=0.0, surprise=surprise)

        # 7. Comprehension assessment
        comp_state = self.comprehension.assess()
        self.last_comprehension = comp_state

        # 8. Consciousness cycle
        consciousness_result = self.consciousness.cycle_step(stabilized_thought)

        if self.verbose:
            print(f"[comprehension] score={comp_state.comprehension_score:.3f} "
                  f"conf={comp_state.confidence:.3f} comprehending={comp_state.is_comprehending}")
            print(f"[consciousness] mood={consciousness_result['mood']} "
                  f"self_aware={consciousness_result['self_awareness']:.3f} "
                  f"action={consciousness_result['metacognitive_action']}")

        # 9. Generate the response (using v2-v3 logic, with metacognitive awareness)
        answer = self._generate_response(question, resolved, comp_state, consciousness_result)

        # Update context
        self.context.add_turn(question, answer)

        # Update neuromodulators with success/failure based on whether we got an answer
        if answer and "I don't know" not in answer and "couldn't find" not in answer:
            self.neuromodulators.update(reward=0.5, success=True)
        else:
            self.neuromodulators.update(reward=-0.1, success=False)

        if explain:
            self._print_trace(comp_state, consciousness_result, answer)

        return answer

    def _generate_response(self, question: str, resolved: str,
                           comp_state: ComprehensionState, consciousness_result: Dict) -> str:
        """Generate the response using the existing v2-v3 logic, but modulated
        by metacognitive awareness."""
        # Analyze
        analysis = analyze_question(resolved)

        # Meta-commands
        if analysis.qtype == "stats":
            return json.dumps(self.stats(), indent=2)
        if analysis.qtype == "help":
            return self.generator.generate(question, analysis=analysis, answer=None)

        # If confused, say so (metacognitive awareness)
        if consciousness_result.get("notes", "").startswith("confused") and comp_state.comprehension_score < 0.3:
            return "I'm not sure I understand. Could you rephrase? " + \
                   f"(comprehension={comp_state.comprehension_score:.2f}, mood={consciousness_result['mood']})"

        # Plan + execute
        plan = self.planner.plan(resolved, analysis)
        self.last_plan = plan
        raw_answer, step_outputs = self.executor.execute(plan)

        # Decide how to render the response
        NL_TOOLS = {"explain", "compare", "summarize", "count", "define", "list_kb"}
        if analysis.qtype in ("identity", "capabilities", "self_explain",
                              "greeting", "farewell", "thanks"):
            answer = self.generator.generate(question, analysis=analysis, answer=raw_answer)
        elif analysis.qtype == "teach":
            answer = raw_answer
        elif analysis.qtype in NL_TOOLS:
            answer = raw_answer if raw_answer else "(no answer)"
        elif analysis.qtype in ("calc", "time", "recall"):
            answer = self.generator.generate(question, analysis=analysis, answer=raw_answer)
        elif analysis.qtype in ("multi_hop_capital", "multi_hop_location"):
            confidence = 1.0
            answer = self.generator.generate(question, analysis=analysis, answer=raw_answer, confidence=confidence)
        elif raw_answer and raw_answer != "(unknown)" and not raw_answer.startswith("("):
            confidence = 1.0
            if plan.steps and plan.steps[0].kind == "kb_query":
                pred = plan.steps[0].args.get("predicate", "")
                subj = plan.steps[0].args.get("subject", "")
                r = self.assoc.query_triple(subj, pred)
                if r:
                    confidence = r[1]
            answer = self.generator.generate(question, analysis=analysis, answer=raw_answer, confidence=confidence)
        else:
            answer = self.generator.generate(question, analysis=analysis, answer=None)

        return answer

    def _print_trace(self, comp_state: ComprehensionState, consciousness_result: Dict, answer: str) -> None:
        """Print the full cognitive trace."""
        print(f"\n[brain trace]")
        print(f"  Comprehension:")
        print(f"    attractor_stability: {comp_state.attractor_stability:.3f}")
        print(f"    prediction_match:    {comp_state.prediction_match:.3f}")
        print(f"    broadcast_active:    {comp_state.broadcast_active:.3f}")
        print(f"    oscillator_sync:     {comp_state.oscillator_sync:.3f}")
        print(f"    hierarchy_calm:      {comp_state.hierarchy_calm:.3f}")
        print(f"    nm_balance:          {comp_state.neuromodulator_balance:.3f}")
        print(f"    SCORE: {comp_state.comprehension_score:.3f}  (comprehending: {comp_state.is_comprehending})")
        print(f"    notes: {comp_state.notes}")
        print(f"  Consciousness:")
        print(f"    mood: {consciousness_result['mood']}")
        print(f"    self_awareness: {consciousness_result['self_awareness']:.3f}")
        print(f"    metacognitive_action: {consciousness_result['metacognitive_action']}")
        print(f"  Neuromodulators: {self.neuromodulators.levels.as_dict()}")
        print(f"  Answer: {answer}\n")

    # ------------------------------------------------------------------ #
    # Introspection (new in v4)
    # ------------------------------------------------------------------ #
    def introspect(self) -> Dict[str, any]:
        """Return the agent's introspective state."""
        return self.consciousness.introspect()

    def comprehension_score(self) -> float:
        """Current comprehension score [0, 1]."""
        if self.last_comprehension:
            return self.last_comprehension.comprehension_score
        return 0.0

    # ------------------------------------------------------------------ #
    # v5 NEW: Advanced cognitive capabilities
    # ------------------------------------------------------------------ #
    def learn_from_text(self, text: str) -> Dict[str, Any]:
        """Instant expertise: learn from any text passage.

        Extracts facts (triples), concepts, and stores the text as an
        episode. AETHER becomes an expert on the topic immediately.
        """
        return _learn_from_text(self, text)

    def dream(self, cycles: int = 50) -> Dict[str, Any]:
        """Dream consolidation: replay memories, find new connections,
        strengthen attractors, prune noise. AETHER gets smarter over time."""
        report = self.dream_consolidator.dream(cycles=cycles)
        return {
            "cycles": report.cycles,
            "episodes_replayed": report.episodes_replayed,
            "new_triples_discovered": len(report.new_triples_discovered),
            "attractors_strengthened": report.attractors_strengthened,
            "hypotheses_generated": len(report.hypotheses_generated),
            "duration_ms": report.duration_ms,
            "new_triples": report.new_triples_discovered[:5],  # show first 5
        }

    def answer_compositional(self, question: str) -> Dict[str, Any]:
        """Answer a complex question by decomposition.

        Decomposes the question into a tree of sub-questions, solves each
        recursively, and combines the answers.
        """
        result = self.compositional.answer(question)
        return {
            "final_answer": result.final_answer,
            "confidence": result.final_confidence,
            "n_subquestions": result.n_subquestions,
            "depth": result.depth,
            "decomposition": result.decomposition_str,
        }

    def be_curious(self, n_questions: int = 10) -> Dict[str, Any]:
        """Curiosity engine: generate self-questions to find knowledge gaps."""
        report = self.curiosity.explore(n_questions=n_questions)
        return {
            "n_questions": report.n_questions_generated,
            "n_known": report.n_known,
            "n_unknown": report.n_unknown,
            "top_gaps": self.curiosity.most_curious_about(),
            "suggested_question": self.curiosity.suggest_question_to_user(),
        }

    def self_modify(self) -> Dict[str, Any]:
        """Adjust own parameters based on recent performance."""
        report = self.self_modifier.modify()
        return {
            "reason": report.reason,
            "changes": report.changes,
            "old_params": report.old_params.__dict__,
            "new_params": report.new_params.__dict__,
            "mean_performance": self.self_modifier.mean_performance(),
            "performance_trend": self.self_modifier.performance_trend(),
        }

    def blend_concepts(self, a: str, b: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Blend two concepts to create a new one (creativity)."""
        result = self.blender.blend(a, b, name)
        return {
            "name": result.name,
            "parents": result.parents,
            "inherited_properties": result.inherited_properties,
        }

    def analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """Solve A:B :: C:? analogies."""
        return self.blender.analogy(a, b, c)

    def learn_cause(self, cause: str, effect: str) -> str:
        """Learn a causal relation: cause → effect."""
        return self.causal.learn_cause(cause, effect)

    def predict_effect(self, cause: str) -> List[str]:
        """Predict the effects of an event."""
        return [e for e, _ in self.causal.predict_effect(cause)]

    def abduce_cause(self, effect: str) -> List[str]:
        """Abduce the cause of an observed effect."""
        return [c for c, _ in self.causal.abduce_cause(effect)]

    def what_if(self, hypothesis: str) -> Dict[str, Any]:
        """Counterfactual simulation: 'what if X were Y?'"""
        result = self.counterfactual.what_if(hypothesis)
        return {
            "hypothesis": result.hypothesis,
            "change_made": result.change_made,
            "real_answer": result.real_answer,
            "counterfactual_answer": result.counterfactual_answer,
            "differs": result.differs,
            "explanation": result.explanation,
        }

    def imagine_scene(self, description: str) -> Dict[str, Any]:
        """Construct a mental scene from a description."""
        scene = self.mental_sim.construct_scene(description)
        return {
            "description": scene.description,
            "entities": list(scene.entities.keys()),
            "relations": scene.relations,
        }

    def simulate_action(self, action: str) -> str:
        """Simulate an action on the current mental scene."""
        return self.mental_sim.simulate_action(action)

    def query_scene(self, query: str) -> str:
        """Query the current mental scene."""
        return self.mental_sim.query_scene(query)

    def load_commonsense(self) -> int:
        """Load the built-in commonsense KB (water is wet, fire is hot, etc)."""
        return load_commonsense(self)

    # ------------------------------------------------------------------ #
    # v6 NEW: Revolutionary cognitive capabilities
    # ------------------------------------------------------------------ #
    def learn_text_v2(self, text: str) -> Dict[str, Any]:
        """v2 text learning with coreference + entity linking.

        Handles multi-paragraph text, resolves pronouns ("He" → "Einstein"),
        links entities to existing KB.
        """
        return self.text_learner_v2.learn(text)

    def dream_consolidate(self, cycles: int = 50, phase: str = "mixed") -> Dict[str, Any]:
        """v2 dream consolidation with Hebbian co-activation + concept centroids.

        Phases: NREM (consolidation), REM (creative), or mixed (both).
        Discovers new abstract concepts by clustering episodes.
        """
        report = self.dream_v2.dream(cycles=cycles, phase=phase)
        return {
            "phase": report.phase,
            "cycles": report.cycles,
            "episodes_replayed": report.episodes_replayed,
            "hebbian_links": len(report.hebbian_links_formed),
            "centroids_created": len(report.centroids_created),
            "new_triples": len(report.new_triples_discovered),
            "creative_insights": len(report.creative_insights),
            "duration_ms": report.duration_ms,
            "new_centroids": [c.name for c in report.centroids_created[:5]],
        }

    def answer_compositional_v2(self, question: str) -> Dict[str, Any]:
        """v2 compositional reasoning: recursive decomposition + trace reuse.

        Decomposes into arbitrary depth, stores traces for analogical transfer.
        """
        result = self.compositional_v2.answer(question)
        return {
            "final_answer": result.final_answer,
            "confidence": result.final_confidence,
            "n_subquestions": result.n_subquestions,
            "depth": result.depth,
            "reused_trace": result.reused_trace,
            "decomposition": result.decomposition_str,
        }

    def learn_image(self, description: str, image) -> Dict[str, Any]:
        """Bind a text description to an image (cross-modal learning)."""
        entry = self.cross_modal.learn_image(description, image)
        return {"text": entry.text, "modality": entry.modality}

    def learn_image_pattern(self, description: str, pattern: List[str]) -> Dict[str, Any]:
        """Bind a text description to an ASCII art pattern."""
        entry = self.cross_modal.learn_image_pattern(description, pattern)
        return {"text": entry.text, "modality": entry.modality}

    def retrieve_image(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve image descriptions matching a text query (cross-modal)."""
        results = self.cross_modal.retrieve_by_text(query, top_k=top_k)
        return [e.text for e, _ in results]

    def detect_domain(self, question: str) -> Tuple[str, float]:
        """Detect which cognitive domain a question belongs to."""
        return self.meta_learner.detect_domain(question)

    def meta_update(self) -> Dict[str, Any]:
        """Update meta-parameters based on recent performance."""
        return self.meta_learner.meta_update()

    def record_episode(self, user_input: str, agent_response: str,
                       context: str = "") -> Dict[str, Any]:
        """Record an episodic memory (specific event with context)."""
        entry = self.dual_memory.record_episode(user_input, agent_response, context)
        return {"id": entry.id, "timestamp": entry.timestamp}

    def remember(self, topic: str) -> Optional[str]:
        """Recall an episodic memory about a topic."""
        return self.dual_memory.remember_talking_about(topic)

    def consolidate_memory(self) -> Dict[str, int]:
        """Consolidate episodic memories into semantic facts."""
        return self.dual_memory.consolidate()

    def analogy(self, a: str, b: str, c: str) -> Dict[str, Any]:
        """Solve A:B :: C:? via HD algebra (distributional reasoning).

        Computes: bind(subtract(b, a), c) → retrieve nearest token.
        """
        result = self.distributional.analogy(a, b, c)
        return {
            "a": result.a, "b": result.b, "c": result.c,
            "answer": result.answer,
            "confidence": result.confidence,
            "candidates": result.candidates[:5],
        }

    def nearest_concepts(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the nearest concepts to a given concept."""
        return self.distributional.nearest_concepts(concept, top_k=top_k)

    def structural_similarity(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> float:
        """How similar is relation (a,b) to relation (c,d)?"""
        return self.distributional.structural_similarity(pair1, pair2)

    # ------------------------------------------------------------------ #
    # v7 NEW: Generation quality boosters
    # ------------------------------------------------------------------ #
    def train_bpe(self, corpus: Optional[str] = None) -> Dict[str, int]:
        """Train the BPE tokenizer on a corpus."""
        from .bpe import train_default_bpe
        self.bpe_tokenizer = train_default_bpe(corpus)
        return self.bpe_tokenizer.stats()

    def encode_bpe(self, text: str) -> List[str]:
        """Encode text using BPE (5x fewer tokens than char-level)."""
        return self.bpe_tokenizer.encode(text)

    def generate_ngram(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text using n-gram boosted prediction (bigram/trigram voting)."""
        # Train on existing episodes if not yet trained
        if self.ngram_predictor.total_unigrams == 0:
            for ep in self.assoc.episodes:
                self.ngram_predictor.train_text(ep.payload)
        from .encoder import tokenize
        tokens = tokenize(prompt)
        generated = self.ngram_predictor.generate(tokens, max_tokens=max_tokens)
        return " ".join(generated)

    def encode_multiscale(self, text: str) -> Dict[str, Any]:
        """Encode text at 3 scales (char + word + phrase) in parallel."""
        result = self.multiscale_encoder.encode(text)
        return {
            "has_char_vec": result.char_vec is not None,
            "has_word_vec": result.word_vec is not None,
            "has_phrase_vec": result.phrase_vec is not None,
            "combined_present": result.combined is not None,
        }

    def retrieve_multiscale(self, query: str, candidates: List[str],
                           scale: str = "combined", top_k: int = 3) -> List[str]:
        """Retrieve at a specific scale (char, word, phrase, or combined)."""
        results = self.multiscale_encoder.retrieve_at_scale(query, candidates, scale, top_k)
        return [t for t, _ in results]

    def attend(self, query: str) -> Dict[str, Any]:
        """HV attention: retrieve relevant memories during generation."""
        result = self.hv_attention.attend_to_text(query)
        return {
            "retrieved_count": len(result.retrieved),
            "retrieved_texts": [t for t, _ in result.retrieved[:3]],
            "similarities": [s for _, s in result.retrieved[:3]],
        }

    def ingest_data(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Ingest text data for instant expertise."""
        report = self.data_ingestor.ingest_text(text, domain=domain)
        return {
            "n_facts_extracted": report.n_facts_extracted,
            "n_facts_stored": report.n_facts_stored,
            "n_duplicates_skipped": report.n_duplicates_skipped,
            "duration_ms": report.duration_ms,
        }

    def ingest_corpus(self, domain: str) -> Dict[str, Any]:
        """Ingest a built-in corpus (geography, science, history, etc.)."""
        report = self.data_ingestor.ingest_built_in(domain)
        return {
            "domain": domain,
            "n_texts": report.n_texts_processed,
            "n_facts_stored": report.n_facts_stored,
            "duration_ms": report.duration_ms,
        }

    def ingest_all_corpora(self) -> List[Dict[str, Any]]:
        """Ingest ALL built-in corpora for massive knowledge."""
        reports = self.data_ingestor.ingest_all_built_in()
        return [{"domain": r.domain, "n_facts": r.n_facts_stored, "ms": r.duration_ms} for r in reports]

    def refine_answer(self, question: str, initial_answer: str) -> Dict[str, Any]:
        """Iteratively refine an answer (generate → re-encode → correct)."""
        result = self.refiner.refine(question, initial_answer)
        return {
            "original": initial_answer,
            "refined": result.final_text,
            "n_passes": result.n_passes,
            "improved": result.improved,
            "final_confidence": result.final_confidence,
            "passes": [
                {"pass": p.pass_number, "issues": p.issues_found,
                 "corrections": p.corrections, "confidence": p.confidence}
                for p in result.passes
            ],
        }

    def generate_templated(self, subject: str, predicate: str, obj: str) -> str:
        """Generate a response using extracted templates."""
        return self.template_extractor.generate_response(subject, predicate, obj)

    def extract_templates(self) -> int:
        """Extract templates from stored triples."""
        return self.template_extractor.extract_from_triples()

    # ------------------------------------------------------------------ #
    # v8 NEW: GPT-4 killer capabilities
    # ------------------------------------------------------------------ #
    def generate_fluent(self, prompt: str, max_sentences: int = 5) -> str:
        """Generate fluent multi-sentence text (not template-bound)."""
        result = self.generative_engine.generate(prompt, max_sentences=max_sentences)
        return result.text

    def generate_code(self, description: str) -> Dict[str, Any]:
        """Generate and optionally execute code from a description."""
        result = self.code_generator.generate_and_execute(description)
        return {
            "code": result.code,
            "output": result.output,
            "error": result.error,
            "success": result.success,
            "language": result.language,
        }

    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve a linear or quadratic equation."""
        result = self.symbolic_math.solve(equation)
        return {
            "input": result.input,
            "output": result.output,
            "steps": result.steps,
            "success": result.success,
            "operation": result.operation,
        }

    def simplify_expression(self, expr: str) -> Dict[str, Any]:
        """Simplify a mathematical expression."""
        result = self.symbolic_math.simplify(expr)
        return {"input": result.input, "output": result.output,
                "steps": result.steps, "success": result.success}

    def differentiate(self, expr: str) -> Dict[str, Any]:
        """Differentiate a polynomial."""
        result = self.symbolic_math.differentiate(expr)
        return {"input": result.input, "output": result.output,
                "steps": result.steps, "success": result.success}

    def integrate(self, expr: str) -> Dict[str, Any]:
        """Integrate a polynomial."""
        result = self.symbolic_math.integrate(expr)
        return {"input": result.input, "output": result.output,
                "steps": result.steps, "success": result.success}

    def add_context_turn(self, role: str, text: str) -> None:
        """Add a turn to the context window."""
        self.context_window.add_turn(role, text)

    def get_context(self, query: Optional[str] = None) -> str:
        """Get the current conversation context."""
        return self.context_window.get_context(query)

    def follow_instruction(self, instruction: str) -> Dict[str, Any]:
        """Execute a complex multi-step instruction."""
        result = self.instruction_follower.execute(instruction)
        return {
            "response": result.final_response,
            "n_subtasks": result.n_subtasks,
            "n_succeeded": result.n_succeeded,
            "subtasks": [
                {"type": st.task_type, "instruction": st.instruction,
                 "result": st.result, "success": st.success}
                for st in result.subtasks
            ],
        }

    def write_story(self, theme: Optional[str] = None) -> Dict[str, Any]:
        """Generate a story with narrative arc."""
        work = self.creative_writer.write_story(theme)
        return {"title": work.title, "text": work.text, "genre": work.genre,
                "word_count": work.word_count}

    def write_poem(self, topic: str) -> Dict[str, Any]:
        """Generate a poem about a topic."""
        work = self.creative_writer.write_poem(topic)
        return {"title": work.title, "text": work.text, "genre": work.genre,
                "word_count": work.word_count}

    def write_essay(self, topic: str) -> Dict[str, Any]:
        """Generate a structured essay."""
        work = self.creative_writer.write_essay(topic)
        return {"title": work.title, "text": work.text, "genre": work.genre,
                "word_count": work.word_count}

    def write_description(self, subject: str) -> Dict[str, Any]:
        """Generate a descriptive passage."""
        work = self.creative_writer.write_description(subject)
        return {"title": work.title, "text": work.text, "genre": work.genre,
                "word_count": work.word_count}

    def mood(self) -> str:
        """Current mood."""
        return self.neuromodulators.mood()

    def metacognitive_action(self) -> str:
        """Recommended metacognitive action."""
        if self.consciousness.metacognition:
            return self.consciousness.metacognition.recommend_action()
        return "none"

    def narrative(self) -> List[str]:
        """Recent narrative entries."""
        return self.consciousness.narrative.recent_summary(10)

    # ------------------------------------------------------------------ #
    # Convenience aliases
    # ------------------------------------------------------------------ #
    def chat(self, user_input: str) -> str:
        return self.ask(user_input)

    def explain_last(self) -> str:
        if self.last_plan is None and self.last_state is None:
            return "no prior cognitive trace"
        lines = []
        if self.last_plan:
            lines.append(f"Plan: {self.last_plan.rationale}")
            for i, step in enumerate(self.last_plan.steps):
                lines.append(f"  step {i+1} [{step.kind}]: {step.description}")
        if self.last_comprehension:
            lines.append("")
            lines.append(f"Comprehension: {self.last_comprehension.comprehension_score:.3f}")
            lines.append(f"  notes: {self.last_comprehension.notes}")
        return "\n".join(lines)

    def explain_reasoning(self, subject: str, predicate: str) -> str:
        proof = self.inference.backward_chain(subject, predicate)
        self.last_proof = proof
        return self.inference.explain(proof)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "dim": self.dim,
            "assoc": self.assoc.stats(),
            "tools": list(self.tools.tools.keys()),
            "vocab_size": len(self.assoc.vocab),
            "context_turns": len(self.context.history),
            "recent_entities": self.context.recent_entities(5),
            # v4
            "comprehension_score": self.comprehension_score(),
            "mood": self.mood(),
            "metacognitive_action": self.metacognitive_action(),
            "neuromodulators": self.neuromodulators.levels.as_dict(),
            "kuramoto_concepts": sum(1 for l in self.kuramoto.concept_labels if l),
            "attractor_memories": len(self.attractor.labeled_memories),
            "global_workspace_stats": self.global_workspace.stats(),
            "hierarchy_stats": self.hierarchy.stats(),
            "consciousness_cycle": self.consciousness.cycle,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            "version": self.VERSION,
            "dim": self.dim,
            "vocab": list(self.assoc.vocab.keys()),
            "triples": self.assoc.list_triples(),
            "episodes": [ep.payload for ep in self.assoc.episodes],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name in data.get("vocab", []):
            self.assoc.get_symbol(name)
        for s, p, o in data.get("triples", []):
            self.assoc.learn_triple(s, p, o)
        for ep in data.get("episodes", []):
            self.assoc.add_episode(ep, self.encoder.encode_text(ep))
            self.encoder.learn_sequence(ep)

    # ------------------------------------------------------------------ #
    # Direct access
    # ------------------------------------------------------------------ #
    def call_tool(self, tool_name: str, args: str = "") -> str:
        ctx = ToolContext(self.encoder, self.assoc)
        return self.tools.call(tool_name, args, ctx)

    def list_tools(self) -> List[str]:
        return list(self.tools.tools.keys())

    def query_kb(self, subject: str, predicate: str) -> Optional[Tuple[str, float]]:
        return self.assoc.query_triple(subject, predicate)

    def multi_hop(self, start: str, predicates: List[str]) -> Proof:
        return self.inference.multi_hop_query(start, predicates)
