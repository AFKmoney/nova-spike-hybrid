"""
AETHER v4 — Adaptive Emergent Thinking Hyperdimensional Engine for Reasoning
============================================================================

A non-transformer, GPU-free, instant-learning, brain-inspired cognitive
architecture.

v4 adds brain-inspired cognition on top of v3:
  - Kuramoto oscillator network (cognitive binding via synchrony)
  - Attractor networks (discrete + continuous, stable thoughts)
  - Global Workspace Theory (Baars — conscious broadcast + ignition)
  - Predictive coding (Friston — free energy minimization)
  - Hierarchical predictive cortex (4 levels: sensory/feature/concept/abstract)
  - Neuromodulator system (dopamine, serotonin, ACh, norepinephrine)
  - Comprehension integrator (multi-indicator "real understanding")
  - Consciousness module (self-model, metacognition, attention, narrative)

No transformers. No external LLM. No GPU. Pure numpy on CPU.

v4.0.0 — brain-inspired edition
"""

# v1-v3 imports
from .hd import HDVector, DIM
from .memory import SparseDistributedMemory, AssociativeMemory
from .encoder import TextEncoder
from .semantic import SemanticEncoder, char_ngrams, tag_token, tag_encode
from .reasoning import CognitiveLoop
from .tools import ToolRegistry, default_tools
from .inference import InferenceEngine, Proof, ProofStep
from .planner import Planner, PlanExecutor, Plan, PlanStep
from .context import ConversationContext, Turn
from .generator import ResponseGenerator, analyze_question, QuestionAnalysis, parse_triple

# v3 modules
from .pretrained import SemanticKB, CONCEPT_TAXONOMY, SYNONYMS, ANTONYMS, integrate_into_agent
from .language_model import HDLanguageModel, encode_ngram_context
from .composer import ToolComposer, ComposedExecutor, ToolNode
from .web import HDRAGStore, WebFetcher, WebDoc, OFFLINE_WIKI, get_global_rag, reset_global_rag
from .multimodal import ImageHDEncoder, AudioHDEncoder, CrossModalSpace

# v4 brain-inspired modules
from .kuramoto import KuramotoNetwork, KuramotoState, hd_to_phase, hd_to_frequency
from .attractor import (DiscreteAttractorNetwork, RingAttractor, SheetAttractor,
                         PatternCompleter, AttractorState)
from .global_workspace import (GlobalWorkspace, Specialist, SpecialistOutput,
                                WorkspaceState, make_language_specialist,
                                make_memory_specialist, make_tool_specialist,
                                make_inference_specialist)
from .predictive import PredictiveModel, SequencePredictor, PredictionError, compute_prediction_error
from .hierarchy import PredictiveHierarchy, HierarchyLevel, LevelState
from .neuromodulators import NeuromodulatorSystem, NeuromodulatorLevels, RewardSignal
from .comprehension import ComprehensionIntegrator, ComprehensionState
from .consciousness import (ConsciousnessModule, SelfModel, Metacognition,
                             AttentionDirector, ForwardModel, NarrativeBuffer,
                             MetacognitiveReading, NarrativeEntry)

from .agent import AETHER

__version__ = "4.0.0"
__all__ = [
    # v1
    "HDVector", "DIM",
    "SparseDistributedMemory", "AssociativeMemory",
    "TextEncoder", "CognitiveLoop",
    "ToolRegistry", "default_tools",
    "AETHER",
    # v2
    "SemanticEncoder", "char_ngrams", "tag_token", "tag_encode",
    "InferenceEngine", "Proof", "ProofStep",
    "Planner", "PlanExecutor", "Plan", "PlanStep",
    "ConversationContext", "Turn",
    "ResponseGenerator", "analyze_question", "QuestionAnalysis", "parse_triple",
    # v3
    "SemanticKB", "CONCEPT_TAXONOMY", "SYNONYMS", "ANTONYMS", "integrate_into_agent",
    "HDLanguageModel", "encode_ngram_context",
    "ToolComposer", "ComposedExecutor", "ToolNode",
    "HDRAGStore", "WebFetcher", "WebDoc", "OFFLINE_WIKI",
    "get_global_rag", "reset_global_rag",
    "ImageHDEncoder", "AudioHDEncoder", "CrossModalSpace",
    # v4 brain-inspired
    "KuramotoNetwork", "KuramotoState", "hd_to_phase", "hd_to_frequency",
    "DiscreteAttractorNetwork", "RingAttractor", "SheetAttractor",
    "PatternCompleter", "AttractorState",
    "GlobalWorkspace", "Specialist", "SpecialistOutput", "WorkspaceState",
    "make_language_specialist", "make_memory_specialist",
    "make_tool_specialist", "make_inference_specialist",
    "PredictiveModel", "SequencePredictor", "PredictionError", "compute_prediction_error",
    "PredictiveHierarchy", "HierarchyLevel", "LevelState",
    "NeuromodulatorSystem", "NeuromodulatorLevels", "RewardSignal",
    "ComprehensionIntegrator", "ComprehensionState",
    "ConsciousnessModule", "SelfModel", "Metacognition",
    "AttentionDirector", "ForwardModel", "NarrativeBuffer",
    "MetacognitiveReading", "NarrativeEntry",
]


