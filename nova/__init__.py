"""
NOVA — Neural Oscillatory Vector Architecture
=============================================

Un paradigme d'IA sans transformer, sans GPU, sans LLM externe.

Piliers:
  1. Calcul hyperdimensionnel (HDC) — vecteurs bipolaires D-dim
  2. Mémoire distribuée sparse (SDM de Kanerva) — apprentissage one-shot
  3. Résonance continue — champ dynamique qui raisonne dans le temps
  4. Couche agentique — appel d'outils symbolique
  5. Adaptation Hebbienne locale — pas de backprop

Auteur: construit en brainstorm-code session.
"""

from .hd import (
    HDVector,
    hd_random,
    hd_bind,
    hd_bundle,
    hd_permute,
    hd_similarity,
    hd_distance,
    hd_cleanup,
)
from .memory import SparseDistributedMemory
from .tokenizer import SimpleTokenizer
from .encoder import HDEncoder
from .decoder import HDDecoder
from .resonator import Resonator, ResonatorConfig
from .agent import Agent, Tool
from .brain import Nova, NovaConfig

__version__ = "0.1.0"
__all__ = [
    "HDVector",
    "hd_random",
    "hd_bind",
    "hd_bundle",
    "hd_permute",
    "hd_similarity",
    "hd_distance",
    "hd_cleanup",
    "SparseDistributedMemory",
    "SimpleTokenizer",
    "HDEncoder",
    "HDDecoder",
    "Resonator",
    "ResonatorConfig",
    "Agent",
    "Tool",
    "Nova",
    "NovaConfig",
]
