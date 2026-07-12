"""
SPIKE — Spiking Pattern Intelligence with Kernel Execution
=========================================================

Un cerveau artificiel à impulsions (SNN) couplé à une couche agentique.

Piliers:
  1. Neurones LIF (Leaky Integrate-and-Fire) — biologique, temporel
  2. Matrices CSR sparse — 1% de connectivité, RAM minimal
  3. Event-driven — on ne calcule QUE les neurones qui spikent
  4. STDP — apprentissage local, asynchrone, sans backprop
  5. Populations sensorielle/associative/motrice — décodage par compte
  6. Couche agentique — déclenchement d'outils par activité motrice

CPU-only, zéro multiplication matricielle, zéro GPU.
"""

from .core import (
    LIFNeuron,
    LIFParams,
    Spike,
    SimulationClock,
)
from .network import SpikingNetwork, PopulationType
from .stdp import STDPTracker, STDPConfig, imprint_path
from .coder import SpikeCoder, PopulationDecoder
from .agent import SpikeAgent
from .brain import SpikeBrain, SpikeConfig

__version__ = "0.1.0"
__all__ = [
    "LIFNeuron", "LIFParams", "Spike", "SimulationClock",
    "SpikingNetwork", "PopulationType",
    "STDPSynapse", "STDPTracker", "STDPConfig", "imprint_path",
    "SpikeCoder", "PopulationDecoder",
    "SpikeAgent",
    "SpikeBrain", "SpikeConfig",
]
