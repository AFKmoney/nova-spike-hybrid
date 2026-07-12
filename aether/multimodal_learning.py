"""
multimodal_learning.py — Cross-modal learning: bind text to images/audio.

PROBLEM
-------
Transformer LLMs need separate architectures for vision (ViT), audio
(Whisper), and text (LLM), plus an alignment layer (CLIP) to connect them.

SOLUTION
--------
In HD space, all modalities are just hypervectors. We can BIND a text
description to an image HD vector — they become associated in the same
memory. No separate architecture needed.

  learn_image("a red ball", image_hv)
    → bind(text_hv("a red ball"), image_hv)
    → stored in cross-modal memory

  retrieve("show me a ball")
    → encode query → find nearest cross-modal HD vector
    → return the associated image

This is true multi-modal fusion — not a bolt-on, but native.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import logging

from .hd import HDVector, DIM, bundle
from .multimodal import ImageHDEncoder, AudioHDEncoder

log = logging.getLogger(__name__)


@dataclass
class CrossModalEntry:
    """A cross-modal entry: text + image (and/or audio) bound together."""
    text: Optional[str] = None
    text_vec: Optional[HDVector] = None
    image_vec: Optional[HDVector] = None
    audio_vec: Optional[HDVector] = None
    bound_vec: Optional[HDVector] = None  # the bound representation
    modality: str = "text"  # "text", "image", "audio", "multi"


class CrossModalLearner:
    """Learn and retrieve cross-modal associations."""

    def __init__(self, agent):
        self.agent = agent
        self.image_encoder = ImageHDEncoder(dim=agent.dim)
        self.audio_encoder = AudioHDEncoder(dim=agent.dim)
        self.entries: List[CrossModalEntry] = []

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def learn_image(self, description: str, image: np.ndarray) -> CrossModalEntry:
        """Associate a text description with an image.

        The image HD vector and the text HD vector are BOUND together,
        creating a cross-modal memory entry.
        """
        text_vec = self.agent.encoder.encode_text(description)
        image_vec = self.image_encoder.encode(image)
        # Bind: the result is ~orthogonal to both, but contains both
        bound_vec = text_vec.bind(image_vec)
        entry = CrossModalEntry(
            text=description, text_vec=text_vec,
            image_vec=image_vec, bound_vec=bound_vec,
            modality="multi",
        )
        self.entries.append(entry)
        # Also store in the agent's episode memory for general retrieval
        self.agent.assoc.add_episode(f"[image] {description}", bound_vec)
        # Store the description as a fact too
        self.agent.teach(description, silent=True)
        log.info(f"learned image: {description!r}")
        return entry

    def learn_image_pattern(self, description: str, pattern: List[str]) -> CrossModalEntry:
        """Associate a text description with an ASCII art pattern."""
        text_vec = self.agent.encoder.encode_text(description)
        image_vec = self.image_encoder.encode_from_pattern(pattern)
        bound_vec = text_vec.bind(image_vec)
        entry = CrossModalEntry(
            text=description, text_vec=text_vec,
            image_vec=image_vec, bound_vec=bound_vec,
            modality="multi",
        )
        self.entries.append(entry)
        self.agent.assoc.add_episode(f"[image] {description}", bound_vec)
        log.info(f"learned image pattern: {description!r}")
        return entry

    def learn_audio(self, description: str, waveform: np.ndarray) -> CrossModalEntry:
        """Associate a text description with an audio waveform."""
        text_vec = self.agent.encoder.encode_text(description)
        audio_vec = self.audio_encoder.encode(waveform)
        bound_vec = text_vec.bind(audio_vec)
        entry = CrossModalEntry(
            text=description, text_vec=text_vec,
            audio_vec=audio_vec, bound_vec=bound_vec,
            modality="multi",
        )
        self.entries.append(entry)
        self.agent.assoc.add_episode(f"[audio] {description}", bound_vec)
        log.info(f"learned audio: {description!r}")
        return entry

    def learn_sine(self, description: str, frequency: float) -> CrossModalEntry:
        """Associate a text description with a sine wave (for testing)."""
        text_vec = self.agent.encoder.encode_text(description)
        audio_vec = self.audio_encoder.encode_sine(frequency)
        bound_vec = text_vec.bind(audio_vec)
        entry = CrossModalEntry(
            text=description, text_vec=text_vec,
            audio_vec=audio_vec, bound_vec=bound_vec,
            modality="multi",
        )
        self.entries.append(entry)
        self.agent.assoc.add_episode(f"[audio] {description}", bound_vec)
        return entry

    # ------------------------------------------------------------------ #
    # Cross-modal retrieval
    # ------------------------------------------------------------------ #
    def retrieve_by_text(self, query: str, top_k: int = 3) -> List[Tuple[CrossModalEntry, float]]:
        """Retrieve cross-modal entries matching a text query."""
        q_vec = self.agent.encoder.encode_text(query)
        sims = [(e, q_vec.similarity(e.text_vec)) for e in self.entries if e.text_vec]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def retrieve_by_image(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[CrossModalEntry, float]]:
        """Retrieve cross-modal entries matching an image."""
        q_vec = self.image_encoder.encode(image)
        sims = [(e, q_vec.similarity(e.image_vec)) for e in self.entries if e.image_vec]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def retrieve_by_audio(self, waveform: np.ndarray, top_k: int = 3) -> List[Tuple[CrossModalEntry, float]]:
        """Retrieve cross-modal entries matching an audio waveform."""
        q_vec = self.audio_encoder.encode(waveform)
        sims = [(e, q_vec.similarity(e.audio_vec)) for e in self.entries if e.audio_vec]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def retrieve_cross_modal(self, query: str, from_modality: str = "image",
                            top_k: int = 3) -> List[Tuple[CrossModalEntry, float]]:
        """Cross-modal retrieval: query with text, get entries of another modality.

        Example: retrieve_cross_modal("ball", from_modality="image")
        → returns image entries whose text description matches "ball"
        """
        text_vec = self.agent.encoder.encode_text(query)
        results = []
        for entry in self.entries:
            if entry.text_vec and entry.image_vec and from_modality == "image":
                sim = text_vec.similarity(entry.text_vec)
                results.append((entry, sim))
            elif entry.text_vec and entry.audio_vec and from_modality == "audio":
                sim = text_vec.similarity(entry.text_vec)
                results.append((entry, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        n_image = sum(1 for e in self.entries if e.image_vec)
        n_audio = sum(1 for e in self.entries if e.audio_vec)
        n_multi = sum(1 for e in self.entries if e.modality == "multi")
        return {
            "n_entries": len(self.entries),
            "n_with_image": n_image,
            "n_with_audio": n_audio,
            "n_multi_modal": n_multi,
        }
