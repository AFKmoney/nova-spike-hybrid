"""
Multi-modal — encodeur d'images → spikes.

Principe biologique: la rétine encode les images en trains de spikes.
Chaque pixel est converti en un taux de décharge Poisson proportionnel
à son intensité.

Architecture:
  - Population visuelle (n_visual neurones) — distincte de sensory/assoc/motor
  - L'image est redimensionnée à une grille (ex: 20x20 = 400 neurones)
  - Chaque neurone reçoit un courant proportionnel à l'intensité du pixel
  - Couplage à la couche associative du SNN existant

Usage:
    encoder = ImageEncoder(n_visual=400, grid_size=(20, 20))
    I_visual = encoder.encode_poisson(image_array, gain=2.0, duration=10)
    # I_visual: liste de 10 arrays (n_visual,) à injecter pendant 10 ticks
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageEncoder:
    """
    Encodeur image → spikes (rate coding Poisson).

    Args:
        n_visual: nombre de neurones visuels (doit matcher grid_size)
        grid_size: dimensions de la grille (H, W)
    """
    n_visual: int = 400
    grid_size: tuple[int, int] = (20, 20)
    normalize: bool = True  # normalise l'image 0-1

    _rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self):
        h, w = self.grid_size
        assert h * w == self.n_visual, f"grid_size {self.grid_size} != n_visual {self.n_visual}"
        self._rng = np.random.default_rng()

    # ---------------------------------------------------------------- #
    # Prétraitement image
    # ---------------------------------------------------------------- #
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite l'image:
          - conversion en niveaux de gris si nécessaire
          - redimensionnement à grid_size
          - normalisation 0-1
        """
        # Conversion en niveaux de gris
        if image.ndim == 3:
            # Moyenne pondérée RGB
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        # Redimensionnement naïf (nearest neighbor)
        h, w = self.grid_size
        orig_h, orig_w = image.shape
        # Indices échantillonnés
        row_idx = np.linspace(0, orig_h - 1, h).astype(int)
        col_idx = np.linspace(0, orig_w - 1, w).astype(int)
        resized = image[np.ix_(row_idx, col_idx)]
        # Normalisation
        if self.normalize:
            mx = resized.max()
            if mx > 0:
                resized = resized / mx
        return resized.astype(np.float32)

    # ---------------------------------------------------------------- #
    # Encodage
    # ---------------------------------------------------------------- #
    def encode_static(self, image: np.ndarray, gain: float = 2.0) -> np.ndarray:
        """
        Encode l'image en un courant statique (n_visual,).
        """
        proc = self.preprocess(image)
        # Flatten en ordre row-major
        flat = proc.flatten()
        # Convertit en courant: pixels sombres = 0, pixels clairs = fort
        return (flat * gain).astype(np.float32)

    def encode_poisson(self, image: np.ndarray, duration: int = 10,
                        rate_scale: float = 1.0,
                        gain: float = 2.0) -> list[np.ndarray]:
        """
        Encode l'image en train de spikes Poisson sur `duration` ticks.

        À chaque tick, chaque neurone tire un spike avec proba
        proportionnelle à l'intensité du pixel correspondant.

        Args:
            image: image d'entrée (H, W) ou (H, W, 3)
            duration: nombre de ticks
            rate_scale: facteur d'échelle du taux de décharge
            gain: gain du courant injecté

        Return:
            Liste de `duration` arrays (n_visual,) — courant à injecter
        """
        I_static = self.encode_static(image, gain=gain)
        currents = []
        for _ in range(duration):
            # Pour chaque neurone, proba de spike = intensité * rate_scale
            proba = np.clip(I_static / gain * rate_scale, 0, 1)
            spikes = (self._rng.random(self.n_visual) < proba).astype(np.float32)
            # Convertit en courant (spike = gain, pas de spike = 0)
            I_tick = spikes * gain
            currents.append(I_tick)
        return currents

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def save_grid_image(self, image: np.ndarray, path: str) -> None:
        """Sauvegarde l'image prétraitée pour debug."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        proc = self.preprocess(image)
        plt.imsave(path, proc, cmap="gray")


# ---------------------------------------------------------------------- #
# Extension de SpikeBrain pour supporter la modalité visuelle
# ---------------------------------------------------------------------- #

class MultiModalBrain:
    """
    SpikeBrain étendu avec une population visuelle.

    Le cerveau peut maintenant:
      - Recevoir du texte (population sensory standard)
      - Recevoir des images (population visuelle)
      - Combiner les deux (multi-modal)

    Les deux populations projettent sur la même couche associative.
    """

    def __init__(self, spike_brain, n_visual: int = 400,
                 grid_size: tuple[int, int] = (20, 20)):
        """
        Args:
            spike_brain: SpikeBrain existant à étendre
            n_visual: nombre de neurones visuels
            grid_size: dimensions de la grille
        """
        self.brain = spike_brain
        self.encoder = ImageEncoder(n_visual=n_visual, grid_size=grid_size)
        # Synapses vis→assoc (vers la population associative existante)
        from spike.network import SynapseGroup, PopulationType
        self.syn_visual_assoc = SynapseGroup(
            "vis→assoc", PopulationType.SENSORY, PopulationType.ASSOCIATIVE,
            n_visual, spike_brain.net.n_associative, plastic=True,
        )
        self.syn_visual_assoc.random_connect(density=0.10, w_mean=1.5,
                                              w_std=0.3, seed=10)

    # ---------------------------------------------------------------- #
    def process_image(self, image: np.ndarray, duration: int = 10,
                       gain: float = 2.0) -> dict:
        """
        Traite une image: encode en spikes, propage au SNN, décode l'activité.

        Args:
            image: image d'entrée (H, W) ou (H, W, 3)
            duration: nombre de ticks
            gain: gain du courant

        Return:
            dict avec stats sur l'activité générée
        """
        # Reset mémoire de travail
        self.brain.net.reset(soft=False)

        # Encode l'image en train de spikes
        currents = self.encoder.encode_poisson(image, duration=duration,
                                                 gain=gain)
        # Simule: à chaque tick, on propage les spikes visuels
        total_assoc = 0
        total_motor = 0
        motor_log = []
        for I_vis in currents:
            # Propagation vis→assoc
            spikes_vis = I_vis > 0
            I_assoc_from_vis = self.syn_visual_assoc.propagate(spikes_vis)
            # Step associative (avec input visuel uniquement)
            # On doit simuler manuellement car le réseau standard ne gère pas vis
            spikes_assoc = self.brain.net.associative.step(I_assoc_from_vis,
                                                            self.brain.net.clock.t)
            # Propagation assoc→motor
            I_motor = self.brain.net.syn_assoc_to_motor.propagate(spikes_assoc)
            spikes_motor = self.brain.net.motor.step(I_motor,
                                                      self.brain.net.clock.t)
            # STDP
            if self.brain.cfg.stdp_enabled:
                self.brain.stdp_assoc_assoc.update(spikes_assoc, spikes_assoc)
                self.brain.stdp_assoc_motor.update(spikes_assoc, spikes_motor)
            self.brain.net.clock.tick()
            self.brain.net.last_spikes = {
                "sensory": np.zeros(self.brain.cfg.n_sensory, dtype=bool),
                "associative": spikes_assoc.copy(),
                "motor": spikes_motor.copy(),
                "t": self.brain.net.clock.t,
            }
            total_assoc += int(spikes_assoc.sum())
            total_motor += int(spikes_motor.sum())
            motor_log.append(spikes_motor.copy())

        total_motor_counts = np.stack(motor_log).sum(axis=0).astype(np.int32) if motor_log else np.zeros(self.brain.cfg.n_motor, dtype=np.int32)
        top_tokens = self.brain.coder.decode_top_k(total_motor_counts, k=5,
                                                     min_count=1)
        return {
            "duration": duration,
            "total_assoc_spikes": total_assoc,
            "total_motor_spikes": total_motor,
            "top_tokens": top_tokens,
            "image_shape": image.shape if hasattr(image, "shape") else "unknown",
        }

    # ---------------------------------------------------------------- #
    def process_text_and_image(self, text: str, image: np.ndarray,
                                text_weight: float = 1.0,
                                image_weight: float = 1.0,
                                duration: int = 20) -> dict:
        """
        Traite simultanément du texte et une image (vraiment multi-modal).

        Stratégie simple: on alterne les ticks — texte puis image — pour
        éviter les conflits STDP. Les deux modalités activent la même couche
        associative et convergent vers le même décodage moteur.
        """
        # Reset
        self.brain.net.reset(soft=False)

        # Encode les deux modalités
        I_text = self.brain.coder.encode_text_to_current(text, gain=self.brain.cfg.input_gain * text_weight)
        image_currents = self.encoder.encode_poisson(image, duration=duration,
                                                       gain=2.0 * image_weight)

        motor_log = []
        # Phase 1: texte seul (première moitié)
        half = duration // 2
        for tick in range(half):
            mask_t = (self.brain.rng.random(self.brain.cfg.n_sensory) < self.brain.cfg.poisson_rate).astype(np.float32)
            I_text_tick = I_text * mask_t
            self.brain.net.tick(I_text_tick)
            if self.brain.cfg.stdp_enabled:
                self.brain._apply_stdp()
            motor_log.append(self.brain.net.last_spikes["motor"].copy())

        # Phase 2: image seule (seconde moitié) via process_image
        for tick in range(half, duration):
            I_vis = image_currents[tick] if tick < len(image_currents) else np.zeros(self.encoder.n_visual, dtype=np.float32)
            spikes_vis = I_vis > 0
            I_assoc_from_vis = self.syn_visual_assoc.propagate(spikes_vis)
            spikes_assoc = self.brain.net.associative.step(I_assoc_from_vis,
                                                            self.brain.net.clock.t)
            I_motor = self.brain.net.syn_assoc_to_motor.propagate(spikes_assoc)
            spikes_motor = self.brain.net.motor.step(I_motor,
                                                      self.brain.net.clock.t)
            self.brain.net.clock.tick()
            self.brain.net.last_spikes = {
                "sensory": np.zeros(self.brain.cfg.n_sensory, dtype=bool),
                "associative": spikes_assoc.copy(),
                "motor": spikes_motor.copy(),
                "t": self.brain.net.clock.t,
            }
            motor_log.append(spikes_motor.copy())

        total_motor = np.stack(motor_log).sum(axis=0).astype(np.int32) if motor_log else np.zeros(self.brain.cfg.n_motor, dtype=np.int32)
        top_tokens = self.brain.coder.decode_top_k(total_motor, k=5, min_count=1)
        return {
            "text": text,
            "duration": duration,
            "total_motor_spikes": int(total_motor.sum()),
            "top_tokens": top_tokens,
        }


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Test ImageEncoder...")

    # Génère une image synthétique (carré blanc sur fond noir)
    image = np.zeros((100, 100), dtype=np.float32)
    image[30:70, 30:70] = 1.0  # carré central blanc
    # Ajoute du bruit
    image += np.random.random((100, 100)) * 0.1
    image = np.clip(image, 0, 1)

    encoder = ImageEncoder(n_visual=400, grid_size=(20, 20))
    I = encoder.encode_static(image, gain=2.0)
    print(f"Image shape: {image.shape}")
    print(f"Courant visuel: shape={I.shape}, max={I.max():.3f}, sum={I.sum():.3f}")

    # Train de spikes
    currents = encoder.encode_poisson(image, duration=10, gain=2.0)
    print(f"\nTrain de spikes (10 ticks):")
    for i, c in enumerate(currents):
        print(f"  tick {i}: {(c > 0).sum()} neurones actifs")

    # Test MultiModalBrain
    print("\nTest MultiModalBrain...")
    from spike import SpikeBrain, SpikeConfig
    brain = SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=30,
    ))
    mm = MultiModalBrain(brain, n_visual=400, grid_size=(20, 20))

    # Traite l'image seule
    r = mm.process_image(image, duration=20)
    print(f"Image seule: {r}")

    # Traite texte + image
    r2 = mm.process_text_and_image("chat", image, duration=20)
    print(f"Texte + image: {r2}")
