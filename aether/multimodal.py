"""
multimodal.py — Multi-modal HD encoding: images + audio.

PROBLEM
-------
AETHER v2 only handles text. To be a real cognitive agent, it needs to
perceive images and audio too. Transformers do this with separate vision
encoders (ViT, CLIP) and audio encoders (Whisper).

AETHER'S SOLUTION
-----------------
Encode images and audio DIRECTLY as HD vectors, using the same VSA
algebra as text. No neural networks. No GPU.

  IMAGE ENCODING:
    1. Downscale image to NxN (e.g., 16x16) grayscale.
    2. For each pixel (i, j), create an HD vector:
         pixel_vec = bind(position_vec(i, j), intensity_vec(gray_value))
       - position_vec: HD vector seeded by "pos:i,j" (deterministic)
       - intensity_vec: HD vector seeded by "int:N" where N is the gray level
    3. Bundle all pixel vectors -> single HD vector representing the image.
    4. Two images with similar pixel patterns will have high HD similarity.

  AUDIO ENCODING:
    1. Compute a simple spectrogram (FFT over short windows).
    2. Bin frequencies into N bands (e.g., 16 bands).
    3. For each time step t, create an HD vector:
         frame_vec = bind(time_vec(t), bind(band_vec(b), level_vec(energy)))
    4. Bundle all frame vectors -> single HD vector representing the audio.

Both produce 4096-dim bipolar HD vectors that can be stored in the same
memory as text. AETHER can then compare text and images in the SAME space.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass

from .hd import HDVector, DIM, bundle, _sign


# ---------------------------------------------------------------------------
# Image HD encoder
# ---------------------------------------------------------------------------

class ImageHDEncoder:
    """Encode images as HD vectors via position-bound intensity bundling.

    The image is downscaled to a fixed grid (default 16x16) and converted
    to grayscale. Each pixel becomes an HD vector; the image is the bundle.
    """

    def __init__(self, dim: int = DIM, grid_size: int = 16, intensity_bins: int = 8):
        self.dim = dim
        self.grid_size = grid_size
        self.intensity_bins = intensity_bins
        # Pre-compute position vectors (deterministic from coordinates)
        self._position_vecs: Dict[Tuple[int, int], HDVector] = {}
        # Pre-compute intensity vectors (deterministic from bin)
        self._intensity_vecs: Dict[int, HDVector] = {}

    def _get_position_vec(self, i: int, j: int) -> HDVector:
        key = (i, j)
        if key not in self._position_vecs:
            self._position_vecs[key] = HDVector.from_text_seed(f"pos:{i},{j}", self.dim)
        return self._position_vecs[key]

    def _get_intensity_vec(self, intensity: int) -> HDVector:
        """Get the HD vector for an intensity bin (0 to intensity_bins-1)."""
        if intensity not in self._intensity_vecs:
            self._intensity_vecs[intensity] = HDVector.from_text_seed(f"int:{intensity}", self.dim)
        return self._intensity_vecs[intensity]

    def encode(self, image: np.ndarray) -> HDVector:
        """Encode a 2D grayscale image as an HD vector.

        Args:
            image: 2D numpy array of pixel intensities (any range).

        Returns:
            HDVector representing the image.
        """
        # Convert to grayscale if needed (take mean of channels)
        if image.ndim == 3:
            image = image.mean(axis=2)
        # Downscale to grid_size x grid_size by averaging blocks
        h, w = image.shape
        gh, gw = self.grid_size, self.grid_size
        # Simple block averaging
        # Reshape to (gh, h//gh, gw, w//gw) and mean
        h_step = h // gh
        w_step = w // gw
        if h_step == 0 or w_step == 0:
            # Image is smaller than grid — pad with zeros
            padded = np.zeros((gh, gw), dtype=np.float64)
            padded[:min(h, gh), :min(w, gw)] = image[:min(h, gh), :min(w, gw)]
            downscaled = padded
        else:
            # Crop to multiple of step
            cropped = image[:h_step * gh, :w_step * gw]
            # Reshape and mean
            downscaled = cropped.reshape(gh, h_step, gw, w_step).mean(axis=(1, 3))

        # Normalize to [0, intensity_bins-1]
        dmin, dmax = downscaled.min(), downscaled.max()
        if dmax > dmin:
            normalized = ((downscaled - dmin) / (dmax - dmin) * (self.intensity_bins - 1)).astype(int)
        else:
            normalized = np.zeros_like(downscaled, dtype=int)

        # Build HD vector: bundle of bind(pos(i,j), intensity(b)) for each pixel
        vecs = []
        for i in range(gh):
            for j in range(gw):
                pos_vec = self._get_position_vec(i, j)
                int_vec = self._get_intensity_vec(int(normalized[i, j]))
                pixel_vec = pos_vec.bind(int_vec)
                vecs.append(pixel_vec)

        return bundle(vecs)

    def encode_from_pattern(self, pattern: List[str]) -> HDVector:
        """Encode an ASCII art pattern as an HD vector.

        Each character in the pattern is treated as a pixel intensity:
          ' ' = 0, '.' = 1, ':' = 2, '#' = 3, '@' = 4 (etc.)
        """
        if not pattern:
            return HDVector.zero(self.dim)
        # Convert to numpy array
        h = len(pattern)
        w = max(len(row) for row in pattern)
        img = np.zeros((h, w), dtype=np.float64)
        char_to_intensity = {
            ' ': 0, '.': 1, ':': 2, '-': 2, '=': 3, '+': 3,
            '*': 4, '#': 5, '%': 6, '@': 7, 'X': 7, 'M': 7,
        }
        for i, row in enumerate(pattern):
            for j, ch in enumerate(row):
                img[i, j] = char_to_intensity.get(ch, 0)
        return self.encode(img)


# ---------------------------------------------------------------------------
# Audio HD encoder
# ---------------------------------------------------------------------------

class AudioHDEncoder:
    """Encode audio waveforms as HD vectors via spectrogram bundling.

    The audio is split into N short frames. For each frame, compute the
    FFT and bin the magnitude spectrum into B frequency bands. Each
    (frame, band) pair becomes an HD vector; the audio is the bundle.
    """

    def __init__(self, dim: int = DIM, n_frames: int = 16, n_bands: int = 16,
                 energy_bins: int = 8):
        self.dim = dim
        self.n_frames = n_frames
        self.n_bands = n_bands
        self.energy_bins = energy_bins
        # Pre-compute frame and band vectors
        self._frame_vecs: Dict[int, HDVector] = {}
        self._band_vecs: Dict[int, HDVector] = {}
        self._energy_vecs: Dict[int, HDVector] = {}

    def _get_frame_vec(self, t: int) -> HDVector:
        if t not in self._frame_vecs:
            self._frame_vecs[t] = HDVector.from_text_seed(f"frame:{t}", self.dim)
        return self._frame_vecs[t]

    def _get_band_vec(self, b: int) -> HDVector:
        if b not in self._band_vecs:
            self._band_vecs[b] = HDVector.from_text_seed(f"band:{b}", self.dim)
        return self._band_vecs[b]

    def _get_energy_vec(self, e: int) -> HDVector:
        if e not in self._energy_vecs:
            self._energy_vecs[e] = HDVector.from_text_seed(f"energy:{e}", self.dim)
        return self._energy_vecs[e]

    def encode(self, waveform: np.ndarray, sample_rate: int = 16000) -> HDVector:
        """Encode a 1D audio waveform as an HD vector.

        Args:
            waveform: 1D numpy array of audio samples (any range).
            sample_rate: sample rate (used only to size the FFT window).

        Returns:
            HDVector representing the audio.
        """
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)  # stereo -> mono

        n = len(waveform)
        if n == 0:
            return HDVector.zero(self.dim)

        # Split into n_frames frames
        frame_size = max(n // self.n_frames, 8)
        frames = []
        for i in range(self.n_frames):
            start = i * frame_size
            end = min(start + frame_size, n)
            if start >= n:
                break
            frame = waveform[start:end]
            if len(frame) > 0:
                frames.append(frame)
        if not frames:
            return HDVector.zero(self.dim)

        # For each frame, compute magnitude spectrum and bin into n_bands
        vecs = []
        for t, frame in enumerate(frames):
            # Apply Hamming window
            windowed = frame * np.hamming(len(frame))
            # FFT (real)
            spectrum = np.abs(np.fft.rfft(windowed))
            # Bin into n_bands frequency bands (log scale for better coverage)
            n_freqs = len(spectrum)
            if n_freqs == 0:
                continue
            # Log-spaced bin edges
            bin_edges = np.logspace(0, np.log10(max(n_freqs, 2)), self.n_bands + 1).astype(int)
            bin_edges = np.clip(bin_edges, 1, n_freqs)
            band_energies = []
            for b in range(self.n_bands):
                start = bin_edges[b] - 1
                end = bin_edges[b + 1]
                if end > start:
                    band_energies.append(spectrum[start:end].mean())
                else:
                    band_energies.append(0.0)
            band_energies = np.array(band_energies)
            # Normalize and bin
            emin, emax = band_energies.min(), band_energies.max()
            if emax > emin:
                normalized = ((band_energies - emin) / (emax - emin) * (self.energy_bins - 1)).astype(int)
            else:
                normalized = np.zeros_like(band_energies, dtype=int)
            # Build HD vectors for this frame: bind(frame_vec(t), bind(band_vec(b), energy_vec(e)))
            for b in range(self.n_bands):
                frame_vec = self._get_frame_vec(t)
                band_vec = self._get_band_vec(b)
                energy_vec = self._get_energy_vec(int(normalized[b]))
                vec = frame_vec.bind(band_vec.bind(energy_vec))
                vecs.append(vec)

        if not vecs:
            return HDVector.zero(self.dim)
        return bundle(vecs)

    def encode_sine(self, frequency: float, duration: float = 1.0, sample_rate: int = 16000) -> HDVector:
        """Encode a pure sine wave (useful for testing)."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t)
        return self.encode(waveform, sample_rate)

    def encode_chord(self, frequencies: List[float], duration: float = 1.0, sample_rate: int = 16000) -> HDVector:
        """Encode a chord (sum of sine waves)."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = np.zeros_like(t)
        for freq in frequencies:
            waveform += np.sin(2 * np.pi * freq * t)
        waveform /= max(len(frequencies), 1)
        return self.encode(waveform, sample_rate)


# ---------------------------------------------------------------------------
# Cross-modal similarity (text ↔ image ↔ audio)
# ---------------------------------------------------------------------------

class CrossModalSpace:
    """A shared HD space where text, images, and audio can be compared.

    The trick: each modality has its own "modality role" vector. When we
    store a perception, we bind it with its modality role. When we query
    across modalities, we can either:
      - query within a modality (bind with that modality's role)
      - query across modalities (use the raw perception vector, ignoring role)
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.modality_roles = {
            "text":  HDVector.from_text_seed("modality:text", dim),
            "image": HDVector.from_text_seed("modality:image", dim),
            "audio": HDVector.from_text_seed("modality:audio", dim),
        }
        self.image_encoder = ImageHDEncoder(dim=dim)
        self.audio_encoder = AudioHDEncoder(dim=dim)
        # Store: list of (modality, content, vector)
        self.store: List[Tuple[str, str, HDVector]] = []

    def add_text(self, text: str, vector: HDVector) -> None:
        """Add a text perception (vector already encoded by text encoder)."""
        # Bind with text modality role
        mod_vec = vector.bind(self.modality_roles["text"])
        self.store.append(("text", text, mod_vec))

    def add_image(self, image: np.ndarray, label: str = "") -> None:
        """Add an image perception."""
        vec = self.image_encoder.encode(image)
        mod_vec = vec.bind(self.modality_roles["image"])
        self.store.append(("image", label, mod_vec))

    def add_image_pattern(self, pattern: List[str], label: str = "") -> None:
        """Add an ASCII art image."""
        vec = self.image_encoder.encode_from_pattern(pattern)
        mod_vec = vec.bind(self.modality_roles["image"])
        self.store.append(("image", label, mod_vec))

    def add_audio(self, waveform: np.ndarray, label: str = "") -> None:
        """Add an audio perception."""
        vec = self.audio_encoder.encode(waveform)
        mod_vec = vec.bind(self.modality_roles["audio"])
        self.store.append(("audio", label, mod_vec))

    def add_sine(self, frequency: float, label: str = "") -> None:
        """Add a pure sine wave."""
        vec = self.audio_encoder.encode_sine(frequency)
        mod_vec = vec.bind(self.modality_roles["audio"])
        self.store.append(("audio", label, mod_vec))

    def find_similar(self, query_vec: HDVector, modality: Optional[str] = None, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find similar perceptions. If modality is None, search all modalities."""
        results = []
        for mod, label, vec in self.store:
            if modality is not None and mod != modality:
                continue
            sim = query_vec.similarity(vec)
            results.append((mod, label, sim))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    def find_similar_image(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find perceptions similar to the given image."""
        vec = self.image_encoder.encode(image).bind(self.modality_roles["image"])
        return self.find_similar(vec, top_k=top_k)

    def find_similar_image_pattern(self, pattern: List[str], top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find perceptions similar to the given ASCII art pattern."""
        vec = self.image_encoder.encode_from_pattern(pattern).bind(self.modality_roles["image"])
        return self.find_similar(vec, top_k=top_k)

    def find_similar_audio(self, waveform: np.ndarray, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find perceptions similar to the given audio."""
        vec = self.audio_encoder.encode(waveform).bind(self.modality_roles["audio"])
        return self.find_similar(vec, top_k=top_k)

    def find_similar_sine(self, frequency: float, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find perceptions similar to a pure sine wave."""
        vec = self.audio_encoder.encode_sine(frequency).bind(self.modality_roles["audio"])
        return self.find_similar(vec, top_k=top_k)

    def stats(self) -> Dict[str, int]:
        modality_counts: Dict[str, int] = {}
        for mod, _, _ in self.store:
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        return {
            "total": len(self.store),
            **{f"{mod}_count": cnt for mod, cnt in modality_counts.items()},
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Image HD Encoder ===")
    img_enc = ImageHDEncoder(dim=4096, grid_size=16, intensity_bins=8)

    # Create two synthetic images: a square and a circle
    square = np.zeros((32, 32))
    square[8:24, 8:24] = 1.0
    circle = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if (i - 16) ** 2 + (j - 16) ** 2 < 100:
                circle[i, j] = 1.0
    noise = np.random.rand(32, 32)

    v_sq = img_enc.encode(square)
    v_ci = img_enc.encode(circle)
    v_no = img_enc.encode(noise)
    v_sq2 = img_enc.encode(square.copy())  # identical

    print(f"  sim(square, square_copy) = {v_sq.similarity(v_sq2):.3f}  (should be ~1.0)")
    print(f"  sim(square, circle)      = {v_sq.similarity(v_ci):.3f}  (should be moderate)")
    print(f"  sim(square, noise)       = {v_sq.similarity(v_no):.3f}  (should be ~0)")
    print(f"  sim(circle, noise)       = {v_ci.similarity(v_no):.3f}  (should be ~0)")

    print("\n=== Audio HD Encoder ===")
    aud_enc = AudioHDEncoder(dim=4096, n_frames=16, n_bands=16, energy_bins=8)

    v_440 = aud_enc.encode_sine(440)   # A4
    v_440b = aud_enc.encode_sine(440)  # same
    v_880 = aud_enc.encode_sine(880)   # A5 (octave)
    v_220 = aud_enc.encode_sine(220)   # A3 (octave below)
    v_chord = aud_enc.encode_chord([440, 554, 659])  # A major chord

    print(f"  sim(440, 440_copy) = {v_440.similarity(v_440b):.3f}  (should be ~1.0)")
    print(f"  sim(440, 880)      = {v_440.similarity(v_880):.3f}  (should be moderate)")
    print(f"  sim(440, 220)      = {v_440.similarity(v_220):.3f}  (should be moderate)")
    print(f"  sim(440, chord)    = {v_440.similarity(v_chord):.3f}  (should be lower, chord has more freqs)")

    print("\n=== Cross-modal Space ===")
    cms = CrossModalSpace(dim=4096)

    # Add some perceptions
    cms.add_image_pattern([
        "                                ",
        "         ###########            ",
        "       ###############          ",
        "      #################         ",
        "     ###################        ",
        "     ###################        ",
        "     ###################        ",
        "      #################         ",
        "       ###############          ",
        "         ###########            ",
        "                                ",
    ], label="circle_pattern")

    cms.add_image_pattern([
        "                                ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "       ################         ",
        "                                ",
        "                                ",
        "                                ",
        "                                ",
        "                                ",
    ], label="rectangle_pattern")

    cms.add_sine(440, label="A4_note")
    cms.add_sine(880, label="A5_note")

    print(f"  CrossModal stats: {cms.stats()}")
    print()

    # Find similar to circle pattern
    circle_pattern = [
        "                                ",
        "         ###########            ",
        "       ###############          ",
        "      #################         ",
        "     ###################        ",
        "     ###################        ",
        "     ###################        ",
        "      #################         ",
        "       ###############          ",
        "         ###########            ",
        "                                ",
    ]
    results = cms.find_similar_image_pattern(circle_pattern, top_k=3)
    print("  Find similar to circle_pattern:")
    for mod, label, sim in results:
        print(f"    [{mod}] {label!r:25s} sim={sim:.3f}")

    # Find similar to A4 note
    results = cms.find_similar_sine(440, top_k=3)
    print("\n  Find similar to A4 (440 Hz):")
    for mod, label, sim in results:
        print(f"    [{mod}] {label!r:25s} sim={sim:.3f}")

    results = cms.find_similar_sine(450, top_k=3)  # close to 440
    print("\n  Find similar to 450 Hz (close to A4):")
    for mod, label, sim in results:
        print(f"    [{mod}] {label!r:25s} sim={sim:.3f}")
