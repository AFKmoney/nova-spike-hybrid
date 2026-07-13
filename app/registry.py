"""
Model Registry — manages all AI brains with hot-loading and toggle.

Each model can be:
  - ENABLED/DISABLED (hot toggle without restart)
  - LOADED/UNLOADED (lazy init / unload to free memory)
  - QUERIED (generate, reason, chat)

The registry tracks metrics per model:
  - n_calls, total_time_ms, avg_time_ms
  - last_response, last_latency
  - status (enabled/disabled, loaded/unloaded)
"""

from __future__ import annotations
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class ModelMetrics:
    """Per-model metrics."""
    n_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_time_ms: float = 0.0
    last_response: str = ""
    last_error: str = ""
    enabled: bool = False
    loaded: bool = False
    load_time_s: float = 0.0
    memory_mb: float = 0.0


class ModelEntry:
    """A model entry in the registry."""

    def __init__(self, name: str, description: str,
                 loader_fn, color: str = "#00d2ff"):
        self.name = name
        self.description = description
        self.loader_fn = loader_fn  # callable that returns the brain
        self.color = color
        self.brain = None
        self.metrics = ModelMetrics()
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Hot-load the model."""
        if self.brain is not None:
            return True
        try:
            t0 = time.time()
            self.brain = self.loader_fn()
            self.metrics.load_time_s = time.time() - t0
            self.metrics.loaded = True
            self.metrics.memory_mb = self._estimate_memory()
            return True
        except Exception as e:
            self.metrics.last_error = str(e)
            return False

    def unload(self) -> None:
        """Unload the model to free memory."""
        self.brain = None
        self.metrics.loaded = False
        self.metrics.memory_mb = 0.0

    def enable(self) -> None:
        self.metrics.enabled = True

    def disable(self) -> None:
        self.metrics.enabled = False

    def _estimate_memory(self) -> float:
        """Estimate memory in MB."""
        if self.brain is None:
            return 0.0
        try:
            import sys
            return sys.getsizeof(self.brain) / (1024 * 1024)
        except Exception:
            return 0.0

    def chat(self, message: str) -> dict:
        """Send a message to this model. Returns {response, time_ms, error}."""
        if not self.metrics.enabled:
            return {"error": "Model disabled", "response": "", "time_ms": 0}
        if not self.metrics.loaded:
            ok = self.load()
            if not ok:
                return {"error": f"Load failed: {self.metrics.last_error}",
                        "response": "", "time_ms": 0}

        with self._lock:
            t0 = time.time()
            try:
                response = self.brain.chat(message)
                dt = (time.time() - t0) * 1000
                self.metrics.n_calls += 1
                self.metrics.total_time_ms += dt
                self.metrics.avg_time_ms = self.metrics.total_time_ms / self.metrics.n_calls
                self.metrics.last_time_ms = dt
                self.metrics.last_response = response
                self.metrics.last_error = ""
                return {"response": response, "time_ms": dt, "error": ""}
            except Exception as e:
                dt = (time.time() - t0) * 1000
                self.metrics.last_error = str(e)
                return {"error": str(e), "response": "", "time_ms": dt}

    def generate(self, prompt: str, max_tokens: int = 20) -> dict:
        """Generate text."""
        if not self.metrics.enabled:
            return {"error": "Model disabled", "response": "", "time_ms": 0}
        if not self.metrics.loaded:
            ok = self.load()
            if not ok:
                return {"error": f"Load failed: {self.metrics.last_error}",
                        "response": "", "time_ms": 0}

        with self._lock:
            t0 = time.time()
            try:
                if hasattr(self.brain, "generate"):
                    response = self.brain.generate(prompt, max_tokens=max_tokens)
                elif hasattr(self.brain, "chat"):
                    response = self.brain.chat(prompt)
                else:
                    response = str(self.brain)
                dt = (time.time() - t0) * 1000
                self.metrics.n_calls += 1
                self.metrics.total_time_ms += dt
                self.metrics.avg_time_ms = self.metrics.total_time_ms / self.metrics.n_calls
                self.metrics.last_time_ms = dt
                self.metrics.last_response = response
                return {"response": response, "time_ms": dt, "error": ""}
            except Exception as e:
                dt = (time.time() - t0) * 1000
                self.metrics.last_error = str(e)
                return {"error": str(e), "response": "", "time_ms": dt}


class ModelRegistry:
    """
    Registry of all AI brains with hot-loading and toggle.

    Usage:
        registry = ModelRegistry()
        registry.register("Generative", "Fast generative AI", loader_fn)
        registry.enable("Generative")
        result = registry.chat("Generative", "Hello")
    """

    def __init__(self):
        self.models: dict[str, ModelEntry] = {}
        self._lock = threading.Lock()

    def register(self, name: str, description: str, loader_fn,
                  color: str = "#00d2ff", auto_enable: bool = False) -> None:
        """Register a new model."""
        entry = ModelEntry(name, description, loader_fn, color)
        if auto_enable:
            entry.enable()
        self.models[name] = entry

    def enable(self, name: str) -> bool:
        if name in self.models:
            self.models[name].enable()
            return True
        return False

    def disable(self, name: str) -> bool:
        if name in self.models:
            self.models[name].disable()
            return True
        return False

    def load(self, name: str) -> bool:
        if name in self.models:
            return self.models[name].load()
        return False

    def unload(self, name: str) -> bool:
        if name in self.models:
            self.models[name].unload()
            return True
        return False

    def chat(self, name: str, message: str) -> dict:
        if name not in self.models:
            return {"error": f"Unknown model: {name}", "response": "", "time_ms": 0}
        return self.models[name].chat(message)

    def generate(self, name: str, prompt: str, max_tokens: int = 20) -> dict:
        if name not in self.models:
            return {"error": f"Unknown model: {name}", "response": "", "time_ms": 0}
        return self.models[name].generate(prompt, max_tokens)

    def chat_combined(self, message: str) -> dict:
        """
        Combined mode: query all ENABLED+LOADED models, return all responses.
        Picks the fastest response as primary, others as alternatives.
        """
        responses = {}
        fastest_name = None
        fastest_time = float("inf")

        for name, entry in self.models.items():
            if not entry.metrics.enabled:
                continue
            result = entry.chat(message)
            responses[name] = result
            if result.get("time_ms", float("inf")) < fastest_time and not result.get("error"):
                fastest_time = result["time_ms"]
                fastest_name = name

        return {
            "responses": responses,
            "primary": fastest_name,
            "n_models": len(responses),
        }

    def get_status(self) -> dict:
        """Get status of all models."""
        status = {}
        for name, entry in self.models.items():
            status[name] = {
                "name": name,
                "description": entry.description,
                "color": entry.color,
                "enabled": entry.metrics.enabled,
                "loaded": entry.metrics.loaded,
                "n_calls": entry.metrics.n_calls,
                "avg_time_ms": round(entry.metrics.avg_time_ms, 1),
                "last_time_ms": round(entry.metrics.last_time_ms, 1),
                "last_response": entry.metrics.last_response[:200],
                "last_error": entry.metrics.last_error,
                "load_time_s": round(entry.metrics.load_time_s, 2),
                "memory_mb": round(entry.metrics.memory_mb, 2),
            }
        return status

    def get_metrics_history(self) -> dict:
        """Get aggregate metrics."""
        total_calls = sum(e.metrics.n_calls for e in self.models.values())
        total_memory = sum(e.metrics.memory_mb for e in self.models.values())
        n_enabled = sum(1 for e in self.models.values() if e.metrics.enabled)
        n_loaded = sum(1 for e in self.models.values() if e.metrics.loaded)
        return {
            "total_calls": total_calls,
            "total_memory_mb": round(total_memory, 2),
            "n_enabled": n_enabled,
            "n_loaded": n_loaded,
            "n_models": len(self.models),
        }
