"""
NOVA-SPIKE-HYBRID App — FastAPI server with model registry, real-time metrics,
hot toggle, and combined mode.

Endpoints:
  GET  /                  — Dashboard HTML
  GET  /api/status        — All models status (enabled, loaded, metrics)
  POST /api/chat          — Chat with a specific model
  POST /api/chat_combined — Combined mode (all enabled models respond)
  POST /api/generate      — Generate text
  POST /api/enable/{name}    — Enable a model (hot)
  POST /api/disable/{name}   — Disable a model (hot)
  POST /api/load/{name}      — Hot-load a model
  POST /api/unload/{name}    — Unload a model (free memory)
  WS   /ws/metrics        — Real-time metrics stream
"""

from __future__ import annotations
import os
import sys
import json
import time
import asyncio
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.registry import ModelRegistry


# ---------------------------------------------------------------------- #
# Model loaders (lazy — only called when model is hot-loaded)
# ---------------------------------------------------------------------- #

def load_generative():
    """Load the GenerativeBrain (fast predictor, ~10s)."""
    from generative import GenerativeBrain, GenerativeConfig
    return GenerativeBrain(GenerativeConfig(verbose=False))


def load_aether():
    """Load AETHER (cognitive loop, ~15s)."""
    from aether import AETHER
    return AETHER()


def load_nova():
    """Load NOVA (HDC + SDM)."""
    from nova import Nova, NovaConfig
    return Nova(NovaConfig(D=3000, sdm_locations=5000))


def load_spike():
    """Load SPIKE (SNN)."""
    from spike import SpikeBrain, SpikeConfig
    return SpikeBrain(SpikeConfig(
        n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25,
    ))


def load_hybrid():
    """Load HYBRID (NOVA + SPIKE)."""
    from hybrid import HybridBrain, HybridConfig
    from spike import SpikeConfig
    from nova import NovaConfig
    return HybridBrain(HybridConfig(
        spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25),
        nova=NovaConfig(D=3000, sdm_locations=5000),
    ))


# ---------------------------------------------------------------------- #
# Build registry
# ---------------------------------------------------------------------- #

def build_registry() -> ModelRegistry:
    """Build the model registry with all brains."""
    reg = ModelRegistry()

    reg.register(
        "Generative",
        "Fast generative AI (BPE + 7-gram + Kneser-Ney + beam search). "
        "Best for: generation, reasoning, creative writing.",
        load_generative,
        color="#00d2ff",
        auto_enable=True,  # enabled by default
    )
    reg.register(
        "AETHER",
        "Brain-inspired cognitive agent (Kuramoto + attractors + GWT + consciousness). "
        "Best for: deep reasoning, cognitive trace.",
        load_aether,
        color="#5f27cd",
    )
    reg.register(
        "NOVA",
        "Hyperdimensional Computing (HDC + SDM + resonator). "
        "Best for: memory recall, robustness to noise.",
        load_nova,
        color="#feca57",
    )
    reg.register(
        "SPIKE",
        "Spiking Neural Network (LIF + STDP + R-STDP). "
        "Best for: temporal reasoning, tool calling.",
        load_spike,
        color="#ff6b6b",
    )
    reg.register(
        "HYBRID",
        "Orchestrator (NOVA + SPIKE). "
        "Best for: combining memory + temporal.",
        load_hybrid,
        color="#54a0ff",
    )

    return reg


# ---------------------------------------------------------------------- #
# FastAPI app
# ---------------------------------------------------------------------- #

app = FastAPI(title="NOVA-SPIKE-HYBRID App", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global registry
registry: ModelRegistry = None
metrics_clients: set[WebSocket] = set()


@app.on_event("startup")
async def startup():
    global registry
    print("=" * 60)
    print("  NOVA-SPIKE-HYBRID App — Starting up")
    print("=" * 60)
    registry = build_registry()
    print(f"  Registry: {len(registry.models)} models registered")
    print(f"  Models: {list(registry.models.keys())}")
    print(f"  Default enabled: Generative")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #

@app.get("/")
async def index():
    """Dashboard HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def get_status():
    """Get status of all models."""
    return {
        "models": registry.get_status(),
        "aggregate": registry.get_metrics_history(),
        "timestamp": time.time(),
    }


class ChatRequest(BaseModel):
    message: str
    model: str = "Generative"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with a specific model."""
    t0 = time.time()
    result = registry.chat(req.model, req.message)
    total_ms = (time.time() - t0) * 1000
    return {
        "input": req.message,
        "model": req.model,
        "response": result.get("response", ""),
        "error": result.get("error", ""),
        "time_ms": result.get("time_ms", 0),
        "total_ms": total_ms,
    }


class ChatCombinedRequest(BaseModel):
    message: str


@app.post("/api/chat_combined")
async def chat_combined(req: ChatCombinedRequest):
    """Combined mode: query all enabled models."""
    t0 = time.time()
    result = registry.chat_combined(req.message)
    total_ms = (time.time() - t0) * 1000
    return {
        "input": req.message,
        "responses": result["responses"],
        "primary": result["primary"],
        "n_models": result["n_models"],
        "total_ms": total_ms,
    }


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "Generative"
    max_tokens: int = 20


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Generate text with a specific model."""
    result = registry.generate(req.model, req.prompt, max_tokens=req.max_tokens)
    return {
        "prompt": req.prompt,
        "model": req.model,
        "response": result.get("response", ""),
        "error": result.get("error", ""),
        "time_ms": result.get("time_ms", 0),
    }


@app.post("/api/enable/{name}")
async def enable_model(name: str):
    """Enable a model (hot)."""
    ok = registry.enable(name)
    return {"ok": ok, "name": name, "action": "enable"}


@app.post("/api/disable/{name}")
async def disable_model(name: str):
    """Disable a model (hot)."""
    ok = registry.disable(name)
    return {"ok": ok, "name": name, "action": "disable"}


@app.post("/api/load/{name}")
async def load_model(name: str):
    """Hot-load a model."""
    t0 = time.time()
    ok = registry.load(name)
    dt = time.time() - t0
    return {"ok": ok, "name": name, "action": "load", "load_time_s": dt}


@app.post("/api/unload/{name}")
async def unload_model(name: str):
    """Unload a model (free memory)."""
    ok = registry.unload(name)
    return {"ok": ok, "name": name, "action": "unload"}


# ---------------------------------------------------------------------- #
# WebSocket — real-time metrics
# ---------------------------------------------------------------------- #

@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    """Stream metrics every 1 second."""
    await ws.accept()
    metrics_clients.add(ws)
    try:
        while True:
            status = {
                "models": registry.get_status(),
                "aggregate": registry.get_metrics_history(),
                "timestamp": time.time(),
            }
            await ws.send_json(status)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        metrics_clients.discard(ws)
    except Exception:
        metrics_clients.discard(ws)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    import uvicorn
    print("\n" + "=" * 60)
    print("  NOVA-SPIKE-HYBRID App — Dashboard")
    print("  http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
