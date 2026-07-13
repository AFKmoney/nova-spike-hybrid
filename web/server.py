"""
SPIKE Web — serveur FastAPI + WebSocket pour visualisation temps réel.

Endpoints:
  GET  /                — dashboard HTML
  GET  /api/stats       — stats JSON du cerveau
  POST /api/chat        — envoie un message, retourne la réponse
  POST /api/learn       — apprentissage explicite
  POST /api/dream       — déclenche le mode rêve
  POST /api/reward      — applique une récompense R-STDP
  POST /api/reset       — reset le réseau
  WS   /ws/spikes       — stream temps réel des spikes
  WS   /ws/chat         — stream temps réel d'une conversation

Le dashboard se connecte aux WebSockets et affiche:
  - Raster plot des spikes (sensory/assoc/motor)
  - Compteur d'activité par population
  - Poids synaptiques (heatmap)
  - Log de conversation
  - Stats en temps réel
"""

from __future__ import annotations
import os
import sys
import json
import asyncio
import time
import numpy as np
from typing import Optional

# Ajoute le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spike import SpikeBrain, SpikeConfig
from nova import Nova, NovaConfig
from hybrid import HybridBrain, HybridConfig


# ---------------------------------------------------------------------- #
# Modèles de requête
# ---------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    message: str
    brain: str = "spike"  # "spike" | "nova" | "hybrid"


class LearnRequest(BaseModel):
    fact: str
    value: Optional[str] = None
    brain: str = "spike"


class DreamRequest(BaseModel):
    n_replays: int = 5
    ticks_per_replay: int = 20
    brain: str = "spike"


class RewardRequest(BaseModel):
    reward: float = 1.0


class ResetRequest(BaseModel):
    brain: str = "spike"


# ---------------------------------------------------------------------- #
# Cerveaux globaux (partagés entre les requêtes)
# ---------------------------------------------------------------------- #

class BrainManager:
    """Gère les instances des 3 cerveaux."""
    def __init__(self):
        print("Initialisation des cerveaux...")
        t0 = time.time()
        # SPIKE — config par défaut (rapide)
        self.spike = SpikeBrain(SpikeConfig(
            n_sensory=400, n_associative=1000, n_motor=400,
            sim_ticks=30, rstdp_enabled=True,
        ))
        # NOVA — config moyenne
        self.nova = Nova(NovaConfig(D=5000, sdm_locations=10000))
        # HYBRID — combine les deux
        self.hybrid = HybridBrain(HybridConfig(
            spike=SpikeConfig(n_sensory=300, n_associative=800, n_motor=300, sim_ticks=25),
            nova=NovaConfig(D=3000, sdm_locations=5000),
        ))
        print(f"Prêt en {time.time()-t0:.2f}s")

    def get(self, name: str):
        if name == "spike":
            return self.spike
        if name == "nova":
            return self.nova
        if name == "hybrid":
            return self.hybrid
        raise ValueError(f"Unknown brain: {name}")


brains: Optional[BrainManager] = None


# ---------------------------------------------------------------------- #
# FastAPI app
# ---------------------------------------------------------------------- #

app = FastAPI(title="SPIKE Web", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup():
    global brains
    brains = BrainManager()


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #

@app.get("/")
async def index():
    """Dashboard HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/stats")
async def get_stats():
    """Stats JSON de tous les cerveaux."""
    return {
        "spike": brains.spike.stats(),
        "nova": brains.nova.stats(),
        "hybrid": brains.hybrid.stats(),
        "time": time.time(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Envoie un message au cerveau choisi."""
    brain = brains.get(req.brain)
    t0 = time.time()
    response = brain.chat(req.message)
    t1 = time.time()
    return {
        "input": req.message,
        "response": response,
        "time_ms": (t1 - t0) * 1000,
        "brain": req.brain,
    }


@app.post("/api/learn")
async def learn(req: LearnRequest):
    """Apprentissage explicite."""
    brain = brains.get(req.brain)
    result = brain.learn(req.fact, req.value)
    return {"brain": req.brain, "result": result}


@app.post("/api/dream")
async def dream(req: DreamRequest):
    """Déclenche le mode rêve."""
    brain = brains.get(req.brain)
    if hasattr(brain, "dream"):
        result = brain.dream(req.n_replays, req.ticks_per_replay)
    else:
        result = {"error": "brain does not support dream"}
    return {"brain": req.brain, "result": result}


@app.post("/api/reward")
async def reward(req: RewardRequest):
    """Applique une récompense R-STDP (SPIKE seulement)."""
    if hasattr(brains.spike, "give_reward"):
        result = brains.spike.give_reward(req.reward)
    else:
        result = {"error": "no reward method"}
    return {"result": result}


@app.post("/api/reset")
async def reset(req: ResetRequest):
    """Reset le réseau."""
    brain = brains.get(req.brain)
    if hasattr(brain, "net"):
        brain.net.reset()
    elif hasattr(brain, "resonator"):
        brain.resonator.reset()
    return {"status": "reset", "brain": req.brain}


@app.get("/api/tools")
async def list_tools():
    """Liste les outils disponibles."""
    return {"tools": [t.name for t in brains.spike.agent.tools]}


# ---------------------------------------------------------------------- #
# WebSocket — stream des spikes en temps réel
# ---------------------------------------------------------------------- #

@app.websocket("/ws/spikes")
async def ws_spikes(ws: WebSocket):
    """
    Stream temps réel des spikes du cerveau SPIKE.

    Le client peut envoyer des messages pour:
      - {"cmd": "start", "input": "calcule 2+2"} — démarre une simulation
      - {"cmd": "stop"} — arrête
      - {"cmd": "status"} — demande l'état

    Le serveur envoie à chaque tick:
      - {"type": "tick", "t": 42, "sensory": [0,1,0,...],
         "assoc": [...], "motor": [...]}
      - {"type": "done", "response": "..."}
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_json({"error": "invalid json"})
                continue

            cmd = data.get("cmd")
            if cmd == "start":
                input_text = data.get("input", "")
                # Lance la simulation tick par tick
                brain = brains.spike
                brain.net.reset(soft=False)
                I_static = brain.coder.encode_text_to_current(input_text,
                                                                gain=brain.cfg.input_gain)
                # Patrons d'apprentissage
                import re
                learn_match = None
                for pat in [r"apprends?\s+(?:que\s+)?(.+)",
                            r"mémorise\s+(?:que\s+)?(.+)"]:
                    m = re.search(pat, input_text, re.IGNORECASE)
                    if m:
                        learn_match = m.group(1)
                        break
                if learn_match:
                    if " est " in learn_match:
                        k, v = learn_match.split(" est ", 1)
                        result = brain.learn(k.strip(), v.strip())
                    else:
                        result = brain.learn(learn_match.strip())
                    await ws.send_json({
                        "type": "learn",
                        "result": {k: v for k, v in result.items()
                                    if not isinstance(v, list)},
                    })
                    continue

                n_ticks = brain.cfg.sim_ticks
                for tick in range(n_ticks):
                    mask = (brain.rng.random(brain.cfg.n_sensory) < brain.cfg.poisson_rate).astype(np.float32)
                    I_tick = I_static * mask
                    brain.net.tick(I_tick)
                    if brain.cfg.stdp_enabled:
                        brain._apply_stdp()
                    # Envoie l'état (sous-échantillonné pour ne pas saturer)
                    sensory = brain.net.last_spikes["sensory"].astype(np.int8).tolist()
                    # Pour l'associative, on sous-échantillonne (trop grand sinon)
                    assoc = brain.net.last_spikes["associative"].astype(np.int8)
                    # On envoie seulement les premiers 200 neurones
                    assoc_sample = assoc[:200].tolist()
                    motor = brain.net.last_spikes["motor"].astype(np.int8).tolist()
                    # Poids moyens
                    w_sens = float(brain.net.syn_sens_to_assoc.W.data.mean()) if brain.net.syn_sens_to_assoc.W.nnz > 0 else 0
                    w_motor = float(brain.net.syn_assoc_to_motor.W.data.mean()) if brain.net.syn_assoc_to_motor.W.nnz > 0 else 0
                    w_direct = float(brain.syn_sens_to_motor.W.data.mean()) if (brain.syn_sens_to_motor and brain.syn_sens_to_motor.W.nnz > 0) else 0
                    await ws.send_json({
                        "type": "tick",
                        "t": tick,
                        "sensory": sensory,
                        "assoc": assoc_sample,
                        "motor": motor,
                        "counts": {
                            "sensory": int(brain.net.last_spikes["sensory"].sum()),
                            "assoc": int(brain.net.last_spikes["associative"].sum()),
                            "motor": int(brain.net.last_spikes["motor"].sum()),
                        },
                        "weights": {
                            "sens_assoc": w_sens,
                            "assoc_motor": w_motor,
                            "sens_motor_direct": w_direct,
                        },
                    })
                    await asyncio.sleep(0.02)  # 50 fps max

                # Réponse finale
                brain.n_calls += 1
                response = brain.chat(input_text)
                await ws.send_json({
                    "type": "done",
                    "response": response,
                    "stats": brain.stats(),
                })
            elif cmd == "stop":
                await ws.send_json({"type": "stopped"})
            elif cmd == "status":
                await ws.send_json({
                    "type": "status",
                    "stats": brains.spike.stats(),
                })
            else:
                await ws.send_json({"error": f"unknown cmd: {cmd}"})
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------- #
# WebSocket — chat stream
# ---------------------------------------------------------------------- #

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """Chat bidirectionnel — envoie le texte, reçoit la réponse en stream."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            input_text = data.get("input", "")
            brain_name = data.get("brain", "spike")
            brain = brains.get(brain_name)
            t0 = time.time()
            response = brain.chat(input_text)
            t1 = time.time()
            await ws.send_json({
                "input": input_text,
                "response": response,
                "time_ms": (t1 - t0) * 1000,
                "brain": brain_name,
            })
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    import uvicorn
    print("\n" + "=" * 60)
    print("  SPIKE WEB — Dashboard temps réel")
    print("=" * 60)
    print("  http://localhost:4141")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=4141, log_level="info")


if __name__ == "__main__":
    main()
