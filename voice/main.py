"""Local-only openWakeWord sidecar for ATLAS's custom wake word."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from openwakeword.model import Model

WAKE_WORD = os.getenv("ATLAS_WAKE_WORD", "ATLAS").strip() or "ATLAS"
MODEL_PATH = Path(os.getenv("ATLAS_WAKE_WORD_MODEL_PATH", "/models/atlas.onnx"))
THRESHOLD = float(os.getenv("ATLAS_WAKE_WORD_THRESHOLD", "0.5"))
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # openWakeWord's efficient 80 ms inference frame.
COOLDOWN_SECONDS = 3.0

app = FastAPI(title="ATLAS Voice", version="0.1.0")


def _model_available() -> bool:
    return MODEL_PATH.is_file() and MODEL_PATH.suffix.lower() == ".onnx"


def _new_detector() -> Model:
    """Load only the supplied ATLAS model; bundled wake words are never used."""
    return Model(wakeword_models=[str(MODEL_PATH)], inference_framework="onnx")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ready" if _model_available() else "waiting_for_model",
            "wake_word": WAKE_WORD,
            "model_available": _model_available(),
            "sample_rate": SAMPLE_RATE,
        }
    )


@app.websocket("/wake")
async def wake_listener(websocket: WebSocket) -> None:
    await websocket.accept()
    if not _model_available():
        await websocket.send_json(
            {
                "type": "unavailable",
                "detail": "The custom ATLAS ONNX wake-word model is not mounted at /models/atlas.onnx.",
            }
        )
        await websocket.close(code=1013)
        return
    try:
        detector = await asyncio.to_thread(_new_detector)
    except Exception:
        await websocket.send_json(
            {"type": "unavailable", "detail": "ATLAS could not load the custom wake-word model."}
        )
        await websocket.close(code=1011)
        return

    await websocket.send_json({"type": "ready", "wake_word": WAKE_WORD, "sample_rate": SAMPLE_RATE})
    buffered = bytearray()
    last_activation = 0.0
    try:
        while True:
            buffered.extend(await websocket.receive_bytes())
            while len(buffered) >= FRAME_SAMPLES * 2:
                frame = bytes(buffered[: FRAME_SAMPLES * 2])
                del buffered[: FRAME_SAMPLES * 2]
                scores = await asyncio.to_thread(
                    detector.predict, np.frombuffer(frame, dtype=np.int16)
                )
                score = max((float(value) for value in scores.values()), default=0.0)
                now = time.monotonic()
                if score >= THRESHOLD and now - last_activation >= COOLDOWN_SECONDS:
                    last_activation = now
                    await websocket.send_json(
                        {"type": "wake_detected", "wake_word": WAKE_WORD, "score": round(score, 3)}
                    )
    except WebSocketDisconnect:
        return
