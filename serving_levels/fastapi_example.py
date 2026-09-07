from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

REGISTRY_URI = "models:/RideDurationModel@production"
PICKLE_PATH = Path(__file__).resolve().parent.parent / "models" / "rf_model.pkl"


class PredictRequest(BaseModel):
    distance_km: float = Field(gt=0, le=500)
    passengers: int = Field(ge=1, le=8)


class PredictResponse(BaseModel):
    duration_min: float
    eta_band: str
    model_version: str
    batch_size: int


_lock = threading.Lock()

_requests = 0
_batched_items = 0
_batches = 0
_predict_s = 0.0


def _band(minutes: float) -> str:
    lo = int(minutes // 10) * 10
    return f"{lo}-{lo + 10} min"


def _load_model():
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow.sklearn

            model = mlflow.sklearn.load_model(REGISTRY_URI)
            return model, "mlflow:production"
        except Exception:
            pass

    return joblib.load(PICKLE_PATH), f"pickle:{PICKLE_PATH.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, version = _load_model()

    app.state.model = model
    app.state.version = version

    # Warm-up
    app.state.model.predict(np.array([[5.0, 1.0]]))

    yield


app = FastAPI(
    title="Ride Duration API — baseline",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    global _requests, _batched_items, _batches, _predict_s

    features = np.array(
        [[req.distance_km, req.passengers]],
        dtype=float,
    )

    t0 = time.perf_counter()

    pred = float(app.state.model.predict(features)[0])

    dt = time.perf_counter() - t0

    with _lock:
        _requests += 1
        _batches += 1
        _batched_items += 1
        _predict_s += dt

    return PredictResponse(
        duration_min=round(pred, 2),
        eta_band=_band(pred),
        model_version=app.state.version,
        batch_size=1,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": app.state.version,
    }


@app.get("/metrics")
def metrics():
    with _lock:
        r = _requests
        b = _batches
        i = _batched_items
        s = _predict_s

    return {
        "requests": r,
        "avg_batch_size": round(i / max(b, 1), 2),
        "avg_predict_ms": round(
            s / max(r, 1) * 1000,
            3,
        ),
    }
