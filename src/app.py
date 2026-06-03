"""
app.py — FastAPI inference + management server.

Endpoints:
    GET  /health        — liveness check
    GET  /model/info    — loaded model metadata
    POST /predict       — single or batch prediction
    POST /retrain       — trigger full training pipeline, hot-reload model
"""

import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from src.config_loader import load_config
from src.train import run_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model store (in-process singleton) ───────────────────────────────────────

class ModelStore:
    """Holds the loaded pipeline and its metadata."""

    def __init__(self):
        self.pipeline       = None
        self.model_path:  Path = None
        self.loaded_at:   float = None
        self.is_retraining: bool = False

    def load(self, path: Path) -> None:
        self.pipeline   = joblib.load(path)
        self.model_path = path
        self.loaded_at  = time.time()
        log.info("Model loaded from %s", path)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def model_name(self) -> str:
        if not self.ready:
            return "none"
        # SklearnPipeline step named "model"
        try:
            return type(self.pipeline.named_steps["model"]).__name__
        except Exception:
            return type(self.pipeline).__name__


store = ModelStore()


# ── Lifespan — load model on startup ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    if not model_path.exists():
        log.warning("No model found at %s — run /retrain first.", model_path)
    else:
        store.load(model_path)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MLOps AutoML API",
    description="Auto model comparison pipeline — predict, retrain, inspect.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Pass a list of records (dicts). Each dict is one row.

    Single row example:
        {"data": [{"feature_1": 5.1, "feature_2": "cat_a"}]}

    Batch example:
        {"data": [{"feature_1": 5.1}, {"feature_1": 3.2}]}
    """
    data: list[dict[str, Any]]

    @model_validator(mode="after")
    def must_not_be_empty(self):
        if not self.data:
            raise ValueError("'data' must contain at least one record.")
        return self


class PredictResponse(BaseModel):
    predictions: list[Any]
    model_name:  str
    record_count: int


class RetrainResponse(BaseModel):
    status:  str
    message: str


class HealthResponse(BaseModel):
    status:     str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_name:  str
    model_path:  str
    loaded_at:   float | None
    is_retraining: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness check — always returns 200 so load balancers know the pod is up."""
    return {"status": "ok", "model_ready": store.ready}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name":    store.model_name,
        "model_path":    str(store.model_path) if store.model_path else "none",
        "loaded_at":     store.loaded_at,
        "is_retraining": store.is_retraining,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """
    Run inference on one or more records.

    The pipeline handles preprocessing automatically — send raw feature
    values exactly as they appear in your training CSV.
    """
    if not store.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. POST /retrain to train one first.",
        )
    if store.is_retraining:
        raise HTTPException(
            status_code=503,
            detail="Retraining in progress. Try again in a moment.",
        )

    try:
        df = pd.DataFrame(request.data)
        predictions = store.pipeline.predict(df).tolist()
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predictions":  predictions,
        "model_name":   store.model_name,
        "record_count": len(predictions),
    }


def _retrain_job() -> None:
    """Blocking training job — runs in a background thread."""
    config = load_config()
    model_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    try:
        store.is_retraining = True
        log.info("Retraining started …")
        run_pipeline()                  # runs full train.py pipeline
        store.load(model_path)          # hot-reload the new artifact
        log.info("Retraining complete. New model: %s", store.model_name)
    except Exception:
        log.exception("Retraining failed")
    finally:
        store.is_retraining = False


@app.post("/retrain", response_model=RetrainResponse, tags=["ops"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger a full retrain in the background.

    Returns immediately with 202. Poll GET /model/info to check
    is_retraining — when it flips to false, the new model is live.
    """
    if store.is_retraining:
        raise HTTPException(
            status_code=409,
            detail="A retrain is already running.",
        )

    background_tasks.add_task(_retrain_job)
    return JSONResponse(
        status_code=202,
        content={
            "status":  "accepted",
            "message": "Retraining started. Poll GET /model/info until is_retraining is false.",
        },
    )