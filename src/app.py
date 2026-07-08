"""
app.py — FastAPI inference + management server.

Endpoints:
    GET  /health        — liveness check
    GET  /model/info    — loaded model metadata
    GET  /metrics       — Prometheus metrics
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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, model_validator

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.config_loader import load_config
from src.train import run_pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Prometheus metrics ────────────────────────────────────────────────────────

# Total prediction requests (labelled by model name)
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total number of prediction requests",
    ["model_name", "status"],  # status: success | error
)

# Prediction latency in seconds
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Time taken to run a prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Total HTTP requests by endpoint and method
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# Number of times model was retrained
RETRAIN_TOTAL = Counter(
    "model_retrain_total",
    "Total number of model retrains triggered",
)

# Is model currently retraining (1 = yes, 0 = no)
RETRAINING_IN_PROGRESS = Gauge(
    "model_retraining_in_progress",
    "1 if model is currently retraining, 0 otherwise",
)

# Is model loaded and ready (1 = yes, 0 = no)
MODEL_READY = Gauge(
    "model_ready",
    "1 if model is loaded and ready for predictions",
)


# ── Model store ───────────────────────────────────────────────────────────────


class ModelStore:
    """Holds the loaded pipeline and its metadata."""

    def __init__(self):
        self.pipeline = None
        self.model_path: Path = None
        self.loaded_at: float = None
        self.is_retraining: bool = False

    def load(self, path: Path) -> None:
        self.pipeline = joblib.load(path)
        self.model_path = path
        self.loaded_at = time.time()
        MODEL_READY.set(1)
        log.info("Model loaded from %s", path)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def model_name(self) -> str:
        if not self.ready:
            return "none"
        try:
            return type(self.pipeline.named_steps["model"]).__name__
        except Exception:
            return type(self.pipeline).__name__


store = ModelStore()


# ── Middleware — track every request ──────────────────────────────────────────


async def metrics_middleware(request: Request, call_next):
    """Count every HTTP request with method, path, and status code."""
    start = time.time()
    response = await call_next(request)
    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    log.debug(
        "%.3fs %s %s %s",
        time.time() - start,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    if not model_path.exists():
        log.warning("No model found at %s — run /retrain first.", model_path)
        MODEL_READY.set(0)
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

app.middleware("http")(metrics_middleware)


# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """
    Pass a list of records (dicts). Each dict is one row.

    Single row:  {"data": [{"Pclass": 3, "Age": 22, "Sex": "male", ...}]}
    Batch:       {"data": [{"Pclass": 3, ...}, {"Pclass": 1, ...}]}
    """

    data: list[dict[str, Any]]

    @model_validator(mode="after")
    def must_not_be_empty(self):
        if not self.data:
            raise ValueError("'data' must contain at least one record.")
        return self


class PredictResponse(BaseModel):
    predictions: list[Any]
    model_name: str
    record_count: int


class RetrainResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    loaded_at: float | None
    is_retraining: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness check — always returns 200."""
    return {"status": "ok", "model_ready": store.ready}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": store.model_name,
        "model_path": str(store.model_path) if store.model_path else "none",
        "loaded_at": store.loaded_at,
        "is_retraining": store.is_retraining,
    }


@app.get("/metrics", tags=["ops"])
def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus scrapes this every 15s to collect:
      - predict_requests_total
      - predict_latency_seconds
      - http_requests_total
      - model_retrain_total
      - model_retraining_in_progress
      - model_ready
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """
    Run inference on one or more records.
    Send raw feature values — pipeline handles preprocessing automatically.
    """
    if not store.ready:
        PREDICT_REQUESTS.labels(model_name="none", status="error").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. POST /retrain to train one first.",
        )
    if store.is_retraining:
        raise HTTPException(
            status_code=503,
            detail="Retraining in progress. Try again in a moment.",
        )

    start = time.time()
    try:
        df = pd.DataFrame(request.data)
        predictions = store.pipeline.predict(df).tolist()

        # Record metrics
        PREDICT_LATENCY.observe(time.time() - start)
        PREDICT_REQUESTS.labels(model_name=store.model_name, status="success").inc()

    except Exception as exc:
        PREDICT_REQUESTS.labels(model_name=store.model_name, status="error").inc()
        log.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predictions": predictions,
        "model_name": store.model_name,
        "record_count": len(predictions),
    }


def _retrain_job() -> None:
    """Blocking training job — runs in a background thread."""
    config = load_config()
    model_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    try:
        store.is_retraining = True
        RETRAINING_IN_PROGRESS.set(1)
        RETRAIN_TOTAL.inc()
        log.info("Retraining started ...")
        run_pipeline()
        store.load(model_path)
        log.info("Retraining complete. New model: %s", store.model_name)
    except Exception:
        log.exception("Retraining failed")
    finally:
        store.is_retraining = False
        RETRAINING_IN_PROGRESS.set(0)


@app.post("/retrain", response_model=RetrainResponse, tags=["ops"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger a full retrain in the background.
    Returns 202 immediately. Poll /model/info until is_retraining is false.
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
            "status": "accepted",
            "message": "Retraining started. Poll GET /model/info until is_retraining is false.",
        },
    )
