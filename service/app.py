"""Lab 3 — inference service.

Provider-neutral by construction: the model arrives through the adapter, and the same
container image deploys to SageMaker, Azure ML, or Vertex AI. Route paths differ per
platform; that difference belongs in cloudlayer/, never here.

Run locally:  uvicorn service.app:app --port 8080
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
log = logging.getLogger("service")

STATE: dict[str, Any] = {"model": None, "version": os.environ.get("MODEL_VERSION", "unknown")}


def _load_model():
    """Load once, at startup. Never per request.

    Loading per request is the commonest cause of a p99 that looks nothing like p50, and
    it is the first thing to check when your latency distribution has a long tail.
    """
    name = os.environ.get("MODEL_REGISTRY_NAME")
    version = os.environ.get("MODEL_VERSION")
    if name and version:
        import mlflow.sklearn  # imported lazily so tests can run without a registry

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        uri = f"models:/{name}/{version}"
        try:
            return mlflow.sklearn.load_model(uri)
        except Exception as exc:
            if "Untrusted types" not in str(exc):
                raise
            # The serialization risk this lab is about, met in practice. MLflow 3 stores
            # scikit-learn models with skops, and skops refuses to rebuild
            # sklearn.tree._tree.Tree unless the caller names it: the type holds raw node
            # indices that scikit-learn reads without bounds checking, so a hostile file
            # could crash the process. The list belongs in the MLmodel file, written at
            # log time by log_model(skops_trusted_types=[...]) — this model was registered
            # before that argument was added upstream, so the reader has to supply it.
            # Trusting it here is a statement about provenance: this version's lineage tags
            # name the commit, the data version and the training job that produced it.
            import skops.io

            local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
            model_file = next(Path(local).rglob("model.skops"))
            log.warning('"loading %s with skops, trusting sklearn.tree._tree.Tree"', uri)
            return skops.io.load(model_file, trusted=["sklearn.tree._tree.Tree"])

    # Fallback for local development and tests only. Submitting this is not acceptable:
    # your deployed service must load a registered version.
    import joblib

    path = Path(os.environ.get("MODEL_PATH", "reports/model.joblib"))
    if not path.exists():
        raise RuntimeError(
            "No model available. Set MODEL_REGISTRY_NAME and MODEL_VERSION, or MODEL_PATH."
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = _load_model()
        log.info('"model loaded, version=%s"', STATE["version"])
    except Exception as exc:  # readiness stays false; liveness still passes
        STATE["model"] = None
        log.error('"model load failed: %s"', exc)
    yield
    STATE["model"] = None


app = FastAPI(title="ITCS355 inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-model-version"] = str(STATE["version"])
    log.info(
        '{"request_id":"%s","path":"%s","status":%d,"latency_ms":%.2f,"model_version":"%s"}',
        request_id, request.url.path, response.status_code, latency_ms, STATE["version"],
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is up. Says nothing about whether it can serve."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness. The model is loaded and can score.

    These two are genuinely different, and confusing them causes a specific production
    failure: traffic routed to a container whose model has not finished loading. All three
    providers distinguish them, and Quiz 3 asks about it.
    """
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "model not loaded"})
    return {"status": "ready", "model_version": STATE["version"]}


def _score(rows: list[dict]) -> list[float]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import pandas as pd

    from src.data import FEATURES

    frame = pd.DataFrame(rows)[FEATURES]
    return [float(p) for p in STATE["model"].predict_proba(frame)[:, 1]]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    score = _score([payload.model_dump()])[0]
    return PredictResponse(probability=score, model_version=str(STATE["version"]))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    scores = _score([row.model_dump() for row in payload.rows])
    return BatchResponse(probabilities=scores, model_version=str(STATE["version"]))
