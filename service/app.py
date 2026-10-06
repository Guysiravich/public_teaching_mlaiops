"""Lab 3 — inference service.

Provider-neutral by construction: the model arrives through the adapter, and the same
container image deploys to SageMaker, Azure ML, or Vertex AI. Route paths differ per
platform; that difference belongs in cloudlayer/, never here.

Run locally:  uvicorn service.app:app --port 8080
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
log = logging.getLogger("service")

STATE: dict[str, Any] = {"model": None, "version": os.environ.get("MODEL_VERSION", "unknown")}

# --- Lab 4: metrics -----------------------------------------------------------------
# Exposed at /metrics in Prometheus text format and scraped by monitoring/compose.yaml.
# The names match the queries in monitoring/dashboard.json; a metric emitted under one
# name and charted under another is the "dashboard shows nothing" failure in the handout.
REQUESTS = Counter("http_requests_total", "Requests served", ["path", "status_class"])
LATENCY = Histogram("request_latency_ms", "Request latency in milliseconds", ["path"],
                    buckets=(5, 10, 25, 50, 100, 200, 300, 500, 1000, 2500, 5000, 10000))
MODEL_VERSION_INFO = Gauge("model_version_info", "1 for the model version this replica serves",
                           ["version"])
FEATURE_MEAN = Gauge("feature_rolling_mean", "Mean of a feature over the last inputs",
                     ["feature"])
FEATURE_STD = Gauge("feature_rolling_std", "Standard deviation of a feature over the last inputs",
                    ["feature"])
ROLLING_INPUTS = Gauge("feature_rolling_window_rows", "Inputs currently in the rolling window")

# 500 rows: the smallest window whose sampling noise stays well under the drift threshold
# (reports/lab4/threshold-study.md). A shorter window charts normal variance as drift.
ROLLING_WINDOW = int(os.environ.get("ROLLING_WINDOW", "500"))
RECENT: deque[dict[str, float]] = deque(maxlen=ROLLING_WINDOW)


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
    path = request.url.path
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled error never reaches the code below; count it here or the 5xx
        # panel reads zero during exactly the incident it exists for.
        REQUESTS.labels(path, "5xx").inc()
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    if path != "/metrics":  # the scraper's own requests are not traffic
        REQUESTS.labels(path, f"{response.status_code // 100}xx").inc()
        LATENCY.labels(path).observe(latency_ms)
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
    scores = [float(p) for p in STATE["model"].predict_proba(frame)[:, 1]]
    for row in rows:
        RECENT.append(row)
        # One line per scored input: the record the scheduled drift job reads back from
        # the platform's log store (monitoring/drift_job.py). Inputs only — no identifiers.
        log.info('{"event":"input","features":%s}', json.dumps({f: row[f] for f in FEATURES}))
    return scores


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape target. The rolling feature statistics are computed here, at scrape
    time, so the request path pays for an append and nothing else."""
    from src.data import FEATURES

    MODEL_VERSION_INFO.clear()
    MODEL_VERSION_INFO.labels(str(STATE["version"])).set(1)
    window = list(RECENT)
    ROLLING_INPUTS.set(len(window))
    if window:
        import numpy as np

        for feature in FEATURES:
            values = np.fromiter((row[feature] for row in window), dtype=float)
            FEATURE_MEAN.labels(feature).set(float(values.mean()))
            FEATURE_STD.labels(feature).set(float(values.std()))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    score = _score([payload.model_dump()])[0]
    return PredictResponse(probability=score, model_version=str(STATE["version"]))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    scores = _score([row.model_dump() for row in payload.rows])
    return BatchResponse(probabilities=scores, model_version=str(STATE["version"]))
