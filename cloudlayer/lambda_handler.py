"""The serving logic of service/app.py, as a Lambda function handler. Lab 5 Task 4.

The same request schema (service/schemas.py), the same feature list (src/data.py) and the same
model file; only the transport differs. FastAPI receives an HTTP request; a Lambda function
receives an event dict and returns one. That difference is the whole content of this file,
which is why it lives in cloudlayer/ and not in service/.

The response mirrors an HTTP response (statusCode, headers, body) so the same handler would sit
unchanged behind a function URL or API Gateway.
"""
from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from pydantic import ValidationError

from service.schemas import BatchRequest, PredictRequest
from src.data import FEATURES

MODEL = joblib.load(os.environ.get("MODEL_PATH", "/var/task/model.joblib"))  # once, at cold start
VERSION = os.environ.get("MODEL_VERSION", "unknown")


def _reply(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"x-model-version": VERSION,
                                              "content-type": "application/json"},
            "body": json.dumps(body)}


def handler(event: dict, context) -> dict:
    try:
        payload = json.loads(event.get("body") or "{}")
        if "rows" in payload:
            rows = [r.model_dump() for r in BatchRequest(**payload).rows]
        else:
            rows = [PredictRequest(**payload).model_dump()]
    except (ValidationError, ValueError, TypeError) as exc:
        return _reply(422, {"detail": str(exc)[:500]})
    scores = [float(p) for p in MODEL.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]]
    if "rows" in payload:
        return _reply(200, {"probabilities": scores, "model_version": VERSION})
    return _reply(200, {"probability": scores[0], "model_version": VERSION})
