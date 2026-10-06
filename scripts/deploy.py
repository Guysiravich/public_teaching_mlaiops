"""Lab 3 — build, push and deploy the serving image, then smoke-test it.

    make deploy VERSION=1            # build, push, deploy that registered version
    make smoke                       # three known payloads against the live endpoint

Provider-neutral: every cloud call goes through the adapter.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config

SMOKE_ROWS = [
    {"temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
     "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0},
    {"temp_c": 95.0, "vibration_mm_s": 7.5, "pressure_kpa": 290.0,
     "hours_since_service": 9500.0, "load_pct": 95.0, "ambient_humidity": 80.0},
    {"temp_c": 62.0, "vibration_mm_s": 1.2, "pressure_kpa": 330.0,
     "hours_since_service": 100.0, "load_pct": 20.0, "ambient_humidity": 35.0},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="registered model version to serve")
    ap.add_argument("--alias", help="serve the version this registry alias points to now "
                                    "(Lab 4 CD: --alias staging)")
    ap.add_argument("--endpoint-name", default="itcs355-predict")
    ap.add_argument("--instance", default="0.5/1.0Gi", help="<cpu>/<memory> for Container Apps")
    ap.add_argument("--image", help="local serving image tag to push (default: skip the push)")
    ap.add_argument("--endpoint-file", default="reports/lab3-endpoint.txt")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    cfg = config.load()
    adapter = get_adapter(cfg)

    version = args.version
    if args.alias:
        # Resolve once, here, and deploy the number. The alias says which version is
        # approved; the revision must say which version is running, and keep saying it
        # after a restart even if someone moves the alias in between.
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        version = MlflowClient().get_model_version_by_alias(cfg.model_registry_name,
                                                            args.alias).version
        print(f"alias {args.alias} -> version {version}")

    if args.smoke_only:
        endpoint = os.environ["ENDPOINT_URL"]
    else:
        if args.image:
            print("pushing serving image ...")
            image_ref = adapter.push_image(args.image)
            print("  ", image_ref)
            os.environ["SERVE_IMAGE"] = image_ref
        print("deploying ...")
        endpoint = adapter.deploy(f"models:/{cfg.model_registry_name}/{version}",
                                  args.endpoint_name, args.instance)
        print("endpoint:", endpoint)
        Path(args.endpoint_file).write_text(endpoint + "\n")

    wait_ready(endpoint)
    print("smoke test:")
    for row in SMOKE_ROWS:
        result = adapter.invoke(endpoint, row)
        print(f"  p(failure)={result['probability']:.4f}  model_version={result['model_version']}")
        check_version(result, version)
    batch = adapter.invoke(endpoint, {"rows": SMOKE_ROWS})
    print(f"  batch of {len(batch['probabilities'])} rows, version {batch['model_version']}")
    check_version(batch, version)
    return 0


def wait_ready(endpoint: str, timeout_s: int = 300) -> None:
    """A scale-to-zero endpoint takes ~35 s to answer its first request (Lab 3), longer
    than one invoke() waits. Poll readiness first so the smoke test tests the model, not
    the cold start."""
    import time
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/ready", timeout=60) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.time() > deadline:
            raise SystemExit(f"{endpoint} not ready after {timeout_s} s")
        time.sleep(5)


def check_version(result: dict, version: str | None) -> None:
    """A smoke test that accepts any version would pass while the old revision answers."""
    if version is None:
        return
    for seen in (result["model_version"], result.get("model_version_header", version)):
        if str(seen) != str(version):
            raise SystemExit(f"smoke test: expected model version {version}, got {seen}")


if __name__ == "__main__":
    raise SystemExit(main())
