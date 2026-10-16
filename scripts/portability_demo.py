"""Lab 5 Task 4 Part B — the swap, demonstrated end to end on the second provider.

    make swap-demo        # starts the emulator, provisions, runs this, writes reports/lab5/

Run with the AWS environment loaded (portability/aws.env): CLOUD_PROVIDER=aws and an s3://
BLOB_URI. Everything below goes through get_adapter(config.load()) — the same call the rest of
the repository makes — so the only thing that changed is configuration:

  upload    the training data, to S3
  download  it back, byte-identical
  invoke    the serving function, with the Lab 3 smoke rows, single and batch, compared
            number for number with the same model scored locally; then a malformed request,
            which must come back as the same 422 the Azure endpoint gives
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config, data  # noqa: E402

BUILD = ROOT / "build" / "lambda"
MODEL = ROOT / "reports" / "model.joblib"
FUNCTION = "itcs355-6688067"  # = PROJECT_ID, which is what the course's swap check invokes
SMOKE_ROWS = [
    {"temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
     "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0},
    {"temp_c": 95.0, "vibration_mm_s": 7.5, "pressure_kpa": 290.0,
     "hours_since_service": 9500.0, "load_pct": 95.0, "ambient_humidity": 80.0},
    {"temp_c": 62.0, "vibration_mm_s": 1.2, "pressure_kpa": 330.0,
     "hours_since_service": 100.0, "load_pct": 20.0, "ambient_humidity": 35.0},
]
DEPS = ["scikit-learn==1.8.0", "pandas==3.0.2", "numpy==2.4.4", "joblib==1.4.2",
        "pydantic==2.13.5", "PyYAML==6.0.3"]


def build_bundle() -> Path:
    """Function code: the handler, the code it imports, the model, and Linux cp312 wheels."""
    if not (BUILD / "sklearn").exists():
        BUILD.mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "run", "--rm", "-v", f"{BUILD}:/out", "python:3.12-slim",
                        "pip", "install", "-q", "--only-binary", ":all:", "--target", "/out",
                        *DEPS], check=True)
    for package in ("src", "service"):
        shutil.copytree(ROOT / package, BUILD / package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (BUILD / "cloudlayer").mkdir(exist_ok=True)
    (BUILD / "cloudlayer" / "__init__.py").write_text("")
    shutil.copy(ROOT / "cloudlayer" / "lambda_handler.py", BUILD / "cloudlayer")
    shutil.copy(MODEL, BUILD / "model.joblib")
    return BUILD


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cfg = config.load(strict=False)
    if cfg.provider != "aws":
        raise SystemExit("load portability/aws.env first: CLOUD_PROVIDER must be aws")
    lines = [f"# Lab 5 Task 4 — swap demonstration, {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
             "", f"CLOUD_PROVIDER={cfg.provider}  BLOB_URI={cfg.blob_uri}  "
             f"AWS_ENDPOINT_URL={os.environ.get('AWS_ENDPOINT_URL', '(real AWS)')}", ""]

    def log(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    from cloudlayer.aws_demo import provision
    provision(cfg.blob_uri, FUNCTION, build_bundle(), model_version=f"local-{sha256(MODEL)[:8]}")
    adapter = get_adapter(cfg)
    log(f"adapter: {type(adapter).__name__}")

    source = ROOT / "data" / "raw" / "sensors.csv"
    uri = adapter.upload(str(source), "data/raw/sensors.csv")
    log(f"upload    {source.name} ({source.stat().st_size:,} bytes) -> {uri}")
    with tempfile.TemporaryDirectory() as tmp:
        back = Path(tmp) / "nested" / "sensors.csv"
        adapter.download(uri, str(back))
        same = sha256(back) == sha256(source)
        log(f"download  {uri} -> {back.name}, sha256 identical: {same}")

    import joblib
    import pandas as pd
    local = joblib.load(MODEL).predict_proba(pd.DataFrame(SMOKE_ROWS)[data.FEATURES])[:, 1]
    started = time.perf_counter()
    first = adapter.invoke(FUNCTION, SMOKE_ROWS[0])
    log(f"invoke    single row: p={first['probability']:.6f} version={first['model_version']} "
        f"header={first['model_version_header']} (first call {time.perf_counter() - started:.1f} s, "
        "cold start)")
    started = time.perf_counter()
    batch = adapter.invoke(FUNCTION, {"rows": SMOKE_ROWS})
    log(f"invoke    batch of 3: {[round(p, 6) for p in batch['probabilities']]} "
        f"({(time.perf_counter() - started) * 1000:.0f} ms warm)")
    diff = max(abs(a - b) for a, b in zip(batch["probabilities"], local))
    log(f"          same model scored locally: {[round(float(p), 6) for p in local]}  "
        f"max difference {diff:.2e}")
    try:
        adapter.invoke(FUNCTION, {**SMOKE_ROWS[0], "load_pct": 250})
        rejected = False
    except RuntimeError as exc:
        rejected = "422" in str(exc)
        log(f"invoke    out-of-range load_pct -> {str(exc)[:90]}...")
    ok = same and diff < 1e-9 and first["probability"] == batch["probabilities"][0] and rejected
    log("")
    log("SWAP DEMONSTRATED: upload, download and invoke through the AWS adapter, same model, "
        "same answers." if ok else "SWAP NOT DEMONSTRATED — see above")
    out = ROOT / "reports" / "lab5" / "portability-demo.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
