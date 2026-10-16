"""Lab 5 Task 1 — one entry point for every step of pipeline/pipeline.yaml.

    python scripts/pipeline_step.py ingest   --out <dir>
    python scripts/pipeline_step.py validate --data <dir>
    python scripts/pipeline_step.py train    --data <dir> --out <dir>
    python scripts/pipeline_step.py evaluate --metrics <dir> --out <dir>
    python scripts/pipeline_step.py register --metrics <dir> --decision <dir> --out <dir>
    python scripts/pipeline_step.py deploy   --registered <dir>

The step logic lives in the image, as the handout asks; the pipeline definition only wires
directories between steps. Each step exits non-zero when it fails, and Azure ML does not run
a step whose upstream failed — which is how "on_failure: abort" and the register condition
are enforced (cloudlayer/pipelines.py explains the mapping).

Credentials: none in the job's environment. The tracking server's username and password are
read from Key Vault through the adapter, as the cluster's own identity, at the start of the
steps that need them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402

RAW = ROOT / "data" / "raw" / "sensors.csv"


def tracking_credentials() -> None:
    """Put the tracking server's credentials into this process's environment, from the
    secret store, unless they are already there (a laptop run)."""
    if os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return
    adapter = get_adapter(config.load(strict=False))
    os.environ["MLFLOW_TRACKING_USERNAME"] = adapter.secret("mlflow-username")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = adapter.secret("mlflow-password")


def stage_data(data_dir: Path) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "sensors.csv", RAW)


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ingest", "validate", "train", "evaluate", "register",
                                     "deploy"])
    for name in ("out", "data", "metrics", "decision", "registered", "after"):
        ap.add_argument(f"--{name}", type=Path)  # --after: an upstream folder, for ordering only
    args = ap.parse_args()

    # Azure ML overwrites MLFLOW_TRACKING_URI with its own store in every job (Lab 2); the
    # pipeline passes ours under another name.
    if os.environ.get("ITCS355_MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = os.environ["ITCS355_MLFLOW_TRACKING_URI"]
        for key in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
                    "MLFLOW_TRACKING_TOKEN"):
            os.environ.pop(key, None)

    if args.step == "ingest":
        args.out.mkdir(parents=True, exist_ok=True)
        run("python", "scripts/make_dataset.py", "--out", str(args.out / "sensors.csv"))

    elif args.step == "validate":
        stage_data(args.data)
        run("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_data.py")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "contract.txt").write_text("tests/test_data.py passed\n")

    elif args.step == "train":
        stage_data(args.data)
        tracking_credentials()
        os.environ.setdefault("TRAINING_JOB_ID", os.environ.get("AZUREML_RUN_ID", "local"))
        run("python", "-m", "src.train", "--experiment", "itcs355-lab5-pipeline",
            "--metrics-out", str(args.out / "metrics.json"))

    elif args.step == "evaluate":
        tracking_credentials()
        forced = ["--incumbent", os.environ["GATE_INCUMBENT"]] if os.environ.get("GATE_INCUMBENT") else []
        run("python", "scripts/evaluation_gate.py", "--metrics", str(args.metrics / "metrics.json"),
            "--decision-out", str(args.out / "decision.json"), *forced)

    elif args.step == "register":
        decision = json.loads((args.decision / "decision.json").read_text())
        if not decision["pass"]:  # unreachable while evaluate exits non-zero; kept as a guard
            print("gate did not pass: not registering")
            return 1
        tracking_credentials()
        run_id = json.loads((args.metrics / "metrics.json").read_text())["run_id"]
        cfg = config.load(strict=False)
        version = get_adapter(cfg).register_model(f"runs:/{run_id}/model", cfg.model_registry_name)
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.MlflowClient().set_registered_model_alias(cfg.model_registry_name, "staging",
                                                         version)
        print(f"registered {cfg.model_registry_name} version {version}; alias staging -> {version}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "registered.json").write_text(json.dumps({"version": version,
                                                              "run_id": run_id}))

    elif args.step == "deploy":
        registered = json.loads((args.registered / "registered.json").read_text())
        cfg = config.load(strict=False)
        app = os.environ.get("STAGING_APP", "itcs355-staging")
        revision = get_adapter(cfg).rollout(app, registered["version"])
        print(f"staging app {app}: new revision {revision} serving version {registered['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
