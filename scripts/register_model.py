"""Lab 2, Task 4 — register a chosen run's model with lineage, and optionally promote it.

    python scripts/register_model.py --run-id <run id>                 # register only
    python scripts/register_model.py --version <n> --alias staging     # promote an existing version

Registration goes through the adapter (`register_model`), which copies the eight lineage
fields onto the model VERSION, not only the run, and refuses to register if any is missing.
Promotion is a separate, deliberate step: MLflow 3 uses aliases in place of stages.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="registered model name (default MODEL_REGISTRY_NAME)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="register runs:/<run-id>/model as a new version")
    group.add_argument("--version", help="an existing version to promote")
    ap.add_argument("--alias", default=None, help="promote the version to this alias, e.g. staging")
    args = ap.parse_args()

    cfg = config.load()
    name = args.name or cfg.model_registry_name
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    version = args.version
    if args.run_id:
        version = get_adapter(cfg).register_model(f"runs:/{args.run_id}/model", name)
        print(f"registered {name} version {version}")
        for key, value in sorted(client.get_model_version(name, version).tags.items()):
            print(f"  {key} = {value}")

    if args.alias:
        client.set_registered_model_alias(name, args.alias, version)
        print(f"promoted {name} version {version} -> alias '{args.alias}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
