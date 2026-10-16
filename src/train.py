"""Training entry point.

Run locally:      python -m src.train --n-estimators 200 --max-depth 8
Run in Docker:    make reproduce

Every run logs: all hyperparameters, the seed, validation AND test metrics separately,
the data fingerprint, and the Git commit. A metric that cannot be traced to code and
data is not evidence of anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, data, seeds


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        # Inside a managed job the image has no .git (see .dockerignore); the submitter
        # passes the commit it built from.
        return os.environ.get("GIT_COMMIT", "unknown")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 1 — reproducible training")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    p.add_argument("--experiment", default="itcs355-lab1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Write final metrics as JSON. Used by `make verify`.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config.load(strict=False)
    seed = seeds.set_all(args.seed)

    df = data.load_raw(cfg.raw_path)
    fingerprint = data.data_fingerprint(cfg.raw_path)
    dvc_md5 = data.dvc_hash(cfg.data_dir / "raw.dvc")
    train_df, val_df, test_df = data.split(df, seed=seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "seed": seed,
            "n_features": len(data.FEATURES),
        })
        # Provenance. This is what makes the metric traceable.
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_fingerprint": fingerprint,
            "data_dvc_md5": dvc_md5,
            "split_strategy": "group_by_machine_id",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_test_rows": len(test_df),
            # The three lineage fields register_model requires, as src/tune.py sets them. In the
            # Lab 5 pipeline the data comes from the ingest step, not DVC, so its version is
            # the content fingerprint of the file this run actually read.
            "data_version": dvc_md5 if dvc_md5 not in ("", "untracked") else f"sha256:{fingerprint}",
            "training_job_id": os.environ.get("TRAINING_JOB_ID", "local"),
            "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
        })

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])

        metrics: dict[str, float] = {}
        for name, part in (("val", val_df), ("test", test_df)):
            proba = model.predict_proba(part[data.FEATURES])[:, 1]
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(part[data.TARGET], proba))
            metrics[f"{name}_pr_auc"] = float(average_precision_score(part[data.TARGET], proba))
        mlflow.log_metrics(metrics)
        # skops 0.15 refuses to serialise sklearn tree models unless the caller names
        # the types it trusts. sklearn.tree._tree.Tree stores raw node indices that
        # scikit-learn indexes without bounds checking, so a malicious file can segfault
        # the process on .predict(). We built this model in this process from our own
        # data, so trusting it here is a statement about provenance, not a bypass — and
        # it is scoped to the one type rather than everything skops reports.
        mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(json.dumps({"seed": seed, "data_fingerprint": fingerprint, **metrics}, indent=2))
        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            # run_id: the Lab 5 pipeline's register step registers exactly this run.
            args.metrics_out.write_text(json.dumps(
                {"seed": seed, "data_fingerprint": fingerprint, **metrics,
                 "run_id": mlflow.active_run().info.run_id}, indent=2))


if __name__ == "__main__":
    main()
