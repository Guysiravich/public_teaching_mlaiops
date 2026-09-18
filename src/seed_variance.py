"""Lab 2, Task 3 — seed variance of the study's leading configurations.

Run (as a managed job):
    python scripts/train_remote.py --image <image> --module src.seed_variance -- --top 3 --seeds 5

Takes the `--top` finished trials of the study by val_roc_auc and refits each configuration
with `--seeds` different model seeds. Only the model's random_state changes; the train/val/test
split keeps the study's seed, so every refit is scored on the same validation rows as the study.
The question it answers: is the gap between the leading trials bigger than seed noise?
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src import config, costs, data, seeds
from src.train import git_commit
from src.tune import FIXED_PARAMS, SEARCH_SPACE, grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 2 — seed variance")
    p.add_argument("--study", default="itcs355-lab2", help="experiment holding the study")
    p.add_argument("--experiment", default="itcs355-lab2-seeds")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--instance", default="local", help="key into src/costs.py PRICE_TABLE")
    return p.parse_args()


def typed_params(run_params: dict) -> dict:
    """MLflow stores params as strings; find the matching grid point to get real types."""
    for point in grid(SEARCH_SPACE):
        if all(str(v) == run_params.get(k) for k, v in point.items()):
            return point
    raise ValueError(f"run params {run_params} match no point of SEARCH_SPACE")


def main() -> None:
    args = parse_args()
    cfg = config.load(strict=False)
    split_seed = seeds.set_all(seeds.DEFAULT_SEED)

    df = data.load_raw(cfg.raw_path)
    train_df, val_df, _ = data.split(df, seed=split_seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    leaders = mlflow.search_runs(
        experiment_names=[args.study],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_roc_auc DESC"],
        max_results=args.top,
    )
    mlflow.set_experiment(args.experiment)
    rate = costs.hourly_rate(cfg.provider, args.instance)

    summary = []
    for _, lead in leaders.iterrows():
        params = typed_params({k[len("params."):]: v for k, v in lead.items()
                               if k.startswith("params.") and v is not None})
        scores = []
        for k in range(args.seeds):
            model_seed = split_seed + 1000 + k
            started = time.perf_counter()
            with mlflow.start_run(run_name=f"seed-{lead['run_id'][:8]}-{k}"):
                model = RandomForestClassifier(random_state=model_seed, n_jobs=-1,
                                               **FIXED_PARAMS, **params)
                model.fit(train_df[data.FEATURES], train_df[data.TARGET])
                proba = model.predict_proba(val_df[data.FEATURES])[:, 1]
                score = float(roc_auc_score(val_df[data.TARGET], proba))
                elapsed_s = time.perf_counter() - started
                mlflow.log_params({**FIXED_PARAMS, **params, "model_seed": model_seed,
                                   "split_seed": split_seed, "instance": args.instance})
                mlflow.log_metrics({"val_roc_auc": score, "duration_s": round(elapsed_s, 3),
                                    "cost_thb": round(elapsed_s / 3600 * rate, 4)})
                mlflow.set_tags({"source_run_id": lead["run_id"], "git_commit": git_commit(),
                                 "training_job_id": os.environ.get("TRAINING_JOB_ID", "local"),
                                 "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
                                 "lab": "2"})
            scores.append(score)
        row = {"source_run_id": lead["run_id"], "params": params,
               "study_val_roc_auc": round(float(lead["metrics.val_roc_auc"]), 4),
               "seed_mean": round(statistics.mean(scores), 4),
               "seed_std": round(statistics.stdev(scores), 4),
               "seed_min": round(min(scores), 4), "seed_max": round(max(scores), 4)}
        summary.append(row)
        print(json.dumps(row))

    print("\nSEED VARIANCE SUMMARY")
    for r in summary:
        print(f"  {r['source_run_id'][:8]} {r['params']}  study={r['study_val_roc_auc']}  "
              f"mean={r['seed_mean']} std={r['seed_std']} range=[{r['seed_min']}, {r['seed_max']}]")


if __name__ == "__main__":
    main()
