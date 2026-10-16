"""Lab 5 — the registration gate.

Compares a candidate against the currently registered production model. Exits non-zero
when the candidate should NOT be registered, which is what makes the pipeline condition
real rather than decorative.

    python scripts/evaluation_gate.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chosen from this project's own measurements, not convention:
#   MIN_IMPROVEMENT - Lab 2 trained the chosen configuration on five seeds: 0.8408 +/- 0.0013.
#     Retraining the SAME model on another seed moves the score by up to about 2 sigma,
#     0.0026. A candidate must beat the incumbent by more than that, or the "improvement"
#     is the seed, and every scheduled retrain would churn the registry for nothing.
#   MIN_ABSOLUTE - Lab 3's canary served version 2 (validation ROC-AUC 0.826 against 0.841)
#     and production saw it: Brier 0.0918 against 0.0808, caught at 3 sigma. A model below
#     0.80 is worse than one already measured as harmful, whatever the incumbent is.
MIN_ABSOLUTE = 0.80
MIN_IMPROVEMENT = 0.003


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--metric", default="test_roc_auc")
    ap.add_argument("--incumbent", type=float, default=None,
                    help="current production score; omit to read from the registry")
    ap.add_argument("--alias", default="staging",
                    help="the registry alias whose version is the incumbent")
    ap.add_argument("--decision-out", type=Path, default=None,
                    help="write {'pass': bool} here for the register step")
    args = ap.parse_args()
    code = decide(args)
    if args.decision_out:
        args.decision_out.parent.mkdir(parents=True, exist_ok=True)
        args.decision_out.write_text(json.dumps({"pass": code == 0}))
    return code


def registry_incumbent(alias: str, metric: str) -> float | None:
    """The score of the version the alias points to, from that version's lineage tags
    (metric_test, copied on at registration). None when nothing holds the alias yet."""
    import mlflow
    from mlflow import MlflowClient

    from src import config

    cfg = config.load(strict=False)
    if not cfg.mlflow_tracking_uri:
        return None
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    try:
        version = MlflowClient().get_model_version_by_alias(cfg.model_registry_name, alias)
    except Exception as exc:  # no such alias or model: a first registration
        print(f"no incumbent under alias {alias!r} ({type(exc).__name__})")
        return None
    tag = {"test_roc_auc": "metric_test", "val_roc_auc": "metric_val"}.get(metric, metric)
    value = version.tags.get(tag)
    print(f"incumbent: version {version.version} (alias {alias!r}), {tag} = {value}")
    return float(value) if value is not None else None


def decide(args: argparse.Namespace) -> int:

    payload = json.loads(args.metrics.read_text())
    candidate = float(payload[args.metric])
    print(f"candidate {args.metric} = {candidate:.4f}")

    if candidate < MIN_ABSOLUTE:
        print(f"GATE FAIL  below absolute floor {MIN_ABSOLUTE}")
        return 1

    incumbent = args.incumbent
    if incumbent is None:
        incumbent = registry_incumbent(args.alias, args.metric)
    if incumbent is None:
        print("no incumbent in the registry - a first registration")
        print("GATE PASS")
        return 0

    delta = candidate - incumbent
    print(f"incumbent {incumbent:.4f}  delta {delta:+.4f}  required {MIN_IMPROVEMENT:+.4f}")
    if delta < MIN_IMPROVEMENT:
        print("GATE FAIL  improvement within noise; not registering")
        return 1

    print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
