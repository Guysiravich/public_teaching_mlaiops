"""Lab 2, Task 3 evidence — the two tied leaders rank alike; do their probabilities?

    python scripts/calibration_check.py

Refits the study's two depth-4 leaders (class_weight None vs balanced) with the study's seed and
split, then compares them on the validation rows: ranking (ROC-AUC, failures caught when
inspecting the top 100), calibration (mean predicted p(failure) against the actual rate, Brier
score, by risk bucket), and what each threshold would flag. Runs locally in a few seconds; no
cloud, no tracking server.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from src import config, data, seeds
from src.tune import FIXED_PARAMS

CANDIDATES = {
    "a5192935 class_weight=None": {"max_depth": 4, "class_weight": None, "max_features": "sqrt"},
    "f9e0834d class_weight=balanced": {"max_depth": 4, "class_weight": "balanced", "max_features": "sqrt"},
}


def main() -> int:
    cfg = config.load(strict=False)
    seed = seeds.set_all(seeds.DEFAULT_SEED)
    train_df, val_df, _ = data.split(data.load_raw(cfg.raw_path), seed=seed)
    X, y = val_df[data.FEATURES], val_df[data.TARGET].to_numpy()
    print(f"validation rows {len(y)}, actual failures {y.sum()} ({y.mean():.1%})")

    for label, params in CANDIDATES.items():
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **FIXED_PARAMS, **params)
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])
        p = model.predict_proba(X)[:, 1]
        top = np.argsort(-p)[:100]
        print(f"\n{label}")
        print(f"  ranking      roc_auc {roc_auc_score(y, p):.4f}; inspecting the top 100 catches "
              f"{y[top].sum()} of {y.sum()} failures")
        print(f"  calibration  mean predicted {p.mean():.1%} vs actual {y.mean():.1%}; "
              f"brier {brier_score_loss(y, p):.4f}")
        edges = np.quantile(p, [0, .5, .8, .95, 1])
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p >= lo) & (p <= hi)
            print(f"    p in [{lo:.2f}, {hi:.2f}]  predicted {p[sel].mean():5.1%}  "
                  f"actual {y[sel].mean():5.1%}  (n={sel.sum()})")
        for t in (0.5, 0.3, 0.2):
            flag = p >= t
            print(f"  threshold {t}: flags {flag.sum():4d} machines, recall {recall_score(y, flag):.2f}, "
                  f"precision {precision_score(y, flag, zero_division=0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
