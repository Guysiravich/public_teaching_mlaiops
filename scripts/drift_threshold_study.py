"""Lab 4 — measure the two numbers a drift threshold has to sit between.

    python scripts/drift_threshold_study.py      # writes reports/lab4/threshold-study.md

1. Noise. How high does PSI go when NOTHING has drifted? Windows are drawn from the held-out
   machines (never seen in training) and scored against the training reference, as the
   scheduled job does. A threshold below this fires on ordinary sampling variance.
2. Harm. How far does a feature have to move before the model gets measurably worse?
   A temp_c sensor offset is applied to the held-out set with the labels left alone — the
   machines did not change, the reading did — and the Brier score is compared with the
   unshifted set. A threshold above the PSI of a harmful shift misses it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from monitoring.drift import psi
from src import config, data, seeds

WINDOWS = (250, 500, 1000)
DRAWS = 1000
SHIFTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
SCALES = (1.1, 1.25, 1.5, 2.0)
FEATURE = "temp_c"


def main() -> int:
    seed = seeds.set_all()
    df = data.load_raw(config.REPO_ROOT / "data" / "raw" / "sensors.csv")
    train, val, test = data.split(df, seed=seed)
    held_out = pd.concat([val, test], ignore_index=True)
    rng = np.random.default_rng(seed)

    lines = ["# Drift threshold study", "",
             f"Reference: training split, {len(train):,} rows. Held-out pool: validation + test, "
             f"{len(held_out):,} rows from machines never seen in training.", "",
             "## 1. Noise: largest PSI over the six features, no drift", "",
             f"{DRAWS:,} windows per size, drawn without replacement from the held-out pool.", "",
             "| Window (rows) | median | p99 | max | windows above 0.10 | above 0.15 |",
             "|---|---|---|---|---|---|"]
    per_feature: dict[int, dict[str, np.ndarray]] = {}
    for n in WINDOWS:
        scores = {f: [] for f in data.FEATURES}
        for _ in range(DRAWS):
            window = held_out.sample(n, random_state=int(rng.integers(1 << 31)))
            for f in data.FEATURES:
                scores[f].append(psi(train[f].to_numpy(float), window[f].to_numpy(float)))
        per_feature[n] = {f: np.array(v) for f, v in scores.items()}
        worst = np.max(np.vstack(list(per_feature[n].values())), axis=0)
        lines.append(f"| {n} | {np.median(worst):.3f} | {np.percentile(worst, 99):.3f} | "
                     f"{worst.max():.3f} | {(worst >= 0.10).mean():.1%} | {(worst >= 0.15).mean():.1%} |")

    lines += ["", "Per feature, p99 of PSI with no drift:", "",
              "| Feature | " + " | ".join(f"{n} rows" for n in WINDOWS) + " |",
              "|---|" + "---|" * len(WINDOWS)]
    for f in data.FEATURES:
        lines.append(f"| {f} | " + " | ".join(
            f"{np.percentile(per_feature[n][f], 99):.3f}" for n in WINDOWS) + " |")
    whole = [psi(train[f].to_numpy(float), held_out[f].to_numpy(float)) for f in data.FEATURES]
    lines += ["", "Whole held-out pool against the training reference (no sampling noise left; "
              "what remains is the difference between machines):", "",
              "| " + " | ".join(data.FEATURES) + " |", "|" + "---|" * len(data.FEATURES),
              "| " + " | ".join(f"{w:.3f}" for w in whole) + " |"]

    model = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    model.fit(train[data.FEATURES], train[data.TARGET])
    base_brier = brier_score_loss(held_out[data.TARGET],
                                  model.predict_proba(held_out[data.FEATURES])[:, 1])
    base_rate = (model.predict_proba(held_out[data.FEATURES])[:, 1] >= 0.5).mean()

    lines += ["", f"## 2. Harm: a `{FEATURE}` sensor fault, labels unchanged", "",
              f"Unshifted held-out Brier {base_brier:.4f}; share flagged (p >= 0.5) {base_rate:.1%}. "
              "PSI is the mean over 200 windows of 500 rows.", "",
              "| Fault | PSI (500 rows) | Brier | Brier change | Share flagged |",
              "|---|---|---|---|---|"]
    ref = train[FEATURE].to_numpy(float)
    centre = held_out[FEATURE].mean()
    faults = [(f"offset +{s:g} °C", lambda x, s=s: x + s) for s in SHIFTS]
    faults += [(f"spread x{k:g}", lambda x, k=k: centre + (x - centre) * k) for k in SCALES]
    for label, fault in faults:
        shifted = held_out.copy()
        shifted[FEATURE] = fault(shifted[FEATURE].to_numpy(float))
        proba = model.predict_proba(shifted[data.FEATURES])[:, 1]
        brier = brier_score_loss(shifted[data.TARGET], proba)
        scores = [psi(ref, shifted[FEATURE].sample(500, random_state=i).to_numpy(float))
                  for i in range(200)]
        lines.append(f"| {label} | {np.mean(scores):.3f} | {brier:.4f} | "
                     f"{(brier - base_brier) / base_brier:+.1%} | {(proba >= 0.5).mean():.1%} |")

    lines += ["", "## 3. Harm per feature: an offset of one training standard deviation", "",
              "| Feature | 1 SD | PSI (500 rows) | Brier change |", "|---|---|---|---|"]
    for f in data.FEATURES:
        sd = float(train[f].std())
        shifted = held_out.copy()
        shifted[f] = shifted[f] + sd
        brier = brier_score_loss(shifted[data.TARGET],
                                 model.predict_proba(shifted[data.FEATURES])[:, 1])
        score = np.mean([psi(train[f].to_numpy(float),
                             shifted[f].sample(500, random_state=i).to_numpy(float))
                         for i in range(100)])
        lines.append(f"| {f} | {sd:.2f} | {score:.3f} | {(brier - base_brier) / base_brier:+.1%} |")

    text = "\n".join(lines) + "\n"
    out = config.REPO_ROOT / "reports" / "lab4" / "threshold-study.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
