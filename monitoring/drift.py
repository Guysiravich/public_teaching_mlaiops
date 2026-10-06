"""Lab 4 — drift detection.

PSI and Kolmogorov–Smirnov, implemented directly so you can see what the numbers mean.
Evidently is allowed instead, but you must still be able to explain what your chosen
statistic measures and why your threshold is what it is.

    python -m monitoring.drift --reference data/raw/sensors.csv --current data/current.csv

A threshold copied from a tutorial is not a justified threshold, and Lab 4 grades the
justification, not the code.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Conventional PSI reading, and it IS only conventional — it comes from credit scoring,
# where features are stable and volumes are large. Used here only to label the printout.
PSI_NO_CHANGE = 0.10
PSI_MODERATE = 0.25

# The alert thresholds, per feature, for a window of the last WINDOW_ROWS production inputs.
# Measured, not copied: reports/lab4/threshold-study.md scores 1,000 drift-free windows of
# held-out machines against the training reference.
#   * Five features never exceed PSI 0.057 at p99 with no drift, so 0.10 sits clear of their
#     noise and still catches a +3 °C temp_c offset (PSI 0.116), about 0.3 SD.
#   * load_pct differs between machines by design: the whole held-out pool already scores
#     0.134 against training, and drift-free windows reach 0.213 at p99 and 0.252 at worst.
#     At 0.10 it would alert on every run. 0.30 is above the worst drift-free window seen.
# The default 0.25 would miss a +4 °C offset (0.185) and still fire on load_pct noise.
WINDOW_ROWS = 500
THRESHOLDS: dict[str, float] = {
    "temp_c": 0.10,
    "vibration_mm_s": 0.10,
    "pressure_kpa": 0.10,
    "hours_since_service": 0.10,
    "load_pct": 0.30,
    "ambient_humidity": 0.10,
}


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ref_mean: float
    cur_mean: float
    verdict: str


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    Bins the reference into quantiles and compares the proportion of mass falling in each.
    Sensitive to changes in shape, not only in mean — which is why a feature can drift
    badly while its average looks untouched.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing: an empty bin would otherwise make the log term infinite.
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two CDFs.

    Complements PSI. KS is more sensitive to a shift in location; PSI to a change in shape.
    Reporting both, and noticing when they disagree, is worth more than either alone.
    """
    ref = np.sort(reference)
    cur = np.sort(current)
    pooled = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, pooled, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def verdict_for(score: float) -> str:
    if score < PSI_NO_CHANGE:
        return "stable"
    if score < PSI_MODERATE:
        return "moderate"
    return "significant"


def compare(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[FeatureDrift]:
    results = []
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        score = psi(ref, cur)
        results.append(FeatureDrift(
            feature=feature,
            psi=round(score, 5),
            ks_statistic=round(ks_statistic(ref, cur), 5),
            ref_mean=round(float(ref.mean()), 4),
            cur_mean=round(float(cur.mean()), 4),
            verdict=verdict_for(score),
        ))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="one PSI threshold for every feature, overriding THRESHOLDS")
    ap.add_argument("--out", type=Path, default=Path("reports/drift.json"))
    ap.add_argument("--emit", action="store_true", help="send scores as cloud metrics")
    args = ap.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    results = compare(reference, current, data.FEATURES)
    thresholds = ({f: args.threshold for f in data.FEATURES} if args.threshold is not None
                  else THRESHOLDS)
    return report(results, thresholds, args.out,
                  config.load(strict=False) if args.emit else None)


def worst_ratio(results: list[FeatureDrift], thresholds: dict[str, float]) -> float:
    """The largest PSI as a fraction of its own threshold: 1.0 or more means alert. One
    number, so one alert rule covers features whose thresholds differ."""
    return max(r.psi / thresholds[r.feature] for r in results)


def report(results: list[FeatureDrift], thresholds: dict[str, float], out: Path,
           cfg=None, rows: int | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(f"{'feature':<22}{'psi':>10}{'limit':>8}{'ks':>10}  verdict")
    for r in results:
        print(f"{r.feature:<22}{r.psi:>10.5f}{thresholds[r.feature]:>8.2f}"
              f"{r.ks_statistic:>10.5f}  {r.verdict}")
    ratio = worst_ratio(results, thresholds)

    if cfg is not None:
        # Through the adapter, so this reaches Azure Monitor (or CloudWatch / Cloud
        # Monitoring) without the detector knowing which.
        from cloudlayer.factory import get_adapter
        adapter = get_adapter(cfg)
        for r in results:
            adapter.emit_metric(f"drift.psi.{r.feature}", r.psi)
        adapter.emit_metric("drift.threshold_ratio", ratio)
        if rows is not None:
            adapter.emit_metric("drift.window_rows", float(rows), unit="Count")

    breached = [r for r in results if r.psi >= thresholds[r.feature]]
    if breached:
        print(f"\nALERT  {len(breached)} feature(s) above threshold (worst at "
              f"{ratio:.2f}x its limit): " + ", ".join(r.feature for r in breached))
        print("Before you retrain: is this drift, or is it a broken upstream pipeline? "
              "Retraining on corrupted data destroys a working model faster than any "
              "schedule would.")
        return 2
    print(f"\nOK  no feature above its threshold (worst at {ratio:.2f}x its limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
