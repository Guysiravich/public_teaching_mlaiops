"""Lab 4 Task 5 — the scheduled drift detector.

    python -m monitoring.drift_job            # what the Azure ML schedule runs every 15 minutes
    make drift-schedule / make drift-unschedule

Reference: the training split of data/raw/sensors.csv, the rows the model was fitted on.
Current: the last WINDOW_ROWS inputs production actually received, read back through the
adapter. Scores go out as metrics through the adapter; an alert rule on the metric sends the
email. The job itself succeeds whether or not drift is found — a red job should mean the
detector broke, not that it worked.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from monitoring.drift import THRESHOLDS, WINDOW_ROWS, compare, report
from src import config, data, seeds


def main() -> int:
    started = time.time()
    cfg = config.load(strict=False)
    adapter = get_adapter(cfg)

    raw = data.load_raw(config.REPO_ROOT / "data" / "raw" / "sensors.csv")
    reference, _, _ = data.split(raw, seed=seeds.set_all())

    rows = adapter.recent_inputs(WINDOW_ROWS)
    print(f"reference: {len(reference)} training rows; current: {len(rows)} logged inputs")
    if len(rows) < WINDOW_ROWS:
        # Too few rows and PSI measures sampling noise, not drift (threshold-study.md):
        # report the shortfall as its own metric instead of a score nobody should trust.
        adapter.emit_metric("drift.window_rows", float(len(rows)), unit="Count")
        print(f"SKIP  {len(rows)} < {WINDOW_ROWS} inputs; no score this run")
        return 0

    current = pd.DataFrame(rows)[data.FEATURES]
    results = compare(reference, current, data.FEATURES)
    report(results, THRESHOLDS, Path("reports/drift.json"), cfg=cfg, rows=len(rows))
    print(f"done in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
