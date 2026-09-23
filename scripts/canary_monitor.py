"""Lab 3, Task 4 — watch a canary from metrics alone, and say when it was caught.

    python scripts/canary_monitor.py --endpoint https://... --rate 5 --minutes 20

Sends held-out rows whose true label we know, one at a time, and scores every response
against that label. The rule below is declared BEFORE the canary is deployed:

  baseline   the first --baseline responses, while 100% of traffic is on the current
             revision, give a mean Brier score b0 and its standard error
  alert      a cohort is called degraded when its rolling Brier over the last --window
             of ITS OWN responses exceeds b0 + 3 standard errors, twice in a row

Cohorts are the values of the `x-model-version` response header, relabelled A and B in
first-seen order. The monitor never reads which version is new, and never reads the
traffic weights; that mapping is printed once at the end, for the report.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, seeds


def post(endpoint: str, row: dict) -> tuple[float, str, float]:
    body = json.dumps(row).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/predict", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
        cohort = response.headers.get("x-model-version", "unknown")
    return float(payload["probability"]), cohort, (time.perf_counter() - started) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--baseline", type=int, default=200, help="responses used to set the limit")
    ap.add_argument("--window", type=int, default=100, help="rolling window per cohort")
    ap.add_argument("--out", type=Path, default=Path("reports/lab3/canary-raw.jsonl"))
    args = ap.parse_args()

    cfg = config.load(strict=False)
    seed = seeds.set_all()
    _, _, test_df = data.split(data.load_raw(cfg.raw_path), seed=seed)
    rows = test_df.to_dict("records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.open("w")
    started = time.time()
    deadline = started + args.minutes * 60
    interval = 1.0 / args.rate

    baseline_scores: list[float] = []
    limit: float | None = None
    windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=args.window))
    breaches: dict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    alerted: set[str] = set()
    sent = 0

    print(f"start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
          f"rate={args.rate}/s window={args.window} baseline={args.baseline}")
    while time.time() < deadline:
        row = rows[sent % len(rows)]
        features = {k: float(row[k]) for k in data.FEATURES}
        label = int(row[data.TARGET])
        try:
            probability, version, latency_ms = post(args.endpoint, features)
        except Exception as exc:  # a failed request is itself a signal
            print(f"  request error: {exc}")
            time.sleep(interval)
            continue
        sent += 1
        brier = (probability - label) ** 2
        cohort = names.setdefault(version, chr(ord("A") + len(names)))
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "cohort": cohort, "brier": round(brier, 6), "p": round(probability, 6),
                  "label": label, "latency_ms": round(latency_ms, 2)}
        log.write(json.dumps(record) + "\n")
        log.flush()

        if limit is None:
            baseline_scores.append(brier)
            if len(baseline_scores) == args.baseline:
                b0 = statistics.mean(baseline_scores)
                se = statistics.stdev(baseline_scores) / (len(baseline_scores) ** 0.5)
                limit = b0 + 3 * se
                print(f"  baseline set: mean Brier {b0:.4f}, limit {limit:.4f} "
                      f"(after {sent} responses)")
        else:
            window = windows[cohort]
            window.append(brier)
            if len(window) == args.window:
                rolling = statistics.mean(window)
                breaches[cohort] = breaches[cohort] + 1 if rolling > limit else 0
                if breaches[cohort] >= 2 and cohort not in alerted:
                    alerted.add(cohort)
                    elapsed = time.time() - started
                    print(f"  ALERT cohort {cohort}: rolling Brier {rolling:.4f} > {limit:.4f} "
                          f"after {elapsed / 60:.1f} min, {sent} responses total")
        time.sleep(interval)

    log.close()
    print("\nper-cohort summary (the mapping, printed only now):")
    for version, cohort in names.items():
        scores = [b for b in windows[cohort]]
        print(f"  cohort {cohort} = model_version {version}: "
              f"last-window Brier {statistics.mean(scores):.4f}" if scores else
              f"  cohort {cohort} = model_version {version}")
    print(f"raw responses: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
