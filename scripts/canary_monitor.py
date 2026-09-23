"""Lab 3, Task 4 — watch a canary from metrics alone, and record when it was caught.

    python scripts/canary_monitor.py --rate 30 --workers 8 --minutes 20

Sends held-out rows whose true label we know and scores each response with the Brier score
(probability minus outcome, squared). Cohorts are the values of the `x-model-version`
response header, relabelled A and B in first-seen order; the monitor never reads which
version is new, and never reads the traffic weights.

The rule is declared before the canary is deployed:

    alert when one cohort's cumulative mean Brier exceeds the other's by more than
    3 standard errors of the difference, on at least 200 responses in each cohort,
    and stays there for two consecutive checks.

3 sigma on this pipeline needs roughly 1,900 responses in the smaller cohort — at 10% of
30 requests a second that is about ten minutes. That number is the point of the exercise:
a canary this small is a slow detector, and the report says so.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, seeds


class Cohort:
    """Running count, mean and variance of one cohort's Brier scores (Welford)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


def post(endpoint: str, row: dict, timeout: float) -> tuple[float, str, float]:
    body = json.dumps(row).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/predict", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
        version = response.headers.get("x-model-version", "unknown")
    return float(payload["probability"]), version, (time.perf_counter() - started) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None, help="default: reports/lab3-endpoint.txt")
    ap.add_argument("--rate", type=float, default=30.0, help="requests per second, in total")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--min-responses", type=int, default=200, help="per cohort, before judging")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("reports/lab3/canary-raw.jsonl"))
    args = ap.parse_args()

    endpoint = args.endpoint or Path("reports/lab3-endpoint.txt").read_text().strip()
    cfg = config.load(strict=False)
    seed = seeds.set_all()
    _, _, test_df = data.split(data.load_raw(cfg.raw_path), seed=seed)
    rows = [{k: float(r[k]) for k in data.FEATURES} | {"_label": int(r[data.TARGET])}
            for r in test_df.to_dict("records")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.open("w")
    lock = threading.Lock()
    work: queue.Queue = queue.Queue(maxsize=args.workers * 4)
    cohorts: dict[str, Cohort] = defaultdict(Cohort)
    names: dict[str, str] = {}
    started = time.time()
    deadline = started + args.minutes * 60
    state = {"sent": 0, "errors": 0, "alert": None, "streak": 0}

    def worker() -> None:
        while True:
            row = work.get()
            if row is None:
                work.task_done()
                return
            label = row.pop("_label")
            try:
                probability, version, latency_ms = post(endpoint, row, timeout=30)
            except Exception as exc:
                with lock:
                    state["errors"] += 1
                    log.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                          "error": str(exc)[:120]}) + "\n")
                work.task_done()
                continue
            brier = (probability - label) ** 2
            with lock:
                cohort = names.setdefault(version, chr(ord("A") + len(names)))
                cohorts[cohort].add(brier)
                state["sent"] += 1
                log.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "cohort": cohort, "brier": round(brier, 6), "p": round(probability, 6),
                    "label": label, "latency_ms": round(latency_ms, 2)}) + "\n")
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for thread in threads:
        thread.start()

    print(f"start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
          f"endpoint={endpoint} rate={args.rate}/s workers={args.workers} "
          f"rule: {args.sigma} sigma on the cumulative difference, "
          f"min {args.min_responses} responses per cohort")
    interval = 1.0 / args.rate
    index = 0
    next_report = started + 30
    while time.time() < deadline and state["alert"] is None:
        work.put(dict(rows[index % len(rows)]))
        index += 1
        time.sleep(interval)
        now = time.time()
        if now < next_report:
            continue
        next_report = now + 30
        with lock:
            snapshot = {name: (c.n, c.mean, c.var) for name, c in cohorts.items()}
            sent, errors = state["sent"], state["errors"]
        line = " ".join(f"{name}: n={n} brier={mean:.4f}" for name, (n, mean, _) in sorted(snapshot.items()))
        z = float("nan")
        if len(snapshot) == 2:
            (na, ma, va), (nb, mb, vb) = (snapshot[k] for k in sorted(snapshot))
            if min(na, nb) >= args.min_responses:
                se = math.sqrt(va / na + vb / nb)
                z = (mb - ma) / se if se else float("nan")
                if abs(z) > args.sigma:
                    state["streak"] += 1
                    if state["streak"] >= 2:
                        worse = sorted(snapshot)[1] if mb > ma else sorted(snapshot)[0]
                        elapsed = now - started
                        state["alert"] = (worse, elapsed, z)
                        print(f"  ALERT cohort {worse} is worse by {abs(z):.1f} sigma "
                              f"after {elapsed / 60:.1f} min ({sent} responses)")
                else:
                    state["streak"] = 0
        print(f"  [{(now - started) / 60:5.1f} min] sent={sent} errors={errors} {line}"
              + (f" z={z:+.2f}" if z == z else ""))

    for _ in threads:
        work.put(None)
    work.join()
    log.close()

    print("\ncohort to model version (read only now, for the report):")
    for version, cohort in sorted(names.items(), key=lambda kv: kv[1]):
        c = cohorts[cohort]
        print(f"  cohort {cohort} = model_version {version}: n={c.n} mean Brier {c.mean:.4f}")
    if state["alert"]:
        worse, elapsed, z = state["alert"]
        print(f"detection: cohort {worse} flagged after {elapsed:.0f} s ({elapsed / 60:.1f} min), "
              f"{abs(z):.1f} sigma")
    else:
        print("detection: no cohort crossed the rule within the window")
    print(f"raw responses: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
