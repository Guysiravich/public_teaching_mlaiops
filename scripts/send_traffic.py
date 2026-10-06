"""Lab 4 Task 6 — send production-like traffic to the staging endpoint.

    python scripts/send_traffic.py --rows 600                          # normal inputs
    make inject-drift && python scripts/send_traffic.py --source data/current.csv --rows 600

Rows come from the held-out machines only (validation + test split), the machines the model
never saw — what production looks like. With --source data/current.csv they are the same
machines after scripts/inject_drift.py has shifted a feature. Sent in batches of 100 through
adapter.invoke, so the service logs one input line per row for the drift job to read.
Prints the UTC time of the first request: the injection time for Task 6.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from cloudlayer.factory import get_adapter
from src import config, data, seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--endpoint", default=None, help="default: reports/lab4-staging-endpoint.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="send this many rounds")
    ap.add_argument("--pause", type=float, default=30.0, help="seconds between rounds")
    ap.add_argument("--switch-to", type=Path, default=None,
                    help="after --switch-after rounds, draw from this source instead")
    ap.add_argument("--switch-after", type=int, default=0)
    args = ap.parse_args()

    endpoint = args.endpoint or Path("reports/lab4-staging-endpoint.txt").read_text().strip()
    adapter = get_adapter(config.load(strict=False))
    _, val, test = data.split(data.load_raw(Path("data/raw/sensors.csv")), seed=seeds.set_all())
    held_out = set(pd.concat([val, test])[data.GROUP])

    from deploy import wait_ready  # staging scales to zero; a cold start outlasts one invoke
    wait_ready(endpoint)
    for round_no in range(args.repeat):
        switched = args.switch_to is not None and round_no >= args.switch_after
        path = args.switch_to if switched else args.source
        if switched and round_no == args.switch_after:
            print(f"INJECTION  first shifted request at "
                  f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC "
                  f"from {path}", flush=True)
        send_round(adapter, endpoint, path, held_out, args.rows, args.seed + round_no)
        if round_no + 1 < args.repeat:
            time.sleep(args.pause)
    return 0


def send_round(adapter, endpoint: str, path: Path, held_out: set, n: int, seed: int) -> None:
    source = pd.read_csv(path)
    pool = source[source[data.GROUP].isin(held_out)]
    rows = pool.sample(n, replace=len(pool) < n, random_state=seed)
    started = datetime.datetime.now(datetime.timezone.utc)
    sent = 0
    for start in range(0, len(rows), 100):
        batch = rows.iloc[start:start + 100][data.FEATURES].to_dict("records")
        result = adapter.invoke(endpoint, {"rows": batch})
        sent += len(result["probabilities"])
    print(f"{started:%H:%M:%S} UTC  sent {sent} rows from {path.name}  "
          f"temp_c mean {rows['temp_c'].mean():.2f}  version {result['model_version']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
