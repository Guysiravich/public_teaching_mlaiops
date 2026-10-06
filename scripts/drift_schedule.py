"""Lab 4 Task 5 — put the drift detector on a schedule, or take it off.

    python scripts/drift_schedule.py --image itcs355-lab1:<tag>        # every 15 minutes
    python scripts/drift_schedule.py --delete                           # remove it
    python scripts/drift_schedule.py --once --image itcs355-lab1:<tag>  # one run, now

Pushes the training image (which carries monitoring/) by digest and hands the same command
job to adapter.schedule(). 15 minutes because Task 6 measures injection-to-alert time and an
hourly run would make that number mostly waiting; it costs about 0.5 THB a run on the
scale-to-zero cluster, so the schedule exists only while the exercise does.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config

NAME = "itcs355-drift"
CRON = "*/15 * * * *"
PASSED_THROUGH = ("METRICS_RESOURCE_ID", "LOG_WORKSPACE_ID", "STAGING_APP")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="local training image tag (carries monitoring/)")
    ap.add_argument("--cron", default=CRON)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--once", action="store_true", help="submit one run now instead")
    args = ap.parse_args()

    cfg = config.load()
    adapter = get_adapter(cfg)
    if args.delete:
        print("deleted schedule", adapter.schedule(NAME, "", {}, ""))
        return 0
    if not args.image:
        ap.error("--image is required unless --delete")

    missing = [k for k in PASSED_THROUGH if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set in cloud.env first: {', '.join(missing)}")
    image_ref = adapter.push_image(args.image)
    job = {
        "module": "monitoring.drift_job",
        "display_name": "drift check",
        "experiment": "itcs355-lab4-drift",
        "env": {k: os.environ[k] for k in PASSED_THROUGH} | {"CLOUD_PROVIDER": cfg.provider,
                                                             "REGION": cfg.region},
        "lab": 4,
    }
    if args.once:
        job_id = adapter.submit_training(image_ref, job)
        print("submitted", job_id)
        print(adapter.wait_training(job_id))
        return 0
    print("schedule", adapter.schedule(NAME, image_ref, job, args.cron), "cron", args.cron, "UTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
