"""Lab 5 Tasks 1 and 3 — run the managed pipeline once, or put it on its schedule.

    make pipeline                     # push the image, run the DAG once, report every step
    make gate-fail-demo               # the same, against an incumbent no candidate can beat
    make pipeline-schedule            # the cron in pipeline/pipeline.yaml (monthly)
    make pipeline-schedule CRON="*/30 * * * *"   # a short one, to watch a scheduled run happen
    make pipeline-unschedule

The image is pushed by digest and that digest is what every step runs: the pipeline cannot
silently pick up a newer image between steps, and the registered version records it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer import pipelines  # noqa: E402
from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402

SCHEDULE = "itcs355-pipeline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="local training image tag to push and run")
    ap.add_argument("--schedule", nargs="?", const="", default=None,
                    help="create the schedule; a cron here overrides pipeline.yaml")
    ap.add_argument("--unschedule", action="store_true")
    ap.add_argument("--incumbent", help="force the gate's incumbent score (the failing demo)")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    cfg = config.load()
    if args.unschedule:
        client, _ = get_adapter(cfg)._ml_client()
        print("deleted schedule", pipelines.schedule(client, None, SCHEDULE, None))
        return 0

    image = get_adapter(cfg).push_image(args.image)
    os.environ["PIPELINE_IMAGE"] = image
    os.environ["IMAGE_DIGEST"] = image.split("@", 1)[1]
    os.environ["GIT_COMMIT"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                              text=True, cwd=ROOT).stdout.strip()
    if args.incumbent:
        os.environ["GATE_INCUMBENT"] = args.incumbent
    spec = pipelines.load_spec()
    client, job = pipelines.compile_azure(spec)
    print(f"image {image}")

    if args.schedule is not None:
        cron = args.schedule or spec["schedule"]
        print("schedule", pipelines.schedule(client, job, SCHEDULE, cron), "cron", cron, "UTC")
        return 0

    result = pipelines.submit(client, job, wait=not args.no_wait)
    print(json.dumps(result, indent=2))
    out = ROOT / "reports" / "lab5" / f"pipeline-{result['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "image": image, "incumbent_forced": args.incumbent},
                              indent=2))
    return 0 if result.get("status", "Completed") == "Completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
