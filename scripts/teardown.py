"""Delete the resources of one lab, by tag, through the adapter.

    make teardown LAB=3              # delete
    make teardown LAB=3 DRY_RUN=1    # list only

The Makefile target as provided called `teardown(cfg.tags(1))` — lab=1 hardcoded — so
running it during Lab 3, as that handout asks, would have deleted Lab 1's storage account
and container registry. The lab number is now explicit and required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config

# Azure ML schedules are workspace objects without resource tags, so the tag search below
# never finds them — and a schedule that outlives its endpoint keeps running jobs against a
# deleted service (Lab 4 handout). They are deleted by name, explicitly.
SCHEDULES = {"4": ["itcs355-drift"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True, help="lab number whose resources to delete")
    ap.add_argument("--dry-run", action="store_true", help="list what would be deleted")
    args = ap.parse_args()

    cfg = config.load()
    tags = cfg.tags(int(args.lab))
    print(f"teardown scope: {tags}")
    adapter = get_adapter(cfg)
    for name in SCHEDULES.get(str(args.lab), []):
        if args.dry_run:
            print(f"would delete schedule {name}")
        else:
            print(f"deleted schedule {adapter.schedule(name, '', {}, '')}")
    if SCHEDULES.get(str(args.lab)) and hasattr(adapter, "scheduled"):
        print(f"schedules left in the workspace: {adapter.scheduled() or 'none'}")
    deleted = adapter.teardown(tags, dry_run=args.dry_run)
    if not deleted:
        print("nothing carried those tags")
    for item in deleted:
        print(("would delete " if args.dry_run else "deleted ") + item)
    print("Deletion is asynchronous: re-check the portal, and check the bill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
