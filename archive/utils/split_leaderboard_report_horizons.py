#!/usr/bin/env python3
"""Split leaderboard report tabs by horizon using dataset + lookback/horizon filters."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_ARCHIVE_UTILS = os.path.abspath(os.path.dirname(__file__))
if _ARCHIVE_UTILS not in sys.path:
    sys.path.insert(0, _ARCHIVE_UTILS)

from backfill_leaderboard_horizon_fields import backfill_runs
from curate_leaderboard import (
    REPORT_ID,
    fetch_report_view,
    rebuild_report_runsets,
    report_url,
    upsert_report_view,
)
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)


def rebuild_report(spec: dict) -> tuple[int, int]:
    before = 0
    after = 0
    for block in spec.get("blocks") or []:
        if block.get("type") != "panel-grid":
            continue
        md = block.get("metadata") or {}
        before = len(md.get("runSets") or [])
        md["runSets"] = rebuild_report_runsets(md.get("runSets") or [])
        md["openRunSet"] = 0
        block["metadata"] = md
        after = len(md["runSets"])
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-id", default=REPORT_ID)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()

    if not args.skip_backfill:
        scanned, updated = backfill_runs(api, dry_run=args.dry_run)
        print(f"[backfill] scanned={scanned} updated={updated}")

    if not args.skip_report:
        view = fetch_report_view(api, args.report_id)
        spec = json.loads(view["spec"])
        before, after = rebuild_report(spec)
        print(f"[report] runsets {before} -> {after}")
        upsert_report_view(api, view, spec, dry_run=args.dry_run)
        print(f"[report] {report_url(view)}")


if __name__ == "__main__":
    main()
