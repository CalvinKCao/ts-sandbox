"""Delete duplicate phase runs in ts-sandbox-leaderboard.

Keeps one run per (group, job_type). When duplicates exist, prefers runs listed
in curated_wandb_copy_map.json, else the newest created_at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = "calvincao"


def _run_path(run) -> str:
    path = getattr(run, "path", None)
    if isinstance(path, list):
        return "/".join(str(p) for p in path)
    return str(path).lstrip("/")


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "").replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def load_keep_paths() -> Set[str]:
    keep: Set[str] = set()
    for path in (
        os.path.join(REPO, "reports", "sweep_grid_report", "curated_wandb_copy_map.json"),
        os.path.join(REPO, "archive", "reports", "sweep_grid_report", "curated_wandb_copy_map.json"),
    ):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in (data.get("copies") or {}).values():
            if rec.get("status") == "ok" and rec.get("dst_path"):
                keep.add(rec["dst_path"])
    return keep


def phase_key(run) -> Tuple[str, str]:
    return (run.group or "", run.job_type or run.name or "")


def pick_keep(run_a, run_b, keep_paths: Set[str]) -> Tuple[Any, Any]:
    path_a, path_b = _run_path(run_a), _run_path(run_b)
    in_a, in_b = path_a in keep_paths, path_b in keep_paths
    if in_a and not in_b:
        return run_a, run_b
    if in_b and not in_a:
        return run_b, run_a
    ta = _parse_created_at(run_a.created_at)
    tb = _parse_created_at(run_b.created_at)
    if tb > ta:
        return run_b, run_a
    return run_a, run_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    keep_paths = load_keep_paths()
    buckets: Dict[Tuple[str, str], List[Any]] = {}

    for run in api.runs(f"{args.entity}/{args.project}"):
        buckets.setdefault(phase_key(run), []).append(run)

    to_delete: List[Any] = []
    for key, runs in buckets.items():
        if len(runs) <= 1:
            continue
        keeper = runs[0]
        for other in runs[1:]:
            keeper, _ = pick_keep(keeper, other, keep_paths)
        for run in runs:
            if run.id != keeper.id:
                to_delete.append(run)

    print(f"groups: {len(buckets)}, duplicates to delete: {len(to_delete)}")
    deleted = 0
    for run in to_delete:
        path = _run_path(run)
        if args.dry_run:
            print(f"would delete {path} ({run.name}, {run.job_type})")
            deleted += 1
            continue
        print(f"deleting {path} ({run.name})", flush=True)
        run.delete()
        deleted += 1

    print(f"\n{'would delete' if args.dry_run else 'deleted'}: {deleted}")


if __name__ == "__main__":
    main()
