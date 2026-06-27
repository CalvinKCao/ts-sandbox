"""Rename wandb runs in ts-sandbox-leaderboard to {group}-{phase} convention."""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name

PROJECT = "ts-sandbox-leaderboard"
ENTITY = "calvincao"
STEM_RE = re.compile(r"^\d{2}-\d{2}-\d+-")


def expected_name(group: str, job_type: str) -> str:
    return make_phase_run_name(group, job_type)


def needs_rename(current: str, target: str) -> bool:
    if current == target:
        return False
    if STEM_RE.match(current):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    updated = skipped = 0

    for run in api.runs(f"{args.entity}/{args.project}"):
        group = run.group or ""
        job_type = run.job_type or ""
        if not group or not job_type:
            skipped += 1
            continue

        target = expected_name(group, job_type)
        current = run.name or ""

        if not needs_rename(current, target):
            skipped += 1
            continue

        if args.dry_run:
            print(f"would rename: {current!r} -> {target!r}")
            updated += 1
            continue

        run.name = target
        full = f"{group}-{job_type.replace('_', '-')}"
        if target != full and full not in (run.notes or ""):
            notes = (run.notes or "").strip()
            run.notes = f"{notes}\nfull name: {full}".strip() if notes else f"full name: {full}"
        run.update()
        updated += 1
        if updated % 25 == 0:
            print(f"renamed {updated}...", flush=True)

    print(f"\n{'would rename' if args.dry_run else 'renamed'}: {updated}, skipped: {skipped}")


if __name__ == "__main__":
    main()
