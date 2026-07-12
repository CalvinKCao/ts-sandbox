#!/usr/bin/env python3
"""Backfill classical baseline partials into ts-sandbox-leaderboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from utils.load_dotenv import load_repo_dotenv

    load_repo_dotenv(REPO)
except ImportError:
    pass

from utils.run_classical_baselines import (  # noqa: E402
    DEFAULT_CONFIG,
    log_method_to_wandb,
)


def _load_partial(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def log_partials_dir(
    output_dir: Path,
    *,
    config_path: Path,
    job_id: Optional[str],
    dry_run: bool,
    force: bool,
) -> int:
    partial_dir = output_dir / "partials"
    if not partial_dir.is_dir():
        raise FileNotFoundError(f"missing partials dir: {partial_dir}")

    logged = 0
    for path in sorted(partial_dir.glob("*_*.json")):
        if path.name.startswith(".leaderboard_"):
            continue
        payload = _load_partial(path)
        dataset = payload.get("dataset")
        method = payload.get("method")
        metrics = payload.get("metrics")
        config = payload.get("config") or {}
        if not dataset or not method or not metrics:
            print(f"skip {path.name}: incomplete payload")
            continue
        url = log_method_to_wandb(
            dataset,
            method,
            metrics,
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            job_id=job_id,
            dry_run=dry_run,
            force=force,
        )
        if url:
            logged += 1
    return logged


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    n = log_partials_dir(
        args.output_dir.resolve(),
        config_path=args.config,
        job_id=args.job_id,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"logged {n} runs from {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
