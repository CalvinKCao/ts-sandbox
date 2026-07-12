#!/usr/bin/env python3
"""Set lookback/horizon fields on ts-sandbox-leaderboard eval runs for report filters."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

ENTITY = "calvincao"
PROJECT = "ts-sandbox-leaderboard"

LB_HZ_RE = re.compile(r"lb(\d+)[_-]hz(\d+)", re.IGNORECASE)


def _config_dict(run: Any) -> Dict[str, Any]:
    return dict(run.config or {})


def _from_experiment(exp: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(exp, dict):
        return None, None
    for block in (exp, exp.get("value") if isinstance(exp.get("value"), dict) else {}):
        if not isinstance(block, dict):
            continue
        lb = block.get("lookback_length") or block.get("lookback")
        hz = block.get("forecast_length") or block.get("horizon")
        if lb is not None and hz is not None:
            return int(lb), int(hz)
    return None, None


def _from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = LB_HZ_RE.search(text or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def infer_lookback_horizon(run: Any) -> Tuple[int, int]:
    cfg = _config_dict(run)
    lb = cfg.get("leaderboard_lookback")
    hz = cfg.get("leaderboard_horizon")
    if lb is not None and hz is not None:
        return int(lb), int(hz)

    lb, hz = _from_experiment(cfg.get("experiment"))
    if lb is not None and hz is not None:
        return lb, hz

    for text in (
        run.group or "",
        run.name or "",
        str(cfg.get("baseline") or ""),
        str(cfg.get("mmpd_run_config") or ""),
        str(cfg.get("config_nickname") or ""),
    ):
        lb, hz = _from_text(text)
        if lb is not None and hz is not None:
            return lb, hz

    group_l = (run.group or "").lower()
    if "hz720" in group_l or "lb336_hz720" in group_l:
        return 336, 720
    if "hz96" in group_l or "lb336_hz96" in group_l or "lb336-hz96" in group_l:
        return 336, 96
    if "lb96-hz720" in group_l or "lb96_hz720" in group_l:
        return 96, 720

    tags = set(run.tags or [])
    if run.job_type == "mmpd_eval" or "mmpd" in tags:
        return 336, 96

    return 336, 96


def needs_update(run: Any, lb: int, hz: int) -> bool:
    cfg = _config_dict(run)
    exp_lb, exp_hz = _from_experiment(cfg.get("experiment"))
    if cfg.get("leaderboard_lookback") != lb or cfg.get("leaderboard_horizon") != hz:
        return True
    return exp_lb != lb or exp_hz != hz


def apply_horizon_fields(run: Any, lb: int, hz: int) -> None:
    exp = dict(_config_dict(run).get("experiment") or {})
    if not isinstance(exp, dict):
        exp = {}
    exp["lookback_length"] = lb
    exp["forecast_length"] = hz
    run.config["leaderboard_lookback"] = lb
    run.config["leaderboard_horizon"] = hz
    run.config["experiment"] = exp
    run.update()


def backfill_runs(api: Any, *, dry_run: bool) -> Tuple[int, int]:
    updated = 0
    scanned = 0
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        if "eval" not in (run.tags or []):
            continue
        scanned += 1
        lb, hz = infer_lookback_horizon(run)
        if not needs_update(run, lb, hz):
            continue
        if dry_run:
            print(f"would update {run.id} {run.group} -> lb={lb} hz={hz}")
        else:
            apply_horizon_fields(run, lb, hz)
            print(f"updated {run.id} -> lb={lb} hz={hz}")
        updated += 1
    return scanned, updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    scanned, updated = backfill_runs(api, dry_run=args.dry_run)
    print(f"scanned={scanned} {'would update' if args.dry_run else 'updated'}={updated}")


if __name__ == "__main__":
    main()
