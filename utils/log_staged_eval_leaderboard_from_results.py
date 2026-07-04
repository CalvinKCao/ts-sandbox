#!/usr/bin/env python3
"""Backfill staged_eval metrics into ts-sandbox-leaderboard from local results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.leaderboard_config_nicknames import leaderboard_nickname, parse_run_stem
from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from models.diffusion_tsf.pipeline.wandb_utils import (  # noqa: E402
    load_manifest,
    make_phase_run_name,
)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = "calvincao"
JOB_TYPE = "staged_eval"
CKPT_ROOT = REPO / "results" / "ckpts"
DATA_ROOT = REPO / "results" / "datasets"

EVAL_KEYS = (
    "eval/staged_prob_mse",
    "eval/staged_prob_mae",
    "eval/staged_sample_mean_mse",
    "eval/staged_sample_mean_mae",
    "eval/staged_anchor_mse",
    "eval/staged_anchor_mae",
    "eval/staged_crps",
    "eval/staged_top1_mse",
    "eval/staged_top3_mse",
)


def _metrics_from_partial(partial: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "eval/staged_prob_mse": partial.get("mse"),
        "eval/staged_prob_mae": partial.get("mae"),
        "eval/staged_sample_mean_mse": partial.get("sample_mean_mse"),
        "eval/staged_sample_mean_mae": partial.get("sample_mean_mae"),
        "eval/staged_anchor_mse": partial.get("anchor_mse"),
        "eval/staged_anchor_mae": partial.get("anchor_mae"),
        "eval/staged_crps": partial.get("crps"),
        "eval/staged_top1_mse": partial.get("top1_mse"),
        "eval/staged_top3_mse": partial.get("top3_mse"),
        "eval/selected_sampler": partial.get("selected_probabilistic_sampler"),
        "eval/selected_steps": partial.get("selected_probabilistic_num_inference_steps"),
    }


def _partial_path(run_stem: str, dataset: str) -> Path:
    return DATA_ROOT / run_stem / "partials" / f"{dataset}_staged_anchor.json"


def _ckpt_dir(run_stem: str) -> Path:
    return CKPT_ROOT / run_stem


def _run_has_eval_metrics(run) -> bool:
    summary = dict(run.summary)
    return summary.get("eval/staged_crps") is not None


def _find_run_stems(job_ids: List[str]) -> List[str]:
    stems: List[str] = []
    for jid in job_ids:
        matches = sorted(
            p.name for p in CKPT_ROOT.iterdir()
            if p.is_dir() and f"-{jid}-" in p.name
        )
        if not matches:
            raise FileNotFoundError(f"no ckpt dir for job id {jid}")
        stems.append(matches[-1])
    return stems


def backfill_run(
    run_stem: str,
    *,
    api,
    dry_run: bool,
    force: bool,
) -> Tuple[str, Optional[str]]:
    ckpt_dir = _ckpt_dir(run_stem)
    manifest = load_manifest(str(ckpt_dir))
    if not manifest:
        raise FileNotFoundError(f"missing wandb manifest under {ckpt_dir}")

    parsed = parse_run_stem(run_stem)
    if not parsed:
        raise ValueError(f"cannot parse run stem: {run_stem}")
    _job_id, dataset, raw_config = parsed

    partial_path = _partial_path(run_stem, dataset)
    if not partial_path.is_file():
        raise FileNotFoundError(f"missing partial metrics: {partial_path}")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    metrics = _metrics_from_partial(partial)

    group = str(manifest.get("group") or run_stem)
    project = str(manifest.get("project") or PROJECT)
    phase_runs = dict(manifest.get("phase_runs") or {})
    run_id = phase_runs.get(JOB_TYPE)
    tags = list(manifest.get("tags") or [])
    if dataset not in tags:
        tags.insert(0, dataset)
    if "eval" not in tags:
        tags.append("eval")

    existing = None
    if run_id:
        try:
            existing = api.run(f"{ENTITY}/{project}/{run_id}")
        except Exception:
            existing = None
    if existing is None:
        for run in api.runs(f"{ENTITY}/{project}", filters={"group": group}):
            if run.job_type == JOB_TYPE:
                existing = run
                break

    if existing is not None and _run_has_eval_metrics(existing) and not force:
        return "skipped", existing.url

    config_yaml = manifest.get("config_yaml") or ""
    nick = leaderboard_nickname(raw_config=raw_config)
    name = make_phase_run_name(group, JOB_TYPE)

    if dry_run:
        action = "update" if existing is not None else "create"
        print(f"would {action} {name} | {dataset} | crps={metrics.get('eval/staged_crps')}")
        return action, None

    import wandb

    if existing is not None and force:
        run = wandb.init(
            project=project,
            entity=ENTITY,
            id=existing.id,
            resume="must",
            settings=wandb.Settings(console="off"),
        )
    else:
        init_kwargs: Dict[str, Any] = {
            "project": project,
            "entity": ENTITY,
            "name": name,
            "group": group,
            "job_type": JOB_TYPE,
            "tags": tags + ["curated-relog"],
            "notes": f"manual eval relog from {partial_path}",
            "config": {
                "config_nickname": nick,
                "dataset": dataset,
                "experiment_name": raw_config,
                "guidance_type": "patch_decoder",
                "metrics_source": "partials_json",
                "partial_path": str(partial_path),
                "run_stem": run_stem,
                "stub": True,
            },
            "settings": wandb.Settings(console="off"),
        }
        if run_id and existing is None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)

    try:
        wandb.log(metrics, step=0)
        for key in EVAL_KEYS:
            val = metrics.get(key)
            if val is not None:
                run.summary[key] = val
        if metrics.get("eval/selected_sampler") is not None:
            run.summary["eval/selected_sampler"] = metrics["eval/selected_sampler"]
        if metrics.get("eval/selected_steps") is not None:
            run.summary["eval/selected_steps"] = metrics["eval/selected_steps"]
        action = "updated" if existing is not None else "created"
        return action, run.url
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id",
        action="append",
        dest="job_ids",
        help="Slurm job id(s); discovers *-{job_id}-* ckpt dir",
    )
    parser.add_argument(
        "--run-stem",
        action="append",
        dest="run_stems",
        help="Full run stem (MM-DD-jobid-dataset-config)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-log even if metrics exist")
    args = parser.parse_args()

    stems: List[str] = list(args.run_stems or [])
    if args.job_ids:
        stems.extend(_find_run_stems(args.job_ids))
    if not stems:
        parser.error("pass --job-id and/or --run-stem")

    from wandb import Api

    api = Api()
    summary: Dict[str, Any] = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
    for stem in stems:
        try:
            action, url = backfill_run(stem, api=api, dry_run=args.dry_run, force=args.force)
            summary[action] = summary.get(action, 0) + 1
            if url:
                print(f"[{action}] {stem} -> {url}")
            else:
                print(f"[{action}] {stem}")
        except Exception as exc:
            summary["failed"] += 1
            print(f"[failed] {stem}: {exc}", file=sys.stderr)

    print(summary)


if __name__ == "__main__":
    main()
