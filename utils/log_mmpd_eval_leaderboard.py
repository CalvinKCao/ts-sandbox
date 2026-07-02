"""Log MMPD eval partials into ts-sandbox-leaderboard (one mmpd_eval run per dataset)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.leaderboard_config_nicknames import (
    leaderboard_nickname,
    mmpd_leaderboard_run_stem,
    mmpd_stub_wandb_metrics,
)

REPO = Path(__file__).resolve().parents[1]

try:
    from utils.load_dotenv import load_repo_dotenv

    load_repo_dotenv(REPO)
except ImportError:
    pass

PROJECT = "ts-sandbox-leaderboard"
ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
JOB_TYPE = "mmpd_eval"

_OUTPUT_DATE_RE = re.compile(r"^(\d{2}-\d{2})-")


def _api_key_usable() -> bool:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    return bool(key)


def raw_config_from_run_config(path: Optional[Path]) -> str:
    if path is None:
        raise ValueError("mmpd_run_config is required for leaderboard logging")
    return path.stem


def campaign_date_from_output_dir(output_dir: Path) -> str:
    m = _OUTPUT_DATE_RE.match(output_dir.name)
    if m:
        return m.group(1)
    return datetime.now().strftime("%m-%d")


def leaderboard_marker_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "partials" / f".leaderboard_{dataset}.json"


def load_leaderboard_marker(output_dir: Path, dataset: str) -> Optional[Dict[str, Any]]:
    path = leaderboard_marker_path(output_dir, dataset)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_leaderboard_marker(output_dir: Path, dataset: str, payload: Dict[str, Any]) -> None:
    path = leaderboard_marker_path(output_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _leaderboard_group(
    *,
    dataset: str,
    raw_config: str,
    output_dir: Path,
    job_id: Optional[str] = None,
) -> str:
    jid = job_id or os.environ.get("SLURM_JOB_ID")
    if not jid:
        raise RuntimeError(
            "leaderboard logging needs SLURM_JOB_ID or explicit job_id "
            "(set on MMPD worker jobs; skip with --no-mmpd-leaderboard for local runs)"
        )
    return mmpd_leaderboard_run_stem(
        dataset,
        raw_config,
        job_id=jid,
        campaign_date=campaign_date_from_output_dir(output_dir),
    )


def _existing_run(api, *, entity: str, project: str, group: str) -> bool:
    for run in api.runs(f"{entity}/{project}"):
        if run.group == group and run.job_type == JOB_TYPE:
            return True
    return False


def log_mmpd_eval_to_leaderboard(
    *,
    dataset: str,
    metrics: Dict[str, Any],
    output_dir: Path,
    mmpd_run_config: Path,
    job_id: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Optional[str]:
    """Create or skip an mmpd_eval stub in ts-sandbox-leaderboard. Returns run URL if created."""
    if not force and load_leaderboard_marker(output_dir, dataset) is not None:
        print(f"[leaderboard] {dataset}: already logged (marker)")
        return None

    if not _api_key_usable():
        print(f"[leaderboard] {dataset}: skip (WANDB_API_KEY not set)")
        return None

    from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name

    raw_config = raw_config_from_run_config(mmpd_run_config)
    group = _leaderboard_group(
        dataset=dataset,
        raw_config=raw_config,
        output_dir=output_dir,
        job_id=job_id,
    )
    nickname = leaderboard_nickname(yaml_path=str(mmpd_run_config.resolve()))
    stub_metrics = {
        "anchor_mse": metrics.get("anchor_mse", metrics.get("mse")),
        "anchor_mae": metrics.get("anchor_mae", metrics.get("mae")),
        "crps": metrics.get("crps"),
        "raw": metrics,
    }
    if stub_metrics["anchor_mse"] is None or stub_metrics["anchor_mae"] is None or stub_metrics["crps"] is None:
        print(f"[leaderboard] {dataset}: skip (missing anchor_mse/anchor_mae/crps)")
        return None

    summary = mmpd_stub_wandb_metrics(stub_metrics)
    tuning_path = output_dir / "tuning" / f"{dataset}_best.json"
    tuned_hparams = None
    if tuning_path.is_file():
        with tuning_path.open(encoding="utf-8") as f:
            tuned_hparams = json.load(f).get("hparams")

    name = make_phase_run_name(group, JOB_TYPE)
    if dry_run:
        print(f"[leaderboard] would create {name} | {group}")
        return None

    import wandb

    api = wandb.Api()
    if not force and _existing_run(api, entity=ENTITY, project=PROJECT, group=group):
        print(f"[leaderboard] {dataset}: skip (wandb group exists: {group})")
        write_leaderboard_marker(
            output_dir,
            dataset,
            {"group": group, "skipped": "existing_wandb_group"},
        )
        return None

    config: Dict[str, Any] = {
        "config_nickname": nickname,
        "dataset": dataset,
        "baseline": raw_config,
        "job_id": job_id or os.environ.get("SLURM_JOB_ID"),
        "metrics_source": str(output_dir),
        "partial_path": str(output_dir / "partials" / f"{dataset}_mmpd.json"),
        "mmpd_run_config": str(mmpd_run_config.resolve()),
        "stub": True,
    }
    if tuning_path.is_file():
        config["tuning_path"] = str(tuning_path)
    if tuned_hparams:
        config["mmpd_tuned_hparams"] = tuned_hparams

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        group=group,
        job_type=JOB_TYPE,
        tags=[dataset, "eval", "mmpd", "stub", raw_config],
        notes=f"MMPD eval auto-log from {output_dir / 'partials' / f'{dataset}_mmpd.json'}",
        config=config,
        settings=wandb.Settings(console="off"),
    )
    try:
        clean = {k: v for k, v in summary.items() if v is not None}
        wandb.log(clean, step=0)
        for k, v in clean.items():
            run.summary[k] = v
        url = run.url
        write_leaderboard_marker(
            output_dir,
            dataset,
            {"group": group, "run_id": run.id, "url": url},
        )
        print(f"[leaderboard] {dataset}: {url}")
        return url
    finally:
        wandb.finish()


def maybe_log_mmpd_eval_leaderboard(
    args: Any,
    dataset: str,
    metrics: Dict[str, Any],
) -> None:
    if not getattr(args, "mmpd_log_leaderboard", False):
        return
    if getattr(args, "smoke_test", False):
        return
    if args.mmpd_run_config is None:
        return
    try:
        log_mmpd_eval_to_leaderboard(
            dataset=dataset,
            metrics=metrics,
            output_dir=args.output_dir,
            mmpd_run_config=args.mmpd_run_config,
            force=getattr(args, "force_mmpd_leaderboard", False),
        )
    except Exception as exc:
        print(f"[leaderboard] {dataset}: failed ({exc})")
