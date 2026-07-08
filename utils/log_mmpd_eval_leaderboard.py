"""Log MMPD eval partials into ts-sandbox-leaderboard (one mmpd_eval run per dataset)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.leaderboard_config_nicknames import (
    leaderboard_dataset_tags,
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


def _mmpd_horizon_from_yaml(path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
    if path is None or not path.is_file():
        return None, None
    import yaml

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    mmpd = data.get("mmpd") or {}
    lookback = mmpd.get("lookback")
    horizon = mmpd.get("horizon")
    return (
        int(lookback) if lookback is not None else None,
        int(horizon) if horizon is not None else None,
    )


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
    mmpd_run_config: Optional[Path] = None,
    raw_config: Optional[str] = None,
    job_id: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    extra_tags: Optional[list[str]] = None,
    viz_paths: Optional[list[str]] = None,
) -> Optional[str]:
    """Create or skip an mmpd_eval stub in ts-sandbox-leaderboard. Returns run URL if created."""
    if raw_config is None:
        if mmpd_run_config is None:
            raise ValueError("mmpd_run_config or raw_config is required for leaderboard logging")
        raw_config = raw_config_from_run_config(mmpd_run_config)
    if not force and load_leaderboard_marker(output_dir, dataset) is not None:
        print(f"[leaderboard] {dataset}: already logged (marker)")
        return None

    if not _api_key_usable():
        print(f"[leaderboard] {dataset}: skip (WANDB_API_KEY not set)")
        return None

    from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name

    group = _leaderboard_group(
        dataset=dataset,
        raw_config=raw_config,
        output_dir=output_dir,
        job_id=job_id,
    )
    nickname = leaderboard_nickname(yaml_path=str(mmpd_run_config.resolve())) if mmpd_run_config else leaderboard_nickname(raw_config=raw_config)
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
        "stub": True,
    }
    if mmpd_run_config is not None:
        config["mmpd_run_config"] = str(mmpd_run_config.resolve())
        config["baseline"] = raw_config
        lb, hz = _mmpd_horizon_from_yaml(mmpd_run_config)
        if lb is not None or hz is not None:
            exp_block: Dict[str, Any] = {}
            if lb is not None:
                exp_block["lookback_length"] = lb
                config["leaderboard_lookback"] = lb
            if hz is not None:
                exp_block["forecast_length"] = hz
                config["leaderboard_horizon"] = hz
            config["experiment"] = exp_block
    if tuning_path.is_file():
        config["tuning_path"] = str(tuning_path)
    if tuned_hparams:
        config["mmpd_tuned_hparams"] = tuned_hparams

    tags = leaderboard_dataset_tags(dataset) + ["eval", "mmpd", "stub", raw_config]
    if extra_tags:
        tags.extend(t for t in extra_tags if t not in tags)

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        group=group,
        job_type=JOB_TYPE,
        tags=tags,
        notes=f"MMPD eval auto-log from {output_dir / 'partials' / f'{dataset}_mmpd.json'}",
        config=config,
        settings=wandb.Settings(console="off"),
    )
    try:
        clean = {k: v for k, v in summary.items() if v is not None}
        wandb.log(clean, step=0)
        for k, v in clean.items():
            run.summary[k] = v
        if viz_paths:
            from models.diffusion_tsf.pipeline import wandb_utils
            wandb_utils.log_visualization_paths(viz_paths, wandb_key="eval/mmpd_visualizations")
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
    viz_paths: list[str] = []
    if getattr(args, "use_ordinal_window_norm", False):
        try:
            from pathlib import Path as _Path

            from utils.visualize_ordinal_coarse_fine_2d import plot_ordinal_coarse_fine_2d
            from utils.visualize_ordinal_roundtrip import plot_roundtrip

            out_dir = args.output_dir / "viz" / "ordinal_roundtrip"
            out_dir.mkdir(parents=True, exist_ok=True)
            viz_paths.append(
                str(plot_roundtrip(
                    dataset=dataset,
                    config_path=_Path(args.mmpd_run_config),
                    out_dir=out_dir,
                    window_idx=0,
                    variate=0,
                    prefer_ties=False,
                ))
            )
            repr_dir = args.output_dir / "viz" / "ordinal_coarse_fine_2d"
            repr_dir.mkdir(parents=True, exist_ok=True)
            viz_paths.append(
                str(plot_ordinal_coarse_fine_2d(
                    dataset=dataset,
                    config_path=_Path(args.mmpd_run_config),
                    out_dir=repr_dir,
                    window_idx=0,
                    variate=0,
                ))
            )
        except Exception as exc:
            print(f"[leaderboard] {dataset}: ordinal viz skipped ({exc})")
    try:
        log_mmpd_eval_to_leaderboard(
            dataset=dataset,
            metrics=metrics,
            output_dir=args.output_dir,
            mmpd_run_config=args.mmpd_run_config,
            force=getattr(args, "force_mmpd_leaderboard", False),
            viz_paths=viz_paths or None,
        )
    except Exception as exc:
        print(f"[leaderboard] {dataset}: failed ({exc})")
