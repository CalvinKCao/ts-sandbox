#!/usr/bin/env python3
"""Backfill the three completed overlap/value-width pipeline runs into W&B."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import wandb

from models.diffusion_tsf.pipeline.config import build_wandb_config, load_experiment_config
from models.diffusion_tsf.pipeline.wandb_utils import (
    build_pipeline_tags,
    load_manifest,
    record_pipeline_run_id,
)
from utils.leaderboard_config_nicknames import leaderboard_nickname
from utils.load_dotenv import load_repo_dotenv


CONFIG = REPO / "configs/bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp.yaml"
PROJECT = "ts-sandbox-leaderboard"
RUNS = (
    ("ETTh1", "07-21-66077243-ETTh1-bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp"),
    ("traffic", "07-21-66077247-traffic-bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp"),
    ("exchange_rate", "07-21-66077249-exchange_rate-bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp"),
)
VIZ_KEYS = {
    "dual_concat_synthetic_pretrain": "viz/dual_concat_synthetic_pretrain",
    "patch_guidance_finetuned": "viz/patch_guidance_finetuned",
    "staged_eval": "viz/staged_eval",
    "vertical_dual_repr": "viz/vertical_dual_repr",
    "eval_worst": "eval/worst_samples",
    "eval_prob_samples": "eval/probabilistic_samples",
    "eval_dataset_splits": "eval/full_dataset_splits",
    "ordinal_roundtrip": "eval/ordinal_roundtrip",
    "ordinal_coarse_fine_2d": "eval/ordinal_coarse_fine_2d",
}


def staged_result_path(dataset_root: Path) -> Path:
    matches = sorted(dataset_root.glob("*/staged_results.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one staged_results.json under {dataset_root}, found {matches}")
    return matches[0]


def run_state(dataset: str, stem: str, result: dict[str, Any], ckpt_dir: Path, dataset_dir: Path) -> SimpleNamespace:
    experiment = load_experiment_config(str(CONFIG))["experiment"]
    return SimpleNamespace(
        checkpoint_dir=str(ckpt_dir),
        results_dir=str(REPO / "results"),
        datasets_dir=str(dataset_dir),
        smoke_test=False,
        resume=False,
        subset_id=str(result["subset_id"]),
        variate_indices=list(result["variate_indices"]),
        lookback_length=int(experiment["lookback_length"]),
        forecast_length=int(experiment["forecast_length"]),
        dataset=dataset,
        parallel_optuna_workers=1,
    )


def eval_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval/staged_prob_mse": metrics.get("mse"),
        "eval/staged_prob_mae": metrics.get("mae"),
        "eval/staged_sample_mean_mse": metrics.get("sample_mean_mse"),
        "eval/staged_sample_mean_mae": metrics.get("sample_mean_mae"),
        "eval/staged_anchor_mse": metrics.get("anchor_mse"),
        "eval/staged_anchor_mae": metrics.get("anchor_mae"),
        "eval/staged_crps": metrics.get("crps"),
        "eval/staged_top1_mse": metrics.get("top1_mse"),
        "eval/staged_top3_mse": metrics.get("top3_mse"),
        "eval/selected_sampler": metrics.get("selected_probabilistic_sampler"),
        "eval/selected_steps": metrics.get("selected_probabilistic_num_inference_steps"),
    }


def visualization_groups(dataset_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    viz_root = dataset_dir / "viz"
    for child, key in VIZ_KEYS.items():
        paths = sorted(p for p in (viz_root / child).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if paths:
            groups[key] = paths
    return groups


def backfill(dataset: str, stem: str, *, dry_run: bool) -> None:
    ckpt_dir = REPO / "results/ckpts" / stem
    dataset_dir = REPO / "results/datasets" / stem
    log_path = REPO / "results/logs" / f"{stem}.log"
    result_path = staged_result_path(dataset_dir)
    result = json.loads(result_path.read_text())
    metrics = result["eval_metrics"]["staged_anchor"]
    manifest = load_manifest(str(ckpt_dir))
    if manifest is None:
        raise RuntimeError(f"Missing W&B manifest: {ckpt_dir}")
    if manifest.get("run_id"):
        print(f"skip {dataset}: already backfilled as {manifest['run_id']}")
        return

    cfg = load_experiment_config(str(CONFIG))
    phases = cfg["phases"]
    phase = next(p for p in phases if p["phase"] == "staged_eval")
    state = run_state(dataset, stem, result, ckpt_dir, dataset_dir)
    phase_overrides = {key: value for key, value in phase.items() if key != "phase"}
    config = build_wandb_config(
        cfg, state, phase_name="staged_eval", phase_overrides=phase_overrides,
    )
    config["config_nickname"] = leaderboard_nickname(yaml_path=str(CONFIG))
    config["run_stem"] = stem
    config["slurm_job_id"] = re.match(r"^\d\d-\d\d-(\d+)-", stem).group(1)
    config["data_subset"] = result["data_subset"]
    config["binary_length_mode"] = result["binary_length_mode"]
    config["binary_length_g"] = result["binary_length_g"]
    config["backfilled_from_local_results"] = True

    tags = build_pipeline_tags(
        dataset=dataset,
        phase_names=[str(p["phase"]) for p in phases],
        extra_tags=list(manifest.get("tags") or []),
    )
    viz = visualization_groups(dataset_dir)
    if dry_run:
        print(json.dumps({
            "dataset": dataset,
            "group": stem,
            "project": PROJECT,
            "tags": tags,
            "lookback": config["leaderboard_lookback"],
            "horizon": config["leaderboard_horizon"],
            "metrics": eval_metrics(metrics),
            "visualization_counts": {key: len(paths) for key, paths in viz.items()},
        }, indent=2, sort_keys=True))
        return

    run = wandb.init(
        project=PROJECT,
        group=stem,
        job_type="pipeline",
        name=stem,
        tags=tags,
        config=config,
    )
    try:
        record_pipeline_run_id(str(ckpt_dir), run.id, manifest)
        run.log(eval_metrics(metrics), step=0)
        for key, paths in viz.items():
            run.log({key: [wandb.Image(str(path), caption=path.name) for path in paths]})

        config_artifact = wandb.Artifact("experiment-yaml", type="config")
        config_artifact.add_file(str(CONFIG), name=CONFIG.name)
        run.log_artifact(config_artifact)

        files_artifact = wandb.Artifact("pipeline-run-files", type="run-files")
        files_artifact.add_file(str(result_path), name="staged_results.json")
        files_artifact.add_file(str(log_path), name=log_path.name)
        run.log_artifact(files_artifact)
        print(f"backfilled {dataset}: {run.url}")
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_repo_dotenv(str(REPO))
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise RuntimeError("WANDB_API_KEY is unavailable after loading the repo .env")
    for dataset, stem in RUNS:
        backfill(dataset, stem, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
