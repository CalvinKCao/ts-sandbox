#!/usr/bin/env python3
"""Backfill staged_eval metrics for search4_refit20 jobs that failed wandb GroupName>128."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from utils.leaderboard_config_nicknames import (  # noqa: E402
    leaderboard_nickname,
    leaderboard_staged_eval_tags,
)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
YAML_REL = (
    "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_"
    "bs_mid_vertical_dual_per_ds_best_g_search4_refit20.yaml"
)
RAW_CONFIG = Path(YAML_REL).stem
# Short enough for wandb GroupName (<=128) while staying parseable by RUN_STEM_RE.
SHORT_CONFIG = "binary-anchor-ar-patch-decoder-lb336-hz720-ord-unc-bs-mid-vd-per-ds-best-g-s4r20"
YAML_TAGS = [
    "patch_decoder",
    "ordinal_norm",
    "lb336_hz720",
    "uncompressed",
    "vertical_dual",
    "per_ds_best_g",
    "reuse_g1_pretrain",
    "search4_refit20",
    "lr_eff_batch_univariate",
    "repr_t1",
    "backfill",
]

JOBS = [
    {
        "job_id": "4237795",
        "dataset": "ETTh1",
        "results_subdir": "ETTh1",
    },
    {
        "job_id": "4237796",
        "dataset": "electricity",
        "results_subdir": "electricity_4v_s1",
    },
    {
        "job_id": "4237797",
        "dataset": "exchange_rate",
        "results_subdir": "exchange_rate",
    },
    {
        "job_id": "4237798",
        "dataset": "traffic",
        "results_subdir": "traffic_4v_s1",
    },
]


def _stem(job_id: str, dataset: str) -> str:
    return f"07-14-{job_id}-{dataset}-{RAW_CONFIG}"


def _group(job_id: str, dataset: str) -> str:
    return f"07-14-{job_id}-{dataset}-{SHORT_CONFIG}"


def _metrics_to_wandb(m: dict) -> dict:
    out = {
        "eval/staged_prob_mse": float(m["mse"]),
        "eval/staged_prob_mae": float(m["mae"]),
        "eval/staged_sample_mean_mse": float(m["sample_mean_mse"]),
        "eval/staged_sample_mean_mae": float(m["sample_mean_mae"]),
        "eval/staged_anchor_mse": float(m["anchor_mse"]),
        "eval/staged_anchor_mae": float(m["anchor_mae"]),
        "eval/staged_crps": float(m["crps"]),
    }
    if m.get("top1_mse") is not None:
        out["eval/staged_top1_mse"] = float(m["top1_mse"])
    if m.get("top3_mse") is not None:
        out["eval/staged_top3_mse"] = float(m["top3_mse"])
    if m.get("top1_mae") is not None:
        out["eval/staged_top1_mae"] = float(m["top1_mae"])
    if m.get("top3_mae") is not None:
        out["eval/staged_top3_mae"] = float(m["top3_mae"])
    return out


def main() -> None:
    import wandb
    from models.diffusion_tsf.pipeline.wandb_utils import (
        PIPELINE_JOB_TYPE,
        is_binary_eval_run,
        make_pipeline_run_name,
        truncate_wandb_group,
    )

    yaml_path = REPO / YAML_REL
    nickname = leaderboard_nickname(yaml_path=str(yaml_path))
    api = wandb.Api()
    entity_project = f"{ENTITY}/{PROJECT}"

    for job in JOBS:
        job_id = job["job_id"]
        dataset = job["dataset"]
        stem = _stem(job_id, dataset)
        group = truncate_wandb_group(_group(job_id, dataset))
        results_dir = REPO / "results" / "datasets" / stem / job["results_subdir"]
        staged_path = results_dir / "staged_results.json"
        if not staged_path.is_file():
            raise FileNotFoundError(staged_path)

        payload = json.loads(staged_path.read_text())
        metrics = payload["eval_metrics"]["staged_anchor"]
        summary = _metrics_to_wandb(metrics)

        # Skip if a backfill/pipeline run already exists for this short group.
        existing = [
            r
            for r in api.runs(
                entity_project,
                filters={"group": group},
                order="-created_at",
            )
            if is_binary_eval_run(r)
        ]
        if existing:
            print(f"[skip] {dataset}: already have {existing[0].url}")
            continue

        tags = leaderboard_staged_eval_tags(YAML_TAGS, dataset)
        name = make_pipeline_run_name(group)
        config = {
            "config_nickname": nickname,
            "dataset": dataset,
            "raw_config": RAW_CONFIG,
            "job_id": job_id,
            "metrics_source": str(staged_path),
            "checkpoint_stem": stem,
            "wandb_group_short": group,
            "backfill": True,
            "backfill_reason": "original GroupName exceeded wandb 128-char limit",
            "_yaml_path": str(yaml_path),
            "leaderboard_lookback": 336,
            "leaderboard_horizon": 720,
            "experiment": {
                "lookback_length": 336,
                "forecast_length": 720,
                "experiment_name": RAW_CONFIG,
                "name": SHORT_CONFIG,
                "binary_length_g": payload.get("binary_length_g"),
                "binary_length_mode": payload.get("binary_length_mode"),
                "seed": payload.get("seed"),
                "subset_id": payload.get("subset_id"),
            },
            "eval": {
                "sampler": metrics.get("selected_probabilistic_sampler"),
                "num_inference_steps": metrics.get(
                    "selected_probabilistic_num_inference_steps"
                ),
                "n_samples": metrics.get("n_samples"),
                "metrics_profile": metrics.get("metrics_profile"),
            },
        }

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=name,
            group=group,
            job_type=PIPELINE_JOB_TYPE,
            tags=tags,
            notes=(
                f"Backfill staged_eval from {stem} "
                f"(original wandb init failed: GroupName>128)"
            ),
            config=config,
            settings=wandb.Settings(console="off"),
        )
        try:
            wandb.log(summary, step=0)
            for k, v in summary.items():
                run.summary[k] = v
            url = run.url
            marker = REPO / "results" / "datasets" / stem / "partials" / f".leaderboard_{dataset}.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "group": group,
                        "run_id": run.id,
                        "url": url,
                        "job_type": PIPELINE_JOB_TYPE,
                        "backfill": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"[ok] {dataset}: crps={summary['eval/staged_crps']:.4f} "
                f"anchor_mse={summary['eval/staged_anchor_mse']:.4f} -> {url}"
            )
            print(f"     group={group} (len={len(group)}) tags={tags}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
