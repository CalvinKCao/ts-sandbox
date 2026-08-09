#!/usr/bin/env python3
"""Patch / backfill joint_g_lr_batch_s30r20 rows on ts-sandbox-leaderboard.

Finished pipeline runs already exist but use a truncated name + sprawling
auto nickname. This script:
  - sets a short config_nickname
  - ensures dataset + campaign tags (incl. solar alias)
  - ensures leaderboard_lookback/horizon + experiment lookback/forecast
  - logs eval/staged_* from staged_results.json when summary is missing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from utils.leaderboard_config_nicknames import (  # noqa: E402
    leaderboard_staged_eval_tags,
)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
RAW_CONFIG = (
    "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_"
    "bs_mid_vertical_dual_joint_g_lr_batch_s30r20"
)
NICKNAME = "binary VD joint g+LR+batch s30r20"
YAML_TAGS = [
    "patch_decoder",
    "ordinal_norm",
    "lb336_hz720",
    "uncompressed",
    "vertical_dual",
    "joint_g_lr_batch",
    "reuse_g1_pretrain",
    "search30_refit20",
    "train_window_aug",
    "anchor_mse_objective",
    "repr_t1",
    "backfill",
]

# Current wave (4263255–61). results_subdir = folder under results/datasets/<stem>/
JOBS = [
    {
        "job_id": "4263255",
        "dataset": "ETTh2",
        "results_subdir": "ETTh2",
        "wandb_run_id": "ud1i06d5",
    },
    {
        "job_id": "4263256",
        "dataset": "ETTm1",
        "results_subdir": "ETTm1_4v_s3",
        "wandb_run_id": "ici42aue",
    },
    {
        "job_id": "4263257",
        "dataset": "ETTm2",
        "results_subdir": "ETTm2_7v_s4",
        "wandb_run_id": "0g0bb22w",
    },
    {
        "job_id": "4263258",
        "dataset": "weather",
        "results_subdir": "weather_4v_s2",
        "wandb_run_id": "f5fgnr1j",
    },
    {
        "job_id": "4263259",
        "dataset": "PeMS",
        "results_subdir": "PeMS_7v_s1",
        "wandb_run_id": "cnjtsxmw",
    },
    {
        "job_id": "4263260",
        "dataset": "solar_Alabama",
        "results_subdir": "solar_Alabama_2v_s1",
        "wandb_run_id": "rhr458xs",
    },
    {
        "job_id": "4263261",
        "dataset": "dynamic",
        "results_subdir": "dynamic_2v_s480",
        "wandb_run_id": "1nof4d1a",
    },
]


def _stem(job_id: str, dataset: str) -> str:
    return f"07-15-{job_id}-{dataset}-{RAW_CONFIG}"


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
    for k in ("top1_mse", "top3_mse", "top1_mae", "top3_mae"):
        if m.get(k) is not None:
            out[f"eval/staged_{k}"] = float(m[k])
    return out


def _load_staged(job: dict) -> Optional[Dict[str, Any]]:
    stem = _stem(job["job_id"], job["dataset"])
    path = REPO / "results" / "datasets" / stem / job["results_subdir"] / "staged_results.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _load_meta(job: dict) -> Optional[Dict[str, Any]]:
    stem = _stem(job["job_id"], job["dataset"])
    ckpt = REPO / "results" / "ckpts" / stem
    metas = list(ckpt.rglob("vertical_dual/metadata.json"))
    if not metas:
        return None
    return json.loads(metas[-1].read_text())


def main() -> None:
    import wandb

    api = wandb.Api()
    for job in JOBS:
        dataset = job["dataset"]
        run_id = job["wandb_run_id"]
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        payload = _load_staged(job)
        meta = _load_meta(job) or {}
        tp = dict(meta.get("tuned_params") or {})

        tags = leaderboard_staged_eval_tags(YAML_TAGS, dataset)
        run.tags = tags

        run.config["config_nickname"] = NICKNAME
        run.config["dataset"] = dataset
        run.config["raw_config"] = RAW_CONFIG
        run.config["job_id"] = job["job_id"]
        run.config["leaderboard_lookback"] = 336
        run.config["leaderboard_horizon"] = 720
        run.config["backfill"] = True
        run.config["backfill_reason"] = "short nickname + ensure lb/hz/tags for leaderboard UI"

        exp = dict(run.config.get("experiment") or {}) if isinstance(run.config.get("experiment"), dict) else {}
        exp["lookback_length"] = 336
        exp["forecast_length"] = 720
        exp["experiment_name"] = RAW_CONFIG
        exp["name"] = NICKNAME
        if payload:
            exp["binary_length_g"] = payload.get("binary_length_g")
            exp["binary_length_mode"] = payload.get("binary_length_mode")
            exp["seed"] = payload.get("seed")
            exp["subset_id"] = payload.get("subset_id")
        elif tp:
            exp["binary_length_g"] = tp.get("binary_length_g")
            exp["binary_length_mode"] = tp.get("binary_length_mode")
        run.config["experiment"] = exp
        if tp:
            run.config["tuned_params"] = {
                k: tp.get(k)
                for k in (
                    "learning_rate",
                    "binary_length_g",
                    "binary_length_mode",
                    "target_univariate_batch",
                    "effective_univariate_batch",
                    "batch_size",
                )
            }
            run.config["hp_best_selection_score"] = meta.get("best_selection_score")
            run.config["selection_metric"] = meta.get("selection_metric")

        if payload is not None:
            metrics = payload["eval_metrics"]["staged_anchor"]
            summary = _metrics_to_wandb(metrics)
            for k, v in summary.items():
                run.summary[k] = v
            run.update()
            print(
                f"[ok] {dataset}: crps={summary['eval/staged_crps']:.4f} "
                f"anchor_mse={summary['eval/staged_anchor_mse']:.4f} "
                f"g={exp.get('binary_length_g')} -> {run.url}"
            )
            print(f"     nickname={NICKNAME!r} tags={tags}")
        else:
            run.update()
            print(f"[skip-metrics] {dataset}: no staged_results.json yet (run={run_id})")
            print(f"             tags={tags} nickname={NICKNAME} -> {run.url}")


if __name__ == "__main__":
    main()
