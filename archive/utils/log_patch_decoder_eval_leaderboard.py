"""Log patch-decoder staged_eval metrics into ts-sandbox-leaderboard from local logs."""

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
CONFIG_NICK = "binary_anchor_ar_patch_decoder_ctx"
JOB_TYPE = "staged_eval"
LOG_DIR = os.path.join(REPO, "results", "logs")

RUNS = [
    ("07-02-4037033-ETTh1-binary_anchor_ar_patch_decoder_ctx", "ETTh1"),
    ("07-02-4037034-ETTh2-binary_anchor_ar_patch_decoder_ctx", "ETTh2"),
    ("07-02-4037035-ETTm1-binary_anchor_ar_patch_decoder_ctx", "ETTm1"),
    ("07-02-4037036-ETTm2-binary_anchor_ar_patch_decoder_ctx", "ETTm2"),
    ("07-02-4037037-exchange_rate-binary_anchor_ar_patch_decoder_ctx", "exchange_rate"),
    ("07-02-4037038-weather-binary_anchor_ar_patch_decoder_ctx", "weather"),
    ("07-02-4037039-traffic-binary_anchor_ar_patch_decoder_ctx", "traffic"),
]

EVAL_PAT = re.compile(
    r"staged eval done: sampler=(\S+) steps=(\d+) "
    r"prob_mse=([\d.]+) prob_mae=([\d.]+) "
    r"anchor_mse=([\d.]+) anchor_mae=([\d.]+) crps=([\d.]+)"
)


def _find_log(group: str) -> str | None:
    if not os.path.isdir(LOG_DIR):
        return None
    for fn in os.listdir(LOG_DIR):
        if fn.startswith(group) and fn.endswith(".log"):
            return os.path.join(LOG_DIR, fn)
    return None


def _parse_eval(log_path: str) -> re.Match[str] | None:
    last = None
    with open(log_path) as f:
        for line in f:
            m = EVAL_PAT.search(line)
            if m:
                last = m
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import wandb
    from wandb import Api

    api = Api()
    existing = {
        (run.group, run.job_type)
        for run in api.runs(f"{ENTITY}/{PROJECT}")
        if run.job_type == JOB_TYPE and (run.group or "").startswith("07-02-403703")
    }

    created = skipped = 0
    for group, dataset in RUNS:
        key = (group, JOB_TYPE)
        if key in existing:
            print(f"[skip] exists: {group}")
            skipped += 1
            continue

        log_path = _find_log(group)
        if not log_path:
            print(f"[skip] no log: {group}")
            skipped += 1
            continue

        m = _parse_eval(log_path)
        if not m:
            print(f"[skip] no eval line: {log_path}")
            skipped += 1
            continue

        sampler, steps, prob_mse, prob_mae, anchor_mse, anchor_mae, crps = m.groups()
        metrics = {
            "eval/staged_prob_mse": float(prob_mse),
            "eval/staged_prob_mae": float(prob_mae),
            "eval/staged_sample_mean_mse": float(prob_mse),
            "eval/staged_sample_mean_mae": float(prob_mae),
            "eval/staged_anchor_mse": float(anchor_mse),
            "eval/staged_anchor_mae": float(anchor_mae),
            "eval/staged_crps": float(crps),
            "eval/selected_sampler": sampler,
            "eval/selected_steps": int(steps),
        }
        name = make_phase_run_name(group, JOB_TYPE)

        if args.dry_run:
            print(f"would create {name} | {dataset} | {metrics}")
            created += 1
            continue

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=name,
            group=group,
            job_type=JOB_TYPE,
            tags=[dataset, "eval", "curated-relog", "binary"],
            notes=(
                f"manual eval relog from {log_path}; "
                "original wandb project ts-sandbox-binary-anchor-92d3"
            ),
            config={
                "config_nickname": CONFIG_NICK,
                "dataset": dataset,
                "experiment_name": CONFIG_NICK,
                "guidance_type": "patch_decoder",
                "grad_accum_multiplier": 1.5,
                "metrics_source": "log_tail",
                "log_path": log_path,
                "stub": True,
            },
            settings=wandb.Settings(console="off"),
        )
        try:
            wandb.log(metrics, step=0)
            for k, v in metrics.items():
                run.summary[k] = v
            print(f"created {run.url}")
            created += 1
        finally:
            wandb.finish()

    print({"created": created, "skipped": skipped})


if __name__ == "__main__":
    main()
