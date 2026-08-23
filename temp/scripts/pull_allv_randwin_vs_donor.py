#!/usr/bin/env python3
"""Pull wandb summaries for allv-randwin-lr4 vs donor trains vs 4928417 evals."""
from __future__ import annotations

import json
import os
import re

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
PROJECT = "ts-sandbox-leaderboard"

SUMMARY_KEYS = [
    "hp/coarse_diff_ft_best_val_loss",
    "hp/coarse_diff_ft_hp_best_val_loss",
    "hp/coarse_diff_ft_best_lr",
    "hp/coarse_diff_ft_max_scale",
    "hp/coarse_diff_ft_best_trial",
    "hp/patch_refine_diff_ft_best_val_loss",
    "hp/patch_refine_diff_ft_hp_best_val_loss",
    "hp/patch_refine_diff_ft_best_lr",
    "hp/patch_refine_diff_ft_max_scale",
    "hp/patch_refine_diff_ft_best_trial",
    "hp/patch_refine_diff_ft_refit_completed",
    "eval/staged_prob_mse",
    "eval/staged_anchor_mse",
    "eval/staged_crps",
    "eval/staged_n_windows",
]

DONOR_JOBS = {
    "4571065": "ETTh1",
    "4601319": "ETTh2",
    "4597054": "electricity",
    "4597055": "traffic",
    "4597056": "exchange_rate",
    "4623005": "PeMS",
    "4623006": "solar_Alabama",
    "4623007": "ETTm1",
    "4623008": "ETTm2",
    "4849780": "weather",
}
EVAL_JOBS = [str(i) for i in range(4928417, 4928426)]
NEW_JOBS = [str(i) for i in range(4946708, 4946718)] + ["4948103", "4948247"]


def pick(run, keys):
    s = run.summary
    out = {}
    for k in keys:
        v = s.get(k)
        if v is not None:
            out[k] = v
    out["_id"] = run.id
    out["_name"] = run.name
    out["_state"] = run.state
    out["_created"] = str(run.created_at)
    out["_tags"] = list(run.tags or [])
    cfg = run.config or {}
    out["_dataset"] = cfg.get("dataset") or cfg.get("experiment", {}).get("dataset")
    out["_job_type"] = run.job_type
    out["_group"] = run.group
    return out


def main():
    api = wandb.Api(timeout=60)
    path = f"{ENTITY}/{PROJECT}"
    rows = {"new": [], "donor_train": [], "donor_eval": []}

    print("=== tag filter: all_variates + randwin + lr4 ===")
    runs = api.runs(path, filters={"tags": {"$in": ["randwin"]}}, order="-created_at")
    for run in runs:
        tags = set(run.tags or [])
        if "randwin" in tags or "all_variates" in tags and "lr4" in tags:
            rec = pick(run, SUMMARY_KEYS)
            rec["_match"] = "randwin_scan"
            print(
                f"{run.created_at} {run.state:10} {run.id} {run.name} "
                f"fine={run.summary.get('hp/patch_refine_diff_ft_best_val_loss')} "
                f"coarse={run.summary.get('hp/coarse_diff_ft_best_val_loss')} "
                f"prob={run.summary.get('eval/staged_prob_mse')} tags={sorted(tags)}"
            )
            if "lr4" in tags or "randwin" in tags:
                rows["new"].append(rec)

    print("\n=== name contains job ids ===")
    job_ids = list(DONOR_JOBS) + EVAL_JOBS + NEW_JOBS
    for jid in job_ids:
        found = list(api.runs(path, filters={"display_name": {"$regex": jid}}))
        kind = (
            "donor_eval"
            if jid in EVAL_JOBS
            else ("new" if jid in set(NEW_JOBS) else "donor_train")
        )
        if not found:
            print(f"  MISSING {kind} {jid}")
            continue
        for run in found:
            rec = pick(run, SUMMARY_KEYS)
            rec["_job"] = jid
            rec["_kind"] = kind
            rows[kind].append(rec)
            print(
                f"  {kind:12} {jid} {run.state:10} {run.id} {run.name}\n"
                f"    coarse={run.summary.get('hp/coarse_diff_ft_best_val_loss')} "
                f"fine={run.summary.get('hp/patch_refine_diff_ft_best_val_loss')} "
                f"fine_hp={run.summary.get('hp/patch_refine_diff_ft_hp_best_val_loss')} "
                f"fine_lr={run.summary.get('hp/patch_refine_diff_ft_best_lr')} "
                f"ms={run.summary.get('hp/patch_refine_diff_ft_max_scale')} "
                f"prob={run.summary.get('eval/staged_prob_mse')} "
                f"anchor={run.summary.get('eval/staged_anchor_mse')} "
                f"crps={run.summary.get('eval/staged_crps')} "
                f"nwin={run.summary.get('eval/staged_n_windows')}"
            )

    out_path = "/home/cao/ts-sandbox/temp/allv_randwin_vs_donor.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
