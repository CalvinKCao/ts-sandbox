#!/usr/bin/env python3
"""Manually log lean-disc canvas128 binary + MMPD forecast metrics to leaderboard.

Primary table cells: eval/staged_anchor_{mae,mse} + eval/staged_crps (anchor path).
Also logs sample_mean/prob keys for parity with staged_eval stubs.

First summary table filters: dataset tag, not archive, experiment.lookback/forecast
= 96/96, eval/staged_anchor_mse present. True MMPD protocol is lb336/hz96 — recorded
in config notes; experiment fields are normalized to 96/96 like curate_leaderboard.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name  # noqa: E402
from utils.leaderboard_config_nicknames import (  # noqa: E402
    leaderboard_dataset_tags,
    leaderboard_staged_eval_tags,
    mmpd_stub_wandb_metrics,
)

ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
PROJECT = "ts-sandbox-leaderboard"
METRICS_ROOT = REPO / "temp" / "lean_disc_c128_results" / "forecast_metrics"

BINARY_NICKNAME = "Binary canvas128 p64x6"
MMPD_NICKNAME = "MMPD Subset Recent"
MMPD_BASELINE = "mmpd_decoder_flat_subsets_paper_lb336_hz96_matched_binary"

BINARY_SPECS: List[Dict[str, str]] = [
    {
        "dataset": "ETTh1",
        "job_id": "4571065",
        "date": "08-03",
        "config_stem": "binary_window_norm_patch_refine_canvas128_p64x6",
        "ckpt": "results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6",
    },
    {
        "dataset": "ETTh2",
        "job_id": "4601319",
        "date": "08-04",
        "config_stem": "binary_window_norm_patch_refine_canvas128_p64x6_etth2",
        "ckpt": "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2",
    },
    {
        "dataset": "electricity",
        "job_id": "4597054",
        "date": "08-04",
        "config_stem": "binary_window_norm_patch_refine_canvas128_p64x6_electricity",
        "ckpt": "results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity",
    },
    {
        "dataset": "traffic",
        "job_id": "4597055",
        "date": "08-04",
        "config_stem": "binary_window_norm_patch_refine_canvas128_p64x6_traffic",
        "ckpt": "results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic",
    },
    {
        "dataset": "exchange_rate",
        "job_id": "4597056",
        "date": "08-04",
        "config_stem": "binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
        "ckpt": "results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
    },
]

# MMPD rows that need new stubs (missing matched_binary on leaderboard).
MMPD_CREATE = {
    "ETTh2": {
        "job_id": "manual0805",
        "date": "08-05",
        "raw_config": "mmpd_decoder_flat_subsets_paper_lb336_hz96",
        "metrics_file": METRICS_ROOT / "mmpd" / "ETTh2_mmpd_metrics.json",
        "source_note": "08-04-mmpd-decoder-paper-lb336-hz96-ETTh2/metrics.json; n_samples=100",
    },
    "exchange_rate": {
        "job_id": "manual0805",
        "date": "08-05",
        "raw_config": MMPD_BASELINE,
        "metrics_file": METRICS_ROOT / "mmpd" / "exchange_rate_mmpd.json",
        "source_note": "07-29 matched_binary / ordinal-nonordinal pack partial",
    },
}

# Existing matched_binary stubs with correct anchors; need 96/96 for first table.
MMPD_EXISTING_UPDATE = {
    "ETTh1": "vgz5nrj9",
    "electricity": "50pa90n2",
    "traffic": "3w848g26",
}


def _require_api_key() -> None:
    if not (os.environ.get("WANDB_API_KEY") or "").strip():
        raise SystemExit("WANDB_API_KEY is not set; cannot log leaderboard runs.")


def _load_summary() -> Dict[str, Any]:
    return json.loads((METRICS_ROOT / "summary_table.json").read_text())


def _binary_metrics(dataset: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    anchor_path = METRICS_ROOT / "binary" / f"{dataset}_staged_anchor.json"
    raw = json.loads(anchor_path.read_text())
    summary = _load_summary()
    row = next(r for r in summary["rows"] if r["dataset"] == dataset)
    return raw, row


def _mmpd_metrics_from_file(path: Path, dataset: str) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if dataset in data and isinstance(data[dataset], dict):
        block = data[dataset]
        if "mmpd" in block and isinstance(block["mmpd"], dict):
            return block["mmpd"]
        return block
    if "mmpd" in data and isinstance(data["mmpd"], dict) and dataset not in data:
        # flat partial
        return data
    if "anchor_mse" in data or "mse" in data:
        return data
    raise ValueError(f"could not parse MMPD metrics from {path}")


def _wandb_metric_payload(raw: Dict[str, Any]) -> Dict[str, float]:
    """Anchor keys primary; also emit prob/sample_mean from mae/mse."""
    stub = {
        "anchor_mse": raw["anchor_mse"],
        "anchor_mae": raw["anchor_mae"],
        "crps": raw["crps"],
        "raw": raw,
    }
    return mmpd_stub_wandb_metrics(stub)


def _set_short_horizon(config: Dict[str, Any], *, protocol_lb: int, protocol_hz: int) -> None:
    # Report first table filters on experiment.lookback/forecast == 96/96.
    config["leaderboard_lookback"] = 96
    config["leaderboard_horizon"] = 96
    config["experiment"] = {
        "lookback_length": 96,
        "forecast_length": 96,
    }
    config["protocol_lookback"] = protocol_lb
    config["protocol_horizon"] = protocol_hz


def _existing_group(api, *, job_type: str, group: str) -> Optional[Any]:
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group}, per_page=20):
        if run.job_type == job_type and (run.group or "") == group:
            return run
    return None


def create_binary_run(spec: Dict[str, str], *, dry_run: bool) -> Dict[str, Any]:
    import wandb

    dataset = spec["dataset"]
    raw, row = _binary_metrics(dataset)
    metrics = _wandb_metric_payload(raw)
    group = f"{spec['date']}-{spec['job_id']}-{dataset}-{spec['config_stem']}"
    name = make_phase_run_name(group, "staged_eval")
    n_windows = row["binary"]["n_windows"]
    n_samples = row["binary"]["n_samples"]
    protocol_note = row.get("protocol_note") or (
        "binary staged_eval subsample vs MMPD full stride-4 pack"
    )

    tags = leaderboard_staged_eval_tags(
        [
            "canvas128",
            "p64x6",
            "window_norm",
            "patch_refine",
            "manual-log",
            "lean-disc-c128",
            "lb96_hz96",
        ],
        dataset,
    )
    config: Dict[str, Any] = {
        "config_nickname": BINARY_NICKNAME,
        "dataset": dataset,
        "stub": True,
        "job_id": spec["job_id"],
        "baseline": spec["config_stem"],
        "ckpt_path": spec["ckpt"],
        "metrics_source": str(METRICS_ROOT / "binary" / f"{dataset}_staged_anchor.json"),
        "n_windows": n_windows,
        "n_samples": n_samples,
        "eval_pool": "pipeline_staged_eval_subsample",
        "protocol_note": protocol_note,
        "ce_status": "missing — epsilon prediction_target, no canvas CE in staged jsons",
        "manual_log_campaign": "lean_disc_c128_forecast_metrics",
    }
    _set_short_horizon(config, protocol_lb=336, protocol_hz=96)

    out = {"dataset": dataset, "kind": "binary", "group": group, "name": name}
    if dry_run:
        out["dry_run"] = True
        out["metrics"] = metrics
        print(f"[dry-run] binary {dataset}: {name}")
        print(f"  anchor_mse={metrics['eval/staged_anchor_mse']} "
              f"anchor_mae={metrics['eval/staged_anchor_mae']} "
              f"crps={metrics['eval/staged_crps']}")
        return out

    api = wandb.Api()
    existing = _existing_group(api, job_type="staged_eval", group=group)
    if existing is not None:
        out["skipped"] = "existing_group"
        out["run_id"] = existing.id
        out["url"] = existing.url
        print(f"[skip] binary {dataset}: group exists {existing.url}")
        return out

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        group=group,
        job_type="staged_eval",
        tags=tags,
        notes=(
            f"Manual lean-disc canvas128 staged_eval stub. "
            f"ANCHOR metrics from {config['metrics_source']}. "
            f"Pool: subsample n_windows={n_windows} (not full stride-4). "
            f"True protocol noted as lb336/hz96; experiment fields 96/96 for report tab."
        ),
        config=config,
        settings=wandb.Settings(console="off"),
    )
    try:
        wandb.log(metrics, step=0)
        for k, v in metrics.items():
            run.summary[k] = v
        out["run_id"] = run.id
        out["url"] = run.url
        out["metrics"] = metrics
        print(f"[ok] binary {dataset}: {run.url}")
        return out
    finally:
        wandb.finish()


def create_mmpd_run(dataset: str, spec: Dict[str, Any], *, dry_run: bool) -> Dict[str, Any]:
    import wandb

    raw = _mmpd_metrics_from_file(spec["metrics_file"], dataset)
    metrics = _wandb_metric_payload(raw)
    group = f"{spec['date']}-{spec['job_id']}-{dataset}-{spec['raw_config']}"
    name = make_phase_run_name(group, "mmpd_eval")
    summary = _load_summary()
    row = next(r for r in summary["rows"] if r["dataset"] == dataset)
    n_windows = row["mmpd"]["n_windows"]
    n_samples = row["mmpd"]["n_samples"]

    tags = leaderboard_dataset_tags(dataset) + [
        "mmpd",
        "stub",
        "manual-log",
        "lean-disc-c128",
        spec["raw_config"],
    ]
    config: Dict[str, Any] = {
        "config_nickname": MMPD_NICKNAME,
        "dataset": dataset,
        "baseline": spec["raw_config"],
        "stub": True,
        "job_id": spec["job_id"],
        "metrics_source": str(spec["metrics_file"]),
        "source_note": spec["source_note"],
        "n_windows": n_windows,
        "n_samples": n_samples,
        "eval_pool": "full_stride4_pack",
        "protocol_note": row.get("protocol_note"),
        "manual_log_campaign": "lean_disc_c128_forecast_metrics",
    }
    _set_short_horizon(config, protocol_lb=336, protocol_hz=96)

    out = {"dataset": dataset, "kind": "mmpd", "group": group, "name": name}
    if dry_run:
        out["dry_run"] = True
        out["metrics"] = metrics
        print(f"[dry-run] mmpd {dataset}: {name}")
        print(f"  anchor_mse={metrics['eval/staged_anchor_mse']} "
              f"anchor_mae={metrics['eval/staged_anchor_mae']} "
              f"crps={metrics['eval/staged_crps']}")
        return out

    api = wandb.Api()
    existing = _existing_group(api, job_type="mmpd_eval", group=group)
    if existing is not None:
        out["skipped"] = "existing_group"
        out["run_id"] = existing.id
        out["url"] = existing.url
        print(f"[skip] mmpd {dataset}: group exists {existing.url}")
        return out

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        group=group,
        job_type="mmpd_eval",
        tags=tags,
        notes=(
            f"Manual MMPD eval stub for lean-disc canvas128 compare. "
            f"ANCHOR metrics from {spec['metrics_file']}. "
            f"{spec['source_note']}. "
            f"True protocol lb336/hz96; experiment fields 96/96 for first summary table."
        ),
        config=config,
        settings=wandb.Settings(console="off"),
    )
    try:
        wandb.log(metrics, step=0)
        for k, v in metrics.items():
            run.summary[k] = v
        out["run_id"] = run.id
        out["url"] = run.url
        out["metrics"] = metrics
        print(f"[ok] mmpd {dataset}: {run.url}")
        return out
    finally:
        wandb.finish()


def update_existing_mmpd_for_first_table(run_id: str, dataset: str, *, dry_run: bool) -> Dict[str, Any]:
    """Normalize existing matched_binary stub to 96/96 so it hits the first table."""
    import wandb

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    summary = dict(run.summary) if run.summary else {}
    out = {
        "dataset": dataset,
        "kind": "mmpd_update",
        "run_id": run_id,
        "url": run.url,
        "anchor_mse": summary.get("eval/staged_anchor_mse"),
        "anchor_mae": summary.get("eval/staged_anchor_mae"),
        "crps": summary.get("eval/staged_crps"),
    }
    if dry_run:
        out["dry_run"] = True
        print(f"[dry-run] would normalize {run_id} {dataset} -> experiment 96/96")
        return out

    cfg = dict(run.config or {})
    run.config["leaderboard_lookback"] = 96
    run.config["leaderboard_horizon"] = 96
    exp = dict(cfg.get("experiment") or {})
    exp["lookback_length"] = 96
    exp["forecast_length"] = 96
    run.config["experiment"] = exp
    run.config["protocol_lookback"] = 336
    run.config["protocol_horizon"] = 96
    run.config["manual_log_campaign"] = "lean_disc_c128_forecast_metrics"
    run.config["eval_pool"] = "full_stride4_pack"
    if not str(cfg.get("config_nickname") or "").strip():
        run.config["config_nickname"] = MMPD_NICKNAME
    # Ensure dataset tag present (already should be).
    tags = list(run.tags or [])
    for t in leaderboard_dataset_tags(dataset) + ["mmpd", "lean-disc-c128"]:
        if t not in tags:
            tags.append(t)
    run.tags = tags
    run.update()
    print(f"[ok] normalized mmpd {dataset} {run.url} -> 96/96 for first table")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _require_api_key()
    results: List[Dict[str, Any]] = []

    print("=== binary staged_eval stubs (anchor primary) ===")
    for spec in BINARY_SPECS:
        results.append(create_binary_run(spec, dry_run=args.dry_run))

    print("=== mmpd create missing ===")
    for dataset, spec in MMPD_CREATE.items():
        results.append(create_mmpd_run(dataset, spec, dry_run=args.dry_run))

    print("=== mmpd normalize existing for 96/96 tab ===")
    for dataset, run_id in MMPD_EXISTING_UPDATE.items():
        results.append(update_existing_mmpd_for_first_table(run_id, dataset, dry_run=args.dry_run))

    out_path = METRICS_ROOT / "leaderboard_log_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
