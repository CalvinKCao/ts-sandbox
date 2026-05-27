#!/usr/bin/env python3
"""Compare per-window texture/shape metrics: one probabilistic sample vs GT.

For each test window in a shared 50% subset:
  - MMPD: one upstream probabilistic draw (sample_num=1).
  - Gaussian / binary anchor: one DPM++ draw (shape probe; MSE in matrix eval uses anchor-only).

Aggregates the same texture metrics as eval_mmpd_gaussian_anchor (ordinal JSD, RQA,
variogram, path signature) plus pointwise MSE/MAE on that single trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    DATASET_FILES,
    MODEL_KEYS,
    AnchorRun,
    deterministic_metrics,
    explicit_roots_for_variant,
    find_anchor_runs,
    indices_path,
    load_indices_from_disk,
    load_raw_pack,
    load_tsf_test_subset,
    load_tsf_pipeline,
    resolve_storage_paths,
    run_indices_phase,
    run_mmpd_eval,
    stable_dataset_seed,
    texture_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "illness"])
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "datasets" / "prob-shape-texture")
    p.add_argument("--indices-dir", type=Path, default=None, help="raw/ with indices_*.json (default: MMPD_SHARED/raw)")
    p.add_argument("--mmpd-output-root", type=Path, default=None)
    p.add_argument("--mmpd-raw-dir", type=Path, default=None)
    p.add_argument("--mmpd-raw-fallback", type=Path, default=None)
    p.add_argument("--ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    p.add_argument("--mmpd-repo", type=Path, default=REPO_ROOT / "temp" / "MMPD")
    p.add_argument("--mmpd-data-dir", type=Path, default=REPO_ROOT / "temp" / "mmpd_datasets")
    p.add_argument("--anchor-root", action="append", type=Path, default=[])
    p.add_argument("--binary-anchor-root", action="append", type=Path, default=[])
    p.add_argument("--test-fraction", type=float, default=0.5)
    p.add_argument("--test-max-items", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--anchor-batch-size", type=int, default=16)
    p.add_argument("--mmpd-eval-batch-size", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-update-mmpd", action="store_true")
    p.add_argument("--skip-mmpd", action="store_true")
    p.add_argument("--skip-anchors", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def shape_npz_path(out_dir: Path, dataset: str, model_key: str) -> Path:
    return out_dir / "raw" / f"shape_{model_key}_{dataset}.npz"


def evaluate_anchor_prob_sample(
    args: argparse.Namespace,
    run: AnchorRun,
    indices: Sequence[int],
    device: torch.device,
    seed: int,
) -> Dict[str, np.ndarray]:
    """One DPM++ trajectory per window (probabilistic shape probe)."""
    subset = load_tsf_test_subset(
        run.dataset,
        run.metadata["variate_indices"],
        indices,
        args.lookback,
        args.horizon,
    )
    loader = DataLoader(
        subset,
        batch_size=args.anchor_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    from utils.eval_mmpd_gaussian_anchor import load_anchor_model

    model = load_anchor_model(run, args, device)
    y_true: List[np.ndarray] = []
    prob: List[np.ndarray] = []
    det: List[np.ndarray] = []

    gen_kw = {"sampler": "dpmpp", "num_inference_steps": args.num_sampling_steps}
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            past = past.to(device)
            future = future.to(device)
            K = getattr(model.config, "lookback_overlap", 0)
            if K > 0:
                future = future[..., K:]
            y_true.append(future.cpu().numpy())
            torch.manual_seed(seed + batch_idx * 1009)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + batch_idx * 1009)
            prob.append(model.generate(past, **gen_kw)["prediction"].cpu().numpy())
            det.append(model.generate(past, sampler="anchor")["prediction"].cpu().numpy())

    return {
        "y_true": np.concatenate(y_true, axis=0),
        "prob_sample": np.concatenate(prob, axis=0),
        "anchor_det": np.concatenate(det, axis=0),
    }


def summarize_shape_pack(pack: Dict[str, np.ndarray], pred_key: str) -> Dict[str, float]:
    y_true = pack["y_true"]
    pred = pack[pred_key]
    out: Dict[str, float] = {}
    out.update(deterministic_metrics(y_true, pred))
    out.update(texture_metrics(y_true, pred))
    out["n_windows"] = float(y_true.shape[0])
    out["n_variates"] = float(y_true.shape[1])
    return out


def main() -> None:
    args = parse_args()
    args.lookback = 96
    args.horizon = 96
    args.patch_size = 12
    args.datasets = list(dict.fromkeys(args.datasets))
    args.output_dir = args.output_dir.resolve()
    args.ckpt_base = args.ckpt_base.resolve()
    args.mmpd_repo = args.mmpd_repo.resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.resolve()

    if args.indices_dir is not None:
        args.mmpd_raw_dir = args.indices_dir.resolve()
    resolve_storage_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)

    anchors_by_variant = {
        "gaussian": find_anchor_runs(
            args.datasets,
            explicit_roots_for_variant(args, "gaussian"),
            args.ckpt_base,
            variant="gaussian",
        ),
        "binary": find_anchor_runs(
            args.datasets,
            explicit_roots_for_variant(args, "binary"),
            args.ckpt_base,
            variant="binary",
        ),
    }

    indices_path_dir = args.mmpd_raw_dir
    if all((indices_path_dir / f"indices_{d}.json").exists() for d in args.datasets):
        args.mmpd_raw_dir = indices_path_dir
        indices_by_dataset = load_indices_from_disk(args, args.datasets)
    else:
        indices_by_dataset = run_indices_phase(args, anchors_by_variant)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset in args.datasets:
        results[dataset] = {}
        indices = indices_by_dataset[dataset]
        ds_seed = stable_dataset_seed(args.seed, dataset)

        if not args.skip_mmpd:
            key = "mmpd"
            npz = shape_npz_path(args.output_dir, dataset, key)
            if npz.exists() and not args.force:
                pack = load_raw_pack(npz)
            else:
                mmpd_args = argparse.Namespace(**vars(args))
                mmpd_args.sample_num = 1
                mmpd_args.gmm_components = 10
                mmpd_args.gmm_iterations = 10
                mmpd_args.num_workers = 0
                mmpd_args.force_mmpd_eval = True
                full = run_mmpd_eval(mmpd_args, dataset, indices)
                pack = {
                    "y_true": full["y_true"],
                    "prob_sample": full["samples"][:, :, 0, :],
                }
                np.savez_compressed(npz, **pack)
            results[dataset][key] = summarize_shape_pack(pack, "prob_sample")
            print(f"[done] {dataset} mmpd shape metrics")

        if not args.skip_anchors:
            for variant in ("gaussian", "binary"):
                model_key = MODEL_KEYS[variant]
                npz = shape_npz_path(args.output_dir, dataset, model_key)
                if npz.exists() and not args.force:
                    pack = load_raw_pack(npz)
                else:
                    run = anchors_by_variant[variant][dataset]
                    pack = evaluate_anchor_prob_sample(args, run, indices, device, ds_seed)
                    np.savez_compressed(npz, **pack)
                results[dataset][f"{model_key}_prob"] = summarize_shape_pack(pack, "prob_sample")
                results[dataset][f"{model_key}_anchor_det"] = summarize_shape_pack(pack, "anchor_det")
                print(f"[done] {dataset} {variant} prob + anchor_det shape metrics")

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (args.output_dir / "shape_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "results": results}, f, indent=2, sort_keys=True)

    rows = []
    for dataset, by_model in sorted(results.items()):
        for model, metrics in sorted(by_model.items()):
            row = {"dataset": dataset, "model": model}
            row.update(metrics)
            rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r if k not in ("dataset", "model")})
        with (args.output_dir / "shape_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "model"] + keys)
            w.writeheader()
            w.writerows(rows)

    print(f"\nWrote {args.output_dir / 'shape_metrics.json'}")


if __name__ == "__main__":
    main()
