#!/usr/bin/env python3
"""Eval legacy + trend-robust texture metrics on staged guidance iTransformers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import load_itransformer_from_checkpoint
from utils.eval_mmpd_gaussian_anchor import (
    load_tsf_test_subset,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
    stable_dataset_seed,
    summarize_prediction_pack,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import (
    DEFAULT_STAGED_CKPTS,
    dataset_window_lengths_for_run,
    make_indices,
    staged_anchor_run,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration


LEGACY_TEXTURE_KEYS = [
    "texture_ordinal_jsd",
    "texture_rqa_distance",
    "texture_variogram_distance",
    "texture_pathsig_distance",
]

ROBUST_TEXTURE_KEYS = [
    "texture_increment_wasserstein",
    "texture_curvature_wasserstein",
    "texture_haar_detail_jsd",
    "texture_jump_plateau_distance",
    "texture_derivative_motif_jsd",
]

TEXTURE_KEYS = LEGACY_TEXTURE_KEYS + ROBUST_TEXTURE_KEYS
CORE_KEYS = ["mse", "mae", "crps", "top1_mse", "top3_mse"]


def resolve_lookback_overlap(sub: Mapping[str, Any]) -> int:
    coarse_meta = Path(sub["coarse_pt"]).parent / "metadata.json"
    if coarse_meta.is_file():
        with coarse_meta.open(encoding="utf-8") as f:
            tuned = json.load(f).get("tuned_params") or {}
        if "lookback_overlap" in tuned:
            return int(tuned["lookback_overlap"])
    from models.diffusion_tsf.pipeline_config import LOOKBACK_OVERLAP

    return int(LOOKBACK_OVERLAP)


def load_aligned_indices(align_dir: Path, dataset: str) -> List[int]:
    raw_path = align_dir / "raw" / f"binary_staged_{dataset}.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing aligned indices pack: {raw_path}")
    with np.load(raw_path) as data:
        if "indices" not in data.files:
            raise KeyError(f"{raw_path} has no 'indices' array")
        return [int(i) for i in data["indices"].tolist()]


def _itrans_batch_forward(
    model: torch.nn.Module,
    past: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Past (B, C, L) -> forecast (B, C, horizon) in normalized space."""
    past = past.to(device)
    b, c, length = past.shape
    x_enc = past.permute(0, 2, 1)
    seq_sl = getattr(model, "seq_len", length)
    if x_enc.shape[1] > seq_sl:
        x_enc = x_enc[:, -seq_sl:, :]
    x_dec = torch.zeros(b, horizon, c, device=device, dtype=past.dtype)
    out = model(x_enc, None, x_dec, None)
    if isinstance(out, tuple):
        out = out[0]
    return out.permute(0, 2, 1)


def evaluate_itrans_guidance(
    args: argparse.Namespace,
    run,
    sub: Dict[str, Any],
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    raw_path = args.output_dir / "raw" / f"itrans_guidance_{run.dataset}.npz"
    if raw_path.exists() and not args.force_eval:
        with np.load(raw_path) as data:
            return {key: data[key] for key in data.files}

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    overlap_k = resolve_lookback_overlap(sub)
    subset = load_tsf_test_subset(
        run.dataset,
        run_variate_indices(run),
        indices,
        lookback,
        horizon,
        run_train_stride(run),
        run_test_stride(run),
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    guidance = load_itransformer_from_checkpoint(
        str(run.itrans_pt),
        len(run_variate_indices(run)),
        device,
    )
    guidance.eval()

    y_true_all: List[np.ndarray] = []
    det_all: List[np.ndarray] = []
    progress = EvalProgress(f"itrans-guidance/{run.dataset}", len(loader))
    print(
        f"[itrans-guidance] {run.dataset}: ckpt={run.itrans_pt.name} "
        f"windows={len(indices)} batches={len(loader)} stride={run_test_stride(run)} "
        f"overlap_k={overlap_k}",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            t_batch = time.time()
            future = future.to(device)
            if overlap_k > 0:
                future = future[..., overlap_k:]
            y_true_all.append(future.cpu().numpy())

            pred = _itrans_batch_forward(guidance, past, horizon, device)
            if overlap_k > 0:
                pred = pred[..., overlap_k:]
            det_all.append(pred.cpu().numpy())

            progress.maybe_log(
                batch_idx + 1,
                extra=(
                    f"last_batch={fmt_duration(time.time() - t_batch)} "
                    f"elapsed={fmt_duration(time.time() - t0)}"
                ),
            )
    progress.done(extra=f"writing {raw_path}")

    deterministic = np.concatenate(det_all, axis=0)
    pack = {
        "y_true": np.concatenate(y_true_all, axis=0),
        "deterministic": deterministic,
        "samples": deterministic[:, :, np.newaxis, :],
        "indices": np.asarray(indices, dtype=np.int64),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **pack)
    return pack


def summarize(pack: Dict[str, np.ndarray], args: argparse.Namespace, dataset: str) -> Dict[str, float]:
    return summarize_prediction_pack(
        pack,
        gmm_components=args.gmm_components,
        seed=stable_dataset_seed(args.seed, dataset),
        topk_max=args.topk_max,
        texture_per_sample=True,
    )


def write_outputs(args: argparse.Namespace, results: Dict[str, Dict[str, float]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    fields = [
        "dataset",
        "model",
        "n_windows",
        "n_variates",
        "n_samples",
        *CORE_KEYS,
        *TEXTURE_KEYS,
        *[f"prob_{key}" for key in TEXTURE_KEYS],
    ]
    with (args.output_dir / "texture_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in results:
            row = {"dataset": dataset, "model": "itrans_guidance"}
            row.update({key: results[dataset].get(key) for key in fields if key not in row})
            writer.writerow(row)

    print("\nGuidance iTrans texture summary")
    print(",".join(fields))
    for dataset in results:
        metrics = results[dataset]
        row = [dataset, "itrans_guidance"]
        for key in fields[2:]:
            val = metrics.get(key, float("nan"))
            row.append(f"{val:.6f}" if isinstance(val, (float, int)) else "")
        print(",".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_STAGED_CKPTS))
    for dataset, path in DEFAULT_STAGED_CKPTS.items():
        parser.add_argument(f"--{dataset}-ckpt", type=Path, default=Path(path))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "datasets" / "06-03-trend-robust-texture-staged-itrans-guidance",
    )
    parser.add_argument(
        "--align-indices-dir",
        type=Path,
        default=REPO_ROOT / "results" / "datasets" / "06-03-trend-robust-texture-staged-vs-mmpd",
        help="Reuse test window indices from binary_staged_*.npz in this dir (same protocol as staged-vs-mmpd).",
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=2)
    parser.add_argument("--gmm-components", type=int, default=1)
    parser.add_argument("--topk-max", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--no-align-indices", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [dataset for dataset in args.datasets if dataset not in DEFAULT_STAGED_CKPTS]
    if unknown:
        raise ValueError(f"No default staged checkpoint for: {', '.join(unknown)}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}")

    results: Dict[str, Dict[str, float]] = {}
    manifest: Dict[str, Any] = {
        "datasets": args.datasets,
        "test_fraction": args.test_fraction,
        "test_stride": args.test_stride,
        "align_indices_dir": None if args.no_align_indices else str(args.align_indices_dir),
        "staged_ckpts": {},
    }

    for dataset in args.datasets:
        ckpt_dir = getattr(args, f"{dataset}_ckpt").resolve()
        run, sub = staged_anchor_run(dataset, ckpt_dir, args.test_stride)
        manifest["staged_ckpts"][dataset] = {
            "checkpoint_dir": str(ckpt_dir),
            "itrans_pt": str(run.itrans_pt),
            "subset_id": run_subset_id(run),
        }

        if args.no_align_indices:
            indices = make_indices(args, run)
        else:
            indices = load_aligned_indices(args.align_indices_dir.resolve(), dataset)

        print(
            f"\n[{dataset}] subset={run_subset_id(run)} variates={len(run_variate_indices(run))} "
            f"train_stride={run_train_stride(run)} test_stride={run_test_stride(run)} "
            f"indices={len(indices)}",
            flush=True,
        )

        pack = evaluate_itrans_guidance(args, run, sub, indices, device)
        results[dataset] = summarize(pack, args, dataset)
        write_outputs(args, results)

    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
