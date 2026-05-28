#!/usr/bin/env python3
"""Probabilistic (non-anchor) checks for binary-anchor ckpts at bin-h64 / bin-h128.

Modes:
  shape   — one window, assert prediction tensor shape (default)
  texture — 4 texture metrics on one dpmpp sample per test window
            (ordinal JSD, RQA, variogram, path signature)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    find_anchor_runs,
    load_anchor_model,
    load_tsf_pipeline,
    load_tsf_test_subset,
    make_eval_indices,
    stable_dataset_seed,
    texture_metrics,
)

DEFAULT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "illness", "exchange_rate")
TEXTURE_KEYS = (
    "texture_ordinal_jsd",
    "texture_rqa_distance",
    "texture_variogram_distance",
    "texture_pathsig_distance",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("shape", "texture"),
        default="shape",
        help="shape=one-window shape check; texture=4 metrics on prob sample(s).",
    )
    p.add_argument("--date-tag", default="05-27", help="Checkpoint stem prefix (MM-DD).")
    p.add_argument(
        "--heights",
        type=int,
        nargs="+",
        default=[64],
        help="Image heights to scan (e.g. 64 128).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to test (must have best.pt under each stem).",
    )
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--lookback-overlap", type=int, default=8)
    p.add_argument("--ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    p.add_argument("--anchor-prob-sampler", default="dpmpp", choices=["dpmpp", "ddim", "ddpm"])
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--test-fraction", type=float, default=0.5)
    p.add_argument("--test-max-items", type=int, default=None)
    p.add_argument(
        "--indices-dir",
        type=Path,
        default=None,
        help="Reuse raw/indices_{dataset}.json from a prior matrix run.",
    )
    p.add_argument("--anchor-batch-size", type=int, default=8)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write CSV (texture mode) or JSON summary.",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any listed dataset is missing best.pt (default: skip missing).",
    )
    return p.parse_args()


def stem_for(date_tag: str, height: int, dataset: str) -> str:
    return f"{date_tag}-bin-h{height}-{dataset.lower()}"


def ckpt_ready(root: Path, dataset: str) -> bool:
    return (root / dataset / "best.pt").exists() or (root / dataset.lower() / "best.pt").exists()


def discover_completed(
    args: argparse.Namespace,
    height: int,
) -> Tuple[List[str], List[Path], List[str]]:
    roots: List[Path] = []
    datasets: List[str] = []
    pending: List[str] = []
    for ds in args.datasets:
        root = args.ckpt_base / stem_for(args.date_tag, height, ds)
        if ckpt_ready(root, ds):
            roots.append(root)
            datasets.append(ds)
        else:
            pending.append(f"h{height}/{ds}")
    return datasets, roots, pending


def sample_kwargs_for(args: argparse.Namespace) -> Dict:
    if args.anchor_prob_sampler == "ddpm":
        return {"sampler": "ddpm", "use_ddim": False}
    return {
        "sampler": args.anchor_prob_sampler,
        "num_inference_steps": args.num_sampling_steps,
    }


def eval_indices_for_dataset(
    args: argparse.Namespace,
    dataset: str,
    n_test: int,
    variate_indices: Sequence[int],
) -> List[int]:
    if args.indices_dir is not None:
        path = args.indices_dir / "raw" / f"indices_{dataset}.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return list(json.load(f))
    seed = stable_dataset_seed(args.seed, dataset)
    return make_eval_indices(n_test, args.test_fraction, seed, args.test_max_items)


def run_shape_check(
    args: argparse.Namespace,
    height: int,
    datasets: Sequence[str],
    roots: Sequence[Path],
) -> bool:
    eval_args = Namespace(
        lookback=args.lookback,
        horizon=args.horizon,
        anchor_batch_size=1,
        anchor_prob_sampler=args.anchor_prob_sampler,
        num_sampling_steps=args.num_sampling_steps,
        seed=args.seed,
        gpu=args.gpu,
        cpu=args.cpu or not torch.cuda.is_available(),
    )
    device = torch.device("cpu" if eval_args.cpu else f"cuda:{eval_args.gpu}")
    anchors = find_anchor_runs(list(datasets), list(roots), args.ckpt_base, "binary")
    sample_kwargs = sample_kwargs_for(args)
    ok = True

    for ds in datasets:
        run = anchors[ds]
        subset = load_tsf_test_subset(
            ds, run.metadata["variate_indices"], [0], args.lookback, args.horizon
        )
        past, future = next(iter(DataLoader(subset, batch_size=1)))
        k = args.lookback_overlap
        if k > 0:
            future = future[..., k:]

        model = load_anchor_model(run, eval_args, device)
        with torch.no_grad():
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            pred = model.generate(past.to(device), **sample_kwargs)["prediction"]

        exp = tuple(future.shape)
        got = tuple(pred.cpu().shape)
        if got != exp:
            ok = False
            print(
                f"FAIL h{height} {ds}: pred {got} != y_true {exp}  "
                f"image_h={model.config.image_height}  stem={run.root.name}"
            )
        else:
            print(
                f"OK   h{height} {ds}: pred {got}  image_h={model.config.image_height}  "
                f"sampler={args.anchor_prob_sampler}  stem={run.root.name}"
            )
    return ok


def run_texture_eval(
    args: argparse.Namespace,
    height: int,
    datasets: Sequence[str],
    roots: Sequence[Path],
) -> Dict[str, Dict[str, float]]:
    eval_args = Namespace(
        lookback=args.lookback,
        horizon=args.horizon,
        anchor_batch_size=args.anchor_batch_size,
        anchor_prob_sampler=args.anchor_prob_sampler,
        num_sampling_steps=args.num_sampling_steps,
        seed=args.seed,
        gpu=args.gpu,
        cpu=args.cpu or not torch.cuda.is_available(),
    )
    device = torch.device("cpu" if eval_args.cpu else f"cuda:{eval_args.gpu}")
    anchors = find_anchor_runs(list(datasets), list(roots), args.ckpt_base, "binary")
    sample_kwargs = sample_kwargs_for(args)
    out: Dict[str, Dict[str, float]] = {}

    pipeline = load_tsf_pipeline()
    for ds in datasets:
        run = anchors[ds]
        _, _, test_ds, _ = pipeline.load_dataset(
            ds,
            list(run.metadata["variate_indices"]),
            lookback=args.lookback,
            horizon=args.horizon,
            stride=1,
        )
        n_test = len(test_ds)
        indices = eval_indices_for_dataset(args, ds, n_test, run.metadata["variate_indices"])
        subset = load_tsf_test_subset(
            ds, run.metadata["variate_indices"], indices, args.lookback, args.horizon
        )
        loader = DataLoader(
            subset,
            batch_size=args.anchor_batch_size,
            shuffle=False,
            num_workers=0,
        )
        model = load_anchor_model(run, eval_args, device)
        k = args.lookback_overlap
        y_true_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, (past, future) in enumerate(loader):
                past = past.to(device)
                if k > 0:
                    future = future[..., k:]
                batch_seed = args.seed + batch_idx * 1009
                torch.manual_seed(batch_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(batch_seed)
                pred = model.generate(past, **sample_kwargs)["prediction"]
                y_true_parts.append(future.cpu().numpy())
                y_pred_parts.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true_parts, axis=0)
        y_pred = np.concatenate(y_pred_parts, axis=0)
        metrics = texture_metrics(y_true, y_pred)
        metrics["n_windows"] = float(y_true.shape[0])
        metrics["n_variates"] = float(y_true.shape[1])
        out[ds] = metrics
        print(
            f"h{height} {ds}  n={int(metrics['n_windows'])}  "
            + "  ".join(f"{key.split('_', 1)[-1]}={metrics[key]:.6f}" for key in TEXTURE_KEYS)
            + f"  stem={run.root.name}"
        )
    return out


def print_texture_table(all_rows: List[Dict]) -> None:
    print()
    header = ["height", "dataset", "n_windows"] + list(TEXTURE_KEYS)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in all_rows:
        cells = [
            str(row["height"]),
            row["dataset"],
            str(int(row["n_windows"])),
        ]
        for key in TEXTURE_KEYS:
            cells.append(f"{row[key]:.6f}")
        print("| " + " | ".join(cells) + " |")


def write_texture_csv(path: Path, all_rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["height", "dataset", "n_windows", *TEXTURE_KEYS, "stem"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> int:
    args = parse_args()
    args.ckpt_base = args.ckpt_base.resolve()
    if args.indices_dir is not None:
        args.indices_dir = args.indices_dir.resolve()

    all_pending: List[str] = []
    ok = True
    texture_rows: List[Dict] = []

    for height in args.heights:
        datasets, roots, pending = discover_completed(args, height)
        all_pending.extend(pending)
        if not datasets:
            print(f"h{height}: no completed ckpts under {args.ckpt_base}")
            continue

        if args.mode == "shape":
            if not run_shape_check(args, height, datasets, roots):
                ok = False
        else:
            metrics_by_ds = run_texture_eval(args, height, datasets, roots)
            for ds, metrics in metrics_by_ds.items():
                stem = stem_for(args.date_tag, height, ds)
                texture_rows.append(
                    {
                        "height": height,
                        "dataset": ds,
                        "stem": stem,
                        **{k: metrics[k] for k in TEXTURE_KEYS},
                        "n_windows": int(metrics["n_windows"]),
                    }
                )

    if all_pending:
        msg = "skip (no best.pt): " + ", ".join(all_pending)
        if args.require_all:
            print(f"FAIL {msg}", file=sys.stderr)
            return 1
        print(msg)

    if args.mode == "texture":
        if not texture_rows:
            print("FAIL no texture results", file=sys.stderr)
            return 1
        print_texture_table(texture_rows)
        if args.output is not None:
            if args.output.suffix.lower() == ".json":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(texture_rows, indent=2), encoding="utf-8")
            else:
                write_texture_csv(args.output, texture_rows)
            print(f"\nWrote {args.output}")
        return 0

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
