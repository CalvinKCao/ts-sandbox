#!/usr/bin/env python3
"""One-window probabilistic (non-anchor) generate shape check for binary-anchor ckpts."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    find_anchor_runs,
    load_anchor_model,
    load_tsf_test_subset,
)

DEFAULT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "illness", "exchange_rate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date-tag", default="05-27", help="Checkpoint stem prefix (MM-DD).")
    p.add_argument("--height", type=int, default=64, help="Image height in stem (bin-h{height}-*).")
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
    p.add_argument("--num-sampling-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any listed dataset is missing best.pt (default: skip missing).",
    )
    return p.parse_args()


def stem_for(date_tag: str, height: int, dataset: str) -> Path:
    return Path(f"{date_tag}-bin-h{height}-{dataset.lower()}")


def ckpt_ready(root: Path, dataset: str) -> bool:
    return (root / dataset / "best.pt").exists() or (root / dataset.lower() / "best.pt").exists()


def run_shape_test(args: argparse.Namespace) -> int:
    args.ckpt_base = args.ckpt_base.resolve()
    roots: list[Path] = []
    pending: list[str] = []
    for ds in args.datasets:
        root = args.ckpt_base / stem_for(args.date_tag, args.height, ds).name
        if ckpt_ready(root, ds):
            roots.append(root)
        else:
            pending.append(ds)

    if pending:
        msg = f"skip (no best.pt): {', '.join(pending)}"
        if args.require_all:
            print(f"FAIL {msg}", file=sys.stderr)
            return 1
        print(msg)

    if not roots:
        print("FAIL no completed checkpoints found", file=sys.stderr)
        return 1

    root_set = {r.resolve() for r in roots}
    datasets = [
        ds
        for ds in args.datasets
        if (args.ckpt_base / f"{args.date_tag}-bin-h{args.height}-{ds.lower()}").resolve() in root_set
    ]

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

    anchors = find_anchor_runs(datasets, roots, args.ckpt_base, "binary")
    ok = True
    sample_kwargs = {
        "sampler": args.anchor_prob_sampler,
        "num_inference_steps": args.num_sampling_steps,
    }
    if args.anchor_prob_sampler == "ddpm":
        sample_kwargs = {"sampler": "ddpm", "use_ddim": False}

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
                f"FAIL {ds}: pred {got} != y_true {exp}  "
                f"h={model.config.image_height}  stem={run.root.name}"
            )
        else:
            print(
                f"OK   {ds}: pred {got}  h={model.config.image_height}  "
                f"sampler={args.anchor_prob_sampler}  stem={run.root.name}"
            )

    return 0 if ok else 1


def main() -> None:
    sys.exit(run_shape_test(parse_args()))


if __name__ == "__main__":
    main()
