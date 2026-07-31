#!/usr/bin/env python3
"""Visualize discriminator TP/TN/FP/FN examples on test slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_discriminator_texture_staged_vs_mmpd import (
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    build_raw_bundle,
    parse_args as disc_parse_args,
    split_windows,
    stable_hash,
    train_classifier,
)
from utils.eval_mmpd_gaussian_anchor import run_test_stride


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    lookback: int,
    horizon: int,
    slice_len: int,
) -> Tuple[InvertedSliceDiscriminator, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = payload.get("args") or {}
    d_model = int(ckpt_args.get("d_model", 128))
    n_heads = int(ckpt_args.get("n_heads", 4))
    depth = int(ckpt_args.get("depth", 2))
    d_ff = int(ckpt_args.get("d_ff", 256))
    dropout = float(ckpt_args.get("dropout", 0.1))
    candidate_only = bool(ckpt_args.get("candidate_only", False))
    use_offset_embedding = not bool(ckpt_args.get("no_offset_embedding", False))
    seq_len = int(slice_len if candidate_only else lookback + slice_len)
    max_offset = horizon - slice_len

    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=max_offset,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        dropout=dropout,
        use_offset_embedding=use_offset_embedding,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    meta = {
        "dataset": payload["dataset"],
        "fake_source": payload["fake_source"],
        "slice_len": slice_len,
        "seq_len": seq_len,
        "max_offset": max_offset,
        "candidate_only": candidate_only,
    }
    return model, meta


def build_test_dataset(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
) -> HorizonSliceDataset:
    y_true = bundle.y_true_by_source[fake_source]
    seed_base = args.seed + stable_hash(f"{dataset}:{fake_source}:{slice_len}")
    return HorizonSliceDataset(
        bundle.past,
        y_true,
        bundle.fakes[fake_source],
        splits["test"],
        slice_len,
        seed=seed_base + 2,
        offset_stride=(
            int(slice_len)
            if bool(getattr(args, "nonoverlapping_patches", False))
            else args.offset_stride
        ),
        max_examples=args.max_eval_examples,
        include_past=not bool(getattr(args, "candidate_only", False)),
    )


@torch.no_grad()
def collect_test_records(
    model: InvertedSliceDiscriminator,
    ds: HorizonSliceDataset,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    records: List[Dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        x, offsets, labels = batch[0], batch[1], batch[2]
        x = x.to(device)
        offsets = offsets.to(device)
        logits = model(x, offsets)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (logits >= 0.0).cpu().numpy().astype(np.int64)
        labels_np = labels.cpu().numpy().astype(np.int64)
        for i in range(len(labels_np)):
            window, offset, label = ds.items[cursor]
            past = ds.past[window]
            gt = ds.real[window, :, offset : offset + ds.slice_len]
            fake = ds.fake[window, :, offset : offset + ds.slice_len]
            candidate_src = ds.fake if label == 1 else ds.real
            candidate = candidate_src[window, :, offset : offset + ds.slice_len]
            records.append(
                {
                    "window": int(window),
                    "offset": int(offset),
                    "label": int(label),
                    "pred": int(preds[i]),
                    "prob_fake": float(probs[i]),
                    "past": past.copy(),
                    "gt": gt.copy(),
                    "fake": fake.copy(),
                    "candidate": candidate.copy(),
                }
            )
            cursor += 1
    return records


def bucket_name(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def summarize_buckets(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for row in records:
        counts[bucket_name(row["label"], row["pred"])] += 1
    return counts


def pick_examples(
    records: List[Dict[str, Any]],
    per_bucket: int,
    rng: np.random.Generator,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ("TP", "TN", "FP", "FN")}
    for bucket in out:
        pool = [r for r in records if bucket_name(r["label"], r["pred"]) == bucket]
        if not pool:
            continue
        if bucket in ("TP", "FP"):
            pool.sort(key=lambda r: r["prob_fake"], reverse=True)
        else:
            pool.sort(key=lambda r: r["prob_fake"])
        n = min(per_bucket, len(pool))
        if n == len(pool):
            out[bucket] = pool
        else:
            top = pool[: max(1, n // 2)]
            rest = pool[max(1, n // 2) :]
            extra = min(n - len(top), len(rest))
            if extra > 0:
                idx = rng.choice(len(rest), size=extra, replace=False)
                top.extend([rest[i] for i in idx])
            out[bucket] = top[:n]
    return out


def plot_example(
    row: Dict[str, Any],
    *,
    variate: int,
    lookback_tail: int,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    past = row["past"][variate, -lookback_tail:]
    gt = row.get("gt", row["candidate"])[variate]
    fake = row.get("fake", row["candidate"])[variate]
    t_past = np.arange(-len(past), 0)
    t_h = np.arange(0, len(gt))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t_past, past, color="#444444", lw=1.4, label="lookback")
    ax.plot(t_h, gt, color="#1f77b4", lw=2.0, label="GT")
    ax.plot(t_h, fake, color="#d62728", lw=2.0, alpha=0.9, label="model pred")
    ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("time (steps; 0 = horizon start)")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bucket_grid(
    examples: Dict[str, List[Dict[str, Any]]],
    *,
    variate: int,
    lookback_tail: int,
    out_dir: Path,
    stem: str,
) -> None:
    for bucket, rows in examples.items():
        for i, row in enumerate(rows):
            title = (
                f"{bucket} | shown_to_disc={'fake' if row['label'] == 1 else 'real'} "
                f"pred={'fake' if row['pred'] == 1 else 'real'} "
                f"p(fake)={row['prob_fake']:.3f} win={row['window']} off={row['offset']} v={variate}"
            )
            plot_example(
                row,
                variate=variate,
                lookback_tail=lookback_tail,
                title=title,
                out_path=out_dir / f"{stem}_{bucket}_{i:02d}.png",
            )


def checkpoint_path(args: argparse.Namespace, dataset: str, fake_source: str, slice_len: int) -> Path:
    return (
        args.output_dir
        / "checkpoints"
        / f"{dataset}_{fake_source}_L{slice_len}_discriminator.pt"
    )


def default_plot_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "viz_plot_dir", None):
        return Path(args.viz_plot_dir)
    return args.output_dir / "disc_confusions"


def visualize_combo(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    per_bucket: Optional[int] = None,
    plot_dir: Optional[Path] = None,
    variate: Optional[int] = None,
    lookback_tail: Optional[int] = None,
) -> Path:
    ckpt = checkpoint_path(args, dataset, fake_source, slice_len)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint for visualization: {ckpt}")

    y_true = bundle.y_true_by_source[fake_source]
    lookback = int(bundle.past.shape[-1])
    horizon = int(y_true.shape[-1])
    model, _meta = load_checkpoint(
        ckpt,
        device,
        lookback=lookback,
        horizon=horizon,
        slice_len=slice_len,
    )
    ds = build_test_dataset(args, dataset, fake_source, slice_len, bundle, splits)
    records = collect_test_records(model, ds, device, args.batch_size)
    counts = summarize_buckets(records)
    acc = sum(counts[k] for k in ("TP", "TN")) / max(1, len(records))
    print(
        f"[viz] {dataset}/{fake_source}/L{slice_len} test n={len(records)} "
        f"acc={acc:.4f} buckets={counts}",
        flush=True,
    )

    per_bucket = int(per_bucket if per_bucket is not None else getattr(args, "viz_per_bucket", 2))
    variate = int(variate if variate is not None else getattr(args, "viz_variate", 0))
    lookback_tail = int(
        lookback_tail if lookback_tail is not None else getattr(args, "viz_lookback_tail", 32)
    )
    out_root = plot_dir if plot_dir is not None else default_plot_dir(args)
    stem = f"{dataset}_{fake_source}_L{slice_len}"
    out_dir = out_root / stem

    rng = np.random.default_rng(args.seed + stable_hash(f"viz:{dataset}:{fake_source}:{slice_len}"))
    examples = pick_examples(records, per_bucket, rng)
    plot_bucket_grid(
        examples,
        variate=variate,
        lookback_tail=lookback_tail,
        out_dir=out_dir,
        stem=stem,
    )
    summary = {
        "checkpoint": str(ckpt),
        "dataset": dataset,
        "fake_source": fake_source,
        "slice_len": slice_len,
        "n_test": len(records),
        "acc": acc,
        "buckets": counts,
        "plots": str(out_dir),
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[viz] wrote plots under {out_dir}", flush=True)
    # Log when a wandb run is already active (no-op otherwise).
    try:
        from models.diffusion_tsf.pipeline import wandb_utils

        pngs = sorted(out_dir.glob(f"{stem}_*.png"))
        if pngs:
            wandb_utils.log_visualization_paths(
                [str(p) for p in pngs],
                wandb_key=f"disc/confusions/{dataset}/{fake_source}/L{slice_len}",
            )
    except Exception as exc:
        print(f"[viz] wandb log skipped: {exc}", flush=True)
    return out_dir


def parse_viz_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    viz_parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    viz_parser.add_argument("--checkpoint", type=Path, default=None)
    viz_parser.add_argument("--dataset", required=True)
    viz_parser.add_argument("--fake-source", choices=["binary_staged", "mmpd"], required=True)
    viz_parser.add_argument("--slice-len", type=int, default=8)
    viz_parser.add_argument("--per-bucket", type=int, default=2)
    viz_parser.add_argument("--variate", type=int, default=0)
    viz_parser.add_argument("--lookback-tail", type=int, default=32)
    viz_parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "reports/06-03_trend_robust_texture_staged_vs_mmpd/disc_confusions",
        help="Where to write PNG summaries.",
    )
    viz_parser.add_argument("--retrain", action="store_true", help="Train if checkpoint missing.")
    viz_args, remaining = viz_parser.parse_known_args()

    argv = sys.argv
    sys.argv = [argv[0], *remaining]
    try:
        disc_args = disc_parse_args()
    finally:
        sys.argv = argv
    return viz_args, disc_args


def main() -> None:
    viz_args, disc_args = parse_viz_args()
    device = torch.device(
        "cpu" if disc_args.cpu or not torch.cuda.is_available() else f"cuda:{disc_args.gpu}"
    )
    disc_args.datasets = [viz_args.dataset]
    disc_args.fake_sources = [viz_args.fake_source]
    disc_args.slice_lengths = [viz_args.slice_len]
    disc_args.viz_plot_dir = viz_args.plot_dir

    ckpt = viz_args.checkpoint
    if ckpt is None:
        ckpt = checkpoint_path(disc_args, viz_args.dataset, viz_args.fake_source, viz_args.slice_len)
    elif ckpt.is_file():
        disc_args.output_dir = ckpt.parent.parent

    if viz_args.retrain or not ckpt.is_file():
        if not viz_args.retrain and not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Re-run training with --save-checkpoints or pass --retrain."
            )
        disc_args.save_checkpoints = True
        disc_args.force_train = True
        bundle = build_raw_bundle(disc_args, viz_args.dataset, device)
        y_true = bundle.y_true_by_source[viz_args.fake_source]
        splits = split_windows(
            next(iter(bundle.y_true_by_source.values())).shape[0],
            disc_args,
            viz_args.dataset,
            indices=bundle.indices,
            lookback=bundle.past.shape[-1],
            horizon=y_true.shape[-1],
            test_stride=run_test_stride(bundle.run),
            series_starts=bundle.series_starts,
        )
        train_classifier(
            disc_args,
            viz_args.dataset,
            viz_args.fake_source,
            viz_args.slice_len,
            bundle,
            splits,
            device,
        )
        if not ckpt.is_file():
            raise FileNotFoundError(f"Expected checkpoint after retrain: {ckpt}")

    bundle = build_raw_bundle(disc_args, viz_args.dataset, device)
    y_true = bundle.y_true_by_source[viz_args.fake_source]
    n = y_true.shape[0]
    splits = split_windows(
        n,
        disc_args,
        viz_args.dataset,
        indices=bundle.indices,
        lookback=bundle.past.shape[-1],
        horizon=y_true.shape[-1],
        test_stride=run_test_stride(bundle.run),
        series_starts=bundle.series_starts,
    )
    visualize_combo(
        disc_args,
        viz_args.dataset,
        viz_args.fake_source,
        viz_args.slice_len,
        bundle,
        splits,
        device,
        per_bucket=viz_args.per_bucket,
        plot_dir=viz_args.plot_dir,
        variate=viz_args.variate,
        lookback_tail=viz_args.lookback_tail,
    )


if __name__ == "__main__":
    main()
