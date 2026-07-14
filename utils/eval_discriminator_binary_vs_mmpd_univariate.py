#!/usr/bin/env python3
"""Univariate binary-vs-MMPD discriminator (one model per dataset).

Same pack / split / candidate-only / z-score / fair-protocol path as
`eval_discriminator_texture_staged_vs_mmpd.py`, but:

- Each example is a **single-variate** L-length patch (shape [1, L]).
- Label **1 = binary_staged**, **0 = mmpd** (not real-vs-fake).
- One discriminator per dataset (and slice length), trained on patches from
  **all** variates pooled together.

Reuse existing raw packs via `--raw-eval-dir` (no need to regenerate unless forced).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    DEFAULT_DISC_OUTPUT,
    InvertedSliceDiscriminator,
    LOG2,
    apply_smoke_defaults as _apply_smoke_defaults_base,
    binary_auroc,
    build_raw_bundle,
    parse_args as disc_parse_args,
    split_windows,
    stable_hash,
    window_level_metrics,
    write_json,
    zscore_time,
)
from utils.eval_mmpd_gaussian_anchor import run_test_stride  # noqa: E402
from utils.eval_trend_robust_texture_staged_vs_mmpd import (  # noqa: E402
    EvalProgress,
    fmt_duration,
)

DEFAULT_OUTPUT = DEFAULT_DISC_OUTPUT.parent / "disc-lb336-hz720-ordinal-four-binary-vs-mmpd-univariate"
POSITIVE_SOURCE = "binary_staged"
NEGATIVE_SOURCE = "mmpd"


class UnivariateBinaryVsMmpdDataset(Dataset):
    """Balanced univariate patches: label 1=binary_staged, 0=mmpd."""

    def __init__(
        self,
        binary: np.ndarray,
        mmpd: np.ndarray,
        past: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
        include_past: bool = False,
    ) -> None:
        if binary.shape != mmpd.shape:
            raise ValueError(f"binary/mmpd shape mismatch: {binary.shape} vs {mmpd.shape}")
        if binary.shape[0] != past.shape[0]:
            raise ValueError(f"past/binary window mismatch: {past.shape[0]} vs {binary.shape[0]}")
        if slice_len > binary.shape[-1]:
            raise ValueError(f"slice_len={slice_len} exceeds horizon={binary.shape[-1]}")

        self.binary = binary
        self.mmpd = mmpd
        self.past = past
        self.slice_len = int(slice_len)
        self.include_past = bool(include_past)
        n_var = int(binary.shape[1])
        offsets = list(range(0, binary.shape[-1] - slice_len + 1, max(1, int(offset_stride))))
        # (window, offset, variate, label)
        bin_items = [(int(w), int(o), int(v), 1) for w in windows for o in offsets for v in range(n_var)]
        mmpd_items = [(int(w), int(o), int(v), 0) for w in windows for o in offsets for v in range(n_var)]

        rng = np.random.default_rng(seed)
        n = min(len(bin_items), len(mmpd_items))
        if max_examples is not None:
            n = min(n, max(1, int(max_examples) // 2))
        bin_idx = rng.choice(len(bin_items), size=n, replace=False)
        mmpd_idx = rng.choice(len(mmpd_items), size=n, replace=False)
        items = [bin_items[i] for i in bin_idx] + [mmpd_items[i] for i in mmpd_idx]
        rng.shuffle(items)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window, offset, variate, label = self.items[idx]
        src = self.binary if label == 1 else self.mmpd
        candidate = src[window, variate : variate + 1, offset : offset + self.slice_len]
        if self.include_past:
            past = self.past[window, variate : variate + 1]
            x = np.concatenate([zscore_time(past), zscore_time(candidate)], axis=-1).astype(np.float32)
        else:
            x = zscore_time(candidate).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(offset, dtype=torch.long),
            torch.tensor(float(label), dtype=torch.float32),
            torch.tensor(int(window), dtype=torch.long),
        )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    windows_all: List[np.ndarray] = []
    for batch in loader:
        x, offsets, labels, windows = batch
        x = x.to(device)
        offsets = offsets.to(device)
        labels = labels.to(device)
        logits = model(x, offsets)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(labels.numel())
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        windows_all.append(windows.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    windows_np = np.concatenate(windows_all)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (logits_np >= 0.0).astype(np.float32)
    out = {
        "disc_bce": total_loss / max(1, total_count),
        "disc_acc": float((preds == labels_np).mean()),
        "disc_auroc": binary_auroc(labels_np, probs),
        "n_examples": float(total_count),
        "positive_rate": float(labels_np.mean()),
    }
    out.update(window_level_metrics(windows_np, labels_np, probs))
    return out


def train_classifier(
    args: argparse.Namespace,
    dataset: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    binary = bundle.fakes[POSITIVE_SOURCE]
    mmpd = bundle.fakes[NEGATIVE_SOURCE]
    horizon = int(binary.shape[-1])
    n_variates = int(binary.shape[1])
    max_offset = horizon - slice_len
    seed_base = args.seed + stable_hash(f"{dataset}:binary_vs_mmpd_uni:{slice_len}")
    include_past = not bool(getattr(args, "candidate_only", False))
    offset_stride = int(getattr(args, "offset_stride", 1) or 1)
    if bool(getattr(args, "nonoverlapping_patches", False)):
        offset_stride = int(slice_len)
    use_offset_embedding = not bool(getattr(args, "no_offset_embedding", False))
    ds_kwargs = dict(offset_stride=offset_stride, include_past=include_past)

    ds_train = UnivariateBinaryVsMmpdDataset(
        binary, mmpd, bundle.past, splits["train"], slice_len,
        seed=seed_base, max_examples=args.max_train_examples, **ds_kwargs,
    )
    ds_val = UnivariateBinaryVsMmpdDataset(
        binary, mmpd, bundle.past, splits["val"], slice_len,
        seed=seed_base + 1, max_examples=args.max_eval_examples, **ds_kwargs,
    )
    ds_test = UnivariateBinaryVsMmpdDataset(
        binary, mmpd, bundle.past, splits["test"], slice_len,
        seed=seed_base + 2, max_examples=args.max_eval_examples, **ds_kwargs,
    )

    generator = torch.Generator()
    generator.manual_seed(seed_base)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), generator=generator,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    seq_len = int(slice_len if not include_past else bundle.past.shape[-1] + slice_len)
    print(
        f"[disc-uni] {dataset}/L{slice_len}: binary_vs_mmpd univariate "
        f"candidate_only={not include_past} offset_stride={offset_stride} "
        f"offset_emb={use_offset_embedding} seq_len={seq_len} n_variates={n_variates} "
        f"n_train={len(ds_train)} n_val={len(ds_val)} n_test={len(ds_test)}",
        flush=True,
    )
    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=max_offset,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_offset_embedding=use_offset_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    progress = EvalProgress(f"disc-uni/{dataset}/L{slice_len}", args.epochs)
    t0 = time.time()
    epoch = -1
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_idx, batch in enumerate(train_loader):
            x, offsets, labels = batch[0], batch[1], batch[2]
            x = x.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, offsets)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * int(labels.numel())
            train_count += int(labels.numel())
            if args.max_batches_per_epoch and batch_idx + 1 >= args.max_batches_per_epoch:
                break

        val_metrics = evaluate_classifier(model, val_loader, device)
        train_bce = train_loss / max(1, train_count)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        progress.maybe_log(
            epoch + 1,
            extra=(
                f"train_bce={train_bce:.4f} val_bce={val_metrics['disc_bce']:.4f} "
                f"val_auc={val_metrics['disc_auroc']:.3f} "
                f"val_auc_win={val_metrics.get('disc_auroc_window', float('nan')):.3f} "
                f"elapsed={fmt_duration(time.time() - t0)}"
            ),
        )
        if stale >= args.patience:
            break

    progress.done(extra=f"best_epoch={best_epoch} best_val_bce={best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device)

    if args.save_checkpoints:
        ckpt_path = (
            args.output_dir
            / "checkpoints"
            / f"{dataset}_binary_vs_mmpd_L{slice_len}_univariate_discriminator.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "dataset": dataset,
                "task": "binary_vs_mmpd_univariate",
                "positive_source": POSITIVE_SOURCE,
                "negative_source": NEGATIVE_SOURCE,
                "slice_len": slice_len,
            },
            ckpt_path,
        )

    return {
        **test_metrics,
        "best_val_bce": float(best_val),
        "best_epoch": float(best_epoch),
        "epochs_run": float(epoch + 1),
        "n_train": float(len(ds_train)),
        "n_val": float(len(ds_val)),
        "n_test": float(len(ds_test)),
        "n_windows_train": float(len(splits["train"])),
        "n_windows_val": float(len(splits["val"])),
        "n_windows_test": float(len(splits["test"])),
        "slice_len": float(slice_len),
        "horizon": float(horizon),
        "n_variates": float(n_variates),
        "log2_bce_gap": float(abs(test_metrics["disc_bce"] - LOG2)),
        "candidate_only": float(1.0 if not include_past else 0.0),
        "offset_stride": float(offset_stride),
        "no_offset_embedding": float(0.0 if use_offset_embedding else 1.0),
        "native_repr_stride": float(getattr(args, "native_repr_stride", 1) or 1),
        "task_binary_vs_mmpd": 1.0,
        "univariate": 1.0,
    }


def partial_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "partials" / f"{dataset}.json"


def collect_partials(output_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    partial_dir = output_dir / "partials"
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not partial_dir.is_dir():
        return out
    for path in sorted(partial_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.stem
        out[dataset] = {str(k): dict(v) for k, v in payload.items()}
    return out


def write_metrics_csv(output_dir: Path, merged: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    rows = []
    for dataset, by_len in merged.items():
        for slice_len, metrics in by_len.items():
            row = {"dataset": dataset, "slice_len": slice_len, **metrics}
            rows.append(row)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def merge_and_write(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged = collect_partials(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", merged)
    write_metrics_csv(args.output_dir, merged)
    manifest = {
        "task": "binary_vs_mmpd_univariate",
        "positive_source": POSITIVE_SOURCE,
        "negative_source": NEGATIVE_SOURCE,
        "datasets": sorted(merged.keys()),
        "candidate_only": bool(getattr(args, "candidate_only", False)),
        "nonoverlapping_patches": bool(getattr(args, "nonoverlapping_patches", False)),
        "no_offset_embedding": bool(getattr(args, "no_offset_embedding", False)),
        "native_repr_stride": int(getattr(args, "native_repr_stride", 1) or 1),
        "pack_splits": getattr(args, "pack_splits", "test"),
        "pack_fraction": getattr(args, "pack_fraction", None),
        "mmpd_ordinal_quantize": bool(getattr(args, "mmpd_ordinal_quantize", False)),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"[merge] wrote {args.output_dir / 'metrics.json'}", flush=True)
    return merged


def parse_args() -> argparse.Namespace:
    # Reuse the full disc CLI, then override output default + force both fake sources.
    argv_backup = sys.argv
    # Inject defaults only when caller didn't pass them.
    injected = []
    joined = " ".join(sys.argv[1:])
    if "--output-dir" not in joined:
        injected += ["--output-dir", str(DEFAULT_OUTPUT)]
    if "--fake-sources" not in joined:
        injected += ["--fake-sources", "binary_staged", "mmpd"]
    sys.argv = [argv_backup[0], *injected, *argv_backup[1:]]
    try:
        args = disc_parse_args()
    finally:
        sys.argv = argv_backup
    # Always need both forecast sources in the bundle.
    args.fake_sources = [POSITIVE_SOURCE, NEGATIVE_SOURCE]
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    _apply_smoke_defaults_base(args)
    # Base smoke collapses fake_sources to 1; restore both for this task.
    args.fake_sources = [POSITIVE_SOURCE, NEGATIVE_SOURCE]
    if args.smoke_test and (
        args.output_dir == DEFAULT_DISC_OUTPUT or args.output_dir == DEFAULT_OUTPUT
    ):
        args.output_dir = DEFAULT_OUTPUT.parent / f"{DEFAULT_OUTPUT.name}-smoke"


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    print(
        f"[task] univariate binary_vs_mmpd "
        f"(+={POSITIVE_SOURCE}, -={NEGATIVE_SOURCE}) "
        f"output={args.output_dir}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n[{dataset}] loading/materializing raw packs", flush=True)
        bundle = build_raw_bundle(args, dataset, device)
        if POSITIVE_SOURCE not in bundle.fakes or NEGATIVE_SOURCE not in bundle.fakes:
            raise KeyError(
                f"{dataset}: need fakes for {POSITIVE_SOURCE} and {NEGATIVE_SOURCE}; "
                f"got {sorted(bundle.fakes)}"
            )
        y_ref = bundle.fakes[POSITIVE_SOURCE]
        splits = split_windows(
            int(y_ref.shape[0]),
            args,
            dataset,
            indices=bundle.indices,
            lookback=int(bundle.past.shape[-1]),
            horizon=int(y_ref.shape[-1]),
            test_stride=run_test_stride(bundle.run),
            series_starts=bundle.series_starts,
        )
        print(
            f"[{dataset}] windows={y_ref.shape[0]} train/val/test="
            f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
            f"variates={y_ref.shape[1]} horizon={y_ref.shape[-1]}",
            flush=True,
        )

        existing: Dict[str, Dict[str, float]] = {}
        path = partial_path(args.output_dir, dataset)
        if path.is_file() and not args.force_train:
            existing = {str(k): dict(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

        by_len: Dict[str, Dict[str, float]] = dict(existing)
        for slice_len in args.slice_lengths:
            key = str(int(slice_len))
            if key in by_len and not args.force_train:
                print(f"[{dataset}] L{slice_len}: reuse partial", flush=True)
                continue
            if int(slice_len) > int(y_ref.shape[-1]):
                print(f"[{dataset}] skipping L{slice_len} (horizon={y_ref.shape[-1]})", flush=True)
                continue
            metrics = train_classifier(args, dataset, int(slice_len), bundle, splits, device)
            by_len[key] = metrics
            write_json(path, by_len)
            print(
                f"[{dataset}] L{slice_len}: acc={metrics['disc_acc']:.4f} "
                f"auroc={metrics['disc_auroc']:.4f} bce={metrics['disc_bce']:.4f}",
                flush=True,
            )

    if args.merge_metrics:
        merge_and_write(args)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.merge_partials_only:
        merge_and_write(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
