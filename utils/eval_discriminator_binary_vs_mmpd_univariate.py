#!/usr/bin/env python3
"""Univariate real-vs-fake discriminator (binary vs GT, MMPD vs GT).

Same fair protocol as `eval_discriminator_texture_staged_vs_mmpd.py`, but each
example is a **single-variate** L-patch `[1, L]` (z-scored). One model per
`(dataset, fake_source, L)` is trained on patches pooled across **all**
variates.

Label **1 = fake** (binary_staged or mmpd), **0 = GT** — same task as the
multivariate texture disc, not binary-vs-mmpd.
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

from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    DEFAULT_DISC_OUTPUT,
    FAKE_SOURCES,
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
from typing import Literal

ReduceMode = Literal["per_variate", "joint"]

DEFAULT_OUTPUT = (
    DEFAULT_DISC_OUTPUT.parent
    / "disc-lb336-hz720-ordinal-four-patch-only-fair-univariate-bin16"
)


def _unique_absolute_slice_items(
    windows: np.ndarray,
    *,
    horizon: int,
    slice_len: int,
    n_var: int,
    offset_stride: int,
    series_starts: np.ndarray,
    lookback: int,
    seed: int,
) -> List[Tuple[int, int, int, int]]:
    """One random (window, offset) per absolute L-block × variate (real+fake pair).

    Absolute future index ``T`` for offset ``o`` in window ``w`` is
    ``series_starts[w] + lookback + o`` (past starts at ``series_starts[w]``).
    Overlapping 96-horizons that cover the same ``[T, T+L)`` collapse to one draw —
    same spirit as ``UniquePatchSegmentDataset`` for refine training.
    """
    starts = np.asarray(series_starts, dtype=np.int64)
    if starts.ndim != 1:
        raise ValueError(f"series_starts must be 1d, got {starts.shape}")
    offsets = list(range(0, int(horizon) - int(slice_len) + 1, max(1, int(offset_stride))))
    if not offsets:
        raise ValueError(f"no offsets for horizon={horizon} slice_len={slice_len}")
    # (abs_t, variate) -> [(window, offset), ...]
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for w in windows:
        w = int(w)
        if w < 0 or w >= starts.shape[0]:
            raise ValueError(f"window {w} out of range for series_starts len={starts.shape[0]}")
        fut0 = int(starts[w]) + int(lookback)
        for o in offsets:
            abs_t = fut0 + int(o)
            for v in range(int(n_var)):
                groups.setdefault((abs_t, int(v)), []).append((w, int(o)))
    rng = np.random.default_rng(int(seed))
    items: List[Tuple[int, int, int, int]] = []
    for (abs_t, v), parents in groups.items():
        w, o = parents[int(rng.integers(0, len(parents)))]
        items.append((w, o, v, 0))
        items.append((w, o, v, 1))
    rng.shuffle(items)
    return items


class UnivariateRealVsFakeDataset(Dataset):
    """Balanced univariate patches: label 1=fake, 0=GT. Pools all variates."""

    def __init__(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        past: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
        include_past: bool = False,
        apply_zscore: bool = True,
        apply_bin_center_shift: bool = False,
        legal_levels: Optional[np.ndarray] = None,
        bin_center_reduce: ReduceMode = "per_variate",
        unique_absolute_slices: bool = False,
        series_starts: Optional[np.ndarray] = None,
        lookback: Optional[int] = None,
    ) -> None:
        if real.shape != fake.shape:
            raise ValueError(f"real/fake shape mismatch: {real.shape} vs {fake.shape}")
        if real.shape[0] != past.shape[0]:
            raise ValueError(f"past/real window mismatch: {past.shape[0]} vs {real.shape[0]}")
        if slice_len > real.shape[-1]:
            raise ValueError(f"slice_len={slice_len} exceeds horizon={real.shape[-1]}")
        if apply_zscore and apply_bin_center_shift:
            raise ValueError("apply_zscore and apply_bin_center_shift are mutually exclusive")
        if apply_bin_center_shift:
            if legal_levels is None:
                raise ValueError("legal_levels required when apply_bin_center_shift=True")
            levels = np.asarray(legal_levels, dtype=np.float32)
            if levels.shape[:2] != real.shape[:2]:
                raise ValueError(
                    f"legal_levels N,V {levels.shape[:2]} != real {real.shape[:2]}"
                )

        self.real = real
        self.fake = fake
        self.past = past
        self.legal_levels = (
            None if legal_levels is None else np.asarray(legal_levels, dtype=np.float32)
        )
        self.slice_len = int(slice_len)
        self.include_past = bool(include_past)
        self.apply_zscore = bool(apply_zscore)
        self.apply_bin_center_shift = bool(apply_bin_center_shift)
        self.bin_center_reduce: ReduceMode = bin_center_reduce
        n_var = int(real.shape[1])
        horizon = int(real.shape[-1])
        offsets = list(range(0, horizon - slice_len + 1, max(1, int(offset_stride))))
        rng = np.random.default_rng(seed)

        if bool(unique_absolute_slices):
            if series_starts is None or lookback is None:
                raise ValueError(
                    "unique_absolute_slices requires series_starts and lookback "
                    "(absolute past starts + lookback → future timeline)"
                )
            items = _unique_absolute_slice_items(
                np.asarray(windows, dtype=np.int64),
                horizon=horizon,
                slice_len=int(slice_len),
                n_var=n_var,
                offset_stride=int(offset_stride),
                series_starts=np.asarray(series_starts, dtype=np.int64),
                lookback=int(lookback),
                seed=int(seed),
            )
            n_pairs = len(items) // 2
            if max_examples is not None:
                n_keep = min(n_pairs, max(1, int(max_examples) // 2))
                # items are [real, fake] pairs shuffled as individuals; rebuild by pair keys
                pair_keys = [(w, o, v) for (w, o, v, lab) in items if lab == 0]
                pick = rng.choice(len(pair_keys), size=n_keep, replace=False)
                items = []
                for i in pick:
                    w, o, v = pair_keys[int(i)]
                    items.append((w, o, v, 0))
                    items.append((w, o, v, 1))
                rng.shuffle(items)
            self.items = items
            print(
                f"[disc-uni] unique_absolute_slices: {n_pairs} unique (abs_t,variate) "
                f"pairs → {len(self.items)} examples "
                f"(from {len(windows)} windows × {len(offsets)} in-horizon offsets)",
                flush=True,
            )
        else:
            # (window, offset, variate, label)
            real_items = [
                (int(w), int(o), int(v), 0)
                for w in windows
                for o in offsets
                for v in range(n_var)
            ]
            fake_items = [
                (int(w), int(o), int(v), 1)
                for w in windows
                for o in offsets
                for v in range(n_var)
            ]
            n = min(len(real_items), len(fake_items))
            if max_examples is not None:
                n = min(n, max(1, int(max_examples) // 2))
            real_idx = rng.choice(len(real_items), size=n, replace=False)
            fake_idx = rng.choice(len(fake_items), size=n, replace=False)
            items = [real_items[i] for i in real_idx] + [fake_items[i] for i in fake_idx]
            rng.shuffle(items)
            self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _norm_segment(self, segment: np.ndarray, window: int, variate: int) -> np.ndarray:
        """Normalize one (1, T) univariate segment. Bin-center uses this segment's T only."""
        seg = np.asarray(segment, dtype=np.float32)
        if self.apply_bin_center_shift:
            assert self.legal_levels is not None
            levels = self.legal_levels[window, variate : variate + 1, :]  # (1, H)
            # bin_center_shift expects (N,V,T) / (N,V,H); V=1 for univariate disc.
            shifted, _ = bin_center_shift(
                seg[None, :, :],
                levels[None, :, :],
                reduce=self.bin_center_reduce,
            )
            return shifted[0].astype(np.float32)
        if self.apply_zscore:
            return zscore_time(seg)
        return seg

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window, offset, variate, label = self.items[idx]
        src = self.fake if label == 1 else self.real
        candidate = src[window, variate : variate + 1, offset : offset + self.slice_len]
        if self.include_past:
            past = self.past[window, variate : variate + 1]
            x = np.concatenate(
                [
                    self._norm_segment(past, window, variate),
                    self._norm_segment(candidate, window, variate),
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            x = self._norm_segment(candidate, window, variate).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(offset, dtype=torch.long),
            torch.tensor(float(label), dtype=torch.float32),
            torch.tensor(int(window), dtype=torch.long),
            torch.tensor(int(variate), dtype=torch.long),
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
    variates_all: List[np.ndarray] = []
    for batch in loader:
        x, offsets, labels, windows, variates = batch
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
        variates_all.append(variates.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    windows_np = np.concatenate(windows_all)
    variates_np = np.concatenate(variates_all)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (logits_np >= 0.0).astype(np.float32)
    out = {
        "disc_bce": total_loss / max(1, total_count),
        "disc_acc": float((preds == labels_np).mean()),
        "disc_auroc": binary_auroc(labels_np, probs),
        "n_examples": float(total_count),
        "positive_rate": float(labels_np.mean()),
    }
    out.update(
        window_level_metrics(windows_np, labels_np, probs, variates=variates_np)
    )
    return out


@torch.no_grad()
def collect_classifier_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Keep per-patch test scores so forecast plots show classifier decisions."""
    model.eval()
    fields: Dict[str, List[np.ndarray]] = {
        "prob_fake": [], "label": [], "window": [], "variate": [], "offset": [],
    }
    for x, offsets, labels, windows, variates in loader:
        logits = model(x.to(device), offsets.to(device))
        fields["prob_fake"].append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        fields["label"].append(labels.numpy().astype(np.float32))
        fields["window"].append(windows.numpy().astype(np.int64))
        fields["variate"].append(variates.numpy().astype(np.int64))
        fields["offset"].append(offsets.numpy().astype(np.int64))
    return {key: np.concatenate(parts) for key, parts in fields.items()}


def train_classifier(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    fake = bundle.fakes[fake_source]
    y_true = bundle.y_true_by_source[fake_source]
    horizon = int(y_true.shape[-1])
    n_variates = int(y_true.shape[1])
    max_offset = horizon - slice_len
    seed_base = args.seed + stable_hash(f"{dataset}:{fake_source}:uni:{slice_len}")
    include_past = not bool(getattr(args, "candidate_only", False))
    offset_stride = int(getattr(args, "offset_stride", 1) or 1)
    if bool(getattr(args, "nonoverlapping_patches", False)):
        offset_stride = int(slice_len)
    use_offset_embedding = not bool(getattr(args, "no_offset_embedding", False))
    apply_bin_center = bool(getattr(args, "disc_bin_center_shift", False))
    apply_zscore = not apply_bin_center
    legal_levels = getattr(bundle, "legal_levels", None)
    if apply_bin_center and legal_levels is None:
        raise ValueError(
            "disc_bin_center_shift requires bundle.legal_levels (N,V,H) for per-slice centering"
        )
    reduce_mode = str(getattr(args, "disc_bin_center_reduce", "per_variate"))
    if reduce_mode not in ("per_variate", "joint"):
        raise ValueError(f"invalid disc_bin_center_reduce={reduce_mode!r}")
    ds_kwargs = dict(
        offset_stride=offset_stride,
        include_past=include_past,
        apply_zscore=apply_zscore,
        apply_bin_center_shift=apply_bin_center,
        legal_levels=legal_levels,
        bin_center_reduce=reduce_mode,  # type: ignore[arg-type]
        unique_absolute_slices=bool(getattr(args, "unique_absolute_slices", False)),
        series_starts=getattr(bundle, "series_starts", None),
        lookback=int(getattr(args, "lookback", 0) or 0) or None,
    )
    if ds_kwargs["unique_absolute_slices"] and (
        ds_kwargs["series_starts"] is None or ds_kwargs["lookback"] is None
    ):
        raise ValueError(
            "unique_absolute_slices requires bundle.series_starts and args.lookback"
        )

    ds_train = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["train"], slice_len,
        seed=seed_base, max_examples=args.max_train_examples, **ds_kwargs,
    )
    ds_val = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["val"], slice_len,
        seed=seed_base + 1, max_examples=args.max_eval_examples, **ds_kwargs,
    )
    ds_test = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["test"], slice_len,
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
        f"[disc-uni] {dataset}/{fake_source}/L{slice_len}: real-vs-fake univariate "
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
    progress = EvalProgress(f"disc-uni/{dataset}/{fake_source}/L{slice_len}", args.epochs)
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
            if getattr(args, "max_batches_per_epoch", None) and batch_idx + 1 >= args.max_batches_per_epoch:
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

    if bool(getattr(args, "save_classification_scores", False)):
        score_path = (
            args.output_dir / "scores" / f"{dataset}_{fake_source}_L{slice_len}_test_scores.npz"
        )
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path, **collect_classifier_scores(model, test_loader, device))
        print(f"[disc-uni] wrote classifier scores {score_path}", flush=True)

    if bool(getattr(args, "save_checkpoints", False)):
        ckpt_path = (
            args.output_dir
            / "checkpoints"
            / f"{dataset}_{fake_source}_L{slice_len}_univariate_discriminator.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "dataset": dataset,
                "fake_source": fake_source,
                "task": "real_vs_fake_univariate",
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
        "unique_absolute_slices": float(
            1.0 if bool(getattr(args, "unique_absolute_slices", False)) else 0.0
        ),
        "no_offset_embedding": float(0.0 if use_offset_embedding else 1.0),
        "native_repr_stride": float(getattr(args, "native_repr_stride", 1) or 1),
        "univariate": 1.0,
    }


def partial_path(output_dir: Path, dataset: str, fake_source: str) -> Path:
    return output_dir / "partials" / f"{dataset}__{fake_source}.json"


def collect_partials(output_dir: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    partial_dir = output_dir / "partials"
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    if not partial_dir.is_dir():
        return out
    for path in sorted(partial_dir.glob("*.json")):
        stem = path.stem
        if "__" not in stem:
            continue
        dataset, fake_source = stem.split("__", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(dataset, {})[fake_source] = {
            str(k): dict(v) for k, v in payload.items()
        }
    return out


def write_metrics_csv(
    output_dir: Path,
    merged: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> None:
    rows = []
    for dataset, by_src in merged.items():
        for fake_source, by_len in by_src.items():
            for slice_len, metrics in by_len.items():
                rows.append(
                    {"dataset": dataset, "fake_source": fake_source, "slice_len": slice_len, **metrics}
                )
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def merge_and_write(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    merged = collect_partials(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", merged)
    write_metrics_csv(args.output_dir, merged)
    manifest = {
        "task": "real_vs_fake_univariate",
        "datasets": sorted(merged.keys()),
        "candidate_only": bool(getattr(args, "candidate_only", False)),
        "nonoverlapping_patches": bool(getattr(args, "nonoverlapping_patches", False)),
        "no_offset_embedding": bool(getattr(args, "no_offset_embedding", False)),
        "native_repr_stride": int(getattr(args, "native_repr_stride", 1) or 1),
        "pack_splits": getattr(args, "pack_splits", "test"),
        "pack_fraction": getattr(args, "pack_fraction", None),
        "ordinal_ladder_quantize": bool(getattr(args, "ordinal_ladder_quantize", False)),
        "univariate": True,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"[merge] wrote {args.output_dir / 'metrics.json'}", flush=True)
    return merged


def parse_args() -> argparse.Namespace:
    argv_backup = sys.argv
    injected = []
    joined = " ".join(sys.argv[1:])
    if "--output-dir" not in joined:
        injected += ["--output-dir", str(DEFAULT_OUTPUT)]
    sys.argv = [argv_backup[0], *injected, *argv_backup[1:]]
    try:
        args = disc_parse_args()
    finally:
        sys.argv = argv_backup
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    _apply_smoke_defaults_base(args)
    if args.smoke_test and (
        args.output_dir == DEFAULT_DISC_OUTPUT or args.output_dir == DEFAULT_OUTPUT
    ):
        args.output_dir = DEFAULT_OUTPUT.parent / f"{DEFAULT_OUTPUT.name}-smoke"


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    print(
        f"[task] univariate real-vs-fake sources={args.fake_sources} output={args.output_dir}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)

    if not args.datasets:
        raise ValueError("--datasets is empty")
    if not set(args.fake_sources).issubset(set(FAKE_SOURCES)):
        raise ValueError(f"--fake-sources must be within {FAKE_SOURCES}")

    for dataset in args.datasets:
        print(f"\n[{dataset}] loading/materializing raw packs", flush=True)
        # Always load both packs into the bundle when either is requested (shared GT).
        bundle_args = argparse.Namespace(**vars(args))
        bundle_args.fake_sources = list(FAKE_SOURCES)
        bundle = build_raw_bundle(bundle_args, dataset, device)
        y_ref = next(iter(bundle.y_true_by_source.values()))
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

        for fake_source in args.fake_sources:
            path = partial_path(args.output_dir, dataset, fake_source)
            existing: Dict[str, Dict[str, float]] = {}
            if path.is_file() and not args.force_train:
                existing = {
                    str(k): dict(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()
                }
            by_len: Dict[str, Dict[str, float]] = dict(existing)
            for slice_len in args.slice_lengths:
                key = str(int(slice_len))
                if key in by_len and not args.force_train:
                    print(f"[{dataset}/{fake_source}] L{slice_len}: reuse partial", flush=True)
                    continue
                if int(slice_len) > int(y_ref.shape[-1]):
                    print(
                        f"[{dataset}/{fake_source}] skipping L{slice_len} "
                        f"(horizon={y_ref.shape[-1]})",
                        flush=True,
                    )
                    continue
                metrics = train_classifier(
                    args, dataset, fake_source, int(slice_len), bundle, splits, device
                )
                by_len[key] = metrics
                write_json(path, by_len)
                print(
                    f"[{dataset}/{fake_source}] L{slice_len}: "
                    f"acc={metrics['disc_acc']:.4f} auroc={metrics['disc_auroc']:.4f} "
                    f"bce={metrics['disc_bce']:.4f}",
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
