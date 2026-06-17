#!/usr/bin/env python3
"""Learned discriminator texture eval for staged binary vs MMPD outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dual_scale_bin_filter import (
    BIN_MATCH_CHOICES,
    apply_bin_match_to_bundle,
)
from utils.binary_disc_debias import debias_binary_staged_fakes, resolve_dual_scale_bin_params
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    load_tsf_test_subset,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import (
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_CKPT_BASE,
    DEFAULT_MMPD_OUTPUT_ROOT,
    DEFAULT_SUBSET_DATASETS,
    dataset_window_lengths_for_run,
    evaluate_staged_binary,
    make_indices,
    resolve_staged_ckpt_dir,
    staged_anchor_run,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration


FAKE_SOURCES = ("binary_staged", "mmpd")
LOG2 = math.log(2.0)


@dataclass
class RawBundle:
    run: Any
    sub: Dict[str, Any]
    indices: List[int]
    past: np.ndarray
    y_true_by_source: Dict[str, np.ndarray]
    fakes: Dict[str, np.ndarray]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def pack_path(raw_eval_dir: Path, fake_source: str, dataset: str) -> Path:
    if fake_source == "binary_staged":
        return raw_eval_dir / "raw" / f"binary_staged_{dataset}.npz"
    if fake_source == "mmpd":
        return raw_eval_dir / "raw" / f"mmpd_{dataset}.npz"
    raise ValueError(f"unknown fake source: {fake_source}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validate_stochastic_pack(path: Path, pack: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in ("y_true", "samples", "indices") if key not in pack]
    if missing:
        raise KeyError(f"{path} missing required arrays: {missing}")
    samples = pack["samples"]
    if samples.ndim != 4 or samples.shape[2] < 1:
        raise ValueError(f"{path} samples must have shape [N, C, S, H] with S>=1, got {samples.shape}")
    if pack["y_true"].shape != samples[:, :, 0, :].shape:
        raise ValueError(
            f"{path} y_true/sample0 shape mismatch: "
            f"{pack['y_true'].shape} vs {samples[:, :, 0, :].shape}"
        )


def validate_variate_alignment(
    dataset: str,
    run: Any,
    sub: Mapping[str, Any],
    past: np.ndarray,
    y_true_by_source: Mapping[str, np.ndarray],
    fakes: Mapping[str, np.ndarray],
) -> None:
    expected = [int(i) for i in run_variate_indices(run)]
    if not expected:
        raise ValueError(f"{dataset}: staged metadata has no variate_indices")

    for name, value in (
        ("bundle", sub.get("variate_indices")),
        ("fine metadata", (sub.get("fine_metadata") or {}).get("variate_indices")),
        ("coarse metadata", (sub.get("coarse_metadata") or {}).get("variate_indices")),
    ):
        if value is None:
            continue
        observed = [int(i) for i in value]
        if observed != expected:
            raise ValueError(
                f"{dataset}: {name} variate_indices {observed} do not match staged run {expected}"
            )

    n_vars = len(expected)
    if past.shape[1] != n_vars:
        raise ValueError(
            f"{dataset}: past has {past.shape[1]} variates but staged subset has "
            f"{n_vars}: {expected}"
        )
    for fake_source, y_true in y_true_by_source.items():
        fake = fakes[fake_source]
        if y_true.shape[1] != n_vars:
            raise ValueError(
                f"{dataset}/{fake_source}: y_true has {y_true.shape[1]} variates "
                f"but staged subset has {n_vars}: {expected}"
            )
        if fake.shape[1] != n_vars:
            raise ValueError(
                f"{dataset}/{fake_source}: fake has {fake.shape[1]} variates "
                f"but staged subset has {n_vars}: {expected}"
            )


def saved_indices(raw_eval_dir: Path, dataset: str) -> Optional[List[int]]:
    for fake_source in FAKE_SOURCES:
        path = pack_path(raw_eval_dir, fake_source, dataset)
        if path.is_file():
            pack = load_npz(path)
            if "indices" in pack:
                return [int(i) for i in pack["indices"].tolist()]
    index_json = raw_eval_dir / "raw" / f"indices_{dataset}_mmpd_eval.json"
    if index_json.is_file():
        return [int(i) for i in load_json(index_json)]
    return None


def raw_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    out = copy.copy(args)
    out.output_dir = args.raw_eval_dir
    out.mmpd_output_root = args.mmpd_output_root
    out.mmpd_backbone = getattr(args, "mmpd_backbone", "MaskAE")
    out.patch_size = None  # dataset rules (e.g. electricity/traffic patch 24)
    out.force_binary_eval = args.force_raw_eval
    out.force_mmpd_eval = args.force_raw_eval
    out.binary_batch_size = args.raw_binary_batch_size
    out.mmpd_eval_batch_size = args.raw_mmpd_batch_size
    out.sample_num = 1
    return out


def ensure_raw_packs(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any], List[int], Dict[str, Dict[str, np.ndarray]]]:
    ckpt_dir = resolve_staged_ckpt_dir(args.ckpt_base, dataset, args.anchor_config)
    run, sub = staged_anchor_run(dataset, ckpt_dir, args.test_stride)
    indices = saved_indices(args.raw_eval_dir, dataset)
    eval_args = raw_eval_args(args)
    if indices is None:
        indices = make_indices(eval_args, run)

    if "binary_staged" in args.fake_sources:
        binary_path = pack_path(args.raw_eval_dir, "binary_staged", dataset)
        if args.force_raw_eval or not binary_path.is_file():
            print(f"[raw] materializing binary_staged/{dataset} -> {binary_path}", flush=True)
            evaluate_staged_binary(eval_args, run, sub, indices, device)

    if "mmpd" in args.fake_sources:
        mmpd_path = pack_path(args.raw_eval_dir, "mmpd", dataset)
        if args.force_raw_eval or not mmpd_path.is_file():
            print(f"[raw] materializing mmpd/{dataset} -> {mmpd_path}", flush=True)
            ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
            run_mmpd_eval(eval_args, run, indices)

    packs: Dict[str, Dict[str, np.ndarray]] = {}
    for fake_source in args.fake_sources:
        path = pack_path(args.raw_eval_dir, fake_source, dataset)
        if not path.is_file():
            raise FileNotFoundError(f"raw pack missing after materialization: {path}")
        pack = load_npz(path)
        validate_stochastic_pack(path, pack)
        packs[fake_source] = pack

    return run, sub, indices, packs


def load_past_windows(
    args: argparse.Namespace,
    run: Any,
    indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    lookback, horizon = dataset_window_lengths_for_run(args, run)
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
        batch_size=args.raw_load_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    past_all: List[np.ndarray] = []
    for past, _future in loader:
        past_all.append(past.numpy())
    return np.concatenate(past_all, axis=0)


def build_raw_bundle(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> RawBundle:
    run, sub, indices, packs = ensure_raw_packs(args, dataset, device)
    past = load_past_windows(args, run, indices, device)
    y_true_by_source: Dict[str, np.ndarray] = {}
    fakes: Dict[str, np.ndarray] = {}
    ref_shape: Optional[Tuple[int, ...]] = None
    for fake_source, pack in packs.items():
        y_true = pack["y_true"].astype(np.float32)
        fake = pack["samples"][:, :, 0, :].astype(np.float32)
        if ref_shape is None:
            ref_shape = y_true.shape
        elif y_true.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: y_true shape differs from first pack")
        if fake.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: fake shape differs from y_true")
        if not np.array_equal(pack["indices"], np.asarray(indices, dtype=pack["indices"].dtype)):
            raise ValueError(f"{dataset}/{fake_source}: raw pack indices do not match discriminator indices")
        y_true_by_source[fake_source] = y_true
        fakes[fake_source] = fake

    if past.shape[0] != ref_shape[0]:
        raise ValueError(f"{dataset}: past/y_true window mismatch {past.shape[0]} vs {ref_shape[0]}")
    validate_variate_alignment(dataset, run, sub, past, y_true_by_source, fakes)

    if len(y_true_by_source) > 1:
        sources = list(y_true_by_source)
        ref = y_true_by_source[sources[0]]
        for src in sources[1:]:
            other = y_true_by_source[src]
            mse = float(np.mean((ref - other) ** 2))
            if mse > 1e-6:
                print(
                    f"[warn] {dataset}: y_true differs between {sources[0]} and {src} "
                    f"(mse={mse:.6f}); each discriminator uses its own pack GT.",
                    flush=True,
                )

    if args.bin_match_filter:
        print(
            f"[{dataset}] applying dual-scale bin-match filter={args.bin_match_filter} "
            f"(H={args.bin_image_height}, max_scale={args.bin_max_scale}, decoder={args.bin_decoder})",
            flush=True,
        )
        y_true_by_source, fakes = apply_bin_match_to_bundle(
            mode=args.bin_match_filter,
            past=past.astype(np.float32),
            y_true_by_source=y_true_by_source,
            fakes=fakes,
            image_height=args.bin_image_height,
            max_scale=args.bin_max_scale,
            std_floor=args.bin_std_floor,
            decoder=args.bin_decoder,
            device=device,
        )

    if args.binary_debias_quantization and "binary_staged" in fakes:
        max_scale, coarse_h, fine_h = resolve_dual_scale_bin_params(
            dataset,
            sub,
            fallback_max_scale=args.bin_max_scale,
            coarse_height=args.bin_coarse_height,
            fine_height=args.bin_fine_height,
        )
        debiased, debias_stats = debias_binary_staged_fakes(
            fakes["binary_staged"],
            max_scale=max_scale,
            coarse_height=coarse_h,
            fine_height=fine_h,
            seed=args.seed,
            dataset=dataset,
        )
        fakes["binary_staged"] = debiased
        print(
            f"[{dataset}] binary debias: max_scale={max_scale} coarse_h={coarse_h} "
            f"fine_h={fine_h} half_fine_bin={debias_stats['half_fine_bin']:.6f} "
            f"flatline_frac={debias_stats['flatline_frac']:.3f} "
            f"debias_frac={debias_stats['debias_frac']:.3f}",
            flush=True,
        )

    expected_variates = [int(i) for i in run_variate_indices(run)]
    print(
        f"[{dataset}] staged subset={run_subset_id(run)} variates={expected_variates}",
        flush=True,
    )

    return RawBundle(
        run=run,
        sub=sub,
        indices=[int(i) for i in indices],
        past=past.astype(np.float32),
        y_true_by_source=y_true_by_source,
        fakes=fakes,
    )


def window_time_bounds(
    dataset: str,
    indices: Sequence[int],
    lookback: int,
    horizon: int,
    test_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(indices, dtype=np.int64)
    if dataset == "dalia":
        starts = raw
        span = 1
    else:
        starts = raw * max(1, int(test_stride))
        span = int(lookback) + int(horizon)
    ends = starts + span
    return starts, ends


def split_windows(
    n_windows: int,
    args: argparse.Namespace,
    dataset: str,
    *,
    indices: Optional[Sequence[int]] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    test_stride: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if args.max_windows is not None:
        n_windows = min(n_windows, int(args.max_windows))
    if indices is None or lookback is None or horizon is None or test_stride is None:
        raise ValueError("split_windows requires indices, lookback, horizon, and test_stride")
    raw_indices = [int(i) for i in list(indices)[:n_windows]]
    if len(raw_indices) != n_windows:
        raise ValueError(f"{dataset}: got {len(raw_indices)} split indices for {n_windows} windows")

    starts, ends = window_time_bounds(dataset, raw_indices, int(lookback), int(horizon), int(test_stride))
    order = np.argsort(starts, kind="mergesort")
    n_train_target = max(1, int(round(len(order) * args.train_fraction)))
    n_val_target = max(1, int(round(len(order) * args.val_fraction)))
    if n_train_target + n_val_target >= len(order):
        n_val_target = max(1, len(order) - n_train_target - 1)
    n_test = len(order) - n_train_target - n_val_target
    if n_test < 1:
        raise ValueError(f"not enough windows for train/val/test split: {len(order)}")

    test = order[-n_test:]
    test_start = starts[test].min()
    train_val_pool = order[:-n_test]
    train_val_pool = np.asarray([idx for idx in train_val_pool if ends[idx] <= test_start], dtype=np.int64)
    if len(train_val_pool) < 2:
        # illness-like tiny sets: stride-1 windows overlap heavily; index split is fine.
        train_val_pool = np.asarray(order[:-n_test], dtype=np.int64)
        print(
            f"[warn] {dataset}: relaxed train/val split (overlapping windows; "
            f"windows={len(order)}, test={n_test}, test_stride={test_stride})",
            flush=True,
        )
    if len(train_val_pool) < 2:
        raise ValueError(
            f"{dataset}: not enough windows for train/val split "
            f"(windows={len(order)}, test={n_test}, lookback={lookback}, horizon={horizon}, "
            f"test_stride={test_stride})"
        )

    val_ratio = args.val_fraction / max(args.train_fraction + args.val_fraction, 1e-8)
    n_val = max(1, int(round(len(train_val_pool) * val_ratio)))
    if n_val >= len(train_val_pool):
        n_val = len(train_val_pool) - 1
    train = train_val_pool[:-n_val]
    val = train_val_pool[-n_val:]
    return {
        "train": np.sort(train),
        "val": np.sort(val),
        "test": np.sort(test),
    }


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def zscore_time(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-5)


class HorizonSliceDataset(Dataset):
    def __init__(
        self,
        past: np.ndarray,
        real: np.ndarray,
        fake: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
    ) -> None:
        if real.shape != fake.shape:
            raise ValueError(f"real/fake shape mismatch: {real.shape} vs {fake.shape}")
        if real.shape[0] != past.shape[0]:
            raise ValueError(f"past/real window mismatch: {past.shape[0]} vs {real.shape[0]}")
        if slice_len > real.shape[-1]:
            raise ValueError(f"slice_len={slice_len} exceeds horizon={real.shape[-1]}")

        self.past = past
        self.real = real
        self.fake = fake
        self.slice_len = int(slice_len)
        offsets = list(range(0, real.shape[-1] - slice_len + 1, max(1, int(offset_stride))))
        real_items = [(int(w), int(o), 0) for w in windows for o in offsets]
        fake_items = [(int(w), int(o), 1) for w in windows for o in offsets]

        rng = np.random.default_rng(seed)
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, offset, label = self.items[idx]
        candidate_src = self.fake if label == 1 else self.real
        candidate = candidate_src[window, :, offset : offset + self.slice_len]
        past = self.past[window]
        x = np.concatenate([zscore_time(past), zscore_time(candidate)], axis=-1).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(offset, dtype=torch.long),
            torch.tensor(float(label), dtype=torch.float32),
        )


class InvertedSliceDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        max_offset: int,
        d_model: int,
        n_heads: int,
        depth: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.offset_embedding = nn.Embedding(max_offset + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]. Like iTransformer, variates are tokens and time is embedded.
        tokens = self.value_embedding(x)
        tokens = tokens + self.offset_embedding(offsets).unsqueeze(1)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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
    for x, offsets, labels in loader:
        x = x.to(device)
        offsets = offsets.to(device)
        labels = labels.to(device)
        logits = model(x, offsets)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(labels.numel())
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (logits_np >= 0.0).astype(np.float32)
    return {
        "disc_bce": total_loss / max(1, total_count),
        "disc_acc": float((preds == labels_np).mean()),
        "disc_auroc": binary_auroc(labels_np, probs),
        "n_examples": float(total_count),
        "positive_rate": float(labels_np.mean()),
    }


def train_classifier(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: RawBundle,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    fake = bundle.fakes[fake_source]
    y_true = bundle.y_true_by_source[fake_source]
    max_offset = y_true.shape[-1] - slice_len
    seed_base = args.seed + stable_hash(f"{dataset}:{fake_source}:{slice_len}")
    ds_train = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["train"],
        slice_len,
        seed=seed_base,
        offset_stride=args.offset_stride,
        max_examples=args.max_train_examples,
    )
    ds_val = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["val"],
        slice_len,
        seed=seed_base + 1,
        offset_stride=args.offset_stride,
        max_examples=args.max_eval_examples,
    )
    ds_test = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["test"],
        slice_len,
        seed=seed_base + 2,
        offset_stride=args.offset_stride,
        max_examples=args.max_eval_examples,
    )
    generator = torch.Generator()
    generator.manual_seed(seed_base)
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    seq_len = int(bundle.past.shape[-1] + slice_len)
    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=max_offset,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    progress = EvalProgress(f"disc/{dataset}/{fake_source}/L{slice_len}", args.epochs)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_idx, (x, offsets, labels) in enumerate(train_loader):
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
                f"val_auc={val_metrics['disc_auroc']:.3f} elapsed={fmt_duration(time.time() - t0)}"
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
            / f"{dataset}_{fake_source}_L{slice_len}_discriminator.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "dataset": dataset,
                "fake_source": fake_source,
                "slice_len": slice_len,
            },
            ckpt_path,
        )

    out = {
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
        "horizon": float(y_true.shape[-1]),
        "n_variates": float(y_true.shape[1]),
        "log2_bce_gap": float(abs(test_metrics["disc_bce"] - LOG2)),
    }
    return out


def partial_path(output_dir: Path, dataset: str, fake_source: str) -> Path:
    return output_dir / "partials" / f"{dataset}__{fake_source}.json"


def legacy_partial_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "partials" / f"{dataset}.json"


def write_source_partial(
    output_dir: Path,
    dataset: str,
    fake_source: str,
    by_len: Mapping[str, Mapping[str, float]],
) -> None:
    write_json(partial_path(output_dir, dataset, fake_source), dict(by_len))


def collect_partials(output_dir: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    merged: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    partial_dir = output_dir / "partials"
    if not partial_dir.is_dir():
        return merged
    for path in sorted(partial_dir.glob("*.json")):
        stem = path.stem
        data = load_json(path)
        if "__" in stem:
            dataset, fake_source = stem.split("__", 1)
            merged.setdefault(dataset, {})[fake_source] = data
            continue
        if not isinstance(data, dict):
            continue
        if data and all(key in FAKE_SOURCES for key in data):
            merged[stem] = data
    return merged


def existing_combo(
    output_dir: Path,
    dataset: str,
    fake_source: str,
    slice_len: int,
) -> Optional[Dict[str, float]]:
    path = partial_path(output_dir, dataset, fake_source)
    if path.is_file():
        metrics = load_json(path).get(str(slice_len))
        return metrics if isinstance(metrics, dict) else None
    legacy = legacy_partial_path(output_dir, dataset)
    if legacy.is_file():
        metrics = load_json(legacy).get(fake_source, {}).get(str(slice_len))
        return metrics if isinstance(metrics, dict) else None
    return None


def merge_partial_metrics(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    merged = collect_partials(args.output_dir)
    if not merged:
        return {}
    write_json(args.output_dir / "metrics.json", merged)
    fields = [
        "dataset",
        "fake_source",
        "slice_len",
        "disc_bce",
        "log2_bce_gap",
        "disc_acc",
        "disc_auroc",
        "best_val_bce",
        "best_epoch",
        "epochs_run",
        "n_train",
        "n_val",
        "n_test",
        "n_windows_train",
        "n_windows_val",
        "n_windows_test",
        "n_variates",
        "horizon",
    ]
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, by_source in merged.items():
            for fake_source, by_len in by_source.items():
                for slice_key, metrics in by_len.items():
                    row = {"dataset": dataset, "fake_source": fake_source, "slice_len": int(slice_key)}
                    row.update({key: metrics.get(key) for key in fields if key not in row})
                    writer.writerow(row)

    merged_datasets = sorted(merged.keys()) or list(args.datasets)
    merged_sources = sorted({src for by_source in merged.values() for src in by_source})
    manifest = {
        "datasets": merged_datasets,
        "fake_sources": merged_sources or list(args.fake_sources),
        "slice_lengths": args.slice_lengths,
        "raw_eval_dir": str(args.raw_eval_dir),
        "test_fraction": args.test_fraction,
        "test_stride": args.test_stride,
        "staged_ckpts": {
            d: str(resolve_staged_ckpt_dir(args.ckpt_base, d, args.anchor_config))
            for d in merged_datasets
        },
        "anchor_config": args.anchor_config,
        "mmpd_output_root": str(args.mmpd_output_root),
    }
    if getattr(args, "bin_match_filter", None):
        manifest["bin_match_filter"] = args.bin_match_filter
        manifest["bin_image_height"] = args.bin_image_height
        manifest["bin_max_scale"] = args.bin_max_scale
        manifest["bin_decoder"] = args.bin_decoder
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        f"[merge] wrote metrics for datasets={merged_datasets} fake_sources={manifest['fake_sources']}",
        flush=True,
    )
    return merged


def write_outputs(args: argparse.Namespace, results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_results in results.items():
        for fake_source, by_len in dataset_results.items():
            if not by_len:
                continue
            path = partial_path(args.output_dir, dataset, fake_source)
            existing = load_json(path) if path.is_file() else {}
            existing.update(by_len)
            write_source_partial(args.output_dir, dataset, fake_source, existing)
    if args.merge_metrics:
        merge_partial_metrics(args)


def run_merge_only(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_partial_metrics(args)
    if not merged:
        raise FileNotFoundError(f"No partial metrics found under {args.output_dir / 'partials'}")


def valid_slice_lengths(horizon: int, requested: Sequence[int]) -> Tuple[List[int], List[int]]:
    valid = [int(length) for length in requested if int(length) <= horizon]
    skipped = [int(length) for length in requested if int(length) > horizon]
    return valid, skipped


def run_eval(args: argparse.Namespace) -> None:
    if not set(args.fake_sources).issubset(set(FAKE_SOURCES)):
        raise ValueError(f"--fake-sources must be within {FAKE_SOURCES}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}")

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for dataset in args.datasets:
        print(f"\n[{dataset}] loading/materializing raw packs", flush=True)
        bundle = build_raw_bundle(args, dataset, device)
        n = next(iter(bundle.y_true_by_source.values())).shape[0]
        ref_y = next(iter(bundle.y_true_by_source.values()))
        splits = split_windows(
            n,
            args,
            dataset,
            indices=bundle.indices,
            lookback=bundle.past.shape[-1],
            horizon=ref_y.shape[-1],
            test_stride=run_test_stride(bundle.run),
        )
        print(
            f"[{dataset}] windows={n} train/val/test="
            f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
            f"variates={ref_y.shape[1]} horizon={ref_y.shape[-1]} "
            f"subset={run_subset_id(bundle.run)}",
            flush=True,
        )
        results.setdefault(dataset, {})
        horizon = int(ref_y.shape[-1])
        valid_lens, skipped_lens = valid_slice_lengths(horizon, args.slice_lengths)
        if skipped_lens:
            print(
                f"[{dataset}] skipping slice lengths {skipped_lens} (horizon={horizon})",
                flush=True,
            )
        if not valid_lens:
            print(f"[{dataset}] no valid slice lengths for horizon={horizon}; skipping", flush=True)
            continue
        for fake_source in args.fake_sources:
            results[dataset].setdefault(fake_source, {})
            for slice_len in valid_lens:
                trained = False
                if not args.force_train:
                    existing = existing_combo(args.output_dir, dataset, fake_source, int(slice_len))
                    if existing is not None:
                        print(f"[skip] existing metrics dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                        results[dataset][fake_source][str(slice_len)] = existing
                    else:
                        print(f"[train] dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                        metrics = train_classifier(
                            args, dataset, fake_source, int(slice_len), bundle, splits, device
                        )
                        results[dataset][fake_source][str(slice_len)] = metrics
                        trained = True
                else:
                    print(f"[train] dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                    metrics = train_classifier(
                        args, dataset, fake_source, int(slice_len), bundle, splits, device
                    )
                    results[dataset][fake_source][str(slice_len)] = metrics
                    trained = True

                if args.visualize_confusions:
                    try:
                        from utils.visualize_discriminator_texture_confusions import visualize_combo
                    except ModuleNotFoundError as exc:
                        print(f"[viz] skip {dataset}/{fake_source}/L{slice_len}: {exc}", flush=True)
                        continue

                    try:
                        visualize_combo(
                            args,
                            dataset,
                            fake_source,
                            int(slice_len),
                            bundle,
                            splits,
                            device,
                        )
                    except FileNotFoundError as exc:
                        if trained:
                            raise
                        print(f"[viz] skip {dataset}/{fake_source}/L{slice_len}: {exc}", flush=True)
            write_outputs(args, {dataset: {fake_source: results[dataset][fake_source]}})


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n, c, lookback, horizon = 18, 3, 32, 32
    indices = [i * (lookback + horizon) for i in range(n)]
    past = rng.normal(size=(n, c, lookback)).astype(np.float32)
    y = rng.normal(size=(n, c, horizon)).astype(np.float32)
    fake = (0.7 * y + 0.3 * rng.normal(size=(n, c, horizon))).astype(np.float32)
    bundle = RawBundle(
        run=None,
        sub={},
        indices=indices,
        past=past,
        y_true_by_source={"binary_staged": y},
        fakes={"binary_staged": fake},
    )
    args.datasets = ["selftest"]
    args.fake_sources = ["binary_staged"]
    args.slice_lengths = [8]
    args.epochs = min(args.epochs, 2)
    args.patience = 2
    args.max_train_examples = 128
    args.max_eval_examples = 64
    args.batch_size = min(args.batch_size, 32)
    args.max_batches_per_epoch = 2
    device = torch.device("cpu")
    splits = split_windows(
        n,
        args,
        "selftest",
        indices=bundle.indices,
        lookback=lookback,
        horizon=horizon,
        test_stride=1,
    )
    metrics = train_classifier(args, "selftest", "binary_staged", 8, bundle, splits, device)
    print(json.dumps(metrics, indent=2, sort_keys=True))


DEFAULT_DISC_OUTPUT = (
    REPO_ROOT / "results" / "datasets" / "06-14-disc-texture-flat-subsets-ema099-vs-mmpd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_SUBSET_DATASETS))
    parser.add_argument("--anchor-config", type=str, default=DEFAULT_ANCHOR_CONFIG)
    parser.add_argument("--ckpt-base", type=Path, default=DEFAULT_CKPT_BASE)
    parser.add_argument("--fake-sources", nargs="+", default=list(FAKE_SOURCES), choices=list(FAKE_SOURCES))
    parser.add_argument("--slice-lengths", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DISC_OUTPUT,
    )
    parser.add_argument(
        "--raw-eval-dir",
        type=Path,
        default=DEFAULT_DISC_OUTPUT.parent / "06-14-raw-texture-flat-subsets-ema099-vs-mmpd",
    )
    parser.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD_OUTPUT_ROOT)
    parser.add_argument(
        "--mmpd-backbone",
        choices=["Decoder", "MaskAE"],
        default="MaskAE",
        help="MMPD backbone for checkpoint lookup (fair run uses MaskAE).",
    )
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--probabilistic-sampler", choices=["dpmpp", "ddim", "ddpm"], default="dpmpp")
    parser.add_argument("--gmm-components", type=int, default=1)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--topk-max", type=int, default=3)
    parser.add_argument("--raw-binary-batch-size", type=int, default=8)
    parser.add_argument("--raw-mmpd-batch-size", type=int, default=16)
    parser.add_argument("--raw-load-batch-size", type=int, default=64)
    parser.add_argument("--force-raw-eval", action="store_true")
    parser.add_argument("--no-update-mmpd", action="store_true")
    parser.add_argument(
        "--bin-match-filter",
        choices=list(BIN_MATCH_CHOICES),
        default=None,
        help="Round-trip horizons through staged binary 16x16 occupancy lattice. "
        "mmpd=fakes only; both=both fakes; all=GT+fakes.",
    )
    parser.add_argument("--bin-image-height", type=int, default=16)
    parser.add_argument("--bin-coarse-height", type=int, default=16)
    parser.add_argument("--bin-fine-height", type=int, default=16)
    parser.add_argument("--bin-max-scale", type=float, default=3.5)
    parser.add_argument("--bin-std-floor", type=float, default=1e-8)
    parser.add_argument(
        "--binary-debias-quantization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jitter non-flatline binary_staged fakes by up to ±½ fine bin (discriminator only).",
    )
    parser.add_argument(
        "--bin-decoder",
        choices=["mean", "expectation", "pdf_expectation"],
        default="mean",
        help="decode_dual mode for round-trip filter (match staged eval decoder).",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--offset-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--visualize-confusions",
        action="store_true",
        help="After each combo, save TP/TN/FP/FN PNGs under output-dir/disc_confusions/.",
    )
    parser.add_argument("--viz-per-bucket", type=int, default=2)
    parser.add_argument("--viz-variate", type=int, default=0)
    parser.add_argument("--viz-lookback-tail", type=int, default=32)
    parser.add_argument(
        "--viz-plot-dir",
        type=Path,
        default=None,
        help="Override confusion plot root (default: <output-dir>/disc_confusions).",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--merge-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, merge partials into metrics.json (disable for parallel shard jobs).",
    )
    parser.add_argument(
        "--merge-partials-only",
        action="store_true",
        help="Only merge partials/ into metrics.json + CSV + manifest.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-smoke"
    if args.bin_match_filter and args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-binmatch-{args.bin_match_filter}"
    args.datasets = args.datasets[:1]
    args.fake_sources = args.fake_sources[:1]
    args.slice_lengths = args.slice_lengths[:1]
    args.test_max_items = min(args.test_max_items or 8, 8)
    args.max_windows = min(args.max_windows or 8, 8)
    args.max_train_examples = min(args.max_train_examples or 128, 128)
    args.max_eval_examples = min(args.max_eval_examples or 64, 64)
    args.batch_size = min(args.batch_size, 32)
    args.epochs = min(args.epochs, 2)
    args.patience = min(args.patience, 2)
    args.max_batches_per_epoch = min(args.max_batches_per_epoch or 2, 2)


def main() -> None:
    args = parse_args()
    if args.bin_match_filter and args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-binmatch-{args.bin_match_filter}"
    if args.visualize_confusions:
        args.save_checkpoints = True
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test(args)
    elif args.merge_partials_only:
        run_merge_only(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
