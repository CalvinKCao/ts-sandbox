"""Shared data and discriminator protocol for ordinal refinement kill tests.

The binary and MMPD generators are intentionally different, but they must use
the same source windows, strict binary crop eligibility, ordinal support, and
discriminator optimization to make their scores comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.ordinal_patch_refinement_killtest import smoke
from experiments.ordinal_patch_refinement_killtest.nonoverlap_protocol import (
    HORIZON,
    TRAIN_STRIDE,
    build_protocol,
)
from experiments.ordinal_patch_refinement_killtest.ordinal_grid import (
    canonicalize_ranks,
    snap_ranks_to_ladder,
)
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    HorizonSliceDataset,
    InvertedSliceDiscriminator,
    evaluate_classifier,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool


@dataclass(frozen=True)
class SharedOrdinalData:
    """One source-window and ordinal-ladder contract for both generators."""

    protocol: dict[str, Any]
    pool_by: dict[str, Any]
    ladder: Any


@dataclass(frozen=True)
class DiscriminatorConfig:
    lookback: int = 96
    horizon: int = 16
    slice_len: int = 8
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    depth: int = 2
    dropout: float = 0.1


def load_shared_ordinal_data(
    dataset: str,
    n_variates: int,
    *,
    lookback: int = 96,
    horizon: int = HORIZON,
    train_stride: int = TRAIN_STRIDE,
    test_stride: int = 4,
) -> SharedOrdinalData:
    """Load identical source pools, split IDs, and ordinal ladder for both paths."""
    if horizon != HORIZON or train_stride != TRAIN_STRIDE or test_stride != 4:
        raise ValueError(
            "ordinal discriminator protocol is fixed at horizon=16, train_stride=2, test_stride=4"
        )
    protocol = build_protocol(dataset, n_variates, lookback=lookback)
    pool_by = {
        "train": load_tsf_pack_pool(
            dataset, list(range(n_variates)), lookback=lookback, horizon=horizon,
            train_stride=train_stride, test_stride=test_stride, pack_splits=["train"],
        )[0],
        "val": load_tsf_pack_pool(
            dataset, list(range(n_variates)), lookback=lookback, horizon=horizon,
            train_stride=1, test_stride=test_stride, pack_splits=["val"],
        )[0],
        "test": load_tsf_pack_pool(
            dataset, list(range(n_variates)), lookback=lookback, horizon=horizon,
            train_stride=1, test_stride=test_stride, pack_splits=["test"],
        )[0],
    }
    _, _, _, stats = load_dataset(
        dataset, list(range(n_variates)), lookback=lookback, horizon=horizon,
        stride=1, test_stride=test_stride, use_ordinal_window_norm=True,
    )
    return SharedOrdinalData(protocol=protocol, pool_by=pool_by, ladder=stats["ordinal_ladder"])


def strict_binary_patch_filter(
    pool: Any,
    indices: Iterable[int],
    ladder: Any,
    *,
    high_bins: int = 256,
    coarse_bins: int = 16,
    max_selected: int | None = None,
) -> tuple[list[int], dict[str, int | str]]:
    """Retain a window iff the binary 32x8 path has at least one valid crop."""
    if high_bins < smoke.PATCH_H or coarse_bins != 16:
        raise ValueError(
            "strict binary eligibility requires a 16-bin coarse input and "
            f"at least {smoke.PATCH_H} high-resolution bins"
        )
    selected: list[int] = []
    stats: dict[str, int | str] = {
        "candidates": 0,
        "skipped_oob": 0,
        "skipped_oob_canvas": 0,
        "skipped_oob_column_edge": 0,
        "kept": 0,
        "windows_considered": 0,
        "windows_with_patches": 0,
        "rule": (
            f"strict: retain window iff binary {high_bins}-row CDF path has any valid 32x8 crop"
        ),
    }
    for raw_index in indices:
        index = int(raw_index)
        past, future = pool[index]
        past_ord, future_ord, batch_ladder, _ = ordinal_encode(
            past.unsqueeze(0), future.unsqueeze(0)[..., :HORIZON], ladder=ladder,
            apply_ood_shift=True, causal_only=True,
        )
        assert future_ord is not None
        rank_max = batch_ladder.rank_max_per_variate().to(dtype=torch.float32)
        target = smoke._cdf_from_values(future_ord, rank_max, high_bins)
        coarse = smoke._cdf_from_values(future_ord, rank_max, coarse_bins)
        upscaled = smoke._vertical_upsample(coarse, high_bins)
        history = smoke._cdf_from_values(
            past_ord[..., -smoke.PATCH_W :], rank_max, high_bins,
        )
        kept_in_window = 0
        for variate in range(int(upscaled.shape[1])):
            bins = TimeSeriesTo2D.bin_indices_from_cdf(
                coarse[:, variate : variate + 1],
            )[0, 0].long()
            _naive, _hist, _target, _coords, patch_stats = smoke._patch_batch(
                upscaled[0, variate : variate + 1],
                target[0, variate : variate + 1],
                history[0, variate : variate + 1],
                bins,
            )
            for key in (
                "candidates", "skipped_oob", "skipped_oob_canvas",
                "skipped_oob_column_edge", "kept",
            ):
                stats[key] = int(stats[key]) + int(patch_stats[key])
            kept_in_window += int(patch_stats["kept"])
        stats["windows_considered"] = int(stats["windows_considered"]) + 1
        if kept_in_window > 0:
            selected.append(index)
            stats["windows_with_patches"] = int(stats["windows_with_patches"]) + 1
            if max_selected is not None and len(selected) >= max_selected:
                break
    if not selected:
        raise RuntimeError("binary strict-OOB patch filter retained no windows")
    return selected, stats


def canonicalize_disc_pair(
    gt_rank_raw: torch.Tensor,
    fake_rank_raw: torch.Tensor,
    ladder: Any,
    bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalize GT and fake via the identical ordinal-grid transformation."""
    if gt_rank_raw.shape != fake_rank_raw.shape:
        raise ValueError(
            f"GT/fake rank shape mismatch: {tuple(gt_rank_raw.shape)} vs {tuple(fake_rank_raw.shape)}"
        )
    rank_max = ladder.rank_max_per_variate().to(dtype=torch.float32)
    fake_ladder_rank = snap_ranks_to_ladder(fake_rank_raw, ladder)
    gt_rank, gt_high_bin = canonicalize_ranks(gt_rank_raw, rank_max, ladder, bins)
    fake_rank, fake_high_bin = canonicalize_ranks(fake_ladder_rank, rank_max, ladder, bins)
    return gt_rank, fake_rank, gt_high_bin, fake_high_bin


def train_rank_discriminator(
    past_rank: np.ndarray,
    gt_rank: np.ndarray,
    fake_rank: np.ndarray,
    *,
    device: torch.device,
    config: DiscriminatorConfig,
    seed: int,
    smoke: bool,
) -> tuple[InvertedSliceDiscriminator, dict[str, float], HorizonSliceDataset, dict[str, np.ndarray]]:
    """Train the fixed rank-space discriminator on a canonicalized pair."""
    n_windows = len(past_rank)
    if n_windows == 0 or len(gt_rank) != n_windows or len(fake_rank) != n_windows:
        raise ValueError("past, GT, and fake rank arrays must have the same non-zero window count")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    n_train = max(1, int(0.7 * n_windows))
    n_val = max(1, int(0.15 * n_windows))
    split = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, min(n_windows, n_train + n_val)),
        "test": np.arange(min(n_windows, n_train + n_val), n_windows),
    }
    if len(split["val"]) == 0:
        split["val"] = split["train"].copy()
    if len(split["test"]) == 0:
        split["test"] = split["val"].copy()
    dataset_kwargs = dict(slice_len=config.slice_len, include_past=True, offset_stride=1)
    train_data = HorizonSliceDataset(past_rank, gt_rank, fake_rank, split["train"], seed=seed, **dataset_kwargs)
    val_data = HorizonSliceDataset(past_rank, gt_rank, fake_rank, split["val"], seed=seed + 1, **dataset_kwargs)
    test_data = HorizonSliceDataset(past_rank, gt_rank, fake_rank, split["test"], seed=seed + 2, **dataset_kwargs)
    model = InvertedSliceDiscriminator(
        seq_len=config.lookback + config.slice_len,
        max_offset=config.horizon - config.slice_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        depth=config.depth,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, generator=train_generator,
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    for _ in range(1 if smoke else config.epochs):
        model.train()
        for batch in train_loader:
            x, offsets, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, offsets), labels)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_classifier(model, val_loader, device)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, evaluate_classifier(model, test_loader, device), test_data, split
