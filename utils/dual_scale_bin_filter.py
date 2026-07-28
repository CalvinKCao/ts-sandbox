#!/usr/bin/env python3
"""Match GT/MMPD/binary horizons onto binary's ordinal → coarse/fine lattice.

Binary ordinal_norm training:
  1. series already in train-set z-score space
  2. map to global ordinal ranks (with optional OOD constant shift)
  3. subsample by representation_time_stride
  4. encode_dual_heights_bounded on [0, rank_max] per variate
  5. decode, linear-upsample to raw horizon, ordinal → z

Discriminator bin-match applies that same round-trip to whichever sources
``mode`` selects. No per-window instance norm and no ±max_scale clip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.nn.functional as F

from models.diffusion_tsf.ordinal_window_norm import (
    OrdinalLadder,
    build_global_ladder_from_training,
    ordinal_decode,
    ordinal_encode,
)
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D

BinMatchMode = Literal["mmpd", "both", "all"]
BIN_MATCH_CHOICES: Tuple[str, ...] = ("mmpd", "both", "all")


def align_mmpd_to_binary_dataset_norm(
    *,
    binary_y_true: np.ndarray,
    mmpd_y_true: np.ndarray,
    mmpd_fakes: np.ndarray,
    mmpd_mean: np.ndarray,
    mmpd_std: np.ndarray,
    binary_mean: np.ndarray,
    binary_std: np.ndarray,
    max_rmse: float = 2e-5,
    max_abs_error: float = 2e-4,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Map MMPD's dataset z-score convention into binary's convention.

    MMPD and binary packs contain the same raw target windows, but their
    train-split scalers can differ slightly.  Derive the affine map from those
    saved training scalers -- never from evaluation targets -- then validate
    it against the independently produced paired targets.  The caller keeps
    ``binary_y_true`` as the canonical GT tensor only after that validation.
    """
    binary = np.asarray(binary_y_true, dtype=np.float32)
    mmpd_true = np.asarray(mmpd_y_true, dtype=np.float32)
    mmpd_pred = np.asarray(mmpd_fakes, dtype=np.float32)
    if binary.shape != mmpd_true.shape or binary.shape != mmpd_pred.shape:
        raise ValueError(
            "binary/MMPD coordinate alignment requires identical (N,V,T) shapes, got "
            f"binary={binary.shape} mmpd_y_true={mmpd_true.shape} mmpd_fakes={mmpd_pred.shape}"
        )
    if binary.ndim != 3:
        raise ValueError(f"expected (N,V,T) packs, got {binary.shape}")
    if not (np.isfinite(binary).all() and np.isfinite(mmpd_true).all() and np.isfinite(mmpd_pred).all()):
        raise ValueError("cannot align non-finite MMPD/binary coordinate packs")

    n_vars = binary.shape[1]
    def _scaler(name: str, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != n_vars or not np.isfinite(flat).all():
            raise ValueError(f"{name} must contain {n_vars} finite variate values, got {flat.shape}")
        return flat

    src_mean = _scaler("mmpd_mean", mmpd_mean)
    src_std = _scaler("mmpd_std", mmpd_std)
    dst_mean = _scaler("binary_mean", binary_mean)
    dst_std = _scaler("binary_std", binary_std)
    if np.any(src_std <= 0.0) or np.any(dst_std <= 0.0):
        raise ValueError("MMPD/binary training scaler standard deviations must be positive")
    scale = src_std / dst_std
    offset = (src_mean - dst_mean) / dst_std
    aligned_true = mmpd_true * scale[None, :, None] + offset[None, :, None]
    residual = aligned_true - binary
    rmse = np.sqrt(np.mean(residual**2, axis=(0, 2)))
    max_abs = np.max(np.abs(residual), axis=(0, 2))
    if float(rmse.max()) > float(max_rmse) or float(max_abs.max()) > float(max_abs_error):
        raise ValueError(
            "MMPD and binary targets are not related by a dataset-normalization affine map: "
            f"max_rmse={float(rmse.max()):.3e} max_abs={float(max_abs.max()):.3e}. "
            "Refuse to compare predictions from non-corresponding windows or an instance-normalized pack."
        )
    aligned_pred = mmpd_pred * scale[None, :, None] + offset[None, :, None]
    return aligned_pred.astype(np.float32), {
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "offset_min": float(offset.min()),
        "offset_max": float(offset.max()),
        "target_rmse_max": float(rmse.max()),
        "target_max_abs": float(max_abs.max()),
    }


def assert_on_binary_dual_ordinal_lattice(
    horizon: np.ndarray,
    past: np.ndarray,
    *,
    ladder: OrdinalLadder,
    coarse_height: int,
    fine_height: int,
    device: torch.device,
    repr_time_stride: int = 1,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Assert values decode from one of binary's Hc×Hf ordinal bins.

    This is stricter than checking membership in the full training ladder:
    each (window, variate) forecast value must correspond to one of the at
    most ``coarse_height * fine_height`` decoded ordinal bins.  Interpolation
    at representation stride >1 creates non-bin values, so this assertion is
    intentionally unavailable for that representation.
    """
    if int(repr_time_stride) != 1:
        raise ValueError(
            "exact Hc×Hf ordinal-lattice validation requires representation_time_stride=1; "
            f"got {repr_time_stride}"
        )
    values = np.asarray(horizon, dtype=np.float32)
    if values.ndim != 3 or values.shape[:2] != past.shape[:2]:
        raise ValueError(f"expected matching (N,V,T) horizon/past, got {values.shape}/{past.shape}")
    n_bins = int(coarse_height) * int(fine_height)
    if n_bins <= 0:
        raise ValueError(f"invalid dual-scale bin count {coarse_height}×{fine_height}")
    total = 0
    invalid = 0
    max_delta = 0.0
    max_unique = 0
    for start in range(0, values.shape[0], max(1, int(batch_size))):
        end = min(values.shape[0], start + max(1, int(batch_size)))
        legal = binary_dual_decode_levels_dataset_z(
            past[start:end],
            ladder=ladder,
            coarse_height=coarse_height,
            fine_height=fine_height,
            device=device,
        )
        chunk = values[start:end]
        # Compare to decoded dataset-z values directly.  Re-encoding an
        # already decoded level can choose a different ordinal rank when the
        # global ladder has uneven gaps, which is exactly the false positive
        # this assertion is intended to avoid.
        delta = np.min(np.abs(chunk[..., None] - legal[:, :, None, :]), axis=-1)
        valid = delta <= 1e-6
        invalid += int((~valid).sum())
        total += int(valid.size)
        max_delta = max(max_delta, float(delta.max(initial=0.0)))
        max_unique = max(
            max_unique,
            max(int(np.unique(legal[bi, vi]).size) for bi in range(legal.shape[0]) for vi in range(legal.shape[1])),
        )
    if invalid:
        raise AssertionError(
            f"{invalid}/{total} values are outside binary's {n_bins}-bin ordinal decode lattice"
        )
    return {
        "n_bins": float(n_bins),
        "n_values": float(total),
        "max_unique_per_chunk_variate": float(max_unique),
        "max_decode_delta": float(max_delta),
    }


def _resample_1d_time_series(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear resample trailing time axis (same as DiffusionTSF._resample_1d_time_series)."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] == 1:
        return x.expand(*x.shape[:-1], target_len)
    flat = x.reshape(-1, 1, x.shape[-1])
    out = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
    return out.reshape(*x.shape[:-1], target_len)


def dual_scale_ordinal_roundtrip(
    x_ranks: torch.Tensor,
    to_2d: TimeSeriesTo2D,
    *,
    coarse_height: int,
    fine_height: int,
    rank_max: torch.Tensor,
    decoder: str = "mean",
) -> torch.Tensor:
    """Round-trip integer ranks through binary's bounded coarse/fine bins."""
    coarse, fine = to_2d.encode_dual_heights_bounded(
        x_ranks,
        coarse_height=coarse_height,
        fine_height=fine_height,
        value_min=0.0,
        value_max_per_variate=rank_max,
    )
    cdf_decoder = "pdf_expectation" if decoder == "pdf_expectation" else decoder
    return to_2d.decode_dual_heights_bounded(
        coarse,
        fine,
        value_min=0.0,
        value_max_per_variate=rank_max,
        cdf_decoder=cdf_decoder,
        squeeze_univariate=False,
    )


def binary_dual_decode_levels_dataset_z(
    past: np.ndarray,
    *,
    ladder: OrdinalLadder,
    coarse_height: int,
    fine_height: int,
    device: torch.device,
) -> np.ndarray:
    """Decode every Hc×Hf binary ordinal bin into final dataset-z space.

    This deliberately runs rank centers through the same bounded dual-CDF
    encode/decode and ordinal denormalization used by the discriminator
    canonicalization.  The result is window-specific only when the causal OOD
    envelope introduces a shift.
    """
    past_np = np.asarray(past, dtype=np.float32)
    if past_np.ndim != 3:
        raise ValueError(f"expected past (N,V,L), got {past_np.shape}")
    n_bins = int(coarse_height) * int(fine_height)
    if n_bins <= 0:
        raise ValueError(f"invalid dual-scale bin count {coarse_height}×{fine_height}")
    with torch.no_grad():
        past_t = torch.from_numpy(past_np).to(device)
        past_ord, _future_ord, ladder_b, ood_shift = ordinal_encode(
            past_t,
            None,
            ladder=ladder,
            apply_ood_shift=True,
            causal_only=True,
        )
        rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=torch.float32)
        centers = (
            (torch.arange(n_bins, device=device, dtype=torch.float32) + 0.5)
            / float(n_bins)
        )
        rank_centers = centers.view(1, 1, n_bins) * rank_max.view(1, -1, 1)
        rank_centers = rank_centers.expand(past_t.shape[0], -1, -1)
        to_2d = TimeSeriesTo2D(height=int(coarse_height), max_scale=1.0).to(device)
        decoded_ranks = dual_scale_ordinal_roundtrip(
            rank_centers,
            to_2d,
            coarse_height=int(coarse_height),
            fine_height=int(fine_height),
            rank_max=rank_max,
            decoder="mean",
        )
        _past_z, decoded_z = ordinal_decode(
            past_ord[..., :1], decoded_ranks, ladder_b, ood_shift=ood_shift,
        )
        assert decoded_z is not None
        return decoded_z.detach().cpu().numpy().astype(np.float32)


def apply_dual_scale_bin_filter(
    horizon: np.ndarray,
    past: np.ndarray,
    *,
    ladder: OrdinalLadder,
    coarse_height: int,
    fine_height: int,
    decoder: str,
    device: torch.device,
    batch_size: int = 64,
    apply_ood_shift: bool = True,
    margin_frac: float = 0.05,
    repr_time_stride: int = 1,
) -> np.ndarray:
    """Snap a horizon onto binary's ordinal+bounded dual-scale lattice.

    ``past`` / ``horizon`` must already be in the same train-set z-score space
    binary used. Path mirrors staged ordinal encode:
      ordinal_encode (+ causal OOD shift from past)
      → subsample ``::repr_time_stride`` (binary representation_time_stride)
      → encode/decode_dual_heights_bounded on [0, rank_max]
      → linear upsample back to raw horizon length
      → ordinal_decode (undo OOD shift)

    No per-window instance norm and no ±max_scale clip.
    """
    if horizon.shape != past.shape[:2] + (horizon.shape[-1],):
        raise ValueError(f"horizon/past window mismatch: {horizon.shape} vs {past.shape}")
    if horizon.shape[0] != past.shape[0]:
        raise ValueError(f"horizon/past batch mismatch: {horizon.shape[0]} vs {past.shape[0]}")
    if int(ladder.values.shape[1]) != int(horizon.shape[1]):
        raise ValueError(
            f"ladder variates {int(ladder.values.shape[1])} != horizon variates {horizon.shape[1]}"
        )
    stride = max(1, int(repr_time_stride))

    # max_scale unused for bounded encode/decode; keep ctor happy.
    to_2d = TimeSeriesTo2D(height=int(coarse_height), max_scale=1.0).to(device)
    rank_max = ladder.rank_max_per_variate().to(device=device, dtype=torch.float32)
    raw_len = int(horizon.shape[-1])
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, horizon.shape[0], batch_size):
            end = start + batch_size
            past_t = torch.from_numpy(past[start:end].astype(np.float32, copy=False)).to(device)
            fut_t = torch.from_numpy(horizon[start:end].astype(np.float32, copy=False)).to(device)
            past_ord, fut_ord, ladder_b, ood_shift = ordinal_encode(
                past_t,
                fut_t,
                ladder=ladder,
                apply_ood_shift=apply_ood_shift,
                margin_frac=margin_frac,
                causal_only=True,
            )
            assert fut_ord is not None
            fut_repr = fut_ord[..., ::stride] if stride > 1 else fut_ord
            fut_ord_rt = dual_scale_ordinal_roundtrip(
                fut_repr,
                to_2d,
                coarse_height=int(coarse_height),
                fine_height=int(fine_height),
                rank_max=rank_max,
                decoder=decoder,
            )
            if stride > 1:
                fut_ord_rt = _resample_1d_time_series(fut_ord_rt, raw_len)
            _past_z, fut_z = ordinal_decode(
                past_ord,
                fut_ord_rt,
                ladder_b,
                ood_shift=ood_shift,
            )
            assert fut_z is not None
            chunks.append(fut_z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def should_filter_y_true(mode: str, fake_source: str) -> bool:
    return mode == "all"


def should_filter_fake(mode: str, fake_source: str) -> bool:
    if mode in ("both", "all"):
        return True
    if mode == "mmpd":
        return fake_source == "mmpd"
    return False


def apply_bin_match_to_bundle(
    *,
    mode: str,
    past: np.ndarray,
    y_true_by_source: dict[str, np.ndarray],
    fakes: dict[str, np.ndarray],
    ladder: OrdinalLadder,
    coarse_height: int,
    fine_height: int,
    decoder: str,
    device: torch.device,
    apply_ood_shift: bool = True,
    margin_frac: float = 0.05,
    repr_time_stride: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    filter_kwargs = {
        "past": past,
        "ladder": ladder,
        "coarse_height": coarse_height,
        "fine_height": fine_height,
        "decoder": decoder,
        "device": device,
        "apply_ood_shift": apply_ood_shift,
        "margin_frac": margin_frac,
        "repr_time_stride": repr_time_stride,
    }
    y_out = dict(y_true_by_source)
    f_out = dict(fakes)
    for fake_source in y_true_by_source:
        if should_filter_y_true(mode, fake_source):
            y_out[fake_source] = apply_dual_scale_bin_filter(
                y_true_by_source[fake_source], **filter_kwargs
            )
        if should_filter_fake(mode, fake_source):
            f_out[fake_source] = apply_dual_scale_bin_filter(fakes[fake_source], **filter_kwargs)
    return y_out, f_out


def run_self_test() -> None:
    rng = np.random.default_rng(0)
    # Synthetic train ladder in z-score space (unique ranks per variate).
    train = np.stack(
        [
            np.linspace(-1.0, 1.0, 64) ** 3 * 2.0,
            np.exp(np.linspace(-1.5, 1.0, 64)) - 1.0,
            np.sign(np.linspace(-1.0, 1.0, 64)) * np.linspace(-1.0, 1.0, 64) ** 2,
        ],
        axis=1,
    ).astype(np.float32)
    ladder = build_global_ladder_from_training(train, tie_atol=1e-6)
    past = rng.normal(size=(4, 3, 32)).astype(np.float32)
    # Keep past inside train envelope so OOD shift stays zero for idempotence check.
    past = np.clip(past, -0.4, 0.4)
    horizon = rng.normal(size=(4, 3, 20)).astype(np.float32) * 0.3
    device = torch.device("cpu")
    filtered = apply_dual_scale_bin_filter(
        horizon,
        past,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        decoder="mean",
        device=device,
        apply_ood_shift=True,
        repr_time_stride=1,
    )
    again = apply_dual_scale_bin_filter(
        filtered,
        past,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        decoder="mean",
        device=device,
        apply_ood_shift=True,
        repr_time_stride=1,
    )
    err = float(np.max(np.abs(filtered - again)))
    if err > 1e-5:
        raise AssertionError(f"ordinal dual-scale round-trip not idempotent: max err={err}")
    lattice_stats = assert_on_binary_dual_ordinal_lattice(
        filtered,
        past,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        device=device,
    )
    decoded_levels = binary_dual_decode_levels_dataset_z(
        past[:1],
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        device=device,
    )
    if decoded_levels.shape != (1, 3, 256):
        raise AssertionError(f"256-bin decode shape mismatch: {decoded_levels.shape}")
    if not np.isfinite(decoded_levels).all():
        raise AssertionError("256-bin ordinal decode produced non-finite dataset-z values")

    # MMPD's saved outputs are train-set z-scores under its own scaler.  Use
    # known train scalers, not target regression, to recover binary dataset-z.
    mmpd_mean = np.asarray([11.0, -2.0, 4.0], dtype=np.float32)
    mmpd_std = np.asarray([2.8, 1.4, 3.3], dtype=np.float32)
    binary_mean = np.asarray([10.6, -3.0, 4.3], dtype=np.float32)
    binary_std = np.asarray([2.0, 2.0, 3.0], dtype=np.float32)
    mmpd_scale = mmpd_std / binary_std
    mmpd_offset = (mmpd_mean - binary_mean) / binary_std
    binary_target = horizon * mmpd_scale[None, :, None] + mmpd_offset[None, :, None]
    mmpd_fake = (horizon + 0.15).astype(np.float32)
    aligned_fake, align_stats = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_target,
        mmpd_y_true=horizon,
        mmpd_fakes=mmpd_fake,
        mmpd_mean=mmpd_mean,
        mmpd_std=mmpd_std,
        binary_mean=binary_mean,
        binary_std=binary_std,
    )
    expected_fake = mmpd_fake * mmpd_scale[None, :, None] + mmpd_offset[None, :, None]
    if not np.allclose(aligned_fake, expected_fake.astype(np.float32), rtol=0.0, atol=2e-6):
        raise AssertionError("MMPD→binary dataset-z affine mapping changed a known fake incorrectly")

    # Stride-2 path: upsample is lossy, so only check finiteness + shape.
    filtered_s2 = apply_dual_scale_bin_filter(
        horizon,
        past,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        decoder="mean",
        device=device,
        apply_ood_shift=True,
        repr_time_stride=2,
    )
    if filtered_s2.shape != horizon.shape:
        raise AssertionError(f"stride-2 shape mismatch: {filtered_s2.shape} vs {horizon.shape}")
    if not np.isfinite(filtered_s2).all():
        raise AssertionError("stride-2 path produced non-finite values")

    # OOD past should get a constant shift rather than hard-clipping the horizon.
    past_ood = past.copy()
    past_ood[:, 0, :] += 10.0
    horizon_ood = horizon.copy()
    horizon_ood[:, 0, :] += 10.0
    out_ood = apply_dual_scale_bin_filter(
        horizon_ood,
        past_ood,
        ladder=ladder,
        coarse_height=16,
        fine_height=16,
        decoder="mean",
        device=device,
        apply_ood_shift=True,
        repr_time_stride=2,
    )
    if not np.isfinite(out_ood).all():
        raise AssertionError("OOD shift path produced non-finite values")
    print(
        "dual_scale_bin_filter self-test ok "
        f"(max idempotence err={err:.2e}, bins={int(lattice_stats['n_bins'])}, "
        f"decoded_levels={decoded_levels.shape[-1]}, align_rmse={align_stats['target_rmse_max']:.2e})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.self_test:
        raise SystemExit("Pass --self-test")
    run_self_test()


if __name__ == "__main__":
    main()
