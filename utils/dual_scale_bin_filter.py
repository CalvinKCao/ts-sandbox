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
from typing import Literal, Tuple

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
            np.linspace(-2.0, 2.0, 64),
            np.linspace(-1.5, 3.0, 64),
            np.linspace(-0.5, 1.0, 64),
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
    print(f"dual_scale_bin_filter self-test ok (max idempotence err={err:.2e})")


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
