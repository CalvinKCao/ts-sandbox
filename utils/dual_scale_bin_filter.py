#!/usr/bin/env python3
"""Dual-scale encode/decode round-trip to match staged binary occupancy lattice."""

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

from models.diffusion_tsf.preprocessing import TimeSeriesTo2D

BinMatchMode = Literal["mmpd", "both", "all"]
BIN_MATCH_CHOICES: Tuple[str, ...] = ("mmpd", "both", "all")


def window_norm_stats(
    past: np.ndarray,
    std_floor: float,
    *,
    center: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    if center == "last":
        ref = past[..., -1:].astype(np.float32)
    elif center == "mean":
        ref = past.mean(axis=-1, keepdims=True).astype(np.float32)
    else:
        raise ValueError(f"window_norm center must be 'mean' or 'last', got {center!r}")
    std = np.maximum(past.std(axis=-1, keepdims=True), std_floor)
    return ref, std.astype(np.float32)


def dual_scale_roundtrip_norm(
    x_norm: torch.Tensor,
    to_2d: TimeSeriesTo2D,
    *,
    decoder: str = "mean",
) -> torch.Tensor:
    coarse, fine = to_2d.encode_dual(x_norm)
    cdf_decoder = "pdf_expectation" if decoder == "pdf_expectation" else decoder
    return to_2d.decode_dual(
        coarse,
        fine,
        cdf_decoder=cdf_decoder,
        squeeze_univariate=False,
    )


def apply_dual_scale_bin_filter(
    horizon: np.ndarray,
    past: np.ndarray,
    *,
    image_height: int,
    max_scale: float,
    std_floor: float,
    decoder: str,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    if horizon.shape != past.shape[:2] + (horizon.shape[-1],):
        raise ValueError(f"horizon/past window mismatch: {horizon.shape} vs {past.shape}")
    if horizon.shape[0] != past.shape[0]:
        raise ValueError(f"horizon/past batch mismatch: {horizon.shape[0]} vs {past.shape[0]}")

    mean, std = window_norm_stats(past, std_floor)
    x_norm = np.clip((horizon - mean) / std, -max_scale, max_scale).astype(np.float32)

    to_2d = TimeSeriesTo2D(height=image_height, max_scale=max_scale).to(device)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            batch = torch.from_numpy(x_norm[start : start + batch_size]).to(device)
            y_norm = dual_scale_roundtrip_norm(batch, to_2d, decoder=decoder)
            chunks.append(y_norm.cpu().numpy())
    y_norm_out = np.concatenate(chunks, axis=0)
    return (y_norm_out * std + mean).astype(np.float32)


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
    image_height: int,
    max_scale: float,
    std_floor: float,
    decoder: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    filter_kwargs = {
        "past": past,
        "image_height": image_height,
        "max_scale": max_scale,
        "std_floor": std_floor,
        "decoder": decoder,
        "device": device,
    }
    y_out = dict(y_true_by_source)
    f_out = dict(fakes)
    for fake_source in y_true_by_source:
        if should_filter_y_true(mode, fake_source):
            y_out[fake_source] = apply_dual_scale_bin_filter(y_true_by_source[fake_source], **filter_kwargs)
        if should_filter_fake(mode, fake_source):
            f_out[fake_source] = apply_dual_scale_bin_filter(fakes[fake_source], **filter_kwargs)
    return y_out, f_out


def run_self_test() -> None:
    rng = np.random.default_rng(0)
    past = rng.normal(size=(4, 3, 32)).astype(np.float32)
    horizon = rng.normal(size=(4, 3, 20)).astype(np.float32)
    device = torch.device("cpu")
    filtered = apply_dual_scale_bin_filter(
        horizon,
        past,
        image_height=16,
        max_scale=3.5,
        std_floor=1e-8,
        decoder="mean",
        device=device,
    )
    again = apply_dual_scale_bin_filter(
        filtered,
        past,
        image_height=16,
        max_scale=3.5,
        std_floor=1e-8,
        decoder="mean",
        device=device,
    )
    err = float(np.max(np.abs(filtered - again)))
    if err > 1e-5:
        raise AssertionError(f"dual-scale round-trip not idempotent: max err={err}")
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
