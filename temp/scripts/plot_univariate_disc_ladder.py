#!/usr/bin/env python3
"""Plot canonicalized forecasts with their univariate discriminator scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dual_scale_bin_filter import binary_dual_decode_levels_dataset_z
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    build_raw_bundle,
    load_ordinal_ladder_for_run,
    parse_args,
)


def _score_path(output_dir: Path, dataset: str, source: str, slice_len: int) -> Path:
    return output_dir / "scores" / f"{dataset}_{source}_L{slice_len}_test_scores.npz"


def _load_scores(output_dir: Path, dataset: str, source: str, slice_len: int) -> dict[str, np.ndarray]:
    path = _score_path(output_dir, dataset, source, slice_len)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing classifier scores {path}; run the discriminator with --save-classification-scores"
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _fake_score(scores: dict[str, np.ndarray], window: int, variate: int) -> tuple[float, float, int]:
    mask = (
        (scores["label"] == 1.0)
        & (scores["window"] == int(window))
        & (scores["variate"] == int(variate))
    )
    if not np.any(mask):
        return float("nan"), float("nan"), 0
    probs = scores["prob_fake"][mask]
    return float(probs.mean()), float((probs >= 0.5).mean()), int(probs.size)


def _shared_scored_pairs(*score_sets: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    pairs = []
    for scores in score_sets:
        mask = scores["label"] == 1.0
        pairs.append(
            set(zip(scores["window"][mask].astype(int), scores["variate"][mask].astype(int)))
        )
    return sorted(set.intersection(*pairs)) if pairs else []


def _binary_256_levels(
    past: np.ndarray,
    *,
    ladder: object,
    coarse_height: int,
    fine_height: int,
    variate: int,
    device: torch.device,
) -> np.ndarray:
    """The exact binary dual-scale decode levels in this window's dataset-z space."""
    decoded = binary_dual_decode_levels_dataset_z(
        past,
        ladder=ladder,
        coarse_height=coarse_height,
        fine_height=fine_height,
        device=device,
    )
    return decoded[0, variate]


def plot_dataset(args: argparse.Namespace, dataset: str, *, slice_len: int, n_windows: int, variate: int) -> Path:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    bundle = build_raw_bundle(args, dataset, device)
    ladder = load_ordinal_ladder_for_run(args, bundle.run)
    binary_scores = _load_scores(args.output_dir, dataset, "binary_staged", slice_len)
    mmpd_scores = _load_scores(args.output_dir, dataset, "mmpd", slice_len)
    pairs = _shared_scored_pairs(binary_scores, mmpd_scores)
    if variate >= 0:
        pairs = [pair for pair in pairs if pair[1] == variate]
    pairs = pairs[:n_windows]
    if not pairs:
        raise RuntimeError(f"{dataset}: no common scored fake windows for binary/MMPD")
    if variate >= bundle.past.shape[1]:
        raise ValueError(f"{dataset}: variate {variate} outside [0,{bundle.past.shape[1] - 1}]")

    fig, axes = plt.subplots(len(pairs), 1, figsize=(13, 3.5 * len(pairs)), squeeze=False)
    horizon_x = np.arange(bundle.fakes["mmpd"].shape[-1])
    past_x = np.arange(-bundle.past.shape[-1], 0)
    for ax, (window, local_variate) in zip(axes[:, 0], pairs):
        p_bin, r_bin, n_bin = _fake_score(binary_scores, window, local_variate)
        p_mmpd, r_mmpd, n_mmpd = _fake_score(mmpd_scores, window, local_variate)
        levels = _binary_256_levels(
            bundle.past[window : window + 1],
            ladder=ladder,
            coarse_height=int(args.bin_coarse_height or args.bin_image_height),
            fine_height=int(args.bin_fine_height or args.bin_image_height),
            variate=local_variate,
            device=device,
        )
        ax.hlines(
            levels,
            -bundle.past.shape[-1],
            bundle.fakes["mmpd"].shape[-1] - 1,
            color="0.55",
            lw=0.25,
            alpha=0.16,
            zorder=0,
        )
        ax.plot(past_x, bundle.past[window, local_variate], color="0.55", lw=1.2, label="lookback")
        ax.plot(horizon_x, bundle.y_true_by_source["binary_staged"][window, local_variate], color="black", lw=1.4, label="GT")
        ax.plot(horizon_x, bundle.fakes["binary_staged"][window, local_variate], color="#1f77b4", lw=1.1, label="binary")
        ax.plot(horizon_x, bundle.fakes["mmpd"][window, local_variate], color="#ff7f0e", lw=1.1, label="MMPD")
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_title(
            f"window={window}, variate={local_variate} | "
            f"256 binary bin paths ({len(np.unique(levels))} unique dataset-z rungs) | "
            f"binary P(fake)={p_bin:.3f}, patch-positive={r_bin:.2%} (n={n_bin}) | "
            f"MMPD P(fake)={p_mmpd:.3f}, patch-positive={r_mmpd:.2%} (n={n_mmpd})"
        )
        ax.set_ylabel("binary dataset-z")
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time step (forecast starts at 0)")
    axes[0, 0].legend(loc="upper left", ncol=4)
    fig.tight_layout()
    out_dir = args.output_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_univariate_disc_ladder.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plot-slice-len", type=int, default=16)
    parser.add_argument("--plot-windows", type=int, default=2)
    parser.add_argument("--plot-variate", type=int, default=-1,
                        help="Optional global variate filter; default picks scored (window,variate) pairs.")
    known, remaining = parser.parse_known_args()
    saved = sys.argv
    sys.argv = [saved[0], *remaining]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    for dataset in args.datasets:
        plot_dataset(
            args,
            dataset,
            slice_len=known.plot_slice_len,
            n_windows=known.plot_windows,
            variate=known.plot_variate,
        )


if __name__ == "__main__":
    main()
