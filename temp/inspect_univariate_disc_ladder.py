#!/usr/bin/env python3
"""Validate the canonicalized packs consumed by the univariate discriminator.

Invoke with the same arguments as
``utils/eval_discriminator_binary_vs_mmpd_univariate.py``.  It materializes
the raw binary/MMPD packs when needed, applies the selected shared
canonicalization, then fails if shapes, GT coordinates, finiteness, or the
ordinal ladder do not agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dual_scale_bin_filter import (
    assert_on_binary_dual_ordinal_lattice,
    binary_dual_decode_levels_dataset_z,
)
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    build_raw_bundle,
    load_ordinal_ladder_for_run,
    parse_args,
    zscore_time,
)


def _summary(name: str, values: np.ndarray) -> None:
    normalized = zscore_time(values)
    flat_fraction = float((values.std(axis=-1) < 1e-5).mean())
    print(
        f"[stats] {name}: shape={tuple(values.shape)} "
        f"raw_mean={values.mean():.6f} raw_std={values.std():.6f} "
        f"disc_mean={normalized.mean():.6f} disc_std={normalized.std():.6f} "
        f"flat_fraction={flat_fraction:.6f}",
        flush=True,
    )


def _decoded_256_levels(
    past: np.ndarray,
    ladder: object,
    *,
    coarse_height: int,
    fine_height: int,
    variate: int,
    device: torch.device,
) -> np.ndarray:
    decoded = binary_dual_decode_levels_dataset_z(
        past,
        ladder=ladder,
        coarse_height=coarse_height,
        fine_height=fine_height,
        device=device,
    )
    # Keep every one of the 16×16 paths: a small ladder can decode adjacent
    # bins to the same value, but the grid still represents all 256 choices.
    return decoded[0, variate]


def _write_ordinal_grid_plot(dataset: str, bundle: object, ladder: object, args: object, device: torch.device) -> None:
    n_windows = min(2, len(bundle.indices))
    variate = 0
    fig, axes = plt.subplots(n_windows, 1, figsize=(13, 3.4 * n_windows), squeeze=False)
    past_x = np.arange(-bundle.past.shape[-1], 0)
    future_x = np.arange(bundle.fakes["mmpd"].shape[-1])
    for local_idx, ax in enumerate(axes[:, 0]):
        levels = _decoded_256_levels(
            bundle.past[local_idx : local_idx + 1],
            ladder,
            coarse_height=int(args.bin_coarse_height or args.bin_image_height),
            fine_height=int(args.bin_fine_height or args.bin_image_height),
            variate=variate,
            device=device,
        )
        ax.hlines(levels, past_x[0], future_x[-1], color="0.55", lw=0.25, alpha=0.16, zorder=0)
        ax.plot(past_x, bundle.past[local_idx, variate], color="0.55", lw=1.0, label="lookback")
        ax.plot(future_x, bundle.y_true_by_source["binary_staged"][local_idx, variate], color="black", lw=1.2, label="GT")
        ax.plot(future_x, bundle.fakes["binary_staged"][local_idx, variate], color="#1f77b4", lw=1.0, label="binary")
        ax.plot(future_x, bundle.fakes["mmpd"][local_idx, variate], color="#ff7f0e", lw=1.0, label="MMPD")
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_title(
            f"{dataset}: canonical window={local_idx}, variate=0; "
            f"faint grid from all 256 binary bin paths ({len(np.unique(levels))} unique dataset-z rungs)"
        )
        ax.set_ylabel("binary dataset-z")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(loc="upper left", ncol=4)
    axes[-1, 0].set_xlabel("time step (forecast starts at 0)")
    fig.tight_layout()
    out_dir = args.output_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset}_ordinal_256_grid_smoke.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[grid] wrote {out}", flush=True)


def main() -> None:
    args = parse_args()
    if set(args.fake_sources) != {"binary_staged", "mmpd"}:
        raise ValueError("pass exactly: --fake-sources binary_staged mmpd")
    if args.bin_match_filter != "all":
        raise ValueError("pass --bin-match-filter all so GT and both fakes share the binary ordinal path")
    if not args.ordinal_ladder_quantize:
        raise ValueError("pass --ordinal-ladder-quantize for exact shared-rung validation")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    for dataset in args.datasets:
        print(f"[{dataset}] materializing and validating canonicalized packs", flush=True)
        bundle = build_raw_bundle(args, dataset, device)
        ladder = load_ordinal_ladder_for_run(args, bundle.run)
        gt_binary = bundle.y_true_by_source["binary_staged"]
        gt_mmpd = bundle.y_true_by_source["mmpd"]
        if not np.array_equal(gt_binary, gt_mmpd):
            max_abs = float(np.max(np.abs(gt_binary - gt_mmpd)))
            raise AssertionError(f"{dataset}: canonicalized binary/MMPD GT differs (max_delta={max_abs})")

        for name, values in (
            ("gt", gt_binary),
            ("binary_staged", bundle.fakes["binary_staged"]),
            ("mmpd", bundle.fakes["mmpd"]),
        ):
            if not np.isfinite(values).all():
                raise AssertionError(f"{dataset}/{name}: non-finite values")
            _summary(f"{dataset}/{name}", values)
            lattice_stats = assert_on_binary_dual_ordinal_lattice(
                values,
                bundle.past,
                ladder=ladder,
                coarse_height=int(args.bin_coarse_height or args.bin_image_height),
                fine_height=int(args.bin_fine_height or args.bin_image_height),
                device=device,
                repr_time_stride=int(getattr(args, "_resolved_bin_repr_time_stride", 1) or 1),
            )
            print(
                f"[dual-lattice] {dataset}/{name}: exact {int(lattice_stats['n_bins'])}-bin "
                f"decode set (values={int(lattice_stats['n_values'])}, "
                f"max_unique={int(lattice_stats['max_unique_per_chunk_variate'])}, "
                f"max_decode_delta={lattice_stats['max_decode_delta']:.1e})",
                flush=True,
            )

        _write_ordinal_grid_plot(dataset, bundle, ladder, args, device)

        print(f"[{dataset}] PASS: same GT coordinates, finite values, and one exact ordinal ladder", flush=True)


if __name__ == "__main__":
    main()
