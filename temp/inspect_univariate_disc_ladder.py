#!/usr/bin/env python3
"""Validate the canonicalized packs consumed by the univariate discriminator.

Invoke with the same arguments as
``utils/eval_discriminator_binary_vs_mmpd_univariate.py``.  It materializes
the raw binary/MMPD packs when needed, applies the selected shared
canonicalization, then fails if shapes, GT coordinates, finiteness, or the
ordinal ladder do not agree.
"""

from __future__ import annotations

import numpy as np
import torch

from utils.binary_disc_debias import quantize_to_ordinal_ladder
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


def _assert_on_ladder(name: str, values: np.ndarray, ladder: object) -> None:
    snapped, _stats = quantize_to_ordinal_ladder(values, ladder)
    max_abs = float(np.max(np.abs(values - snapped)))
    if max_abs != 0.0:
        raise AssertionError(f"{name} is not exactly on the shared ordinal ladder: max_delta={max_abs}")
    print(f"[ladder] {name}: exact (max_delta={max_abs:.1f})", flush=True)


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
            _assert_on_ladder(f"{dataset}/{name}", values, ladder)

        print(f"[{dataset}] PASS: same GT coordinates, finite values, and one exact ordinal ladder", flush=True)


if __name__ == "__main__":
    main()
