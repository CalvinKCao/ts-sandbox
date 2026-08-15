#!/usr/bin/env python3
"""Verify ETTh1 per-refine-crop flatline undersample for the refine-only leaf.

Classifies each unique absolute patch crop × active variate with the same defs
as training (true-flat predicate restricted to the patch_width span), then
applies the seeded keep (100% wiggle crops + keep_frac flat crops). Prints
per-variate and total counts; fail-fast if flat keep rate is far from config
or any wiggle crop was dropped.

Examples:
  source .venv/bin/activate
  python temp/scripts/verify_etth1_flat_undersample.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CFG = (
    "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample.yaml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "temp"
        / "lean_disc_c128_results"
        / "etth1_flat_undersample_verify.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from models.diffusion_tsf.flatline_windows import (
        classify_timeseries_flatline_windows,
        classify_unique_segment_flatline_crops,
        select_flatline_undersample_pairs,
        undersample_flatline_refine_crops,
    )
    from models.diffusion_tsf.patch_refine_segments import iter_unique_segment_starts
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.train_multivariate_pipeline import (
        load_dataset,
        resolve_pipeline_data_subset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    import numpy as np

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": args.dataset})
    state = PipelineState.from_config(cfg)
    resolve_pipeline_data_subset(state)

    keep_frac = float(state.patch_refine_flatline_keep_frac)
    seed = (
        int(state.patch_refine_flatline_seed)
        if state.patch_refine_flatline_seed is not None
        else int(state.seed) + 91
    )
    max_scale = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = int(subset_meta.get("test_stride", 1))
    patch_width = int(state.patch_refine_patch_width)
    seg_stride = max(1, int(train_stride))

    train_ds, _, _, _ = load_dataset(
        state.dataset,
        state.variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    data = train_ds.data
    data_np = (
        data.detach().cpu().numpy()
        if hasattr(data, "detach")
        else np.asarray(data, dtype=np.float32)
    )
    overlap = int(getattr(train_ds, "lookback_overlap", state.lookback_overlap))
    segment_starts = iter_unique_segment_starts(
        int(data_np.shape[0]),
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        overlap=overlap,
        patch_width=patch_width,
        segment_stride=seg_stride,
    )
    if args.max_segments is not None:
        segment_starts = segment_starts[: int(args.max_segments)]

    print(
        f"classify n_seg={len(segment_starts)} pw={patch_width} "
        f"seg_stride={seg_stride} lb={state.lookback_length} "
        f"hz={state.forecast_length} ms={max_scale} "
        f"Hc={state.coarse_image_height} keep_frac={keep_frac} seed={seed}"
    )
    print(
        f"crop geometry: {patch_width} timesteps = {patch_width} coarse bins "
        f"(col_stride={state.patch_refine_col_stride} for AR primaries; "
        f"unique-seg train indexes absolute starts at segment_stride)"
    )

    flat_mask = classify_unique_segment_flatline_crops(
        data_np,
        segment_starts,
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        overlap=overlap,
        patch_width=patch_width,
        max_scale=max_scale,
        coarse_h=int(state.coarse_image_height),
        std_floor=float(state.window_norm_std_floor),
        min_run=int(state.patch_refine_flatline_min_run),
        flat_eps_frac=float(state.patch_refine_flatline_eps_frac),
    )
    if flat_mask.ndim != 2:
        raise SystemExit(f"FAIL: expected (N,V) flat mask, got {flat_mask.shape}")

    # Legacy wrong baseline: full-horizon (parent window, var) OR-flat.
    parent_flat = classify_timeseries_flatline_windows(
        train_ds,
        max_scale=max_scale,
        coarse_h=int(state.coarse_image_height),
        std_floor=float(state.window_norm_std_floor),
        forecast_length=int(state.forecast_length),
        lookback_overlap=overlap,
        min_run=int(state.patch_refine_flatline_min_run),
        flat_eps_frac=float(state.patch_refine_flatline_eps_frac),
    )
    n_parent = int(parent_flat.shape[0])
    n_parent_flat_pairs = int(parent_flat.sum())
    n_parent_wiggle_pairs = int((~parent_flat).sum())

    kept_pairs = select_flatline_undersample_pairs(
        flat_mask, keep_frac=keep_frac, seed=seed
    )
    kept_set = {(int(i), int(v)) for i, v in kept_pairs}
    n_seg = int(flat_mask.shape[0])
    n_flat = int(flat_mask.sum())
    n_wiggle = int((~flat_mask).sum())
    n_flat_kept = sum(1 for i, v in kept_pairs if flat_mask[i, v])
    n_wiggle_kept = sum(1 for i, v in kept_pairs if not flat_mask[i, v])
    rate = (n_flat_kept / n_flat) if n_flat else float("nan")

    per_var = []
    print("\nper-active-variate refine crops (unique abs segment × active var):")
    print(
        f"{'var':>4} {'flat':>8} {'wiggle':>8} {'kept_flat':>10} "
        f"{'kept_wig':>10} {'kept':>8} {'flat_rate':>10} {'pct_kept':>9}"
    )
    for vi in range(int(flat_mask.shape[1])):
        n_f = int(flat_mask[:, vi].sum())
        n_w = int((~flat_mask[:, vi]).sum())
        n_fk = sum(1 for i in range(n_seg) if flat_mask[i, vi] and (i, vi) in kept_set)
        n_wk = sum(
            1 for i in range(n_seg) if (not flat_mask[i, vi]) and (i, vi) in kept_set
        )
        n_k = n_fk + n_wk
        fr = (n_fk / n_f) if n_f else float("nan")
        pct = (100.0 * n_k / n_seg) if n_seg else float("nan")
        row = {
            "variate": vi,
            "n_flatline": n_f,
            "n_wiggle": n_w,
            "n_flatline_kept": n_fk,
            "n_wiggle_kept": n_wk,
            "n_kept": n_k,
            "flat_keep_rate": fr,
            "pct_kept": pct,
        }
        per_var.append(row)
        print(
            f"{vi:4d} {n_f:8d} {n_w:8d} {n_fk:10d} {n_wk:10d} "
            f"{n_k:8d} {fr:10.3f} {pct:8.1f}%"
        )

    # Sanity: full undersample API returns the same allow-set size.
    if args.max_segments is None:
        allowed, api_stats = undersample_flatline_refine_crops(
            train_ds,
            patch_width=patch_width,
            segment_stride=seg_stride,
            max_scale=max_scale,
            coarse_h=int(state.coarse_image_height),
            std_floor=float(state.window_norm_std_floor),
            keep_frac=keep_frac,
            seed=seed,
            min_run=int(state.patch_refine_flatline_min_run),
            flat_eps_frac=float(state.patch_refine_flatline_eps_frac),
        )
        n_api = sum(len(vs) for vs in allowed.values())
        if n_api != len(kept_pairs):
            raise SystemExit(
                f"FAIL: API kept_crops={n_api} != verify kept={len(kept_pairs)}"
            )
        if int(api_stats["n_kept_crops"]) != len(kept_pairs):
            raise SystemExit("FAIL: api_stats n_kept_crops mismatch")

    out = {
        "config": args.config,
        "dataset": args.dataset,
        "semantics": "per_refine_crop",
        "patch_width": patch_width,
        "segment_stride": seg_stride,
        "col_stride": int(state.patch_refine_col_stride),
        "n_segments": n_seg,
        "n_variates": int(flat_mask.shape[1]),
        "n_crops": int(flat_mask.size),
        "n_flatline": n_flat,
        "n_wiggle": n_wiggle,
        "n_flatline_kept": n_flat_kept,
        "n_wiggle_kept": n_wiggle_kept,
        "n_kept_crops": len(kept_pairs),
        "keep_frac": keep_frac,
        "seed": seed,
        "flat_keep_rate": rate,
        "legacy_parent_windows": n_parent,
        "legacy_parent_flat_pairs": n_parent_flat_pairs,
        "legacy_parent_wiggle_pairs": n_parent_wiggle_pairs,
        "per_variate": per_var,
        "lookback": int(state.lookback_length),
        "forecast": int(state.forecast_length),
        "max_scale": max_scale,
        "coarse_h": int(state.coarse_image_height),
        "std_floor": float(state.window_norm_std_floor),
        "series_len": int(data_np.shape[0]),
    }
    print("\ntotals:")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n_segments",
                    "n_variates",
                    "n_crops",
                    "n_flatline",
                    "n_wiggle",
                    "n_flatline_kept",
                    "n_wiggle_kept",
                    "n_kept_crops",
                    "flat_keep_rate",
                    "legacy_parent_windows",
                    "legacy_parent_flat_pairs",
                    "legacy_parent_wiggle_pairs",
                    "patch_width",
                    "segment_stride",
                    "keep_frac",
                    "seed",
                )
            },
            indent=2,
        )
    )
    parent_pairs = n_parent * int(flat_mask.shape[1])
    print(
        f"\nnote: legacy parent×var pairs={parent_pairs} "
        f"({n_parent}×{flat_mask.shape[1]}); crop pairs={int(flat_mask.size)} "
        f"({n_seg} segments × {flat_mask.shape[1]}). "
        f"crop/parent ratio ≈ {int(flat_mask.size) / max(1, parent_pairs):.2f}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")

    if n_wiggle_kept != n_wiggle:
        raise SystemExit(
            f"FAIL: expected all wiggle crops kept, got {n_wiggle_kept}/{n_wiggle}"
        )
    if n_flat > 0 and abs(rate - keep_frac) > 0.05:
        raise SystemExit(
            f"FAIL: flat keep rate {rate:.3f} far from keep_frac={keep_frac}"
        )
    for row in per_var:
        if row["n_wiggle_kept"] != row["n_wiggle"]:
            raise SystemExit(
                f"FAIL: var={row['variate']} dropped wiggles "
                f"{row['n_wiggle_kept']}/{row['n_wiggle']}"
            )
    if int(flat_mask.size) <= parent_pairs:
        print(
            "NOTE: crop count is not >> parent×var; check segment enumeration "
            f"(n_seg={n_seg}, n_parent={n_parent})."
        )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
