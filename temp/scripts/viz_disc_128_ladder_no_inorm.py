# Pipeline integration: Prefer --viz-sanity pre_post (assert window_norm_grid + canvas 128 on snapped tensors).
#!/usr/bin/env python3
"""Prove GT / binary / MMPD sit on the canvas128 window-norm 128-row lattice.

Uses existing ablation forecast packs (dataset-z past/y_true/samples). Snaps with
``legal_window_norm_patch_refine_levels_dataset_z`` — the same finite H-row grid
canvas128 training emits — NOT the ordinal absolute ladder.

Disc inputs stay in **global dataset-z** (no instance / window z-score of the
series). Ladder rungs are window-specific via past mean/std (training geometry)
but values plotted are still dataset-z.

Output: ``temp/viz_disc_128_ladder_no_inorm/<dataset>/`` + README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.disc_bin_center_shift import bin_center_shift, nearest_bin_indices  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    binary_mmpd_train_scaler_map,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    DEFAULT_MMPD_DATA,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402
from utils.patch_refine_ordinal_ladder import snap_to_patch_refine_levels  # noqa: E402
from utils.patch_refine_value_grid import (  # noqa: E402
    legal_window_norm_patch_refine_levels_dataset_z,
)
from utils.visualize_staged_eval_2d_preds import _build_state  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _flat_mask_from_ckpt,
    _max_scale_from_ckpt_metadata,
    _plot_snap_proof_panel,
    _window_norm_grid_config,
    load_ablation_run,
)

DEFAULT_DATASETS = ("ETTh1", "ETTh2", "electricity", "traffic", "exchange_rate")

# Prefer latest local ablation packs (pre-snap dataset-z forecasts).
DEFAULT_PACK_ROOTS = {
    "ETTh1": REPO_ROOT
    / "results/datasets/08-04-1843-ablation-disc-l8-l16-ETTh1-c128-valtest80-byvar",
    "ETTh2": REPO_ROOT
    / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar",
    "electricity": REPO_ROOT
    / "results/datasets/08-04-1845-ablation-disc-l8-l16-electricity-c128-valtest80-byvar",
    "traffic": REPO_ROOT
    / "results/datasets/08-04-1845-ablation-disc-l8-l16-traffic-c128-valtest80-byvar",
    "exchange_rate": REPO_ROOT
    / "results/datasets/08-04-1545-ablation-disc-l8-l16-exchange_rate-c128-valtest80",
}

DEFAULT_CKPTS = {
    "ETTh1": REPO_ROOT
    / "results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6",
    "ETTh2": REPO_ROOT
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2",
    "electricity": REPO_ROOT
    / "results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity",
    "traffic": REPO_ROOT
    / "results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic",
    "exchange_rate": REPO_ROOT
    / "results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
}

DEFAULT_CFGS = {
    "ETTh1": "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
    "ETTh2": "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml",
    "electricity": "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml",
    "traffic": "configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml",
    "exchange_rate": "configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "temp/viz_disc_128_ladder_no_inorm")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--zoom-steps", type=int, default=16)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument(
        "--write-horizon96",
        action="store_true",
        help="Also write full H=96 GT/binary/MMPD line plots per picked window.",
    )
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument(
        "--pack-root",
        type=Path,
        default=None,
        help="Override pack root (single-dataset runs; requires --datasets LEN=1).",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Override ckpt root (single-dataset runs).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config path (single-dataset runs).",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _find_pack(raw_dir: Path, prefix: str, dataset: str) -> Path:
    hits = sorted(raw_dir.glob(f"{prefix}*{dataset}*.npz"))
    if not hits:
        raise FileNotFoundError(f"no {prefix} pack for {dataset} under {raw_dir}")
    # Prefer val-test tagged packs.
    preferred = [p for p in hits if "val-test" in p.name or "val_test" in p.name]
    return preferred[-1] if preferred else hits[-1]


def _snap_residual(values_1d: np.ndarray, levels_1d: np.ndarray) -> float:
    vals = np.asarray(values_1d, dtype=np.float32)
    lev = np.asarray(levels_1d, dtype=np.float32)
    return float(np.abs(vals[:, None] - lev[None, :]).min(axis=1).max(initial=0.0))


def _run_dataset(
    *,
    dataset: str,
    pack_root: Path,
    ckpt_root: Path,
    config_path: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    raw_dir = pack_root / "raw"
    binary_path = _find_pack(raw_dir, "binary_", dataset)
    mmpd_path = _find_pack(raw_dir, "mmpd_", dataset)
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)

    run, _stages, kind = load_ablation_run(dataset, ckpt_root)
    state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
    if bool(getattr(state, "use_ordinal_window_norm", False)):
        raise RuntimeError(
            f"{dataset}: config is ordinal; this viz is for window-norm canvas128 leaves"
        )
    if not bool(getattr(state, "use_window_normalization", False)):
        raise RuntimeError(f"{dataset}: use_window_normalization must be True")

    canvas_height = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if "canvas_height" in binary_pack:
        canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    if canvas_height != 128:
        raise RuntimeError(f"{dataset}: expected canvas_height=128, got {canvas_height}")

    max_scale = _max_scale_from_ckpt_metadata(ckpt_root, dataset)
    flat_mask = _flat_mask_from_ckpt(ckpt_root, dataset)
    grid_cfg = _window_norm_grid_config(
        state,
        canvas_height=canvas_height,
        max_scale=max_scale,
        skip_window_norm_variate_mask=flat_mask,
    )
    snap_mode = (
        "window_norm_grid_hybrid_flat"
        if flat_mask and any(flat_mask)
        else "window_norm_grid"
    )

    # Align index lists (fail if packs disagree).
    b_idx = np.asarray(binary_pack["indices"], dtype=np.int64)
    m_idx = np.asarray(mmpd_pack["indices"], dtype=np.int64)
    if not np.array_equal(b_idx, m_idx):
        raise RuntimeError(f"{dataset}: binary/MMPD indices differ")

    past = np.asarray(binary_pack["past"], dtype=np.float32)
    gt_raw = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_raw = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    mmpd_raw = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)

    ns = SimpleNamespace(
        lookback=args.lookback,
        horizon=args.horizon,
        pack_test_stride=args.pack_test_stride,
        mmpd_data_dir=args.mmpd_data_dir,
    )
    scalers = binary_mmpd_train_scaler_map(ns, run)
    mmpd_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=gt_raw,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_raw,
        **scalers,
    )

    legal = legal_window_norm_patch_refine_levels_dataset_z(past, grid_cfg)
    gt, gt_st = snap_to_patch_refine_levels(gt_raw, legal)
    binary, bin_st = snap_to_patch_refine_levels(np.asarray(binary_raw, dtype=np.float32), legal)
    mmpd, mmpd_st = snap_to_patch_refine_levels(mmpd_z, legal)

    n = int(gt.shape[0])
    rng = np.random.default_rng(int(args.seed))
    n_pick = min(int(args.n_windows), n)
    picks = np.sort(rng.choice(n, size=n_pick, replace=False)).tolist()
    variate = int(args.variate)
    if variate < 0 or variate >= gt.shape[1]:
        raise ValueError(f"variate={variate} out of range V={gt.shape[1]}")

    colors = {"gt": "#222222", "binary": "#1f77b4", "mmpd": "#d62728"}
    ds_out = out_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    panels: List[str] = []
    residuals: List[Dict[str, float]] = []

    for local_i, pool_i in enumerate(picks):
        levels_1d = legal[pool_i, variate]
        series_full = {
            "gt": gt[pool_i, variate],
            "binary": binary[pool_i, variate],
            "mmpd": mmpd[pool_i, variate],
        }
        z1 = min(int(args.zoom_steps), int(gt.shape[-1]))
        path = ds_out / (
            f"{dataset}_v{variate}_local{local_i}_pool{pool_i}_t0-{z1}_snapproof.png"
        )
        stats = _plot_snap_proof_panel(
            out_path=path,
            title=(
                f"{dataset} pool={pool_i} v={variate} | window_norm 128-row training lattice "
                f"(max_scale={max_scale}) | dataset-z | NO instance norm | pre bin_center"
            ),
            levels_1d=levels_1d,
            series={k: v[:z1] for k, v in series_full.items()},
            colors=colors,
            t0=0,
        )
        panels.append(str(path.relative_to(REPO_ROOT)))
        residuals.append({"pool": int(pool_i), "panel": "pre_bc", **stats})

        # Optional L-slice after bin-center (same alphabet).
        for L in args.slice_lengths:
            L = int(L)
            if L > gt.shape[-1]:
                continue
            off = max(0, (int(gt.shape[-1]) - L) // 2)
            z0, z1s = off, off + L
            vals = np.stack(
                [series_full[k][z0:z1s] for k in ("gt", "binary", "mmpd")], axis=0
            )[:, None, :]  # (3,1,L) — shift each series separately via (1,1,L)
            shifted = {}
            for name in ("gt", "binary", "mmpd"):
                v = series_full[name][z0:z1s][None, None, :]
                levels_b = levels_1d[None, None, :]
                out, _ = bin_center_shift(v, levels_b, reduce="per_variate")
                shifted[name] = out[0, 0]
            path_bc = ds_out / (
                f"{dataset}_v{variate}_local{local_i}_pool{pool_i}_L{L}_off{off}_snapproof.png"
            )
            stats_bc = _plot_snap_proof_panel(
                out_path=path_bc,
                title=(
                    f"{dataset} L={L} off={off} | AFTER bin_center_shift only "
                    f"(zscore OFF) | dataset-z | NO instance norm | canvas128"
                ),
                levels_1d=levels_1d,
                series=shifted,
                colors=colors,
                t0=z0,
            )
            panels.append(str(path_bc.relative_to(REPO_ROOT)))
            residuals.append({"pool": int(pool_i), "panel": f"bc_L{L}", **stats_bc})

        if bool(getattr(args, "write_horizon96", False)):
            t = np.arange(gt.shape[-1])
            fig, ax = plt.subplots(figsize=(11.0, 3.6))
            ax.plot(t, series_full["gt"], color="#222222", lw=1.8, label="GT")
            ax.plot(t, series_full["binary"], color="#1f77b4", lw=1.4, alpha=0.9, label="binary")
            ax.plot(t, series_full["mmpd"], color="#d62728", lw=1.4, alpha=0.9, label="MMPD")
            mae_b = float(np.mean(np.abs(series_full["binary"] - series_full["gt"])))
            mae_m = float(np.mean(np.abs(series_full["mmpd"] - series_full["gt"])))
            ax.set_title(
                f"{dataset} pool={pool_i} v={variate} | H={gt.shape[-1]} snapped "
                f"({snap_mode})  MAE(binary)={mae_b:.3g}  MAE(MMPD)={mae_m:.3g}",
                fontsize=10,
            )
            ax.set_xlabel("horizon step t")
            ax.set_ylabel("dataset-z (snapped)")
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path_h = ds_out / (
                f"{dataset}_v{variate}_local{local_i}_pool{pool_i}_H{gt.shape[-1]}.png"
            )
            fig.savefig(path_h, dpi=int(getattr(args, "dpi", 150)))
            plt.close(fig)
            panels.append(str(path_h.relative_to(REPO_ROOT)))

    summary = {
        "dataset": dataset,
        "kind": kind,
        "snap_mode": snap_mode,
        "flat_mask": flat_mask,
        "canvas_height": canvas_height,
        "max_scale": max_scale,
        "binary_pack": str(binary_path.relative_to(REPO_ROOT)),
        "mmpd_pack": str(mmpd_path.relative_to(REPO_ROOT)),
        "ckpt": str(ckpt_root.relative_to(REPO_ROOT)),
        "config": config_path,
        "n_windows_total": n,
        "picks": picks,
        "variate": variate,
        "gt_snap": gt_st,
        "binary_snap": bin_st,
        "mmpd_snap": mmpd_st,
        "mmpd_align": align,
        "panels": panels,
        "panel_residuals": residuals,
        "notes": [
            "Values are global dataset-z (pack pool use_ordinal_window_norm=False).",
            "Lattice = finite 128-row window-norm training grid (not ordinal absolute).",
            "No instance/window z-score applied to disc series; BC-only on L-slice panels.",
            "Binary mean_abs_snap_delta should be ~0 (native lattice).",
        ],
    }
    (ds_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{dataset}] snap_mode={snap_mode} H={canvas_height} max_scale={max_scale} "
        f"binary_delta={bin_st['mean_abs_snap_delta']:.3g} "
        f"gt_delta={gt_st['mean_abs_snap_delta']:.3g} panels={len(panels)}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if (args.pack_root is not None or args.ckpt is not None or args.config is not None):
        if len(args.datasets) != 1:
            raise ValueError("--pack-root/--ckpt/--config require exactly one --datasets entry")
    summaries: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        pack_root = (
            args.pack_root.expanduser().resolve()
            if args.pack_root is not None
            else DEFAULT_PACK_ROOTS[dataset]
        )
        ckpt = (
            args.ckpt.expanduser().resolve()
            if args.ckpt is not None
            else DEFAULT_CKPTS[dataset]
        )
        cfg = args.config if args.config is not None else DEFAULT_CFGS[dataset]
        if not pack_root.is_dir():
            raise FileNotFoundError(f"missing pack root for {dataset}: {pack_root}")
        if not ckpt.is_dir():
            raise FileNotFoundError(f"missing ckpt for {dataset}: {ckpt}")
        summaries.append(
            _run_dataset(
                dataset=dataset,
                pack_root=pack_root,
                ckpt_root=ckpt,
                config_path=cfg,
                args=args,
                out_dir=out_dir,
            )
        )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Disc canvas128 ladder snap proof (no instance norm)",
                "",
                "Protocol:",
                "1. Load GT / binary / MMPD from existing ablation packs in **global dataset-z**.",
                "2. Snap all three onto the **window-norm 128-row training lattice**",
                "   (`legal_window_norm_patch_refine_levels_dataset_z`, `max_scale` from ckpt metadata).",
                "3. **Not** the ordinal absolute ladder (that differs for canvas128 leaves).",
                "4. Disc preprocess shown: **bin-center shift only** (zscore off).",
                "5. Titles emphasize: dataset-z values, **no instance / window norm** on disc inputs.",
                "",
                "Binary forecasts from canvas128 ckpts should already sit on this lattice",
                "(mean abs snap delta ≈ 0). GT/MMPD snap onto the same rungs.",
                "",
                "## Datasets",
                "",
            ]
            + [
                f"- **{s['dataset']}**: canvas={s['canvas_height']} max_scale={s['max_scale']} "
                f"binary_snap_meanΔ={s['binary_snap']['mean_abs_snap_delta']:.3g} "
                f"panels={len(s['panels'])}"
                for s in summaries
            ]
            + ["", f"Wrote under `{out_dir.relative_to(REPO_ROOT)}/`.", ""]
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
