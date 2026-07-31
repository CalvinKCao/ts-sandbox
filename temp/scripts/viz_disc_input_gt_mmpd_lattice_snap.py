#!/usr/bin/env python3
"""Plot GT vs MMPD in the exact discriminator input lattice space (h96 ordinal).

Shows post-snap binary-dataset-z values as stored/fed into disc dataset construction
**before** the classifier's per-slice ``zscore_time``. No extra instance/window norm.

Data path matches ``temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py`` /
``temp/scripts/rerun_ordinal_disc_confusions_h96.py``:
  disc-raw binary pack → MMPD pack (sample0) → align_mmpd_to_binary_dataset_norm
  → legal 256-row ladder from past → snap_to_patch_refine_levels.

Disc-raw (``07-31-0925-h96-ordinal-disc-raw``) stores pre-snap binary ``y_true`` /
``samples`` / ``past``; snapped GT/MMPD are recomputed here (same helpers as the
disc run). Disc default is ``fake_agg=sample0`` (one probabilistic draw).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.eval_univariate_patch_refine_ordinal_vs_mmpd import (  # noqa: E402
    _binary_lattice_atol,
    _mmpd_pack,
)
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    binary_mmpd_train_scaler_map,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    DEFAULT_MMPD_DATA,
    run_train_stride,
    run_test_stride,
    run_variate_indices,
    stage_mmpd_dataset_for_run,
)
from utils.forecast_pack_reduce import (  # noqa: E402
    assert_not_anchor_agg,
    reduce_pack_forecast,
    subset_pack_by_pool_indices,
)
from utils.patch_refine_ordinal_ladder import (  # noqa: E402
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402

DEFAULT_DATASETS = ("electricity", "ETTh1", "dynamic", "traffic")
DEFAULT_BINARY = {
    "electricity": "results/ckpts/07-29-4462979-electricity-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "ETTh1": "results/ckpts/07-29-4462980-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "dynamic": "results/ckpts/07-29-4462981-dynamic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "traffic": "results/ckpts/07-29-4462982-traffic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
}
DEFAULT_DISC_RAW = REPO_ROOT / "results/datasets/07-31-0925-h96-ordinal-disc-raw"
DEFAULT_MMPD = REPO_ROOT / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
DEFAULT_OUT = REPO_ROOT / "results/pulled/h96-disc-input-lattice-snap"
BINARY_CONFIG = REPO_ROOT / "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback.yaml"
LOOKBACK = 336
HORIZON = 96
ORDINAL_TIE_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--variate", type=int, default=0)
    p.add_argument("--zoom-len", type=int, default=24, help="Horizon steps in zoom inset")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--linewidth", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--show-zscore-panel",
        action="store_true",
        help="Optional secondary panel after disc zscore_time (labeled).",
    )
    args = p.parse_args()
    args.datasets = [d for raw in args.datasets for d in str(raw).split(",") if d]
    args.disc_raw_dir = args.disc_raw_dir.expanduser().resolve()
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    assert_not_anchor_agg(args.fake_agg)
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.n_windows = 1
    return args


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _load_run_metadata_only(dataset: str, ckpt_root: Path) -> AnchorRun:
    """Build AnchorRun from patch_refine metadata without requiring .pt weights."""
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"missing ckpt root: {ckpt_root}")
    candidates: List[AnchorRun] = []
    for subset_dir in sorted(ckpt_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        meta_path = subset_dir / "patch_refine" / "metadata.json"
        if not meta_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_name") != dataset:
            continue
        metadata = dict(metadata)
        metadata["dataset_name"] = dataset
        metadata["dataset"] = dataset
        candidates.append(
            AnchorRun(
                variant="binary_patch_refine",
                dataset=dataset,
                root=ckpt_root,
                subset_dir=subset_dir,
                best_pt=None,
                itrans_pt=None,
                metadata=metadata,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"No patch_refine/metadata.json for {dataset} under {ckpt_root} "
            "(weights not required; pull metadata from Killarney if missing)"
        )
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous subsets for {dataset} under {ckpt_root}")
    return candidates[0]


def _zscore_time(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Match disc dataset construction: per-slice mean/std over the time axis."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return ((x - mu) / np.maximum(sd, eps)).astype(np.float32)


def _pick_windows(
    gt: np.ndarray,
    mmpd: np.ndarray,
    *,
    n: int,
    variate: int,
    seed: int,
) -> np.ndarray:
    """Prefer high-structure windows (GT range + |GT-MMPD| energy)."""
    n_win = int(gt.shape[0])
    v = int(variate)
    if v < 0 or v >= gt.shape[1]:
        raise ValueError(f"variate={v} out of range for V={gt.shape[1]}")
    if n_win <= n:
        return np.arange(n_win, dtype=np.int64)
    gt_v = gt[:, v, :]
    mmpd_v = mmpd[:, v, :]
    amp = gt_v.max(axis=-1) - gt_v.min(axis=-1)
    resid = np.mean(np.abs(gt_v - mmpd_v), axis=-1)
    score = 0.65 * amp + 0.35 * resid
    # Soft jitter so ties don't always pick the same early indices.
    rng = np.random.default_rng(int(seed))
    score = score + 1e-6 * rng.random(n_win)
    order = np.argsort(-score)
    return np.sort(order[:n].astype(np.int64))


def _prepare_snapped(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Mapping[str, float], List[int]]:
    binary_path = args.disc_raw_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    if not binary_path.is_file():
        raise FileNotFoundError(
            f"missing disc-raw binary pack: {binary_path}\n"
            "Pull from Killarney: results/datasets/07-31-0925-h96-ordinal-disc-raw/"
        )
    binary_pack = _load_npz(binary_path)
    required = {"y_true", "samples", "past", "indices"}
    missing = sorted(required - set(binary_pack))
    if missing:
        raise KeyError(f"{binary_path} missing {missing}")

    indices = [int(i) for i in binary_pack["indices"].tolist()]
    past = np.asarray(binary_pack["past"], dtype=np.float32)
    if past.shape != (len(indices), past.shape[1], LOOKBACK):
        raise ValueError(f"{dataset}: unexpected past shape {past.shape}")

    mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
    mmpd_pack = subset_pack_by_pool_indices(mmpd_full, np.asarray(indices, dtype=np.int64))

    ckpt_root = (REPO_ROOT / DEFAULT_BINARY[dataset]).resolve()
    run = _load_run_metadata_only(dataset, ckpt_root)
    stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)

    binary_gt = binary_pack["y_true"].astype(np.float32)
    mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
    mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    print(
        f"[{dataset}] fake_agg={args.fake_agg} "
        f"binary_samples={tuple(binary_pack['samples'].shape)} "
        f"mmpd_samples={tuple(mmpd_pack['samples'].shape)} n={len(indices)}",
        flush=True,
    )

    # Minimal namespace for scaler helper (lookback/horizon + mmpd_data_dir).
    scaler_args = argparse.Namespace(
        lookback=LOOKBACK,
        horizon=HORIZON,
        mmpd_data_dir=args.mmpd_data_dir,
    )
    scalers = binary_mmpd_train_scaler_map(scaler_args, run)
    mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_pred,
        **scalers,
    )

    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=LOOKBACK,
        horizon=HORIZON,
        ordinal_tie_atol=ORDINAL_TIE_ATOL,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal ladder missing from load_dataset")

    legal_levels = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
    gt, gt_stats = snap_to_patch_refine_levels(binary_gt, legal_levels)
    mmpd, mmpd_stats = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
    assert_on_patch_refine_levels(gt, legal_levels)
    assert_on_patch_refine_levels(mmpd, legal_levels)

    # Sanity: values land exactly on window-specific ladder rows.
    atol = _binary_lattice_atol(legal_levels)
    row_err_gt = float(np.min(np.abs(gt[..., None] - legal_levels[:, :, None, :]), axis=-1).max())
    row_err_mmpd = float(np.min(np.abs(mmpd[..., None] - legal_levels[:, :, None, :]), axis=-1).max())
    print(
        f"[{dataset}] snap gt_mean_delta={gt_stats['mean_abs_snap_delta']:.4g} "
        f"mmpd_mean_delta={mmpd_stats['mean_abs_snap_delta']:.4g} "
        f"ladder_row_err gt={row_err_gt:.3g} mmpd={row_err_mmpd:.3g} atol={atol:.3g}",
        flush=True,
    )

    tensors = {
        "gt": gt,
        "mmpd": mmpd,
        "past": past,
        "legal_levels": legal_levels,
        "indices": np.asarray(indices, dtype=np.int64),
        "mmpd_pre_snap": mmpd_binary_z.astype(np.float32),
        "gt_pre_snap": binary_gt,
    }
    meta = {
        "align": align,
        "fake_agg": args.fake_agg,
        "n_windows": float(len(indices)),
        "gt_snap": gt_stats,
        "mmpd_snap": mmpd_stats,
        "note": (
            "Main plot = post-snap binary-dataset-z (disc input before zscore_time). "
            "No instance/window norm beyond align+snap. Disc-raw packs are pre-snap; "
            "tensors recomputed with the ordinal evaluator helpers."
        ),
        "binary_config": str(BINARY_CONFIG.relative_to(REPO_ROOT)),
        "disc_raw_dir": str(args.disc_raw_dir),
        "mmpd_output_root": str(args.mmpd_output_root),
    }
    return tensors, meta, indices


def _draw_levels(ax, levels_1d: np.ndarray, x0: float, x1: float) -> None:
    uniq = np.unique(levels_1d.astype(np.float64))
    # Cap line count for readability while still showing the lattice.
    if uniq.size > 96:
        step = max(1, uniq.size // 64)
        uniq = uniq[::step]
    ax.hlines(uniq, x0, x1, colors="0.75", linewidths=0.25, alpha=0.35, zorder=0)


def _plot_dataset(
    args: argparse.Namespace,
    dataset: str,
    tensors: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    out_dir: Path,
) -> List[Path]:
    gt = tensors["gt"]
    mmpd = tensors["mmpd"]
    past = tensors["past"]
    levels = tensors["legal_levels"]
    pool_idx = tensors["indices"]
    v = int(args.variate)
    picks = _pick_windows(gt, mmpd, n=int(args.n_windows), variate=v, seed=int(args.seed))
    paths: List[Path] = []
    lw = float(args.linewidth)
    zoom_len = max(4, min(int(args.zoom_len), HORIZON))

    for local in picks.tolist():
        pool_i = int(pool_idx[local])
        gt_v = gt[local, v]
        mmpd_v = mmpd[local, v]
        past_v = past[local, v]
        levels_v = levels[local, v]
        resid = mmpd_v - gt_v

        n_rows = 3 if args.show_zscore_panel else 2
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(11.5, 2.6 * n_rows + 0.8),
            sharex=False,
            gridspec_kw={"height_ratios": [2.2, 1.0] + ([1.2] if args.show_zscore_panel else [])},
        )
        if n_rows == 2:
            ax_main, ax_resid = axes
        else:
            ax_main, ax_resid, ax_z = axes

        past_x = np.arange(-LOOKBACK, 0)
        fut_x = np.arange(HORIZON)
        _draw_levels(ax_main, levels_v, float(past_x[0]), float(fut_x[-1]))
        ax_main.plot(past_x, past_v, color="0.55", lw=lw, label="lookback (binary dataset-z)")
        ax_main.plot(fut_x, gt_v, color="black", lw=lw, label="GT snapped")
        ax_main.plot(fut_x, mmpd_v, color="#d62728", lw=lw, alpha=0.9, label=f"MMPD snapped ({args.fake_agg})")
        ax_main.axvline(0, color="0.25", lw=0.6)
        ax_main.set_ylabel("binary dataset-z\n(post-snap, pre zscore_time)")
        ax_main.set_title(
            f"{dataset}  local={local} pool_idx={pool_i} variate={v}  "
            f"|  256-row ladder (faint)  |  no instance/window norm"
        )
        ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=3)
        ax_main.grid(alpha=0.12, linewidth=0.4)

        # Zoom inset on early horizon to make discrete levels obvious.
        ax_in = ax_main.inset_axes([0.62, 0.12, 0.35, 0.45])
        z0, z1 = 0, zoom_len
        _draw_levels(ax_in, levels_v, float(z0), float(z1 - 1))
        ax_in.plot(np.arange(z0, z1), gt_v[z0:z1], color="black", lw=lw + 0.15, drawstyle="steps-mid")
        ax_in.plot(np.arange(z0, z1), mmpd_v[z0:z1], color="#d62728", lw=lw + 0.15, drawstyle="steps-mid", alpha=0.9)
        ax_in.set_title(f"zoom t=0..{z1 - 1} (steps-mid)", fontsize=7)
        ax_in.tick_params(labelsize=6)
        ax_in.grid(alpha=0.15, linewidth=0.3)
        # Y-lim tight around zoom segment so rungs separate.
        seg = np.concatenate([gt_v[z0:z1], mmpd_v[z0:z1]])
        pad = 0.08 * max(float(seg.max() - seg.min()), 1e-3)
        ax_in.set_ylim(float(seg.min()) - pad, float(seg.max()) + pad)
        ax_main.indicate_inset_zoom(ax_in, edgecolor="0.4")

        ax_resid.axhline(0.0, color="0.4", lw=0.5)
        ax_resid.plot(fut_x, resid, color="#1f77b4", lw=lw, label="MMPD − GT")
        ax_resid.set_ylabel("residual")
        ax_resid.set_xlabel("forecast offset")
        ax_resid.grid(alpha=0.12, linewidth=0.4)
        ax_resid.legend(loc="upper right", fontsize=8)

        if args.show_zscore_panel:
            gt_z = _zscore_time(gt_v[None, None, :])[0, 0]
            mmpd_z = _zscore_time(mmpd_v[None, None, :])[0, 0]
            ax_z.plot(fut_x, gt_z, color="black", lw=lw, label="GT after zscore_time")
            ax_z.plot(fut_x, mmpd_z, color="#d62728", lw=lw, alpha=0.9, label="MMPD after zscore_time")
            ax_z.set_ylabel("zscore_time\n(secondary)")
            ax_z.set_xlabel("forecast offset")
            ax_z.set_title("OPTIONAL: classifier per-slice z-score (not the main disc-input view)", fontsize=8)
            ax_z.legend(loc="upper left", fontsize=8)
            ax_z.grid(alpha=0.12, linewidth=0.4)

        fig.tight_layout()
        out = out_dir / f"{dataset}_v{v}_local{local}_pool{pool_i}_gt_mmpd_lattice.png"
        fig.savefig(out, dpi=int(args.dpi))
        plt.close(fig)
        paths.append(out)
        print(f"[{dataset}] wrote {out}", flush=True)

    # Compact multi-window strip for the dataset.
    fig, axes = plt.subplots(len(picks), 1, figsize=(11.5, 2.15 * len(picks)), squeeze=False)
    for row, local in enumerate(picks.tolist()):
        ax = axes[row, 0]
        pool_i = int(pool_idx[local])
        gt_v = gt[local, v]
        mmpd_v = mmpd[local, v]
        levels_v = levels[local, v]
        fut_x = np.arange(HORIZON)
        _draw_levels(ax, levels_v, 0.0, float(HORIZON - 1))
        ax.plot(fut_x, gt_v, color="black", lw=lw, label="GT snapped")
        ax.plot(fut_x, mmpd_v, color="#d62728", lw=lw, alpha=0.9, label=f"MMPD ({args.fake_agg})")
        ax.set_ylabel("dataset-z")
        ax.set_title(f"{dataset} local={local} pool={pool_i} v={v}", fontsize=9)
        ax.grid(alpha=0.12, linewidth=0.4)
        if row == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("forecast offset")
    fig.suptitle(
        f"{dataset}: disc-input lattice (post-snap binary-z, pre zscore_time; fake_agg={args.fake_agg})",
        fontsize=10,
    )
    fig.tight_layout()
    strip = out_dir / f"{dataset}_v{v}_strip_gt_mmpd_lattice.png"
    fig.savefig(strip, dpi=int(args.dpi))
    plt.close(fig)
    paths.append(strip)
    print(f"[{dataset}] wrote {strip}", flush=True)

    (out_dir / f"meta_{dataset}.json").write_text(
        json.dumps({**meta, "picked_locals": picks.tolist(), "variate": v}, indent=2),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    # Ladder decode is light; CPU is enough.
    if args.cpu:
        device = torch.device("cpu")
    print(
        f"[device] {device} fake_agg={args.fake_agg} out={args.output_dir}\n"
        f"[sources] disc-raw={args.disc_raw_dir}\n"
        f"          mmpd={args.mmpd_output_root}\n"
        f"          config={BINARY_CONFIG.name}",
        flush=True,
    )

    all_paths: List[str] = []
    for dataset in args.datasets:
        if dataset not in DEFAULT_BINARY:
            raise ValueError(f"unsupported dataset {dataset}; expected one of {list(DEFAULT_BINARY)}")
        print(f"\n===== {dataset} =====", flush=True)
        tensors, meta, _indices = _prepare_snapped(args, dataset, device)
        paths = _plot_dataset(args, dataset, tensors, meta, args.output_dir)
        all_paths.extend(str(p) for p in paths)

    manifest = {
        "datasets": list(args.datasets),
        "fake_agg": args.fake_agg,
        "disc_raw_dir": str(args.disc_raw_dir),
        "mmpd_output_root": str(args.mmpd_output_root),
        "output_dir": str(args.output_dir),
        "pngs": all_paths,
        "coordinate_space": (
            "binary dataset-z after align_mmpd_to_binary_dataset_norm + "
            "snap_to_patch_refine_levels (256-row absolute ladder); "
            "NO disc zscore_time on main panels; NO extra instance/window norm"
        ),
        "tensors_plotted": "snapped GT (from binary y_true) and snapped MMPD (sample0 / first pack draw)",
        "disc_raw_note": (
            "07-31-0925 packs store pre-snap y_true/samples/past; snapped arrays "
            "are recomputed with the same helpers as the ordinal disc evaluator"
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
