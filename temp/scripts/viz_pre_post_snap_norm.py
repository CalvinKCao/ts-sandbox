#!/usr/bin/env python3
"""Pre- vs post- snap / disc-norm panels for GT, binary, MMPD.

Shows the same windows through three spaces (columns):

1. **Pre-snap** — pack storage space (global dataset-z; hybrid flat LULL after
   dataset affine only; non-flat hybrid / wn128 use past mean/std ladder geometry
   but values are still dataset-z).
2. **Post-snap** — nearest rung on the training 128 ladder
   (``window_norm_grid`` / ``window_norm_grid_hybrid_flat``). Occupied rungs drawn.
3. **Post disc-norm** — ``bin_center_shift`` only (zscore OFF; live disc path).
   Optional tiny zscore panel labeled NOT live.

Default focus: hybrid ETTh2 train **4609805** disc pack
``08-05-1057-…hybrid-flat-dsnorm-valtest80-byvar`` — LULL v5 (flat) + HUFL v0
(non-flat). Optional ETTh1 wn128 comparison.

Writes ``temp/viz_pre_post_snap_norm/`` + README.
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

from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _flat_mask_from_ckpt,
    _max_scale_from_ckpt_metadata,
    _window_norm_grid_config,
    load_ablation_run,
)
from utils.disc_bin_center_shift import bin_center_shift, nearest_bin_indices  # noqa: E402
from utils.disc_shared import zscore_time  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    binary_mmpd_train_scaler_map,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    DEFAULT_MMPD_DATA,
    run_subset_id,
)
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402
from utils.patch_refine_ordinal_ladder import snap_to_patch_refine_levels  # noqa: E402
from utils.patch_refine_value_grid import (  # noqa: E402
    legal_window_norm_patch_refine_levels_dataset_z,
)
from utils.visualize_staged_eval_2d_preds import _build_state  # noqa: E402

# ETTh* column order after date.
ETTH_NAMES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

COLORS = {"GT": "#222222", "binary": "#1f77b4", "MMPD": "#d62728"}

HYBRID_SPEC = {
    "tag": "hybrid_ETTh2",
    "dataset": "ETTh2",
    "pack": (
        REPO_ROOT
        / "results/datasets/08-05-1057-ablation-disc-l8-l16-ETTh2-c128-hybrid-flat-dsnorm-valtest80-byvar"
    ),
    "ckpt": (
        REPO_ROOT
        / "results/ckpts/08-05-4609805-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm"
    ),
    "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm.yaml",
    # flat LULL + non-flat HUFL; optional OT.
    "variates": ((5, "LULL"), (0, "HUFL")),
    "preferred_pools": (1116, 37, 1340),
    "space_blurb": (
        "hybrid flat: LULL skips window-norm (dataset affine only); "
        "non-flat vars use past mean/std → window_norm_grid_hybrid_flat"
    ),
}

WN128_ETTH1_SPEC = {
    "tag": "wn128_ETTh1",
    "dataset": "ETTh1",
    "pack": (
        REPO_ROOT
        / "results/datasets/08-04-1843-ablation-disc-l8-l16-ETTh1-c128-valtest80-byvar"
    ),
    "ckpt": (
        REPO_ROOT
        / "results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6"
    ),
    "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
    "variates": ((0, "HUFL"),),
    "preferred_pools": (),
    "space_blurb": (
        "standard canvas128: global dataset-z packs; snap via window_norm_grid "
        "(past mean/std → 128 rungs in dataset-z)"
    ),
}

DEFAULT_OUT = REPO_ROOT / "temp" / "viz_pre_post_snap_norm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--skip-etth1",
        action="store_true",
        help="Skip optional wn128 ETTh1 comparison panels.",
    )
    p.add_argument(
        "--include-zscore-ref",
        action="store_true",
        default=True,
        help="Add a small post-zscore column labeled NOT live (default on).",
    )
    p.add_argument("--no-zscore-ref", action="store_true")
    p.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=None,
        help="Force pool indices for hybrid ETTh2 (else prefer disc-disagree + MAE).",
    )
    return p.parse_args()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _find_pack(raw_dir: Path, prefix: str, dataset: str) -> Path:
    hits = sorted(p for p in raw_dir.glob(f"{prefix}*{dataset}*.npz") if "indices" not in p.name)
    vt = [p for p in hits if "val-test" in p.name or "val_test" in p.name]
    if vt:
        return vt[0]
    if not hits:
        raise FileNotFoundError(f"no {prefix} pack for {dataset} under {raw_dir}")
    return hits[0]


def _var_name(dataset: str, v: int) -> str:
    if dataset.startswith("ETT") and 0 <= v < len(ETTH_NAMES):
        return ETTH_NAMES[v]
    return f"v{v}"


def _space_title(snap_mode: str, flat_mask: Optional[Sequence[bool]], variate: int) -> str:
    is_flat = bool(flat_mask[variate]) if flat_mask and variate < len(flat_mask) else False
    if snap_mode == "window_norm_grid_hybrid_flat":
        if is_flat:
            return (
                "PRE: global dataset-z after hybrid flat dataset affine "
                "(skip window-norm; center=0, std=1 on ladder)"
            )
        return (
            "PRE: global dataset-z (binary train scaler); "
            "ladder uses past mean/std (window-norm geometry)"
        )
    if snap_mode == "window_norm_grid":
        return "PRE: global dataset-z (pack storage); ladder = past mean/std → 128 rungs"
    return f"PRE: pack storage ({snap_mode})"


def _select_locals(
    *,
    indices: np.ndarray,
    gt: np.ndarray,
    mmpd: np.ndarray,
    variate: int,
    n_windows: int,
    preferred: Sequence[int],
    forced: Sequence[int] | None,
    seed: int,
) -> List[int]:
    n = int(gt.shape[0])
    pool_to_local = {int(indices[i]): i for i in range(n)}
    chosen: List[int] = []
    seen: set[int] = set()

    def add_pool(pool: int) -> None:
        local = pool_to_local.get(int(pool))
        if local is None or local in seen:
            return
        seen.add(local)
        chosen.append(local)

    if forced:
        for pool in forced:
            add_pool(int(pool))
    else:
        for pool in preferred:
            add_pool(int(pool))

    mae = np.mean(np.abs(mmpd[:, variate] - gt[:, variate]), axis=-1)
    for local in np.argsort(-mae).tolist():
        if len(chosen) >= n_windows:
            break
        if local in seen:
            continue
        seen.add(local)
        chosen.append(int(local))

    if len(chosen) < n_windows:
        rng = np.random.default_rng(seed)
        rest = [i for i in range(n) if i not in seen]
        need = min(n_windows - len(chosen), len(rest))
        if need:
            chosen.extend(int(x) for x in rng.choice(rest, size=need, replace=False))
    return chosen[:n_windows]


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    """Per-slice zscore (NOT live disc when bin_center_shift is on)."""
    return zscore_time(np.asarray(x, dtype=np.float32)[None, :])[0]


def _load_run_for_snap(dataset: str, ckpt_root: Path) -> AnchorRun:
    """Prefer full patch_refine ckpt; fall back to metadata-only (no best.pt)."""
    try:
        run, _stages, kind = load_ablation_run(dataset, ckpt_root)
        if kind != "patch_refine":
            raise RuntimeError(f"{dataset}: expected patch_refine, got {kind}")
        return run
    except FileNotFoundError as exc:
        # Weights often absent locally; metadata is enough for ladder + scalers.
        meta_path = ckpt_root / dataset / "patch_refine" / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{ckpt_root}: missing best.pt and {meta_path}"
            ) from exc
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = dict(meta)
        meta["dataset_name"] = dataset
        meta["dataset"] = dataset
        subset_dir = ckpt_root / dataset
        print(
            f"[{dataset}] metadata-only AnchorRun (no best.pt under {ckpt_root.name})",
            flush=True,
        )
        return AnchorRun(
            variant="binary_patch_refine",
            dataset=dataset,
            root=ckpt_root,
            subset_dir=subset_dir,
            best_pt=meta_path,
            itrans_pt=None,
            metadata=meta,
        )


def _overlay_series(
    ax: Any,
    *,
    series: Mapping[str, np.ndarray],
    t0: int,
    markers: bool,
    levels_1d: Optional[np.ndarray],
    ylabel: str,
    title: str,
) -> None:
    length = int(next(iter(series.values())).shape[0])
    x = np.arange(t0, t0 + length)
    if levels_1d is not None and markers:
        occupied = np.unique(
            np.concatenate([np.asarray(series[n], dtype=np.float64) for n in series])
        )
        for y in occupied:
            ax.axhline(float(y), color="0.55", lw=0.8, alpha=0.75, zorder=0)
    for name, y in series.items():
        y = np.asarray(y, dtype=np.float64)
        c = COLORS[name]
        ax.plot(x, y, color=c, lw=1.15 if not markers else 0.9, alpha=0.85 if not markers else 0.35, zorder=1, label=name)
        if markers:
            ax.plot(
                x, y, linestyle="none", marker="o", markersize=5.5,
                markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.45, zorder=3,
            )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=3)


def _write_h96_stages(
    *,
    out_path: Path,
    pre: Mapping[str, np.ndarray],
    post: Mapping[str, np.ndarray],
    levels_1d: np.ndarray,
    title_prefix: str,
    space_pre: str,
    snap_mode: str,
    dpi: int,
) -> Path:
    """Two-column H=96: pre-snap vs post-snap (full horizon)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 3.8), sharey=False)
    _overlay_series(
        axes[0],
        series=pre,
        t0=0,
        markers=False,
        levels_1d=None,
        ylabel="dataset-z",
        title=f"{space_pre}",
    )
    _overlay_series(
        axes[1],
        series=post,
        t0=0,
        markers=True,
        levels_1d=levels_1d,
        ylabel="dataset-z (snapped)",
        title=f"POST-SNAP onto {snap_mode} (occupied rungs)",
    )
    mae_b = float(np.mean(np.abs(post["binary"] - post["GT"])))
    mae_m = float(np.mean(np.abs(post["MMPD"] - post["GT"])))
    d_gt = float(np.mean(np.abs(post["GT"] - pre["GT"])))
    d_b = float(np.mean(np.abs(post["binary"] - pre["binary"])))
    d_m = float(np.mean(np.abs(post["MMPD"] - pre["MMPD"])))
    fig.suptitle(
        f"{title_prefix} | H={len(pre['GT'])}  "
        f"MAE(bin/MMPD→GT)={mae_b:.3g}/{mae_m:.3g}  "
        f"mean|Δsnap| GT/bin/MMPD={d_gt:.3g}/{d_b:.3g}/{d_m:.3g}",
        fontsize=10,
    )
    for ax in axes:
        ax.set_xlabel("horizon step t", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _write_l_stages(
    *,
    out_path: Path,
    pre: Mapping[str, np.ndarray],
    post: Mapping[str, np.ndarray],
    post_bc: Mapping[str, np.ndarray],
    post_z: Optional[Mapping[str, np.ndarray]],
    levels_1d: np.ndarray,
    title_prefix: str,
    space_pre: str,
    snap_mode: str,
    offset: int,
    slice_len: int,
    dpi: int,
) -> Path:
    n_cols = 4 if post_z is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.6), sharey=False)
    _overlay_series(
        axes[0],
        series=pre,
        t0=offset,
        markers=False,
        levels_1d=None,
        ylabel="dataset-z",
        title=f"1) PRE-SNAP\n{space_pre}",
    )
    _overlay_series(
        axes[1],
        series=post,
        t0=offset,
        markers=True,
        levels_1d=levels_1d,
        ylabel="dataset-z",
        title=f"2) POST-SNAP\n{snap_mode} (rungs)",
    )
    _overlay_series(
        axes[2],
        series=post_bc,
        t0=offset,
        markers=True,
        levels_1d=levels_1d,
        ylabel="dataset-z (bin-centered)",
        title="3) POST disc-norm\nbin_center_shift ONLY (LIVE)",
    )
    if post_z is not None:
        _overlay_series(
            axes[3],
            series=post_z,
            t0=offset,
            markers=False,
            levels_1d=None,
            ylabel="z-scored (ref)",
            title="4) POST-zscore\nNOT the live disc path",
        )
        axes[3].set_facecolor("#fff8e8")
    fig.suptitle(
        f"{title_prefix} | L={slice_len} mid-horizon off={offset}",
        fontsize=10,
    )
    for ax in axes:
        ax.set_xlabel("horizon step t", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _write_bin_index_compare(
    *,
    out_path: Path,
    post: Mapping[str, np.ndarray],
    post_bc: Mapping[str, np.ndarray],
    levels_1d: np.ndarray,
    title_prefix: str,
    offset: int,
    dpi: int,
) -> Path:
    """Integer bin rows before/after bin_center_shift (proves additive shift)."""
    length = int(next(iter(post.values())).shape[0])
    x = np.arange(offset, offset + length)
    lev = np.asarray(levels_1d, dtype=np.float32)[None, None, :]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, series, lab in (
        (axes[0], post, "POST-SNAP bin index"),
        (axes[1], post_bc, "POST bin_center_shift bin index (LIVE)"),
    ):
        for name, y in series.items():
            bins = nearest_bin_indices(
                np.asarray(y, dtype=np.float32)[None, None, :], lev,
            )[0, 0]
            ax.plot(x, bins, color=COLORS[name], lw=1.0, alpha=0.35)
            ax.plot(
                x, bins, linestyle="none", marker="s", markersize=5.5,
                markerfacecolor=COLORS[name], markeredgecolor="white",
                markeredgewidth=0.4, label=name,
            )
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("horizon step t", fontsize=8)
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=7, ncol=3, framealpha=0.9)
    axes[0].set_ylabel(f"{levels_1d.shape[0]}-row bin index", fontsize=8)
    fig.suptitle(f"{title_prefix} | bin-index view", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _run_spec(
    spec: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
    forced_pools: Sequence[int] | None,
    include_zscore: bool,
) -> Dict[str, Any]:
    dataset = str(spec["dataset"])
    pack_root = Path(spec["pack"]).expanduser().resolve()
    ckpt = Path(spec["ckpt"]).expanduser().resolve()
    config = str(spec["config"])
    tag = str(spec["tag"])
    raw_dir = pack_root / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw/ under {pack_root}")
    if not ckpt.is_dir():
        raise FileNotFoundError(f"missing ckpt: {ckpt}")

    binary_path = _find_pack(raw_dir, "binary_", dataset)
    mmpd_path = _find_pack(raw_dir, "mmpd_", dataset)
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)

    run = _load_run_for_snap(dataset, ckpt)
    state = _build_state(ckpt, dataset, run_subset_id(run), config)
    if bool(getattr(state, "use_ordinal_window_norm", False)):
        raise RuntimeError(f"{tag}: ordinal leaf not supported (use window-norm canvas128)")
    if not bool(getattr(state, "use_window_normalization", False)):
        raise RuntimeError(f"{tag}: use_window_normalization must be True")

    canvas_height = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if "canvas_height" in binary_pack:
        canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    if canvas_height != 128:
        raise RuntimeError(f"{tag}: expected canvas_height=128, got {canvas_height}")

    max_scale = _max_scale_from_ckpt_metadata(ckpt, dataset)
    flat_mask = _flat_mask_from_ckpt(ckpt, dataset)
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

    b_idx = np.asarray(binary_pack["indices"], dtype=np.int64)
    m_idx = np.asarray(mmpd_pack["indices"], dtype=np.int64)
    if not np.array_equal(b_idx, m_idx):
        raise RuntimeError(f"{tag}: binary/MMPD indices differ")

    # Pre-snap (aligned MMPD → binary dataset-z).
    past = np.asarray(binary_pack["past"], dtype=np.float32)
    gt_pre = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pre = np.asarray(reduce_pack_forecast(binary_pack, agg=args.fake_agg), dtype=np.float32)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)
    mmpd_raw = np.asarray(reduce_pack_forecast(mmpd_pack, agg=args.fake_agg), dtype=np.float32)
    snap_args = SimpleNamespace(
        fake_agg=args.fake_agg,
        mmpd_data_dir=args.mmpd_data_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        dataset=dataset,
        pack_test_stride=4,
    )
    scalers = binary_mmpd_train_scaler_map(snap_args, run)
    mmpd_pre, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=gt_pre,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_raw,
        **scalers,
    )

    levels = legal_window_norm_patch_refine_levels_dataset_z(past, grid_cfg)
    gt_post, gt_snap = snap_to_patch_refine_levels(gt_pre, levels)
    binary_post, binary_snap = snap_to_patch_refine_levels(binary_pre, levels)
    mmpd_post, mmpd_snap = snap_to_patch_refine_levels(mmpd_pre, levels)
    indices = b_idx
    print(
        f"[{tag}] snap_mode={snap_mode} canvas={canvas_height} max_scale={max_scale} "
        f"device={device} flat_mask={flat_mask} "
        f"binary_meanΔ={binary_snap['mean_abs_snap_delta']:.3g}",
        flush=True,
    )

    tag_dir = out_root / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    panels: List[str] = []
    var_summaries: List[Dict[str, Any]] = []

    for variate, vname in spec["variates"]:
        v = int(variate)
        name = vname or _var_name(dataset, v)
        is_flat = bool(flat_mask[v]) if flat_mask is not None else False
        space_pre = _space_title(snap_mode, flat_mask, v)
        locals_ = _select_locals(
            indices=indices,
            gt=gt_post,
            mmpd=mmpd_post,
            variate=v,
            n_windows=int(args.n_windows),
            preferred=list(spec.get("preferred_pools") or ()),
            forced=forced_pools if tag.startswith("hybrid") else None,
            seed=int(args.seed) + v * 17,
        )
        pools = [int(indices[i]) for i in locals_]
        v_dir = tag_dir / f"{name}_v{v}"
        v_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{tag}] {name} v={v} flat={is_flat} locals={locals_} pools={pools} "
            f"snap={snap_mode}",
            flush=True,
        )

        for local in locals_:
            pool = int(indices[local])
            prefix = f"{tag}/{dataset} {name} v={v} pool={pool} local={local}"
            flat_tag = "FLAT" if is_flat else "non-flat"
            title = f"{prefix} [{flat_tag}]"

            pre = {
                "GT": gt_pre[local, v],
                "binary": binary_pre[local, v],
                "MMPD": mmpd_pre[local, v],
            }
            post = {
                "GT": gt_post[local, v],
                "binary": binary_post[local, v],
                "MMPD": mmpd_post[local, v],
            }
            levels_1d = levels[local, v]

            h96 = _write_h96_stages(
                out_path=v_dir / f"pool{pool}_local{local}_H96_pre_vs_post_snap.png",
                pre=pre,
                post=post,
                levels_1d=levels_1d,
                title_prefix=title,
                space_pre=space_pre,
                snap_mode=snap_mode,
                dpi=int(args.dpi),
            )
            panels.append(str(h96.relative_to(out_root)))

            h = int(gt_post.shape[-1])
            for L in args.slice_lengths:
                L = int(L)
                if L > h:
                    continue
                off = max(0, (h - L) // 2)
                pre_s = {k: v_[off : off + L] for k, v_ in pre.items()}
                post_s = {k: v_[off : off + L] for k, v_ in post.items()}
                post_bc: Dict[str, np.ndarray] = {}
                for k, seg in post_s.items():
                    shifted, _ = bin_center_shift(
                        seg[None, None, :],
                        levels_1d[None, None, :],
                        reduce="per_variate",
                    )
                    post_bc[k] = shifted[0, 0]
                post_z = None
                if include_zscore:
                    post_z = {k: _zscore_1d(v_) for k, v_ in post_s.items()}

                lpath = _write_l_stages(
                    out_path=v_dir / (
                        f"pool{pool}_local{local}_L{L}_off{off}_pre_snap_bc.png"
                    ),
                    pre=pre_s,
                    post=post_s,
                    post_bc=post_bc,
                    post_z=post_z,
                    levels_1d=levels_1d,
                    title_prefix=title,
                    space_pre=space_pre,
                    snap_mode=snap_mode,
                    offset=off,
                    slice_len=L,
                    dpi=int(args.dpi),
                )
                panels.append(str(lpath.relative_to(out_root)))

                bipath = _write_bin_index_compare(
                    out_path=v_dir / (
                        f"pool{pool}_local{local}_L{L}_off{off}_bin_index.png"
                    ),
                    post=post_s,
                    post_bc=post_bc,
                    levels_1d=levels_1d,
                    title_prefix=title,
                    offset=off,
                    dpi=int(args.dpi),
                )
                panels.append(str(bipath.relative_to(out_root)))

        var_summaries.append(
            {
                "variate": v,
                "name": name,
                "is_flat": is_flat,
                "locals": locals_,
                "pools": pools,
                "space_pre": space_pre,
            }
        )

    return {
        "tag": tag,
        "dataset": dataset,
        "pack": str(pack_root.relative_to(REPO_ROOT))
        if pack_root.is_relative_to(REPO_ROOT)
        else str(pack_root),
        "ckpt": str(ckpt.relative_to(REPO_ROOT))
        if ckpt.is_relative_to(REPO_ROOT)
        else str(ckpt),
        "config": config,
        "binary": binary_path.name,
        "mmpd": mmpd_path.name,
        "snap_mode": snap_mode,
        "flat_mask": list(flat_mask) if flat_mask is not None else None,
        "canvas_height": canvas_height,
        "space_blurb": spec.get("space_blurb"),
        "mmpd_align": align,
        "variates": var_summaries,
        "panels": panels,
        "gt_snap": gt_snap,
        "binary_snap": binary_snap,
        "mmpd_snap": mmpd_snap,
        "max_scale": max_scale,
    }


def _write_readme(out_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Pre- / post- snap & disc-norm visualizations",
        "",
        "Generated by `temp/scripts/viz_pre_post_snap_norm.py`.",
        "",
        "## Stages (same windows / variates)",
        "",
        "1. **Pre-snap / pre-disc-norm** — values as packs store them:",
        "   - canvas128 / hybrid non-flat: **global dataset-z** (binary train scaler).",
        "   - hybrid flat (LULL): **global dataset-z after hybrid flat dataset affine**",
        "     (skip window-norm; ladder center=0, std=1).",
        "   - MMPD is affine-aligned into binary dataset-z before plotting.",
        "2. **Post-snap** — nearest rung on the training 128 ladder",
        "   (`legal_window_norm_patch_refine_levels_dataset_z` →",
        "   `window_norm_grid` or `window_norm_grid_hybrid_flat`). Occupied rungs drawn.",
        "3. **Post disc-norm** — **`bin_center_shift` only** (zscore OFF) — live disc path.",
        "4. Optional **post-zscore** column — mean/std over the L-slice; cream background;",
        "   labeled **NOT the live path**.",
        "",
        "H=96 panels: pre vs post-snap side-by-side.",
        "L=8 / L=16: mid-horizon pre → post-snap → post-bin-center (+ optional zscore).",
        "Bin-index panels: integer ladder rows before/after bin_center_shift.",
        "",
        "## Runs",
        "",
    ]
    for s in summaries:
        lines.append(f"### `{s['tag']}` ({s['dataset']})")
        lines.append("")
        lines.append(f"- pack: `{s['pack']}`")
        lines.append(f"- ckpt: `{s['ckpt']}`")
        lines.append(f"- snap_mode: `{s['snap_mode']}`")
        lines.append(f"- flat_mask: `{s.get('flat_mask')}`")
        lines.append(f"- note: {s.get('space_blurb', '')}")
        lines.append(f"- panels: {len(s['panels'])}")
        for vs in s["variates"]:
            flat = "FLAT" if vs["is_flat"] else "non-flat"
            lines.append(
                f"  - **{vs['name']} v={vs['variate']}** ({flat}) pools={vs['pools']}"
            )
        lines.append("")
        # Starter panel pointers
        starters = [p for p in s["panels"] if "H96_pre_vs_post_snap" in p][:2]
        starters += [p for p in s["panels"] if "_L8_" in p and "pre_snap_bc" in p][:2]
        if starters:
            lines.append("Starter panels:")
            for pth in starters:
                lines.append(f"- `{pth}`")
            lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    include_zscore = bool(args.include_zscore_ref) and not bool(args.no_zscore_ref)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{int(args.gpu)}"
    )
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: List[Mapping[str, Any]] = [HYBRID_SPEC]
    if not args.skip_etth1:
        specs.append(WN128_ETTH1_SPEC)

    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        summaries.append(
            _run_spec(
                spec,
                args=args,
                out_root=out_dir,
                device=device,
                forced_pools=args.pools,
                include_zscore=include_zscore,
            )
        )

    _write_readme(out_dir, summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8",
    )
    print(json.dumps(
        [{k: s[k] for k in ("tag", "snap_mode", "variates", "panels") if k in s}
         for s in summaries],
        indent=2,
        default=str,
    ), flush=True)
    print(f"[done] → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
