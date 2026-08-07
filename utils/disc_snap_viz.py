"""Render-only disc snap / lattice sanity visualizations.

Reuse tensors already produced by ``_snap_bundle`` (legal_levels, pre/post snap,
aligned MMPD). Do **not** rebuild window-norm / ordinal ladders here — fail fast
if required keys are missing.

Hooks (call from ablation / lean after snap):
  - ``viz_disc_snap_sanity`` — H96 + L-slice snapproof post bin-center
  - ``viz_disc_pre_post`` — pre → post-snap → BC (+ optional bin-index)
  - ``viz_gt_encode_bins`` — encode alphabet panels (separate from disc lattice)
  - ``write_disc_disagreement_viz`` — binary-correct/MMPD-wrong L-patches and vice versa
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.disc_bin_center_shift import bin_center_shift, nearest_bin_indices
from utils.disc_shared import write_json

COLORS = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}

# --viz-sanity tokens
SANITY_ALL = frozenset({"snap", "pre_post", "staged_boxes"})
SANITY_ALIASES = {
    "true": "all",
    "1": "all",
    "yes": "all",
    "snap_horizon": "snap",
    "snap_pack": "snap",
    "lattice": "pre_post",
    "prepost": "pre_post",
}
# CLI default for lean + ablation disc jobs (argparse ``default=``).
DEFAULT_VIZ_SANITY = "all"


def parse_viz_sanity(raw: Optional[str]) -> Set[str]:
    """Parse ``--viz-sanity`` into a set of hook names.

    Empty / None / false / ``none`` → empty set (off). ``true`` / ``all`` →
    ``snap`` + ``pre_post`` (CLI default). ``staged_boxes`` stays opt-in even
    under ``all`` (pipeline YAML owns staged boxes). ``encode_bins`` is a
    separate argparse flag (model rebuild; not part of ``all``).
    """
    if raw is None:
        return set()
    text = str(raw).strip().lower()
    if not text or text in ("0", "false", "no", "off", "none"):
        return set()
    parts = [p.strip() for p in re.split(r"[,|\s]+", text) if p.strip()]
    out: Set[str] = set()
    for p in parts:
        p = SANITY_ALIASES.get(p, p)
        if p == "all":
            out.update({"snap", "pre_post"})
            continue
        if p not in ("snap", "pre_post", "staged_boxes", "encode_bins"):
            raise ValueError(
                f"unknown --viz-sanity token {p!r}; "
                "want snap, pre_post, staged_boxes, encode_bins, all/true; "
                "or none/off to disable"
            )
        out.add(p)
    return out


def snap_residual(values_1d: np.ndarray, levels_1d: np.ndarray) -> float:
    vals = np.asarray(values_1d, dtype=np.float32)
    lev = np.asarray(levels_1d, dtype=np.float32)
    return float(np.abs(vals[:, None] - lev[None, :]).min(axis=1).max(initial=0.0))


def assert_snap_foil(
    snapped: Mapping[str, Any],
    *,
    expect_snap_mode: Optional[str] = None,
    expect_snap_mode_prefix: Optional[str] = None,
    expect_canvas_height: Optional[int] = None,
    label: str = "viz",
) -> None:
    """Leaf-foil asserts on pipeline snap metadata (no alternate codepaths)."""
    mode = str(snapped.get("snap_mode", ""))
    h = int(snapped.get("canvas_height", -1))
    if expect_snap_mode is not None and mode != expect_snap_mode:
        raise RuntimeError(f"{label}: expected snap_mode={expect_snap_mode!r}, got {mode!r}")
    if expect_snap_mode_prefix is not None and not mode.startswith(expect_snap_mode_prefix):
        raise RuntimeError(
            f"{label}: expected snap_mode startswith {expect_snap_mode_prefix!r}, got {mode!r}"
        )
    if expect_canvas_height is not None and h != int(expect_canvas_height):
        raise RuntimeError(
            f"{label}: expected canvas_height={expect_canvas_height}, got {h}"
        )


def plot_snap_proof_panel(
    *,
    out_path: Path,
    title: str,
    levels_1d: np.ndarray,
    series: Mapping[str, np.ndarray],
    colors: Optional[Mapping[str, str]] = None,
    t0: int = 0,
) -> Dict[str, float]:
    """Marker + occupied-rung proof that values sit on the absolute ladder."""
    colors = colors or COLORS
    names = list(series.keys())
    length = int(next(iter(series.values())).shape[0])
    x = np.arange(t0, t0 + length)
    n_rows = int(np.asarray(levels_1d).shape[0])

    residuals = {n: snap_residual(series[n], levels_1d) for n in names}
    max_err = float(max(residuals.values()))
    if max_err > 1e-5:
        raise RuntimeError(f"{title}: snap residual {max_err:.3e} — refusing to plot")

    occupied = np.unique(
        np.concatenate([np.asarray(series[n], dtype=np.float64) for n in names])
    )
    bins = {
        n: nearest_bin_indices(
            np.asarray(series[n], dtype=np.float32)[None, None, :],
            np.asarray(levels_1d, dtype=np.float32)[None, None, :],
        )[0, 0]
        for n in names
    }

    fig, (ax_y, ax_b) = plt.subplots(
        2, 1, figsize=(max(9.0, 0.55 * length + 3.5), 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.4]}, sharex=True,
    )
    ax_y.set_facecolor("white")
    for y in occupied:
        ax_y.axhline(float(y), color="0.55", lw=0.9, alpha=0.85, zorder=0)
    for n in names:
        y = np.asarray(series[n], dtype=np.float64)
        c = colors.get(n, "#333333")
        ax_y.plot(x, y, color=c, lw=1.0, alpha=0.35, zorder=1)
        ax_y.plot(
            x, y, linestyle="none", marker="o", markersize=7.5,
            markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.6,
            label=f"{n} (max|Δ|={residuals[n]:.1e})", zorder=3,
        )
    ax_y.set_ylabel("dataset-z (snapped)")
    ax_y.set_title(
        f"{title}\noccupied rungs only ({occupied.size}/{n_rows}); "
        f"all markers on ladder (max residual {max_err:.1e})",
        fontsize=10,
    )
    ax_y.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_y.grid(alpha=0.15)

    for n in names:
        c = colors.get(n, "#333333")
        ax_b.plot(x, bins[n], color=c, lw=1.0, alpha=0.35, zorder=1)
        ax_b.plot(
            x, bins[n], linestyle="none", marker="s", markersize=6.5,
            markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.5,
            label=n, zorder=3,
        )
    ax_b.set_ylabel(f"{n_rows}-row bin index")
    ax_b.set_xlabel("horizon step t")
    ax_b.set_title(
        "integer ladder row (discrete; same alphabet for GT / binary / MMPD)",
        fontsize=9,
    )
    ax_b.legend(loc="best", fontsize=8, framealpha=0.9, ncol=3)
    ax_b.grid(alpha=0.15)
    ax_b.set_yticks(sorted({int(v) for b in bins.values() for v in b.tolist()}))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "max_snap_residual": max_err,
        "n_occupied_rungs": float(occupied.size),
        **{f"residual_{n}": residuals[n] for n in names},
    }


def bin_center_slice(
    snapped: Mapping[str, np.ndarray],
    *,
    local: int,
    variate: int,
    offset: int,
    slice_len: int,
) -> Dict[str, np.ndarray]:
    """Apply live disc bin_center_shift to an L-slice (render-only)."""
    if "legal_levels" not in snapped:
        raise RuntimeError("bin_center_slice requires snapped['legal_levels'] from _snap_bundle")
    levels = np.asarray(snapped["legal_levels"])
    out: Dict[str, np.ndarray] = {}
    for name, key in (("GT", "gt"), ("binary", "binary"), ("MMPD", "mmpd")):
        if key not in snapped:
            raise RuntimeError(f"bin_center_slice missing snapped[{key!r}]")
        seg = np.asarray(snapped[key])[local, variate, offset : offset + slice_len]
        shifted, _ = bin_center_shift(
            seg[None, None, :],
            levels[local : local + 1, variate : variate + 1, :],
            reduce="per_variate",
        )
        out[name] = shifted[0, 0]
    return out


def select_window_locals(
    n: int,
    n_windows: int,
    *,
    seed: int,
) -> List[int]:
    if n < 1:
        raise RuntimeError("select_window_locals: empty pack")
    k = min(int(n_windows), n)
    rng = np.random.default_rng(int(seed) + 17)
    return sorted(int(x) for x in rng.choice(n, size=k, replace=False))


def resolve_variates(
    n_vars: int,
    viz_variates: Optional[Sequence[int]],
    *,
    default: int = 0,
) -> List[int]:
    if viz_variates is None or len(list(viz_variates)) == 0:
        v = int(default)
        if v < 0 or v >= n_vars:
            raise ValueError(f"variate={v} out of range V={n_vars}")
        return [v]
    out: List[int] = []
    for v in viz_variates:
        v = int(v)
        if v < 0 or v >= n_vars:
            raise ValueError(f"variate={v} out of range V={n_vars}")
        out.append(v)
    return out


def write_horizon96(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    locals_: Sequence[int],
    variates: Sequence[int],
    dpi: int = 140,
) -> List[Path]:
    """Full-horizon GT / binary / MMPD (post-snap) panels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    mode = str(snapped.get("snap_mode", "?"))
    paths: List[Path] = []
    for local in locals_:
        for v in variates:
            pool = int(indices[local])
            t = np.arange(gt.shape[-1])
            fig, ax = plt.subplots(figsize=(11.0, 3.6))
            ax.plot(t, gt[local, v], color=COLORS["GT"], lw=1.8, label="GT")
            ax.plot(t, binary[local, v], color=COLORS["binary"], lw=1.4, alpha=0.9, label="binary")
            ax.plot(t, mmpd[local, v], color=COLORS["MMPD"], lw=1.4, alpha=0.9, label="MMPD")
            mae_b = float(np.mean(np.abs(binary[local, v] - gt[local, v])))
            mae_m = float(np.mean(np.abs(mmpd[local, v] - gt[local, v])))
            ax.set_title(
                f"{run_name}/{dataset} pool={pool} local={local} v={v} | "
                f"H={gt.shape[-1]} snapped ({mode})  "
                f"MAE(binary)={mae_b:.3g}  MAE(MMPD)={mae_m:.3g}",
                fontsize=10,
            )
            ax.set_xlabel("horizon step t")
            ax.set_ylabel("dataset-z (snapped)")
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path = out_dir / (
                f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_H{gt.shape[-1]}.png"
            )
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(path)
    return paths


def write_snapproof_slices(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    locals_: Sequence[int],
    variates: Sequence[int],
    slice_lengths: Sequence[int],
    zoom_steps: Optional[int] = None,
    after_bin_center: bool = True,
) -> List[Path]:
    """L-slice occupied-rung snapproof (post-BC by default)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    levels = np.asarray(snapped["legal_levels"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    h = int(gt.shape[-1])
    canvas = snapped.get("canvas_height", "?")
    mode = snapped.get("snap_mode", "?")
    paths: List[Path] = []
    for local in locals_:
        pool = int(indices[local])
        for v in variates:
            levels_v = levels[local, v]
            for L in slice_lengths:
                L = int(L)
                if L > h:
                    continue
                offset = max(0, (h - L) // 2)
                if after_bin_center:
                    series = bin_center_slice(
                        snapped, local=local, variate=v, offset=offset, slice_len=L,
                    )
                    stage = "AFTER bin_center_shift"
                else:
                    series = {
                        "GT": gt[local, v, offset : offset + L],
                        "binary": binary[local, v, offset : offset + L],
                        "MMPD": mmpd[local, v, offset : offset + L],
                    }
                    stage = "post-snap (pre bin_center)"
                # Optional mid-zoom inside the L-slice.
                if zoom_steps is not None:
                    z_steps = min(int(zoom_steps), L)
                    z0 = max(0, (L - z_steps) // 2)
                    z1 = z0 + z_steps
                    series = {k: np.asarray(val)[z0:z1] for k, val in series.items()}
                    t0 = offset + z0
                    t_tag = f"t=[{offset + z0},{offset + z1})"
                else:
                    t0 = offset
                    t_tag = f"t=[{offset},{offset + L})"
                path = out_dir / (
                    f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_"
                    f"L{L}_off{offset}_snapproof.png"
                )
                plot_snap_proof_panel(
                    out_path=path,
                    title=(
                        f"{run_name}/{dataset} pool={pool} local={local} v={v} | "
                        f"disc L={L} off={offset} {t_tag} {stage} "
                        f"(dataset-z; NO instance norm; canvas{canvas} snap={mode})"
                    ),
                    levels_1d=levels_v,
                    series=series,
                    colors=COLORS,
                    t0=t0,
                )
                paths.append(path)
    return paths


def write_early_horizon_snapproof(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    locals_: Sequence[int],
    variates: Sequence[int],
    n_steps: int = 16,
) -> List[Path]:
    """Early-horizon post-snap proof (pre bin-center)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    levels = np.asarray(snapped["legal_levels"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    canvas = snapped.get("canvas_height", "?")
    mode = snapped.get("snap_mode", "?")
    paths: List[Path] = []
    z1 = min(int(n_steps), int(gt.shape[-1]))
    for local in locals_:
        pool = int(indices[local])
        for v in variates:
            path = out_dir / (
                f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_t0-{z1}_snapproof.png"
            )
            plot_snap_proof_panel(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} pool={pool} local={local} v={v} | "
                    f"post-snap (pre bin_center) t=0..{z1 - 1} "
                    f"(dataset-z; NO instance norm; canvas{canvas} snap={mode})"
                ),
                levels_1d=levels[local, v],
                series={
                    "GT": gt[local, v, :z1],
                    "binary": binary[local, v, :z1],
                    "MMPD": mmpd[local, v, :z1],
                },
                colors=COLORS,
                t0=0,
            )
            paths.append(path)
    return paths


def space_title(
    snap_mode: str,
    *,
    flat_mask: Optional[Sequence[bool]] = None,
    variate: int = 0,
) -> str:
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
        return "PRE: global dataset-z (pack storage); ladder = past mean/std → canvas rungs"
    if snap_mode == "ordinal_absolute":
        return "PRE: global dataset-z; ladder = absolute ordinal patch-refine"
    return f"PRE: pack storage ({snap_mode})"


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
        c = COLORS.get(name, "#333333")
        ax.plot(
            x, y, color=c, lw=1.15 if not markers else 0.9,
            alpha=0.85 if not markers else 0.35, zorder=1, label=name,
        )
        if markers:
            ax.plot(
                x, y, linestyle="none", marker="o", markersize=5.5,
                markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.45, zorder=3,
            )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=3)


def write_h96_stages(
    *,
    out_path: Path,
    pre: Mapping[str, np.ndarray],
    post: Mapping[str, np.ndarray],
    levels_1d: np.ndarray,
    title_prefix: str,
    space_pre: str,
    snap_mode: str,
    dpi: int = 140,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 3.8), sharey=False)
    _overlay_series(
        axes[0], series=pre, t0=0, markers=False, levels_1d=None,
        ylabel="dataset-z", title=space_pre,
    )
    _overlay_series(
        axes[1], series=post, t0=0, markers=True, levels_1d=levels_1d,
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


def write_l_stages(
    *,
    out_path: Path,
    pre: Mapping[str, np.ndarray],
    post: Mapping[str, np.ndarray],
    post_bc: Mapping[str, np.ndarray],
    levels_1d: np.ndarray,
    title_prefix: str,
    space_pre: str,
    snap_mode: str,
    offset: int,
    slice_len: int,
    dpi: int = 140,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=False)
    _overlay_series(
        axes[0], series=pre, t0=offset, markers=False, levels_1d=None,
        ylabel="dataset-z", title=f"1) PRE-SNAP\n{space_pre}",
    )
    _overlay_series(
        axes[1], series=post, t0=offset, markers=True, levels_1d=levels_1d,
        ylabel="dataset-z", title=f"2) POST-SNAP\n{snap_mode} (rungs)",
    )
    _overlay_series(
        axes[2], series=post_bc, t0=offset, markers=True, levels_1d=levels_1d,
        ylabel="dataset-z (bin-centered)",
        title="3) POST disc-norm\nbin_center_shift ONLY (LIVE)",
    )
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


def write_bin_index_compare(
    *,
    out_path: Path,
    post: Mapping[str, np.ndarray],
    post_bc: Mapping[str, np.ndarray],
    levels_1d: np.ndarray,
    title_prefix: str,
    offset: int,
    dpi: int = 140,
) -> Path:
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
            c = COLORS.get(name, "#333333")
            ax.plot(x, bins, color=c, lw=1.0, alpha=0.35)
            ax.plot(
                x, bins, linestyle="none", marker="s", markersize=5.5,
                markerfacecolor=c, markeredgecolor="white",
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


def viz_disc_snap_sanity(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, Any],
    n_windows: int = 2,
    variates: Optional[Sequence[int]] = None,
    slice_lengths: Sequence[int] = (8, 16),
    zoom_steps: int = 12,
    seed: int = 42,
    dpi: int = 140,
) -> Dict[str, List[Path]]:
    """Hook A: H96 + L snapproof post-BC + early-horizon pre-BC.

    Supersets ablation ``_write_zoom_viz``. Reuses snapped tensors only.
    """
    for key in ("gt", "binary", "mmpd", "legal_levels", "indices"):
        if key not in snapped:
            raise RuntimeError(f"viz_disc_snap_sanity missing snapped[{key!r}]")
    gt = np.asarray(snapped["gt"])
    locals_ = select_window_locals(int(gt.shape[0]), n_windows, seed=seed)
    vars_ = resolve_variates(int(gt.shape[1]), variates)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h96 = write_horizon96(
        out_dir=out_dir / "horizon96",
        run_name=run_name,
        dataset=dataset,
        snapped=snapped,
        locals_=locals_,
        variates=vars_,
        dpi=dpi,
    )
    snapproof: List[Path] = []
    for L in slice_lengths:
        snapproof.extend(
            write_snapproof_slices(
                out_dir=out_dir / f"L{int(L)}_snapproof",
                run_name=run_name,
                dataset=dataset,
                snapped=snapped,
                locals_=locals_,
                variates=vars_,
                slice_lengths=[int(L)],
                zoom_steps=None,
                after_bin_center=True,
            )
        )
    # Keep zoomed L-slice panels (matches legacy _write_zoom_viz).
    zoomed = write_snapproof_slices(
        out_dir=out_dir / "L_zoom_snapproof",
        run_name=run_name,
        dataset=dataset,
        snapped=snapped,
        locals_=locals_,
        variates=vars_,
        slice_lengths=slice_lengths,
        zoom_steps=zoom_steps,
        after_bin_center=True,
    )
    early = write_early_horizon_snapproof(
        out_dir=out_dir / "early_horizon_snapproof",
        run_name=run_name,
        dataset=dataset,
        snapped=snapped,
        locals_=locals_,
        variates=vars_,
        n_steps=16,
    )
    return {
        "horizon96": h96,
        "snapproof": snapproof,
        "zoom_snapproof": zoomed,
        "early_horizon": early,
    }


def viz_disc_pre_post(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, Any],
    n_windows: int = 2,
    variates: Optional[Sequence[int]] = None,
    slice_lengths: Sequence[int] = (8, 16),
    seed: int = 42,
    dpi: int = 140,
    require_wn128: bool = False,
    flat_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, List[Path]]:
    """Hook B: pre → post-snap → BC panels (+ lattice residual assert for wn128)."""
    for key in ("gt", "binary", "mmpd", "gt_pre", "binary_pre", "mmpd_pre", "legal_levels", "indices"):
        if key not in snapped:
            raise RuntimeError(
                f"viz_disc_pre_post missing snapped[{key!r}] "
                "(extend _snap_bundle to return pre-snap tensors)"
            )
    mode = str(snapped.get("snap_mode", ""))
    canvas = int(snapped.get("canvas_height", -1))
    if require_wn128:
        assert_snap_foil(
            snapped,
            expect_snap_mode_prefix="window_norm_grid",
            expect_canvas_height=128,
            label="viz_disc_pre_post/wn128",
        )

    gt = np.asarray(snapped["gt"])
    locals_ = select_window_locals(int(gt.shape[0]), n_windows, seed=seed)
    vars_ = resolve_variates(int(gt.shape[1]), variates)
    levels = np.asarray(snapped["legal_levels"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h96_paths: List[Path] = []
    l_paths: List[Path] = []
    bin_paths: List[Path] = []
    residual_proofs: List[Path] = []

    for local in locals_:
        pool = int(indices[local])
        for v in vars_:
            levels_v = levels[local, v]
            pre = {
                "GT": np.asarray(snapped["gt_pre"])[local, v],
                "binary": np.asarray(snapped["binary_pre"])[local, v],
                "MMPD": np.asarray(snapped["mmpd_pre"])[local, v],
            }
            post = {
                "GT": gt[local, v],
                "binary": np.asarray(snapped["binary"])[local, v],
                "MMPD": np.asarray(snapped["mmpd"])[local, v],
            }
            # Lattice residual proof on post-snap (no second ladder build).
            for name, arr in post.items():
                err = snap_residual(arr, levels_v)
                if err > 1e-5:
                    raise RuntimeError(
                        f"viz_disc_pre_post {run_name}/{dataset} local={local} v={v} "
                        f"{name} residual={err:.3e} (not on legal_levels)"
                    )

            prefix = f"{run_name}/{dataset} pool={pool} local={local} v={v}"
            space = space_title(mode, flat_mask=flat_mask, variate=v)
            p_h = out_dir / "H96_stages" / (
                f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_H_stages.png"
            )
            write_h96_stages(
                out_path=p_h,
                pre=pre,
                post=post,
                levels_1d=levels_v,
                title_prefix=prefix,
                space_pre=space,
                snap_mode=mode,
                dpi=dpi,
            )
            h96_paths.append(p_h)

            # Early-horizon post-snap markers (lattice proof panel).
            z1 = min(16, int(gt.shape[-1]))
            p_res = out_dir / "lattice_proof" / (
                f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_t0-{z1}_lattice.png"
            )
            plot_snap_proof_panel(
                out_path=p_res,
                title=f"{prefix} | post-snap lattice proof t=0..{z1 - 1} ({mode})",
                levels_1d=levels_v,
                series={k: arr[:z1] for k, arr in post.items()},
                colors=COLORS,
                t0=0,
            )
            residual_proofs.append(p_res)

            h = int(gt.shape[-1])
            for L in slice_lengths:
                L = int(L)
                if L > h:
                    continue
                off = max(0, (h - L) // 2)
                pre_l = {k: arr[off : off + L] for k, arr in pre.items()}
                post_l = {k: arr[off : off + L] for k, arr in post.items()}
                post_bc = bin_center_slice(
                    snapped, local=local, variate=v, offset=off, slice_len=L,
                )
                p_l = out_dir / f"L{L}_stages" / (
                    f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_"
                    f"L{L}_off{off}_stages.png"
                )
                write_l_stages(
                    out_path=p_l,
                    pre=pre_l,
                    post=post_l,
                    post_bc=post_bc,
                    levels_1d=levels_v,
                    title_prefix=prefix,
                    space_pre=space,
                    snap_mode=mode,
                    offset=off,
                    slice_len=L,
                    dpi=dpi,
                )
                l_paths.append(p_l)
                p_b = out_dir / f"L{L}_bin_index" / (
                    f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_"
                    f"L{L}_off{off}_bins.png"
                )
                write_bin_index_compare(
                    out_path=p_b,
                    post=post_l,
                    post_bc=post_bc,
                    levels_1d=levels_v,
                    title_prefix=prefix,
                    offset=off,
                    dpi=dpi,
                )
                bin_paths.append(p_b)

    return {
        "H96_stages": h96_paths,
        "L_stages": l_paths,
        "bin_index": bin_paths,
        "lattice_proof": residual_proofs,
    }


def plot_gt_bins(
    *,
    out_path: Path,
    title: str,
    t: np.ndarray,
    series: Mapping[str, np.ndarray],
    ylabel: str = "model-space (encode alphabet)",
    dpi: int = 140,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.plot(
        t, series["gt_norm"], color="#212121", lw=1.6, alpha=0.85,
        label="GT (model-space, pre-bin)", zorder=3,
    )
    ax.plot(
        t, series["coarse"], color="#E65100", lw=1.8, solid_capstyle="round",
        label="coarse GT bins (solid)", zorder=4,
    )
    ax.plot(
        t, series["fine_refined"], color="#1565C0", lw=1.5, linestyle=":",
        label="fine-refined GT (coarse+residual, dotted)", zorder=5,
    )
    if "fine_hir" in series:
        ax.plot(
            t, series["fine_hir"], color="#2E7D32", lw=1.2, linestyle=":",
            alpha=0.9, label="fine HIR bins (dotted)", zorder=4,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("horizon step t")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def viz_gt_encode_bins(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    past: np.ndarray,
    gt: np.ndarray,
    indices: np.ndarray,
    encode_fn: Callable[[np.ndarray, np.ndarray, int], Mapping[str, np.ndarray]],
    n_windows: int = 2,
    variates: Optional[Sequence[int]] = None,
    slice_lengths: Sequence[int] = (8, 16),
    seed: int = 42,
    dpi: int = 140,
    snap_mode: str = "?",
    ylabel: str = "model-space (encode alphabet; NOT disc lattice)",
) -> Dict[str, List[Path]]:
    """Hook C: GT encode alphabet panels.

    ``encode_fn(past_1, future_1, variate) -> series dict`` with keys
    gt_norm / coarse / fine_refined [/ fine_hir]. Separate from disc snap.
    """
    past = np.asarray(past, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    locals_ = select_window_locals(int(gt.shape[0]), n_windows, seed=seed)
    vars_ = resolve_variates(int(gt.shape[1]), variates)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, List[Path]] = {"horizon96": []}
    for L in slice_lengths:
        written[f"L{int(L)}"] = []

    for local in locals_:
        pool = int(indices[local])
        for v in vars_:
            series = dict(encode_fn(past[local : local + 1], gt[local : local + 1], int(v)))
            for req in ("gt_norm", "coarse", "fine_refined"):
                if req not in series:
                    raise RuntimeError(f"viz_gt_encode_bins encode_fn missing {req!r}")
            h = int(np.asarray(series["gt_norm"]).shape[-1])
            t = np.arange(h)
            path = out_dir / "horizon96" / (
                f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_H{h}_gt_bins.png"
            )
            plot_gt_bins(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} pool={pool} local={local} v={v} | "
                    f"GT encode alphabet (snap_mode={snap_mode}; NOT disc lattice)"
                ),
                t=t,
                series=series,
                ylabel=ylabel,
                dpi=dpi,
            )
            written["horizon96"].append(path)
            for L in slice_lengths:
                L = int(L)
                if L > h:
                    continue
                off = max(0, (h - L) // 2)
                zoom = {
                    k: (np.asarray(val)[off : off + L] if isinstance(val, np.ndarray) else val)
                    for k, val in series.items()
                }
                zpath = out_dir / f"L{L}" / (
                    f"{run_name}_{dataset}_v{v}_local{local}_pool{pool}_"
                    f"L{L}_off{off}_gt_bins.png"
                )
                plot_gt_bins(
                    out_path=zpath,
                    title=(
                        f"{run_name}/{dataset} pool={pool} v={v} | "
                        f"L={L} off={off} GT encode bins"
                    ),
                    t=np.arange(off, off + L),
                    series=zoom,
                    ylabel=ylabel,
                    dpi=dpi,
                )
                written[f"L{L}"].append(zpath)
    return written


def flatten_viz_paths(groups: Mapping[str, Sequence[Path]]) -> List[Path]:
    out: List[Path] = []
    for paths in groups.values():
        out.extend(Path(p) for p in paths)
    return out


def disc_score_index(
    scores: Mapping[str, np.ndarray],
) -> Dict[Tuple[int, int, int, int], Dict[str, float]]:
    """Map (window, offset, variate, label) → {prob_fake, pred, correct}."""
    out: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}
    n = int(scores["label"].shape[0])
    for i in range(n):
        label = int(scores["label"][i])
        prob = float(scores["prob_fake"][i])
        pred = 1 if prob >= 0.5 else 0
        key = (
            int(scores["window"][i]),
            int(scores["offset"][i]),
            int(scores["variate"][i]),
            label,
        )
        out[key] = {"prob_fake": prob, "pred": float(pred), "correct": float(pred == label)}
    return out


def plot_disagreement_panel(
    *,
    out_path: Path,
    title: str,
    past_1d: Optional[np.ndarray],
    gt_1d: np.ndarray,
    binary_1d: np.ndarray,
    mmpd_1d: np.ndarray,
    binary_prob: float,
    mmpd_prob: float,
    label: int,
    offset: int,
) -> None:
    """GT / binary / MMPD L-slice (±lookback) with disc P(fake) annotations."""
    L = int(gt_1d.shape[0])
    t_h = np.arange(offset, offset + L)
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    if past_1d is not None and past_1d.size:
        t_past = np.arange(offset - int(past_1d.shape[0]), offset)
        ax.plot(t_past, past_1d, color="#555555", lw=1.2, label="lookback", alpha=0.85)
        ax.axvline(offset, color="black", ls="--", lw=0.8, alpha=0.45)
    ax.plot(t_h, gt_1d, color="black", lw=2.0, label="GT")
    ax.plot(
        t_h, binary_1d, color="#1f77b4", lw=1.8, alpha=0.9,
        label=f"binary (Pfake={binary_prob:.2f})",
    )
    ax.plot(
        t_h, mmpd_1d, color="#d62728", lw=1.8, alpha=0.9,
        label=f"MMPD (Pfake={mmpd_prob:.2f})",
    )
    ax.axvspan(offset, offset + L - 1, color="#ffe08a", alpha=0.25, zorder=0)
    shown = "FAKE" if label == 1 else "REAL"
    ax.set_title(f"{title}\nshown_to_disc={shown} (label={label})", fontsize=10)
    ax.set_xlabel("horizon step t")
    ax.set_ylabel("dataset-z (snapped)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_disc_disagreement_viz(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    slice_len: int,
    snapped: Mapping[str, np.ndarray],
    binary_scores: Mapping[str, np.ndarray],
    mmpd_scores: Mapping[str, np.ndarray],
    include_past: bool,
    max_panels: int,
    seed: int,
) -> Dict[str, Any]:
    """Panels where one source's disc is correct and the other's is wrong.

    Keys align on (window, offset, variate, label). For label=0 both discs see
    GT; for label=1 each sees its own fake. Cap panels per direction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_ix = disc_score_index(binary_scores)
    mmpd_ix = disc_score_index(mmpd_scores)
    shared = sorted(set(bin_ix) & set(mmpd_ix))
    dirs = {
        "mmpd_wrong_binary_right": [],
        "binary_wrong_mmpd_right": [],
    }
    for key in shared:
        b = bin_ix[key]
        m = mmpd_ix[key]
        if b["correct"] >= 0.5 and m["correct"] < 0.5:
            dirs["mmpd_wrong_binary_right"].append(key)
        elif b["correct"] < 0.5 and m["correct"] >= 0.5:
            dirs["binary_wrong_mmpd_right"].append(key)

    rng = np.random.default_rng(int(seed) + int(slice_len) * 17)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    past = np.asarray(snapped["past"])
    L = int(slice_len)
    lookback_tail = min(32, int(past.shape[-1])) if include_past else 0
    paths: Dict[str, List[str]] = {}
    counts = {k: len(v) for k, v in dirs.items()}

    for direction, keys in dirs.items():
        def _wrong_margin(k: Tuple[int, int, int, int], _dir: str = direction) -> float:
            if _dir.startswith("mmpd_wrong"):
                return abs(float(mmpd_ix[k]["prob_fake"]) - 0.5)
            return abs(float(bin_ix[k]["prob_fake"]) - 0.5)

        keys_sorted = sorted(keys, key=_wrong_margin, reverse=True)
        n = min(int(max_panels), len(keys_sorted))
        if n < len(keys_sorted):
            top = keys_sorted[: max(1, n // 2)]
            rest = keys_sorted[max(1, n // 2) :]
            extra = n - len(top)
            if extra > 0 and rest:
                pick = rng.choice(len(rest), size=min(extra, len(rest)), replace=False)
                top.extend([rest[int(i)] for i in np.atleast_1d(pick)])
            chosen = top[:n]
        else:
            chosen = keys_sorted

        dir_paths: List[str] = []
        for i, (window, offset, variate, label) in enumerate(chosen):
            past_1d = None
            if lookback_tail > 0:
                past_1d = past[window, variate, -lookback_tail:]
            path = out_dir / (
                f"{run_name}_{dataset}_L{L}_{direction}_"
                f"w{window}_off{offset}_v{variate}_lab{label}_{i:02d}.png"
            )
            plot_disagreement_panel(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} L={L} {direction} | "
                    f"w={window} off={offset} v={variate}"
                ),
                past_1d=past_1d,
                gt_1d=gt[window, variate, offset : offset + L],
                binary_1d=binary[window, variate, offset : offset + L],
                mmpd_1d=mmpd[window, variate, offset : offset + L],
                binary_prob=float(bin_ix[(window, offset, variate, label)]["prob_fake"]),
                mmpd_prob=float(mmpd_ix[(window, offset, variate, label)]["prob_fake"]),
                label=int(label),
                offset=int(offset),
            )
            dir_paths.append(str(path))
        paths[direction] = dir_paths
        print(
            f"[disc-disagree] {run_name}/{dataset}/L{L} {direction}: "
            f"pool={counts[direction]} wrote={len(dir_paths)} → {out_dir}",
            flush=True,
        )

    manifest = {
        "run": run_name,
        "dataset": dataset,
        "slice_len": L,
        "n_shared_keys": len(shared),
        "counts": counts,
        "paths": paths,
    }
    write_json(out_dir / f"manifest_L{L}.json", manifest)
    return manifest
