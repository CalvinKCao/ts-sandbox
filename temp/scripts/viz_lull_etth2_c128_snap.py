# Pipeline integration: Prefer --viz-sanity snap,pre_post + --viz-variates 5; encode via --viz-encode-bins.
#!/usr/bin/env python3
"""LULL (ETTh2 v=5) window_norm_grid canvas128 snap viz — foil to ordinal absolute.

Uses the recent non-hybrid canvas128 leaf (train **4601319**). Fail-fast if an
ordinal absolute ladder or hybrid flat mask is selected.

Panels (under ``temp/viz_lull_etth2_c128_snap/``):

1. ``pre_post_snap/LULL_v5/`` — GT / binary / MMPD pre-snap vs post-snap onto the
   128-row window_norm training lattice; L=8/16 pre→snap→bin_center
   (+ optional zscore NOT-live); bin-index panels.
2. ``horizon96/`` + ``L8_snapproof/`` / ``L16_snapproof/`` — forecast overlays
   and occupied-rung snapproof AFTER ``bin_center_shift``.
3. ``gt_coarse_fine/`` — GT coarse (solid) vs fine-refined (dotted) on the
   window-norm encode path for this leaf (not ordinal, not hybrid flat).

Default sources:

  pack  results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar
  ckpt  results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2
  cfg   configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (  # noqa: E402
    stage_state,
)
from models.diffusion_tsf.pipeline.state import PipelineState  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    create_diffusion_model,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _flat_mask_from_ckpt,
    _ladder_only,
    _max_scale_from_ckpt_metadata,
    _plot_snap_proof_panel,
    _snap_bundle,
    load_ablation_run,
)
from temp.scripts.viz_pre_post_snap_norm import (  # noqa: E402
    _write_bin_index_compare,
    _write_h96_stages,
    _write_l_stages,
    _zscore_1d,
)
from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.disc_shared import (  # noqa: E402
    binary_mmpd_train_scaler_map,
)
from utils.eval_mmpd_gaussian_anchor import DEFAULT_MMPD_DATA  # noqa: E402
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402
from utils.patch_refine_ordinal_ladder import snap_to_patch_refine_levels  # noqa: E402
from utils.visualize_staged_eval_2d_preds import _build_state  # noqa: E402

LULL_VARIATE = 5
DATASET = "ETTh2"
RUN_NAME = "window_norm_c128"
# Display blurb — must not be confused with ordinal_absolute (dataset-z ranked ladder).
WN128_LATTICE = (
    "window_norm 128-grid (uniform in window-z → affine to dataset-z)"
)
DEFAULT_PACK = (
    REPO_ROOT
    / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
)
DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
DEFAULT_OUT = REPO_ROOT / "temp" / "viz_lull_etth2_c128_snap"
# Disc-disagreement LULL (v5) windows from the 2009 pack (L8 mmpd_wrong_binary_right).
PREFERRED_POOLS = (1169, 1393, 1310)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-root", type=Path, default=DEFAULT_PACK)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--variate", type=int, default=LULL_VARIATE)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--pools", type=int, nargs="+", default=None)
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--skip-gt-bins", action="store_true")
    p.add_argument(
        "--include-zscore-ref",
        action="store_true",
        default=True,
        help="Add NOT-live zscore column on L-stage panels (default on).",
    )
    p.add_argument("--no-zscore-ref", action="store_true")
    args = p.parse_args()
    args.pack_root = args.pack_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    if args.no_zscore_ref:
        args.include_zscore_ref = False
    return args


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


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _select_locals(
    *,
    indices: np.ndarray,
    gt: np.ndarray,
    binary: np.ndarray,
    mmpd: np.ndarray,
    variate: int,
    n_windows: int,
    forced_pools: Sequence[int] | None,
    seed: int,
) -> List[int]:
    """Prefer forced / known LULL disc pools, then high |binary−MMPD|, then MAE."""
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

    if forced_pools:
        for pool in forced_pools:
            add_pool(int(pool))
    else:
        for pool in PREFERRED_POOLS:
            add_pool(int(pool))

    disagree = np.mean(np.abs(binary[:, variate] - mmpd[:, variate]), axis=-1)
    for local in np.argsort(-disagree).tolist():
        if len(chosen) >= n_windows:
            break
        if local in seen:
            continue
        seen.add(local)
        chosen.append(int(local))

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


def _bin_center_slice(
    snapped: Mapping[str, np.ndarray],
    *,
    local: int,
    variate: int,
    offset: int,
    slice_len: int,
) -> Dict[str, np.ndarray]:
    levels = np.asarray(snapped["legal_levels"])
    out: Dict[str, np.ndarray] = {}
    for name, key in (("GT", "gt"), ("binary", "binary"), ("MMPD", "mmpd")):
        seg = np.asarray(snapped[key])[local, variate, offset : offset + slice_len]
        shifted, _ = bin_center_shift(
            seg[None, None, :],
            levels[local : local + 1, variate : variate + 1, :],
            reduce="per_variate",
        )
        out[name] = shifted[0, 0]
    return out


def _write_horizon96(
    *,
    out_dir: Path,
    snapped: Mapping[str, Any],
    locals_: Sequence[int],
    variate: int,
    dpi: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(snapped["gt"])
    binary = np.asarray(snapped["binary"])
    mmpd = np.asarray(snapped["mmpd"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    paths: List[Path] = []
    for local in locals_:
        pool = int(indices[local])
        t = np.arange(gt.shape[-1])
        fig, ax = plt.subplots(figsize=(11.0, 3.6))
        ax.plot(t, gt[local, variate], color="black", lw=1.8, label="GT")
        ax.plot(t, binary[local, variate], color="#1f77b4", lw=1.4, alpha=0.9, label="binary")
        ax.plot(t, mmpd[local, variate], color="#d62728", lw=1.4, alpha=0.9, label="MMPD")
        mae_b = float(np.mean(np.abs(binary[local, variate] - gt[local, variate])))
        mae_m = float(np.mean(np.abs(mmpd[local, variate] - gt[local, variate])))
        ax.set_title(
            f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} local={local} | "
            f"H={gt.shape[-1]} snapped onto {WN128_LATTICE}  "
            f"[mode={snapped.get('snap_mode')}]  "
            f"MAE(binary)={mae_b:.3g}  MAE(MMPD)={mae_m:.3g}",
            fontsize=9,
        )
        ax.set_xlabel("horizon step t")
        ax.set_ylabel("dataset-z (post wn128 lattice snap; NOT ordinal absolute)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = out_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_H{gt.shape[-1]}.png"
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_snapproof(
    *,
    out_dir: Path,
    snapped: Mapping[str, Any],
    locals_: Sequence[int],
    variate: int,
    slice_len: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}
    levels = np.asarray(snapped["legal_levels"])
    indices = np.asarray(snapped["indices"], dtype=np.int64)
    h = int(np.asarray(snapped["gt"]).shape[-1])
    offset = max(0, (h - slice_len) // 2)
    paths: List[Path] = []
    for local in locals_:
        pool = int(indices[local])
        series = _bin_center_slice(
            snapped, local=local, variate=variate, offset=offset, slice_len=slice_len,
        )
        path = out_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_"
            f"L{slice_len}_off{offset}_snapproof.png"
        )
        title = (
            f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} local={local} | "
            f"L={slice_len} off={offset} AFTER bin_center_shift | "
            f"{WN128_LATTICE} [mode={snapped.get('snap_mode')}]"
        )
        _plot_snap_proof_panel(
            out_path=path,
            title=title,
            levels_1d=levels[local, variate],
            series=series,
            colors=colors,
            t0=offset,
        )
        paths.append(path)
    return paths


def _build_window_norm_encode_model(
    *,
    ckpt: Path,
    config_path: str,
    max_scale: float,
    lookback: int,
    horizon: int,
    n_variates: int,
    device: torch.device,
) -> torch.nn.Module:
    """Encode/decode-only DiffusionTSF on the non-hybrid window_norm path."""
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": DATASET})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(ckpt.resolve())
    state.dataset = DATASET
    state.subset_id = DATASET
    if bool(state.use_ordinal_window_norm):
        raise RuntimeError(
            f"{config_path}: use_ordinal_window_norm must be False "
            "(refusing ordinal ladder for this foil)"
        )
    if not bool(state.use_window_normalization):
        raise RuntimeError(
            f"{config_path}: use_window_normalization must be True for window_norm_grid"
        )
    if bool(getattr(state, "hybrid_flat_dataset_norm", False)):
        raise RuntimeError(
            f"{config_path}: hybrid_flat_dataset_norm=True — refuse hybrid for this foil"
        )
    pipeline_mod.GLOBAL_ORDINAL_LADDER = None
    state.extra.pop("global_ordinal_ladder", None)
    state.extra.pop("hybrid_flat_norm_stats", None)
    state = stage_state(state, "coarse", honor_dataset_windows=True)
    pipeline_mod.DISABLE_CROSS_ATTENTION = True
    pipeline_mod.USE_GUIDANCE_CHANNEL = False
    pipeline_mod.USE_ORDINAL_WINDOW_NORM = False
    pipeline_mod.GLOBAL_ORDINAL_LADDER = None
    model = create_diffusion_model(
        n_variates=n_variates,
        lookback=lookback,
        horizon=horizon,
        guidance_model=None,
        diffusion_stage="coarse",
        use_guidance_channel=False,
    ).to(device)
    if bool(getattr(model.config, "use_ordinal_window_norm", False)):
        raise RuntimeError("create_diffusion_model enabled ordinal window norm")
    if not bool(getattr(model.config, "use_window_normalization", False)):
        raise RuntimeError("create_diffusion_model did not enable window normalization")
    model.config.skip_window_norm_variate_mask = None
    model.config.hybrid_flat_dataset_norm = False
    model.config.max_scale = float(max_scale)
    model.to_2d.max_scale = float(max_scale)
    if model._uses_global_ordinal_encoding():
        raise RuntimeError("model is on global ordinal encode path (wrong foil)")
    model.eval()
    return model


@torch.no_grad()
def _encode_gt_bins_window_norm(
    model: torch.nn.Module,
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    variate: int,
) -> Dict[str, np.ndarray]:
    """Window-norm encode GT → staged coarse/fine in model space."""
    past_norm, future_norm, stats = model._normalize_sequence(past, future)
    assert future_norm is not None
    maps = model._encode_staged_maps(future_norm)
    coarse_h = int(model.config.coarse_image_height)
    coarse_1d = model._decode_coarse_1d_from_map(maps["coarse"], cdf_decoder="mean")
    fine_res = model._decode_fine_1d_from_map(
        maps["fine"], coarse_height=coarse_h, cdf_decoder="mean",
    )
    if coarse_1d.dim() == 2:
        coarse_1d = coarse_1d.unsqueeze(1)
    if fine_res.dim() == 2:
        fine_res = fine_res.unsqueeze(1)
    combined = coarse_1d + fine_res

    canvas_h = int(getattr(model.config, "patch_refine_canvas_height", 0) or 0)
    hir_1d = None
    if canvas_h > 0:
        hir = model._encode_absolute_future_hir(future_norm, canvas_h)
        hir_1d = model._decode_absolute_future_hir(hir)

    h = int(future.shape[-1])

    def _trim(x: torch.Tensor) -> np.ndarray:
        y = x[0, variate].detach().cpu().numpy().astype(np.float32)
        if y.shape[-1] > h:
            y = y[-h:]
        return y

    gt_norm = future_norm[0, variate].detach().cpu().numpy().astype(np.float32)
    if gt_norm.shape[-1] > h:
        gt_norm = gt_norm[-h:]

    out = {
        "gt_norm": gt_norm,
        "coarse": _trim(coarse_1d),
        "fine_refined": _trim(combined),
        "center": float(stats[0][0, variate, 0].item()),
        "std": float(stats[1][0, variate, 0].item()),
    }
    if hir_1d is not None:
        out["fine_hir"] = _trim(hir_1d)
    return out


def _plot_gt_bins(
    *,
    out_path: Path,
    title: str,
    t: np.ndarray,
    series: Mapping[str, np.ndarray],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.plot(
        t, series["gt_norm"], color="#212121", lw=1.6, alpha=0.85,
        label="GT (window-norm model-space, pre-bin)", zorder=3,
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
            alpha=0.9, label="fine absolute HIR bins (dotted)", zorder=4,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("horizon step t")
    ax.set_ylabel("window-z model-space (wn128 encode; NOT ordinal ranked)")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_gt_bins(
    *,
    out_dir: Path,
    past_all: np.ndarray,
    gt_all: np.ndarray,
    indices: np.ndarray,
    locals_: Sequence[int],
    variate: int,
    model: torch.nn.Module,
    slice_lengths: Sequence[int],
    device: torch.device,
    dpi: int,
) -> Dict[str, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h96_dir = out_dir / "horizon96"
    written: Dict[str, List[str]] = {"horizon96": []}
    for L in slice_lengths:
        written[f"L{int(L)}"] = []

    canvas_h = int(getattr(model.config, "patch_refine_canvas_height", 0) or 0)
    for local in locals_:
        pool = int(indices[local])
        past_t = torch.from_numpy(past_all[local : local + 1]).to(device)
        fut_t = torch.from_numpy(gt_all[local : local + 1]).to(device)
        series = _encode_gt_bins_window_norm(model, past_t, fut_t, variate=variate)
        h = int(series["gt_norm"].shape[-1])
        t = np.arange(h)
        path = h96_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_H{h}_gt_bins.png"
        )
        _plot_gt_bins(
            out_path=path,
            title=(
                f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} | "
                f"GT encode on {WN128_LATTICE} "
                f"(Hc={model.config.coarse_image_height}, canvas={canvas_h}; "
                f"NOT ordinal)  center={series['center']:.3g} std={series['std']:.3g}"
            ),
            t=t,
            series=series,
            dpi=dpi,
        )
        written["horizon96"].append(_rel(path))

        for L in slice_lengths:
            L = int(L)
            if L > h:
                continue
            off = max(0, (h - L) // 2)
            zoom = {
                k: (np.asarray(v)[off : off + L] if isinstance(v, np.ndarray) else v)
                for k, v in series.items()
            }
            zpath = out_dir / f"L{L}" / (
                f"{RUN_NAME}_{DATASET}_LULL_v{variate}_local{local}_pool{pool}_"
                f"L{L}_off{off}_gt_bins.png"
            )
            _plot_gt_bins(
                out_path=zpath,
                title=(
                    f"{RUN_NAME}/{DATASET} LULL v={variate} pool={pool} | "
                    f"L={L} off={off} GT bins on {WN128_LATTICE} (NOT ordinal)"
                ),
                t=np.arange(off, off + L),
                series=zoom,
                dpi=dpi,
            )
            written[f"L{L}"].append(_rel(zpath))
    return written


def _write_readme(out_dir: Path, meta: Mapping[str, Any]) -> None:
    starter = meta.get("starter_pngs") or []
    lines = [
        "# LULL (ETTh2 v=5) window_norm_grid canvas128 snap viz",
        "",
        "Generated by `temp/scripts/viz_lull_etth2_c128_snap.py`.",
        "",
        "Foil to ordinal absolute (`temp/viz_lull_ordinal_fine_snap`, snap_mode "
        "`ordinal_absolute`, canvas 256) and hybrid flat "
        "(`temp/viz_lull_etth2_hybrid`, `window_norm_grid_hybrid_flat`).",
        "",
        "## Snap contract",
        "",
        f"- snap_mode: `{meta['snap_mode']}` (must be exactly `window_norm_grid`)",
        f"- canvas_height: **{meta['canvas_height']}**",
        f"- max_scale: **{meta['max_scale']}** (from ckpt metadata)",
        f"- use_ordinal_window_norm: false",
        f"- use_window_normalization: true",
        f"- hybrid_flat: false (no flat_variate_mask)",
        f"- Ladder: **{WN128_LATTICE}** via "
        "`legal_window_norm_patch_refine_levels_dataset_z` "
        "(NOT ordinal absolute / NOT hybrid flat).",
        "",
        "## Sources",
        "",
        f"- pack: `{meta['pack']}`",
        f"- ckpt: `{meta['ckpt']}` (train job **4601319**)",
        f"- config: `{meta['config']}`",
        f"- binary: `{meta['binary']}`",
        f"- mmpd: `{meta['mmpd']}` (indices aligned to binary test pool)",
        f"- kind: `{meta['kind']}` (patch_refine)",
        f"- fallback_note: {meta.get('fallback_note', 'none')}",
        "",
        "## Layout",
        "",
        "- `pre_post_snap/LULL_v5/` — H=96 pre vs post-snap; L=8/16 pre→snap→bin_center "
        "(+ optional zscore NOT-live); bin-index panels",
        "- `horizon96/` — full H=96 GT / binary / MMPD (snapped dataset-z)",
        "- `L8_snapproof/` / `L16_snapproof/` — mid-horizon AFTER `bin_center_shift`",
        "- `gt_coarse_fine/` — GT coarse (solid) vs fine-refined (dotted) window-norm encode",
        "",
        f"## Panels (variate={meta['variate']} LULL)",
        "",
        f"- pools: {meta['pools']}",
        f"- locals: {meta['locals']}",
        f"- pre_post panels: {meta.get('n_pre_post', '?')}",
        f"- H96: {meta.get('n_h96', '?')}",
        f"- L8 snapproof: {meta.get('n_l8', '?')}",
        f"- L16 snapproof: {meta.get('n_l16', '?')}",
        "",
        "## Contrast",
        "",
        "- vs ordinal_fine: absolute ladder / canvas256 / `ordinal_absolute` (ETTh1 leaf)",
        "- vs hybrid 4609805: LULL is flat (dataset affine only) → "
        "`window_norm_grid_hybrid_flat`; this leaf applies real past mean/std window-norm "
        "to LULL as well",
        "",
        "## Open these first",
        "",
    ]
    for s in starter:
        lines.append(f"- `{s}`")
    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```bash",
            "source .venv/bin/activate",
            "python temp/scripts/viz_lull_etth2_c128_snap.py --cpu --n-windows 4",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{int(args.gpu)}"
    )
    raw_dir = args.pack_root / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw/ under {args.pack_root}")
    if not args.ckpt.is_dir():
        raise FileNotFoundError(f"missing ckpt: {args.ckpt}")

    binary_path = _find_pack(raw_dir, "binary_", DATASET)
    mmpd_path = _find_pack(raw_dir, "mmpd_", DATASET)
    binary_pack = _load_npz(binary_path)
    mmpd_pack = _load_npz(mmpd_path)

    run, _stages, kind = load_ablation_run(DATASET, args.ckpt)
    if kind not in ("fine", "patch_refine"):
        raise RuntimeError(f"expected fine|patch_refine, got {kind}")

    state = _build_state(args.ckpt, DATASET, DATASET, args.config)
    if bool(state.use_ordinal_window_norm):
        raise RuntimeError(
            f"{args.config}: use_ordinal_window_norm=True — refusing "
            "(this viz is the window_norm_grid foil; not ordinal absolute)"
        )
    if not bool(state.use_window_normalization):
        raise RuntimeError(
            f"{args.config}: use_window_normalization=False — wrong ladder family"
        )
    flat_mask = _flat_mask_from_ckpt(args.ckpt, DATASET)
    if flat_mask is not None and any(flat_mask):
        raise RuntimeError(
            f"fail-fast: ckpt has hybrid flat_variate_mask={flat_mask}; "
            "use temp/scripts/viz_lull_etth2_hybrid.py for hybrid 4609805"
        )
    if bool(getattr(state, "hybrid_flat_dataset_norm", False)):
        raise RuntimeError(
            f"{args.config}: hybrid_flat_dataset_norm=True — wrong leaf for this foil"
        )

    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))
    canvas_height = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if "canvas_height" in binary_pack:
        canvas_height = int(np.asarray(binary_pack["canvas_height"]).reshape(-1)[0])
    if canvas_height != 128:
        raise RuntimeError(
            f"expected window_norm canvas_height=128, got {canvas_height} "
            "(ordinal canvas256 packs are the wrong foil)"
        )

    # Ladder unused for window_norm snap; keep None so ordinal path cannot sneak in.
    ladder = None
    if bool(state.use_ordinal_window_norm):
        ladder = _ladder_only(
            dataset=DATASET, run=run, lookback=int(args.lookback), horizon=int(args.horizon),
        )

    snap_args = SimpleNamespace(
        fake_agg=args.fake_agg,
        mmpd_data_dir=args.mmpd_data_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        dataset=DATASET,
        pack_test_stride=4,
    )
    print(
        f"[LULL-wn128] snap N={binary_pack['y_true'].shape[0]} "
        f"V={binary_pack['y_true'].shape[1]} kind={kind} canvas={canvas_height} "
        f"max_scale={max_scale} binary={binary_path.name} mmpd={mmpd_path.name} "
        f"device={device}",
        flush=True,
    )
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=ladder,
        args=snap_args,
        device=device,
        canvas_height=canvas_height,
        ckpt_root=args.ckpt,
        config_path=args.config,
    )
    snap_mode = str(snapped.get("snap_mode") or "")
    if snap_mode != "window_norm_grid":
        raise RuntimeError(
            f"fail-fast: expected snap_mode=window_norm_grid, got {snap_mode!r} "
            "(ordinal_absolute / hybrid_flat are wrong for this foil)"
        )
    if "ordinal" in snap_mode:
        raise RuntimeError(f"fail-fast: ordinal ladder selected ({snap_mode})")

    past = np.asarray(binary_pack["past"], dtype=np.float32)
    gt_pre = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pre = np.asarray(reduce_pack_forecast(binary_pack, agg=args.fake_agg), dtype=np.float32)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)
    mmpd_raw = np.asarray(reduce_pack_forecast(mmpd_pack, agg=args.fake_agg), dtype=np.float32)
    scalers = binary_mmpd_train_scaler_map(snap_args, run)
    mmpd_pre, _align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=gt_pre,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_raw,
        **scalers,
    )
    levels = np.asarray(snapped["legal_levels"], dtype=np.float32)
    gt_post = np.asarray(snapped["gt"])
    binary_post = np.asarray(snapped["binary"])
    mmpd_post = np.asarray(snapped["mmpd"])
    for name, pre, post in (
        ("GT", gt_pre, gt_post),
        ("binary", binary_pre, binary_post),
        ("MMPD", mmpd_pre, mmpd_post),
    ):
        re, _ = snap_to_patch_refine_levels(pre, levels)
        err = float(np.max(np.abs(re - post)))
        if err > 1e-5:
            raise RuntimeError(f"{name}: re-snap vs bundle mismatch max_err={err}")

    v = int(args.variate)
    if v < 0 or v >= gt_post.shape[1]:
        raise ValueError(f"variate={v} out of range V={gt_post.shape[1]}")

    locals_ = _select_locals(
        indices=np.asarray(snapped["indices"], dtype=np.int64),
        gt=gt_post,
        binary=binary_post,
        mmpd=mmpd_post,
        variate=v,
        n_windows=int(args.n_windows),
        forced_pools=args.pools,
        seed=int(args.seed),
    )
    pools = [int(snapped["indices"][i]) for i in locals_]
    print(
        f"[LULL-wn128] locals={locals_} pools={pools} snap_mode={snap_mode}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = args.output_dir / "pre_post_snap" / f"LULL_v{v}"
    pre_dir.mkdir(parents=True, exist_ok=True)
    space_pre = (
        "PRE: pack storage = global dataset-z; "
        f"snap lattice = {WN128_LATTICE} from past mean/std "
        "(max_scale from ckpt metadata; NOT ordinal absolute)"
    )
    # Display-only string for panel subtitles (assert already enforced snap_mode).
    snap_label = f"{snap_mode} ≡ {WN128_LATTICE}"
    pre_post_panels: List[str] = []
    for local in locals_:
        pool = int(snapped["indices"][local])
        title = (
            f"{RUN_NAME}/{DATASET} LULL v={v} pool={pool} local={local} | "
            f"{WN128_LATTICE}"
        )
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
            out_path=pre_dir / f"pool{pool}_local{local}_H96_pre_vs_post_snap.png",
            pre=pre,
            post=post,
            levels_1d=levels_1d,
            title_prefix=title,
            space_pre=space_pre,
            snap_mode=snap_label,
            dpi=int(args.dpi),
        )
        pre_post_panels.append(_rel(h96))

        h = int(gt_post.shape[-1])
        for L in args.slice_lengths:
            L = int(L)
            if L > h:
                continue
            off = max(0, (h - L) // 2)
            pre_s = {k: arr[off : off + L] for k, arr in pre.items()}
            post_s = {k: arr[off : off + L] for k, arr in post.items()}
            post_bc: Dict[str, np.ndarray] = {}
            for k, seg in post_s.items():
                shifted, _ = bin_center_shift(
                    seg[None, None, :],
                    levels_1d[None, None, :],
                    reduce="per_variate",
                )
                post_bc[k] = shifted[0, 0]
            post_z = None
            if args.include_zscore_ref:
                post_z = {k: _zscore_1d(arr) for k, arr in post_s.items()}
            lpath = _write_l_stages(
                out_path=pre_dir / f"pool{pool}_local{local}_L{L}_off{off}_pre_snap_bc.png",
                pre=pre_s,
                post=post_s,
                post_bc=post_bc,
                post_z=post_z,
                levels_1d=levels_1d,
                title_prefix=title,
                space_pre=space_pre,
                snap_mode=snap_label,
                offset=off,
                slice_len=L,
                dpi=int(args.dpi),
            )
            pre_post_panels.append(_rel(lpath))
            bipath = _write_bin_index_compare(
                out_path=pre_dir / f"pool{pool}_local{local}_L{L}_off{off}_bin_index.png",
                post=post_s,
                post_bc=post_bc,
                levels_1d=levels_1d,
                title_prefix=title,
                offset=off,
                dpi=int(args.dpi),
            )
            pre_post_panels.append(_rel(bipath))

    h96_paths = _write_horizon96(
        out_dir=args.output_dir / "horizon96",
        snapped=snapped,
        locals_=locals_,
        variate=v,
        dpi=int(args.dpi),
    )
    l8: List[Path] = []
    l16: List[Path] = []
    for L in args.slice_lengths:
        L = int(L)
        dest = args.output_dir / f"L{L}_snapproof"
        paths = _write_snapproof(
            out_dir=dest,
            snapped=snapped,
            locals_=locals_,
            variate=v,
            slice_len=L,
        )
        if L == 8:
            l8 = paths
        elif L == 16:
            l16 = paths

    gt_meta: Dict[str, Any] = {}
    if not args.skip_gt_bins:
        print("[LULL-wn128] encoding GT coarse/fine on window-norm path…", flush=True)
        enc_model = _build_window_norm_encode_model(
            ckpt=args.ckpt,
            config_path=args.config,
            max_scale=max_scale,
            lookback=int(args.lookback),
            horizon=int(args.horizon),
            n_variates=int(gt_pre.shape[1]),
            device=device,
        )
        gt_meta = _write_gt_bins(
            out_dir=args.output_dir / "gt_coarse_fine",
            past_all=past,
            gt_all=gt_pre,
            indices=np.asarray(snapped["indices"], dtype=np.int64),
            locals_=locals_,
            variate=v,
            model=enc_model,
            slice_lengths=args.slice_lengths,
            device=device,
            dpi=int(args.dpi),
        )

    starter: List[str] = []
    if locals_ and pools:
        local0, pool0 = int(locals_[0]), int(pools[0])
        starter = [
            _rel(pre_dir / f"pool{pool0}_local{local0}_H96_pre_vs_post_snap.png"),
            _rel(pre_dir / f"pool{pool0}_local{local0}_L8_off44_pre_snap_bc.png"),
            _rel(
                args.output_dir
                / "L8_snapproof"
                / (
                    f"{RUN_NAME}_{DATASET}_LULL_v{v}_local{local0}_pool{pool0}_"
                    f"L8_off44_snapproof.png"
                )
            ),
        ]
        if gt_meta.get("horizon96"):
            starter.append(gt_meta["horizon96"][0])
    elif pre_post_panels:
        starter = [pre_post_panels[0]]

    meta = {
        "pack": _rel(args.pack_root),
        "ckpt": _rel(args.ckpt),
        "config": args.config,
        "binary": binary_path.name,
        "mmpd": mmpd_path.name,
        "kind": kind,
        "variate": v,
        "variate_name": "LULL",
        "snap_mode": snap_mode,
        "canvas_height": canvas_height,
        "max_scale": max_scale,
        "locals": locals_,
        "pools": pools,
        "n_pre_post": len(pre_post_panels),
        "n_h96": len(h96_paths),
        "n_l8": len(l8),
        "n_l16": len(l16),
        "gt_bins": gt_meta,
        "starter_pngs": starter,
        "foil_to": ["ordinal_absolute", "window_norm_grid_hybrid_flat"],
        "fallback_note": "none — used 4601319 non-hybrid + 08-04-2009 pack (not hybrid 4609805)",
        "mmpd_note": (
            "mmpd_ETTh2.npz / mmpd_ETTh2_val-test.npz shipped with 08-04-2009 disc leaf "
            "(exact index match to binary_window_norm_c128_ETTh2_val-test.npz)"
        ),
    }
    _write_readme(args.output_dir, meta)
    (args.output_dir / "summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
