# Pipeline integration: Prefer ablation --viz-encode-bins (utils.disc_snap_viz.viz_gt_encode_bins).
#!/usr/bin/env python3
"""LULL (ETTh2 v=5) GT coarse/fine bin viz — hybrid flat-dsnorm canvas128.

Encodes real GT the way training does (hybrid flat = dataset affine only for
LULL; no instance norm) into staged coarse+fine CDFs, then decodes back to 1D:

- solid  = coarse bin centers (``_decode_coarse_1d_from_map``)
- dotted = fine-refined (coarse + residual fine) and absolute canvas128 HIR bins

No model predictions / MMPD. Needs config + ckpt ``metadata.json`` only
(encode path; DiT weights optional / unused).

Default pools from temp/viz_lull_etth2_hybrid/README.md: 1116, 37, 1340, 0.

Examples:
  source .venv/bin/activate
  python temp/scripts/viz_lull_gt_coarse_fine.py

  # Killarney (weights not required, but same leaf paths):
  cd "$SCRATCH/ts-sandbox-ordinal-fine"
  source .venv/bin/activate
  python temp/scripts/viz_lull_gt_coarse_fine.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals  # noqa: E402
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (  # noqa: E402
    patch_stage_globals,
)
from models.diffusion_tsf.pipeline.state import PipelineState  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    create_diffusion_model,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _flat_mask_from_ckpt,
    _max_scale_from_ckpt_metadata,
)

LULL_VARIATE = 5
DATASET = "ETTh2"
RUN_NAME = "hybrid_flat_dsnorm"
DEFAULT_PACK = (
    REPO_ROOT
    / "results/datasets/08-05-1057-ablation-disc-l8-l16-ETTh2-c128-hybrid-flat-dsnorm-valtest80-byvar"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-05-4609805-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm"
)
DEFAULT_CFG = (
    "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm.yaml"
)
DEFAULT_OUT = REPO_ROOT / "temp" / "viz_lull_gt_coarse_fine"
PREFERRED_POOLS = (1116, 37, 1340, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-root", type=Path, default=DEFAULT_PACK)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--variate", type=int, default=LULL_VARIATE)
    p.add_argument("--pools", type=int, nargs="+", default=list(PREFERRED_POOLS))
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    args.pack_root = args.pack_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _find_binary_pack(raw_dir: Path, dataset: str) -> Path:
    hits = sorted(p for p in raw_dir.glob(f"binary_*{dataset}*.npz") if "indices" not in p.name)
    vt = [p for p in hits if "val-test" in p.name]
    if vt:
        return vt[0]
    if not hits:
        raise FileNotFoundError(f"no binary pack for {dataset} under {raw_dir}")
    return hits[0]


def _build_encode_model(
    *,
    ckpt: Path,
    config_path: str,
    flat_mask: List[bool],
    max_scale: float,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> torch.nn.Module:
    """Encode/decode-only DiffusionTSF (no DiT weights, no guidance)."""
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": DATASET})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(ckpt.resolve())
    state.dataset = DATASET
    state.subset_id = DATASET
    state.extra["hybrid_flat_norm_stats"] = {"flat_variate_mask": list(flat_mask)}
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state, "coarse", honor_dataset_windows=True)
    # Encode path never needs guidance / cross-attn tokens.
    pipeline_mod.DISABLE_CROSS_ATTENTION = True
    pipeline_mod.USE_GUIDANCE_CHANNEL = False
    model = create_diffusion_model(
        n_variates=len(flat_mask),
        lookback=lookback,
        horizon=horizon,
        guidance_model=None,
        diffusion_stage="coarse",
        use_guidance_channel=False,
    ).to(device)
    model.config.skip_window_norm_variate_mask = list(flat_mask)
    model.config.hybrid_flat_dataset_norm = True
    model.config.max_scale = float(max_scale)
    model.to_2d.max_scale = float(max_scale)
    model.eval()
    return model


@torch.no_grad()
def _encode_gt_bins(
    model: torch.nn.Module,
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    variate: int,
) -> Dict[str, np.ndarray]:
    """Normalize + staged encode GT; return 1D series for one variate (forecast cols)."""
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

    # Pack future is horizon-only (no overlap cols). Trim if encode width > H.
    k = int(getattr(model.config, "lookback_overlap", 0) or 0)
    h = int(future.shape[-1])

    def _trim(x: torch.Tensor) -> np.ndarray:
        y = x[0, variate].detach().cpu().numpy()
        if y.shape[-1] > h:
            y = y[-h:]
        elif k > 0 and y.shape[-1] == h + k:
            y = y[k:]
        return y

    gt_norm = future_norm[0, variate].detach().cpu().numpy()
    if gt_norm.shape[-1] > h:
        gt_norm = gt_norm[-h:]

    out = {
        "gt_norm": gt_norm,
        "coarse": _trim(coarse_1d),
        "fine_residual": _trim(fine_res),
        "fine_refined": _trim(combined),
        "center": float(stats[0][0, variate, 0].item()),
        "std": float(stats[1][0, variate, 0].item()),
    }
    if hir_1d is not None:
        out["fine_hir"] = _trim(hir_1d)
    return out


def _pool_to_local(indices: np.ndarray, pools: Sequence[int]) -> List[Tuple[int, int]]:
    pool_to_local = {int(indices[i]): i for i in range(len(indices))}
    pairs: List[Tuple[int, int]] = []
    missing: List[int] = []
    for pool in pools:
        local = pool_to_local.get(int(pool))
        if local is None:
            missing.append(int(pool))
            continue
        pairs.append((int(pool), int(local)))
    if missing:
        raise KeyError(f"pools not in pack indices: {missing}")
    return pairs


def _plot_panel(
    *,
    out_path: Path,
    title: str,
    t: np.ndarray,
    series: Dict[str, np.ndarray],
    dpi: int,
    zoom: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0 if not zoom else 8.5, 3.8 if not zoom else 3.4))
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
            alpha=0.9, label="fine canvas128 HIR bins (dotted)", zorder=4,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("horizon step t")
    ax.set_ylabel("model-space value (LULL: dataset-z / identity win-norm)")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_readme(out_dir: Path, meta: Dict[str, Any]) -> None:
    lines = [
        "# LULL GT coarse / fine bins (hybrid flat-dsnorm)",
        "",
        "Generated by `temp/scripts/viz_lull_gt_coarse_fine.py`.",
        "",
        "GT only — staged encode of real windows (no binary/MMPD preds).",
        "",
        "## Encoding",
        "",
        "1. `_normalize_sequence` with hybrid flat mask (LULL v=5 → identity window-norm)",
        "2. `_encode_staged_maps` → coarse + residual-fine CDFs (Hc=Hf=16)",
        "3. Decode: `_decode_coarse_1d_from_map` (solid) + residual fine → "
        "**fine-refined** = coarse+residual (dotted)",
        "4. Also absolute `patch_refine` canvas128 HIR encode/decode (dotted green)",
        "",
        "## Sources",
        "",
        f"- pack: `{meta['pack']}`",
        f"- ckpt metadata: `{meta['ckpt']}`",
        f"- config: `{meta['config']}`",
        f"- max_scale: {meta['max_scale']}",
        f"- canvas_height: {meta['canvas_height']}",
        f"- flat_variate_mask[LULL]: {meta['lull_is_flat']}",
        "",
        "## Layout",
        "",
        "- `horizon96/` — full H=96",
        "- `L8/` / `L16/` — mid-horizon zooms",
        "",
        f"## Pools: {meta['pools']}",
        "",
    ]
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
        raise FileNotFoundError(f"missing ckpt dir: {args.ckpt}")

    flat_mask = _flat_mask_from_ckpt(args.ckpt, DATASET)
    if flat_mask is None:
        raise RuntimeError(f"no hybrid flat_variate_mask under {args.ckpt}")
    if not flat_mask[int(args.variate)]:
        raise RuntimeError(
            f"variate {args.variate} is not marked flat in ckpt metadata; "
            f"mask={flat_mask}"
        )
    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))

    binary_path = _find_binary_pack(raw_dir, DATASET)
    pack = _load_npz(binary_path)
    past_all = np.asarray(pack["past"], dtype=np.float32)
    gt_all = np.asarray(pack["y_true"], dtype=np.float32)
    indices = np.asarray(pack["indices"], dtype=np.int64)
    if past_all.shape[1] <= int(args.variate):
        raise ValueError(f"variate={args.variate} out of range V={past_all.shape[1]}")

    pairs = _pool_to_local(indices, args.pools)
    print(
        f"[LULL-GT] pack={binary_path.name} N={past_all.shape[0]} "
        f"pools={[p for p, _ in pairs]} max_scale={max_scale} device={device}",
        flush=True,
    )

    model = _build_encode_model(
        ckpt=args.ckpt,
        config_path=args.config,
        flat_mask=flat_mask,
        max_scale=max_scale,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        device=device,
    )
    canvas_h = int(getattr(model.config, "patch_refine_canvas_height", 0) or 0)
    print(
        f"[LULL-GT] Hc={model.config.coarse_image_height} "
        f"Hf={model.config.fine_image_height} canvas={canvas_h} "
        f"overlap={model.config.lookback_overlap}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h96_dir = args.output_dir / "horizon96"
    h96_paths: List[Path] = []
    zoom_paths: Dict[int, List[Path]] = {int(L): [] for L in args.slice_lengths}

    v = int(args.variate)
    for pool, local in pairs:
        past_t = torch.from_numpy(past_all[local : local + 1]).to(device)
        fut_t = torch.from_numpy(gt_all[local : local + 1]).to(device)
        series = _encode_gt_bins(model, past_t, fut_t, variate=v)
        h = int(series["gt_norm"].shape[-1])
        t = np.arange(h)

        mae_c = float(np.mean(np.abs(series["coarse"] - series["gt_norm"])))
        mae_f = float(np.mean(np.abs(series["fine_refined"] - series["gt_norm"])))
        title = (
            f"{RUN_NAME}/{DATASET} LULL v={v} pool={pool} local={local} | "
            f"H={h} GT bins  center={series['center']:.3g} std={series['std']:.3g}  "
            f"|coarse-GT|={mae_c:.3g} |fine-GT|={mae_f:.3g}"
        )
        path = h96_dir / (
            f"{RUN_NAME}_{DATASET}_LULL_v{v}_local{local}_pool{pool}_H{h}_gt_bins.png"
        )
        _plot_panel(out_path=path, title=title, t=t, series=series, dpi=int(args.dpi))
        h96_paths.append(path)
        print(f"  wrote {path.relative_to(REPO_ROOT)}", flush=True)

        for L in args.slice_lengths:
            L = int(L)
            if L >= h:
                continue
            off = max(0, (h - L) // 2)
            sl = {k: (val[off : off + L] if isinstance(val, np.ndarray) else val)
                  for k, val in series.items()}
            # keep scalars
            for key in ("center", "std"):
                sl[key] = series[key]
            t_sl = np.arange(off, off + L)
            zpath = args.output_dir / f"L{L}" / (
                f"{RUN_NAME}_{DATASET}_LULL_v{v}_local{local}_pool{pool}_"
                f"L{L}_off{off}_gt_bins.png"
            )
            ztitle = (
                f"{RUN_NAME}/{DATASET} LULL v={v} pool={pool} local={local} | "
                f"L={L} off={off} GT bins (zoom)"
            )
            _plot_panel(
                out_path=zpath, title=ztitle, t=t_sl, series=sl,
                dpi=int(args.dpi), zoom=True,
            )
            zoom_paths[L].append(zpath)

    meta = {
        "pack": str(args.pack_root.relative_to(REPO_ROOT))
        if args.pack_root.is_relative_to(REPO_ROOT)
        else str(args.pack_root),
        "ckpt": str(args.ckpt.relative_to(REPO_ROOT))
        if args.ckpt.is_relative_to(REPO_ROOT)
        else str(args.ckpt),
        "config": args.config,
        "binary": binary_path.name,
        "variate": v,
        "variate_name": "LULL",
        "pools": [p for p, _ in pairs],
        "locals": [loc for _, loc in pairs],
        "max_scale": max_scale,
        "canvas_height": canvas_h,
        "coarse_image_height": int(model.config.coarse_image_height),
        "fine_image_height": int(model.config.fine_image_height),
        "lull_is_flat": True,
        "flat_variate_mask": flat_mask,
        "n_h96": len(h96_paths),
        "n_zoom": {str(L): len(paths) for L, paths in zoom_paths.items()},
        "note": "GT encode only; DiT weights unused",
    }
    _write_readme(args.output_dir, meta)
    (args.output_dir / "summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
