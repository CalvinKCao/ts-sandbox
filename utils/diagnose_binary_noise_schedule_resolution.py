#!/usr/bin/env python3
"""Diagnose binary noise schedule at 96/96 vs 336/720 (uncompressed).

This repo uses *per-timestep bit-flip* noise (``BinaryDiffusionScheduler``), not
cumulative Gaussian diffusion. Training samples ``t`` and flips each bit of the
binary CDF map with probability ``beta_t``. At ``beta=0.5`` the map is
independent of ``x0`` by construction. These checks still matter when comparing
resolutions: wider maps (336/720, ``representation_time_stride=1``) retain more
spatial structure at the *same* ``beta_t < 0.5``, so mid-schedule strips can
look under-corrupted even when ``t=T`` is fine.

Check 1 (endpoint): ``x0`` | ``x_T`` (``t=T-1``) | pure Bernoulli(0.5)
Check 2 (progression): strips at t = 0, T/8, …, T for both geometries + beta/SNR plot.

Example:
  python utils/diagnose_binary_noise_schedule_resolution.py --datasets ETTh1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.diffusion import BinaryDiffusionScheduler
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    create_patch_guidance_stack,
    load_dataset,
    resolve_pipeline_data_subset,
    wrap_patch_guidance,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

GEOMETRY_CONFIGS = {
    "96/96": "configs/binary_anchor_ar_patch_decoder_ctx.yaml",
    "336/720_uncompressed": (
        "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed.yaml"
    ),
}

DEFAULT_DATASETS = "ETTh1,weather,electricity,exchange_rate,traffic"
DEFAULT_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def _build_state(config_path: str, dataset: str) -> PipelineState:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = dataset
    resolve_pipeline_data_subset(state)
    state.subset_id = state.subset_id or dataset
    return state


def _load_windows(
    state: PipelineState,
    *,
    n_samples: int,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    meta = state.data_subset_resolved or {}
    train_ds, _, _, norm_stats = load_dataset(
        state.dataset,
        list(state.variate_indices),
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        stride=int(meta.get("train_stride", state.window_stride)),
        test_stride=1,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    idxs = rng.choice(n, size=min(n_samples, n), replace=False)
    return [train_ds[int(i)] for i in idxs], norm_stats


def _make_model(state: PipelineState, stage: str = "coarse", ordinal_ladder=None):
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
    if ordinal_ladder is not None:
        pipeline_mod.GLOBAL_ORDINAL_LADDER = ordinal_ladder
    n_vars = len(state.variate_indices or [])
    lookback = int(state.lookback_length)
    horizon = int(state.forecast_length)
    # Random-init guidance is enough: we only encode CDF maps, no forward/backward.
    stack = create_patch_guidance_stack(n_vars, in_len=lookback, out_len=horizon)
    guidance = wrap_patch_guidance(stack)
    model = create_diffusion_model(
        guidance_model=guidance,
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        diffusion_stage=stage,
        ordinal_ladder=ordinal_ladder,
    )
    model.eval()
    return model, n_vars


@torch.no_grad()
def _encode_coarse_maps(model, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Return coarse binary CDF maps for future: (V, H, W) in {0,1}."""
    past_b = past.unsqueeze(0)
    future_b = future.unsqueeze(0)
    past_norm, future_norm, _ = model._normalize_sequence(past_b, future_b)
    maps = model._encode_staged_maps(future_norm)
    return maps["coarse"][0].detach().cpu().float().clamp(0, 1)


def _scheduler_from_state(state: PipelineState) -> BinaryDiffusionScheduler:
    return BinaryDiffusionScheduler(
        num_steps=int(state.binary_num_steps),
        beta_start=float(state.binary_beta_start),
        beta_end=float(state.binary_beta_end),
        schedule_type=str(state.binary_noise_schedule),
        device="cpu",
    )


def _noise_at_t(
    sched: BinaryDiffusionScheduler,
    x0: torch.Tensor,
    t_idx: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply one bit-flip step at timestep t (training forward process)."""
    t_idx = int(min(max(0, t_idx), sched.num_steps - 1))
    beta = float(sched.betas[t_idx].item())
    if generator is None:
        zt = torch.bernoulli(torch.full_like(x0, beta))
    else:
        zt = torch.bernoulli(torch.full_like(x0, beta), generator=generator)
    return (x0.bool() ^ zt.bool()).float()


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(aa.norm() * bb.norm())
    if denom < 1e-12:
        return 0.0
    return float((aa * bb).sum() / denom)


def _bit_agree(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.round() == b.round()).float().mean())


def _radial_power_1d(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged 2D power spectrum for a single (H, W) map."""
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64) - img.mean()))
    power = (f.real**2 + f.imag**2)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = int(rr.max())
    sums = np.bincount(rr.ravel(), weights=power.ravel(), minlength=rmax + 1)
    counts = np.bincount(rr.ravel(), minlength=rmax + 1).clip(min=1)
    profile = sums / counts
    freqs = np.arange(rmax + 1, dtype=np.float64)
    return freqs, profile


def _pick_var_slice(x: torch.Tensor, var_idx: int = 0) -> np.ndarray:
    # x: (V, H, W)
    return x[var_idx].numpy()


def check1_endpoint(
    *,
    geometry: str,
    subset_id: str,
    maps: Sequence[torch.Tensor],
    sched: BinaryDiffusionScheduler,
    out_dir: Path,
    jpeg_dpi: int,
) -> Dict[str, Any]:
    t_max = sched.num_steps - 1
    beta_max = float(sched.betas[t_max].item())
    n = len(maps)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n), squeeze=False)
    corrs = []
    agrees = []
    for i, x0 in enumerate(maps):
        # show first variate
        xt = _noise_at_t(sched, x0, t_max)
        pure = torch.bernoulli(torch.full_like(x0, 0.5))
        c = _corr(xt, x0)
        a = _bit_agree(xt, x0)
        corrs.append(c)
        agrees.append(a)
        panels = [
            (_pick_var_slice(x0), f"x0 var0"),
            (_pick_var_slice(xt), f"x_T (β={beta_max:.3f})\ncorr={c:.3f}"),
            (_pick_var_slice(pure), "Bernoulli(0.5)"),
        ]
        for j, (img, title) in enumerate(panels):
            axes[i, j].imshow(img, aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis("off")
    fig.suptitle(
        f"Check1 endpoint {geometry} {subset_id} | t={t_max} schedule={sched.schedule_type}",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"check1_endpoint_{subset_id}_{geometry.replace('/', '-')}.jpg"
    save_figure_jpg(fig, str(path), dpi=jpeg_dpi)
    plt.close(fig)

    # power spectrum: mean over samples, var0
    fig2, ax = plt.subplots(figsize=(6, 4))
    for label, maker in (
        ("x_T", lambda x0: _noise_at_t(sched, x0, t_max)),
        ("pure", lambda x0: torch.bernoulli(torch.full_like(x0, 0.5))),
        ("x0", lambda x0: x0),
    ):
        profiles = []
        for x0 in maps:
            freqs, prof = _radial_power_1d(_pick_var_slice(maker(x0)))
            profiles.append(prof)
        # pad to common length
        mlen = max(len(p) for p in profiles)
        stacked = np.stack([np.pad(p, (0, mlen - len(p))) for p in profiles], axis=0)
        mean_p = stacked.mean(axis=0)
        ax.semilogy(np.arange(len(mean_p)), mean_p + 1e-12, label=label)
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("power")
    ax.set_title(f"Check1 power spectrum {geometry} {subset_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = out_dir / f"check1_power_{subset_id}_{geometry.replace('/', '-')}.jpg"
    save_figure_jpg(fig2, str(path2), dpi=jpeg_dpi)
    plt.close(fig2)

    return {
        "geometry": geometry,
        "subset_id": subset_id,
        "t_max": t_max,
        "beta_max": beta_max,
        "mean_corr_xT_x0": float(np.mean(corrs)),
        "mean_bit_agree_xT_x0": float(np.mean(agrees)),
        "endpoint_jpg": str(path),
        "power_jpg": str(path2),
    }


def check2_progression(
    *,
    geometry: str,
    subset_id: str,
    maps: Sequence[torch.Tensor],
    sched: BinaryDiffusionScheduler,
    fractions: Sequence[float],
    out_dir: Path,
    jpeg_dpi: int,
) -> Dict[str, Any]:
    T = sched.num_steps
    t_idxs = [int(round(f * (T - 1))) for f in fractions]
    n = min(len(maps), 6)
    ncols = len(t_idxs)
    fig, axes = plt.subplots(n, ncols, figsize=(1.4 * ncols, 1.8 * n), squeeze=False)
    for i in range(n):
        x0 = maps[i]
        for j, t_idx in enumerate(t_idxs):
            xt = x0 if t_idx == 0 else _noise_at_t(sched, x0, t_idx)
            beta = float(sched.betas[t_idx].item())
            axes[i, j].imshow(
                _pick_var_slice(xt),
                aspect="auto",
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            if i == 0:
                axes[i, j].set_title(f"t={t_idx}\nβ={beta:.3f}", fontsize=7)
            axes[i, j].axis("off")
    fig.suptitle(f"Check2 progression {geometry} {subset_id}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"check2_strip_{subset_id}_{geometry.replace('/', '-')}.jpg"
    save_figure_jpg(fig, str(path), dpi=jpeg_dpi)
    plt.close(fig)

    # beta / signal-retention vs t
    ts = np.arange(T)
    betas = sched.betas.detach().cpu().numpy()
    # E[corr] ≈ 1-2β for bit-flip; "SNR-like" retention (1-2β)^2
    retention = (1.0 - 2.0 * betas) ** 2
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, betas, label="β_t (flip prob)", color="#E91E63")
    ax.set_ylabel("β_t", color="#E91E63")
    ax2 = ax.twinx()
    ax2.semilogy(ts, np.clip(retention, 1e-8, None), label="(1-2β)² retention", color="#2196F3")
    ax2.axhline(1.0, color="#2196F3", linestyle="--", alpha=0.4, label="retention=1")
    # mark ~0 dB analogue: |1-2β| = 1/sqrt(2) → retention=0.5
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    for t_idx in t_idxs:
        ax.axvline(t_idx, color="black", alpha=0.12, linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_title(f"Schedule {geometry} {subset_id} ({sched.schedule_type})")
    ax.grid(True, alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")
    fig2.tight_layout()
    path2 = out_dir / f"check2_schedule_{subset_id}_{geometry.replace('/', '-')}.jpg"
    save_figure_jpg(fig2, str(path2), dpi=jpeg_dpi)
    plt.close(fig2)

    return {
        "geometry": geometry,
        "subset_id": subset_id,
        "timesteps": t_idxs,
        "betas_at_grid": [float(sched.betas[t].item()) for t in t_idxs],
        "strip_jpg": str(path),
        "schedule_jpg": str(path2),
        "map_shape_VHW": list(maps[0].shape),
    }


def compare_strips_side_by_side(
    *,
    subset_id: str,
    maps_old: Sequence[torch.Tensor],
    maps_new: Sequence[torch.Tensor],
    sched_old: BinaryDiffusionScheduler,
    sched_new: BinaryDiffusionScheduler,
    fractions: Sequence[float],
    out_dir: Path,
    jpeg_dpi: int,
) -> Path:
    """One sample: top row 96/96 progression, bottom row 336/720."""
    t_old = [int(round(f * (sched_old.num_steps - 1))) for f in fractions]
    t_new = [int(round(f * (sched_new.num_steps - 1))) for f in fractions]
    ncols = len(fractions)
    fig, axes = plt.subplots(2, ncols, figsize=(1.35 * ncols, 3.6), squeeze=False)
    x0_old, x0_new = maps_old[0], maps_new[0]
    for j, (to, tn) in enumerate(zip(t_old, t_new)):
        xo = x0_old if to == 0 else _noise_at_t(sched_old, x0_old, to)
        xn = x0_new if tn == 0 else _noise_at_t(sched_new, x0_new, tn)
        axes[0, j].imshow(_pick_var_slice(xo), aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].imshow(_pick_var_slice(xn), aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        bo = float(sched_old.betas[to].item())
        bn = float(sched_new.betas[tn].item())
        axes[0, j].set_title(f"96 t={to}\nβ={bo:.3f}", fontsize=6)
        axes[1, j].set_title(f"336 t={tn}\nβ={bn:.3f}", fontsize=6)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("96/96", fontsize=9)
    axes[1, 0].set_ylabel("336/720", fontsize=9)
    fig.suptitle(
        f"{subset_id}: noise progression 96/96 vs 336/720_uncompressed (same β schedule)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / f"compare_strips_{subset_id}.jpg"
    save_figure_jpg(fig, str(path), dpi=jpeg_dpi)
    plt.close(fig)
    return path


def run_dataset(
    *,
    dataset: str,
    geometries: Sequence[str],
    n_samples: int,
    seed: int,
    stage: str,
    out_dir: Path,
    jpeg_dpi: int,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    packed: Dict[str, Any] = {}

    for geometry in geometries:
        config_path = GEOMETRY_CONFIGS[geometry]
        state = _build_state(config_path, dataset)
        windows, norm_stats = _load_windows(state, n_samples=n_samples, seed=seed)
        ladder = norm_stats.get("ordinal_ladder")
        model, n_vars = _make_model(state, stage=stage, ordinal_ladder=ladder)
        maps = [_encode_coarse_maps(model, past, future) for past, future in windows]
        sched = _scheduler_from_state(state)
        print(
            f"[encode] {geometry} {state.subset_id}: "
            f"n={len(maps)} map={tuple(maps[0].shape)} "
            f"schedule={sched.schedule_type} T={sched.num_steps} "
            f"β=[{float(sched.betas[0]):.2e},{float(sched.betas[-1]):.3f}] "
            f"repr_stride={state.representation_time_stride}",
            flush=True,
        )
        r1 = check1_endpoint(
            geometry=geometry,
            subset_id=str(state.subset_id),
            maps=maps,
            sched=sched,
            out_dir=out_dir,
            jpeg_dpi=jpeg_dpi,
        )
        r2 = check2_progression(
            geometry=geometry,
            subset_id=str(state.subset_id),
            maps=maps,
            sched=sched,
            fractions=DEFAULT_FRACTIONS,
            out_dir=out_dir,
            jpeg_dpi=jpeg_dpi,
        )
        row = {
            "dataset": dataset,
            "subset_id": state.subset_id,
            "n_variates": n_vars,
            "lookback": state.lookback_length,
            "horizon": state.forecast_length,
            "repr_stride": state.representation_time_stride,
            "stage": stage,
            **r1,
            "strip_jpg": r2["strip_jpg"],
            "schedule_jpg": r2["schedule_jpg"],
            "map_shape": str(r2["map_shape_VHW"]),
            "betas_at_grid": str(r2["betas_at_grid"]),
        }
        rows.append(row)
        packed[geometry] = {
            "maps": maps,
            "sched": sched,
            "subset_id": str(state.subset_id),
        }
        print(
            f"[check1] {geometry} {state.subset_id}: "
            f"corr(xT,x0)={r1['mean_corr_xT_x0']:.4f} "
            f"bit_agree={r1['mean_bit_agree_xT_x0']:.4f} β_max={r1['beta_max']:.4f}",
            flush=True,
        )

    if "96/96" in packed and "336/720_uncompressed" in packed:
        path = compare_strips_side_by_side(
            subset_id=packed["96/96"]["subset_id"],
            maps_old=packed["96/96"]["maps"],
            maps_new=packed["336/720_uncompressed"]["maps"],
            sched_old=packed["96/96"]["sched"],
            sched_new=packed["336/720_uncompressed"]["sched"],
            fractions=DEFAULT_FRACTIONS,
            out_dir=out_dir,
            jpeg_dpi=jpeg_dpi,
        )
        print(f"[compare] {path}", flush=True)
        for row in rows:
            row["compare_strip_jpg"] = str(path)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument(
        "--geometries",
        default="96/96,336/720_uncompressed",
        help=f"from {sorted(GEOMETRY_CONFIGS)}",
    )
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stage", default="coarse", choices=("coarse", "fine"))
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "noise_schedule_resolution")
    p.add_argument("--jpeg-dpi", type=int, default=100)
    p.add_argument("--device", default="cpu", help="encoding is light; cpu is fine")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    for g in geometries:
        if g not in GEOMETRY_CONFIGS:
            raise ValueError(f"unknown geometry {g!r}")

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        ds_dir = out_root / dataset
        print(f"==== {dataset} -> {ds_dir} ====", flush=True)
        rows = run_dataset(
            dataset=dataset,
            geometries=geometries,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            stage=str(args.stage),
            out_dir=ds_dir,
            jpeg_dpi=int(args.jpeg_dpi),
        )
        all_rows.extend(rows)

    csv_path = out_root / "summary.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    print(f"[summary] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
