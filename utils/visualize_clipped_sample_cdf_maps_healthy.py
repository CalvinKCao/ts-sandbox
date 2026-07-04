#!/usr/bin/env python3
"""Visualize coarse/fine CDF maps under the healthy flat-window clipping heuristic.

Production path (see visualize_clipped_sample_cdf_maps.py):
  std_eff = max(past_std, window_norm_std_floor=0.1)
  max_scale = config max_scale_by_dataset

Healthy heuristic (flat-window aware):
  if past_std <= std_floor: std_eff = 1.0   # z-score units, no 10x blow-up
  else: std_eff = max(past_std, std_floor)
  max_scale = q99.5 of |future_wn| on TRAIN windows under healthy norm

Example:
  python utils/visualize_clipped_sample_cdf_maps_healthy.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.preprocessing import TimeSeriesTo2D  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "reports/representation_roundtrip_floor/clipped_samples"
DEFAULT_OUTPUT = REPO_ROOT / "reports/representation_roundtrip_floor/clipped_samples_2d_healthy"
DEFAULT_CONFIG = REPO_ROOT / "configs/base/binary_staged.yaml"
TEST_STRIDE = 4
CALIB_MAX_WINDOWS = 2048
CALIB_QUANTILE = 0.995
FNAME_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9_]+)_win(?P<win>\d+)_var(?P<var>\d+)_rank(?P<rank>\d+)\.png$"
)


def _load_base_cfg(config_path: Path, dataset: str) -> dict:
    exp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    ms_map = dict(exp.get("max_scale_by_dataset") or {})
    return {
        "max_scale_prod": float(ms_map.get(dataset, exp.get("max_scale", 3.5))),
        "std_floor": float(exp.get("window_norm_std_floor", 1e-8)),
        "center_mode": str(exp.get("window_norm_center", "mean")),
        "coarse_h": int(exp.get("coarse_image_height", exp.get("image_height", 16))),
        "fine_h": int(exp.get("fine_image_height", exp.get("image_height", 16))),
    }


def _window_center(past_z: torch.Tensor, center_mode: str) -> torch.Tensor:
    if center_mode == "last":
        return past_z[..., -1:]
    return past_z.mean(dim=-1, keepdim=True)


def _production_window_norm(
    past_z: torch.Tensor,
    segment_z: torch.Tensor,
    *,
    std_floor: float,
    center_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center = _window_center(past_z, center_mode)
    past_std_raw = past_z.std(dim=-1, keepdim=True)
    std_eff = past_std_raw.clamp_min(std_floor)
    wn = (segment_z - center) / std_eff
    flat = past_std_raw <= std_floor
    return wn, center, std_eff, flat


LOW_VAR_PAST_THRESHOLD = 0.3  # z-score units; below this use std_eff=1.0


def _healthy_window_norm(
    past_z: torch.Tensor,
    segment_z: torch.Tensor,
    *,
    std_floor: float,
    center_mode: str,
    low_var_threshold: float = LOW_VAR_PAST_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat or low-variance past: divide by 1.0 z-unit instead of tiny std / std_floor."""
    center = _window_center(past_z, center_mode)
    past_std_raw = past_z.std(dim=-1, keepdim=True)
    flat = past_std_raw <= std_floor
    low_var = past_std_raw < low_var_threshold
    use_unit_scale = flat | low_var
    std_prod = past_std_raw.clamp_min(std_floor)
    std_healthy = torch.where(use_unit_scale, torch.ones_like(past_std_raw), std_prod)
    wn = (segment_z - center) / std_healthy
    return wn, center, std_healthy, use_unit_scale


def _calibrate_healthy_max_scale(
    train_ds,
    *,
    std_floor: float,
    center_mode: str,
    max_windows: int,
    quantile: float,
) -> float:
    n = min(len(train_ds), max_windows)
    indices = np.linspace(0, len(train_ds) - 1, n, dtype=int).tolist()
    subset = Subset(train_ds, indices)
    abs_vals: list[torch.Tensor] = []
    for past, future in subset:
        segment = torch.cat([past, future], dim=-1).unsqueeze(0)
        past_b = past.unsqueeze(0)
        wn, _, _, _ = _healthy_window_norm(
            past_b, segment, std_floor=std_floor, center_mode=center_mode,
        )
        horizon = future.shape[-1]
        abs_vals.append(wn[..., -horizon:].abs().reshape(-1).cpu())
    pooled = torch.cat(abs_vals).numpy()
    return float(np.quantile(pooled, quantile))


def _parse_samples(input_dir: Path) -> Iterator[Tuple[str, int, int, int, str]]:
    for path in sorted(input_dir.glob("*.png")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        yield (
            m.group("dataset"),
            int(m.group("win")),
            int(m.group("var")),
            int(m.group("rank")),
            path.stem,
        )


def _encode_maps(wn_clip: torch.Tensor, *, max_scale: float, coarse_h: int, fine_h: int):
    encoder = TimeSeriesTo2D(height=coarse_h, max_scale=max_scale)
    coarse, fine = encoder.encode_dual_heights(
        wn_clip.unsqueeze(0).unsqueeze(0),
        coarse_height=coarse_h,
        fine_height=fine_h,
    )
    return coarse[0, 0].cpu().numpy(), fine[0, 0].cpu().numpy()


def _plot_sample(
    *,
    stem: str,
    dataset: str,
    window_idx: int,
    variate: int,
    rank: int,
    cfg: dict,
    healthy_ms: float,
    test_ds,
    out_path: Path,
) -> None:
    past, future = test_ds[window_idx]
    lookback = past.shape[-1]
    horizon = future.shape[-1]
    segment_z = torch.cat([past, future], dim=-1)
    past_b = past.unsqueeze(0)
    seg_b = segment_z.unsqueeze(0)

    wn_h, center, std_h, flat = _healthy_window_norm(
        past_b, seg_b,
        std_floor=cfg["std_floor"],
        center_mode=cfg["center_mode"],
    )
    wn_h = wn_h[0, variate]
    ms = healthy_ms
    wn_clip = torch.clamp(wn_h, -ms, ms)
    clipped_mask = wn_h.abs() > ms
    coarse_np, fine_np = _encode_maps(
        wn_clip, max_scale=ms, coarse_h=cfg["coarse_h"], fine_h=cfg["fine_h"],
    )

    seg_z = segment_z[variate].numpy()
    wn_np = wn_h.numpy()
    wn_clip_np = wn_clip.numpy()
    t = np.arange(len(seg_z))
    t_hor = t[lookback:]
    is_low_var = bool(flat[0, variate, 0].item())
    std_used = float(std_h[0, variate, 0].item())

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 2.2, 2.2], hspace=0.35)
    fig.suptitle(
        f"[HEALTHY] {dataset} win={window_idx} var={variate} rank={rank} | "
        f"MS={ms:.2g} (q99.5 train) std_eff={std_used:.3f} low_var_past={is_low_var} "
        f"clipped={int(clipped_mask[lookback:].sum())}/{horizon} horizon cols",
        fontsize=10,
    )

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, seg_z, color="#1f77b4", lw=1.2, label="global z-score")
    ax0.axvline(lookback - 0.5, color="0.5", ls=":", lw=1)
    ax0.set_ylabel("z-score")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, wn_np, color="#2ca02c", lw=1.0, label="healthy window-norm (pre-clip)")
    ax1.plot(t, wn_clip_np, color="#d62728", lw=1.0, ls="--", label="clipped encoder input")
    ax1.axhline(ms, color="#9467bd", ls=":", lw=1)
    ax1.axhline(-ms, color="#9467bd", ls=":", lw=1)
    ax1.axvline(lookback - 0.5, color="0.5", ls=":", lw=1)
    if clipped_mask.any():
        ax1.scatter(
            t_hor[clipped_mask[lookback:].numpy()],
            wn_np[lookback:][clipped_mask[lookback:].numpy()],
            c="#e377c2",
            s=22,
            zorder=5,
            label="clipped",
        )
    ax1.set_ylabel("window-norm σ")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.25)

    seq_len = coarse_np.shape[1]
    extent = [0, seq_len, 0, coarse_np.shape[0]]

    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(
        coarse_np, aspect="auto", origin="lower", extent=extent,
        cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    ax2.axvline(lookback, color="#ff7f0e", ls="--", lw=1, alpha=0.9)
    ax2.set_ylabel("coarse bin")
    ax2.set_title("coarse CDF — healthy norm should show moving band, not saturated wall", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01, label="occupancy")

    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(
        fine_np, aspect="auto", origin="lower", extent=extent,
        cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    ax3.axvline(lookback, color="#ff7f0e", ls="--", lw=1, alpha=0.9)
    ax3.set_ylabel("fine bin")
    ax3.set_xlabel("time index (orange = lookback | horizon)")
    ax3.set_title("fine CDF — residual texture within coarse bin", fontsize=9)
    fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01, label="occupancy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calib-max-windows", type=int, default=CALIB_MAX_WINDOWS)
    args = parser.parse_args()

    samples = list(_parse_samples(args.input_dir))
    if not samples:
        print(f"no samples in {args.input_dir}")
        return 1

    cfg_cache: Dict[str, dict] = {}
    ds_cache: Dict[str, object] = {}
    healthy_ms: Dict[str, float] = {}

    for dataset, win, var, rank, stem in samples:
        if dataset not in cfg_cache:
            cfg_cache[dataset] = _load_base_cfg(args.config, dataset)
            train_ds, _, test_ds, _ = load_dataset(
                dataset, variate_indices=None, test_stride=TEST_STRIDE,
            )
            ds_cache[dataset] = test_ds
            ms = _calibrate_healthy_max_scale(
                train_ds,
                std_floor=cfg_cache[dataset]["std_floor"],
                center_mode=cfg_cache[dataset]["center_mode"],
                max_windows=args.calib_max_windows,
                quantile=CALIB_QUANTILE,
            )
            healthy_ms[dataset] = ms
            print(
                f"  {dataset}: healthy MS={ms:.2f} "
                f"(prod {cfg_cache[dataset]['max_scale_prod']:.2f})",
                flush=True,
            )

        out_path = args.output_dir / f"{stem}_cdf_healthy.png"
        print(f"plot {stem}...", flush=True)
        _plot_sample(
            stem=stem,
            dataset=dataset,
            window_idx=win,
            variate=var,
            rank=rank,
            cfg=cfg_cache[dataset],
            healthy_ms=healthy_ms[dataset],
            test_ds=ds_cache[dataset],
            out_path=out_path,
        )

    summary_lines = [
        "# Healthy clipping heuristic — calibrated max_scale",
        "",
        "Norm: if `past_std < 0.3` z-units (flat/low-variance past), use `std_eff=1.0`; "
        "else `max(past_std, std_floor)`.",
        f"`max_scale` = q{CALIB_QUANTILE} of |healthy future_wn| on train (≤{args.calib_max_windows} windows).",
        "",
        "| dataset | production MS | healthy MS |",
        "| --- | ---: | ---: |",
    ]
    for ds in sorted(healthy_ms):
        summary_lines.append(
            f"| {ds} | {cfg_cache[ds]['max_scale_prod']:.2f} | {healthy_ms[ds]:.2f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"wrote {len(samples)} figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
