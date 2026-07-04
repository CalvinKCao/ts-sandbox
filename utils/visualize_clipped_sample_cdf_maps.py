#!/usr/bin/env python3
"""Visualize coarse/fine binary CDF maps for clipped round-trip samples.

Reads window specs from existing clipped_samples/*.png filenames and writes
companion figures under reports/representation_roundtrip_floor/clipped_samples_2d/.

Example:
  python utils/visualize_clipped_sample_cdf_maps.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.preprocessing import TimeSeriesTo2D  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "reports/representation_roundtrip_floor/clipped_samples"
DEFAULT_OUTPUT = REPO_ROOT / "reports/representation_roundtrip_floor/clipped_samples_2d"
DEFAULT_CONFIG = REPO_ROOT / "configs/base/binary_staged.yaml"
TEST_STRIDE = 4
FNAME_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9_]+)_win(?P<win>\d+)_var(?P<var>\d+)_rank(?P<rank>\d+)\.png$"
)


def _load_rep_cfg(config_path: Path, dataset: str) -> dict:
    exp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    ms_map = dict(exp.get("max_scale_by_dataset") or {})
    return {
        "max_scale": float(ms_map.get(dataset, exp.get("max_scale", 3.5))),
        "std_floor": float(exp.get("window_norm_std_floor", 1e-8)),
        "center_mode": str(exp.get("window_norm_center", "mean")),
        "coarse_h": int(exp.get("coarse_image_height", exp.get("image_height", 16))),
        "fine_h": int(exp.get("fine_image_height", exp.get("image_height", 16))),
    }


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


def _window_norm(past_z: torch.Tensor, segment_z: torch.Tensor, *, std_floor: float, center_mode: str):
    if center_mode == "last":
        center = past_z[..., -1:]
    else:
        center = past_z.mean(dim=-1, keepdim=True)
    std = past_z.std(dim=-1, keepdim=True).clamp_min(std_floor)
    wn = (segment_z - center) / std
    return wn, center, std


def _plot_sample(
    *,
    stem: str,
    dataset: str,
    window_idx: int,
    variate: int,
    rank: int,
    cfg: dict,
    test_ds,
    out_path: Path,
) -> None:
    past, future = test_ds[window_idx]
    lookback = past.shape[-1]
    horizon = future.shape[-1]
    segment_z = torch.cat([past, future], dim=-1)
    wn, center, std = _window_norm(
        past.unsqueeze(0),
        segment_z.unsqueeze(0),
        std_floor=cfg["std_floor"],
        center_mode=cfg["center_mode"],
    )
    wn = wn[0, variate]
    ms = cfg["max_scale"]
    wn_clip = torch.clamp(wn, -ms, ms)
    clipped_mask = wn.abs() > ms

    encoder = TimeSeriesTo2D(height=cfg["coarse_h"], max_scale=ms)
    coarse, fine = encoder.encode_dual_heights(
        wn_clip.unsqueeze(0).unsqueeze(0),
        coarse_height=cfg["coarse_h"],
        fine_height=cfg["fine_h"],
    )
    coarse_np = coarse[0, 0].cpu().numpy()
    fine_np = fine[0, 0].cpu().numpy()

    seg_z = segment_z[variate].numpy()
    wn_np = wn.numpy()
    wn_clip_np = wn_clip.numpy()
    t = np.arange(len(seg_z))
    t_hor = t[lookback:]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 2.2, 2.2], hspace=0.35)
    fig.suptitle(
        f"{dataset} win={window_idx} var={variate} rank={rank} | "
        f"MS={ms:.1g} past_std={std[0, variate, 0].item():.3f} "
        f"clipped={int(clipped_mask.sum())}/{horizon} horizon cols",
        fontsize=10,
    )

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, seg_z, color="#1f77b4", lw=1.2, label="global z-score")
    ax0.axvline(lookback - 0.5, color="0.5", ls=":", lw=1)
    ax0.set_ylabel("z-score")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, wn_np, color="#2ca02c", lw=1.0, label="window-norm (pre-clip)")
    ax1.plot(t, wn_clip_np, color="#d62728", lw=1.0, ls="--", label="clipped input to encoder")
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
            label="clipped timesteps",
        )
    ax1.set_ylabel("window-norm σ")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.25)

    seq_len = coarse_np.shape[1]
    extent = [0, seq_len, 0, coarse_np.shape[0]]

    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(
        coarse_np,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax2.axvline(lookback, color="#ff7f0e", ls="--", lw=1, alpha=0.9)
    ax2.set_ylabel("coarse bin")
    ax2.set_title("coarse CDF (16 rows × time): filled rows = occupancy staircase", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01, label="occupancy")

    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(
        fine_np,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax3.axvline(lookback, color="#ff7f0e", ls="--", lw=1, alpha=0.9)
    ax3.set_ylabel("fine bin")
    ax3.set_xlabel("time index (orange line = lookback | horizon boundary)")
    ax3.set_title(
        "fine CDF (within-coarse-bin residual): flat past → flat; clipped horizon → pinned top bin",
        fontsize=9,
    )
    fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01, label="occupancy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    samples = list(_parse_samples(args.input_dir))
    if not samples:
        print(f"no samples found in {args.input_dir}")
        return 1

    cfg_cache: dict = {}
    ds_cache: dict = {}
    for dataset, win, var, rank, stem in samples:
        if dataset not in cfg_cache:
            cfg_cache[dataset] = _load_rep_cfg(args.config, dataset)
            _, _, ds_cache[dataset], _ = load_dataset(
                dataset, variate_indices=None, test_stride=TEST_STRIDE,
            )
        out_path = args.output_dir / f"{stem}_cdf.png"
        print(f"plot {stem}...", flush=True)
        _plot_sample(
            stem=stem,
            dataset=dataset,
            window_idx=win,
            variate=var,
            rank=rank,
            cfg=cfg_cache[dataset],
            test_ds=ds_cache[dataset],
            out_path=out_path,
        )

    print(f"wrote {len(samples)} figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
