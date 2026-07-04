#!/usr/bin/env python3
"""Measure dual-scale CDF encode→decode round-trip error on test windows.

Uses production window norm (train z-score, mean center, std floor) and
TimeSeriesTo2D dual 16×16 CDF maps with the mean column-sum decoder.

Reports errors in window-norm σ units and global train z-score units.
Also saves clipped-window visualizations under output_dir/clipped_samples/.

Example:
  python utils/measure_representation_roundtrip_floor.py
  python utils/measure_representation_roundtrip_floor.py --datasets electricity,dynamic
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.preprocessing import TimeSeriesTo2D  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    load_dataset,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "base" / "binary_staged.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "representation_roundtrip_floor"
DEFAULT_DATASETS = [k for k in DATASET_REGISTRY if k != "dalia"]
TEST_STRIDE = 4
BATCH_SIZE = 256
VIZ_DATASETS = (
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "dynamic", "weather", "electricity", "illness",
)
VIZ_PER_DATASET = 3


@dataclass
class DecompositionAccum:
    """Round-trip error split in z-score space: total = clip + quant + cross."""

    n_values: float = 0.0
    mse_total: float = 0.0
    mae_total: float = 0.0
    mse_clip: float = 0.0
    mse_quant: float = 0.0
    mse_cross: float = 0.0
    mae_clip: float = 0.0
    mae_quant: float = 0.0
    n_clipped: float = 0.0

    def merge(self, other: "DecompositionAccum") -> None:
        for field_name in (
            "n_values", "mse_total", "mae_total", "mse_clip", "mse_quant", "mse_cross",
            "mae_clip", "mae_quant", "n_clipped",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass(frozen=True)
class RepConfig:
    max_scale: float
    window_norm_std_floor: float
    window_norm_center: str
    coarse_height: int
    fine_height: int
    finer_height: int

    @property
    def epsilon_max(self) -> float:
        return self.max_scale / (self.coarse_height * self.fine_height)


@dataclass
class ErrorAccum:
    n_values: float = 0.0
    mse_total: float = 0.0
    mae_total: float = 0.0
    mse_clip_only: float = 0.0
    mse_quant_unclipped: float = 0.0
    n_clipped: float = 0.0
    n_unclipped: float = 0.0

    def merge(self, other: "ErrorAccum") -> None:
        self.n_values += other.n_values
        self.mse_total += other.mse_total
        self.mae_total += other.mae_total
        self.mse_clip_only += other.mse_clip_only
        self.mse_quant_unclipped += other.mse_quant_unclipped
        self.n_clipped += other.n_clipped
        self.n_unclipped += other.n_unclipped


def _load_rep_config(config_path: Path, dataset: str) -> RepConfig:
    with config_path.open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)["experiment"]
    ms_map = dict(exp.get("max_scale_by_dataset") or {})
    max_scale = float(ms_map.get(dataset, exp.get("max_scale", 3.5)))
    return RepConfig(
        max_scale=max_scale,
        window_norm_std_floor=float(exp.get("window_norm_std_floor", 1e-8)),
        window_norm_center=str(exp.get("window_norm_center", "mean")),
        coarse_height=int(exp.get("coarse_image_height", exp.get("image_height", 16))),
        fine_height=int(exp.get("fine_image_height", exp.get("image_height", 16))),
        finer_height=int(exp.get("finer_image_height", exp.get("image_height", 16))),
    )


def _window_norm_params(
    past_z: torch.Tensor,
    *,
    std_floor: float,
    center_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if center_mode == "last":
        center = past_z[..., -1:]
    elif center_mode == "mean":
        center = past_z.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {center_mode!r}")
    std = past_z.std(dim=-1, keepdim=True).clamp_min(std_floor)
    return center, std, (past_z - center) / std


def _roundtrip_dual(
    encoder: TimeSeriesTo2D,
    x: torch.Tensor,
    *,
    coarse_h: int,
    fine_h: int,
) -> torch.Tensor:
    coarse, fine = encoder.encode_dual_heights(
        x, coarse_height=coarse_h, fine_height=fine_h,
    )
    return encoder.decode_dual(coarse, fine, cdf_decoder="mean", squeeze_univariate=False)


def _accumulate_decomposition(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    max_scale: float,
    scale_to_z: torch.Tensor,
) -> DecompositionAccum:
    x_clipped = torch.clamp(x, -max_scale, max_scale)
    e_clip = (x_clipped - x) * scale_to_z
    e_quant = (x_hat - x_clipped) * scale_to_z
    e_total = (x_hat - x) * scale_to_z
    clipped = x.abs() > max_scale

    acc = DecompositionAccum()
    acc.n_values = float(x.numel())
    acc.mse_total = float((e_total * e_total).sum().item())
    acc.mae_total = float(e_total.abs().sum().item())
    acc.mse_clip = float((e_clip * e_clip).sum().item())
    acc.mse_quant = float((e_quant * e_quant).sum().item())
    acc.mse_cross = float((e_total * e_total - e_clip * e_clip - e_quant * e_quant).sum().item())
    acc.mae_clip = float(e_clip.abs().sum().item())
    acc.mae_quant = float(e_quant.abs().sum().item())
    acc.n_clipped = float(clipped.sum().item())
    return acc


def _finalize_decomposition(acc: DecompositionAccum, *, prefix: str) -> Dict[str, float]:
    n = acc.n_values
    mse_total = acc.mse_total / n
    mae_total = acc.mae_total / n
    mse_clip = acc.mse_clip / n
    mse_quant = acc.mse_quant / n
    mse_cross = acc.mse_cross / n
    mae_clip = acc.mae_clip / n
    mae_quant = acc.mae_quant / n
    mae_denom = mae_clip + mae_quant
    return {
        f"{prefix}_mse": mse_total,
        f"{prefix}_rmse": mse_total ** 0.5,
        f"{prefix}_mae": mae_total,
        f"{prefix}_mse_clip": mse_clip,
        f"{prefix}_mse_quant": mse_quant,
        f"{prefix}_mse_cross": mse_cross,
        f"{prefix}_pct_mse_clip": 100.0 * mse_clip / mse_total if mse_total > 0 else 0.0,
        f"{prefix}_pct_mse_quant": 100.0 * mse_quant / mse_total if mse_total > 0 else 0.0,
        f"{prefix}_pct_mse_cross": 100.0 * mse_cross / mse_total if mse_total > 0 else 0.0,
        f"{prefix}_mae_clip": mae_clip,
        f"{prefix}_mae_quant": mae_quant,
        f"{prefix}_pct_mae_clip": 100.0 * mae_clip / mae_denom if mae_denom > 0 else 0.0,
        f"{prefix}_pct_mae_quant": 100.0 * mae_quant / mae_denom if mae_denom > 0 else 0.0,
        f"{prefix}_frac_clipped": acc.n_clipped / n,
    }


def _accumulate_errors(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    max_scale: float,
    scale_to_z: Optional[torch.Tensor] = None,
) -> ErrorAccum:
    """Accumulate round-trip errors; optionally scale errors to z-score space."""
    err = x_hat - x
    abs_err = err.abs()
    clipped = x.abs() > max_scale
    clip_err = torch.clamp(x, -max_scale, max_scale) - x
    unclipped = ~clipped

    if scale_to_z is not None:
        err = err * scale_to_z
        abs_err = abs_err * scale_to_z
        clip_err = clip_err * scale_to_z
        quant_err = err[unclipped] if unclipped.any() else err.new_zeros(0)
    else:
        quant_err = err[unclipped] if unclipped.any() else err.new_zeros(0)

    acc = ErrorAccum()
    acc.n_values = float(x.numel())
    acc.mse_total = float((err * err).sum().item())
    acc.mae_total = float(abs_err.sum().item())
    acc.mse_clip_only = float((clip_err * clip_err).sum().item())
    acc.mse_quant_unclipped = float((quant_err * quant_err).sum().item())
    acc.n_clipped = float(clipped.sum().item())
    acc.n_unclipped = float(unclipped.sum().item())
    return acc


def _finalize(acc: ErrorAccum, *, epsilon_max: float, space_label: str) -> Dict[str, float]:
    n = acc.n_values
    out = {
        f"{space_label}_mse": acc.mse_total / n,
        f"{space_label}_rmse": (acc.mse_total / n) ** 0.5,
        f"{space_label}_mae": acc.mae_total / n,
        f"{space_label}_mse_clip_only": acc.mse_clip_only / n,
        f"{space_label}_rmse_clip_only": (acc.mse_clip_only / n) ** 0.5,
        f"{space_label}_frac_clipped": acc.n_clipped / n,
    }
    if acc.n_unclipped > 0:
        out[f"{space_label}_mse_quant_unclipped"] = acc.mse_quant_unclipped / acc.n_unclipped
        out[f"{space_label}_rmse_quant_unclipped"] = (
            acc.mse_quant_unclipped / acc.n_unclipped
        ) ** 0.5
    else:
        out[f"{space_label}_mse_quant_unclipped"] = float("nan")
        out[f"{space_label}_rmse_quant_unclipped"] = float("nan")
    if space_label == "wn":
        out["epsilon_max"] = epsilon_max
    return out


@dataclass
class ClipVizCandidate:
    dataset: str
    window_idx: int
    variate: int
    n_clipped_horizon: int
    max_abs_wn: float
    past_std: float
    segment_z: torch.Tensor
    segment_z_hat: torch.Tensor
    wn_horizon: torch.Tensor
    wn_hat_horizon: torch.Tensor
    clipped_horizon: torch.Tensor
    lookback: int


def _plot_clipped_sample(sample: ClipVizCandidate, out_path: Path, max_scale: float) -> None:
    lookback = sample.lookback
    horizon = sample.wn_horizon.shape[-1]
    t_seg = torch.arange(lookback + horizon)
    t_hor = torch.arange(lookback, lookback + horizon)

    seg_z = sample.segment_z[sample.variate].numpy()
    seg_z_hat = sample.segment_z_hat[sample.variate].numpy()
    wn_h = sample.wn_horizon[sample.variate].numpy()
    wn_hat_h = sample.wn_hat_horizon[sample.variate].numpy()
    clipped = sample.clipped_horizon[sample.variate].numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle(
        f"{sample.dataset} window={sample.window_idx} variate={sample.variate} | "
        f"past_std={sample.past_std:.3f} max|wn|={sample.max_abs_wn:.2f} "
        f"MS={max_scale:.1f} clipped_horizon={sample.n_clipped_horizon}/{horizon}",
        fontsize=10,
    )

    ax = axes[0]
    ax.plot(t_seg, seg_z, color="#1f77b4", lw=1.2, label="z-score original")
    ax.plot(t_seg, seg_z_hat, color="#ff7f0e", lw=1.0, ls="--", label="z-score round-trip")
    ax.axvline(lookback - 0.5, color="0.5", ls=":", lw=1)
    ax.set_ylabel("global z-score")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(t_hor, wn_h, color="#2ca02c", lw=1.2, label="window-norm original")
    ax.plot(t_hor, wn_hat_h, color="#d62728", lw=1.0, ls="--", label="window-norm round-trip")
    ax.axhline(max_scale, color="#9467bd", ls=":", lw=1, label="±max_scale")
    ax.axhline(-max_scale, color="#9467bd", ls=":", lw=1)
    if clipped.any():
        ax.scatter(
            t_hor[clipped],
            wn_h[clipped],
            c="#e377c2",
            s=28,
            zorder=5,
            label="clipped (original)",
        )
    ax.set_ylabel("window-norm σ")
    ax.set_xlabel("time index")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure_dataset(
    dataset: str,
    *,
    config_path: Path,
    test_stride: int,
    batch_size: int,
    max_windows: Optional[int],
    viz_dir: Optional[Path],
    viz_per_dataset: int,
) -> Dict[str, Any]:
    cfg = _load_rep_config(config_path, dataset)
    encoder = TimeSeriesTo2D(height=cfg.coarse_height, max_scale=cfg.max_scale)

    _, _, test_ds, _ = load_dataset(dataset, variate_indices=None, test_stride=test_stride)
    n_windows = len(test_ds)
    if max_windows is not None:
        n_windows = min(n_windows, int(max_windows))

    wn_acc = ErrorAccum()
    z_acc = ErrorAccum()
    wn_horizon_acc = ErrorAccum()
    z_horizon_acc = ErrorAccum()
    z_horizon_decomp = DecompositionAccum()
    viz_candidates: List[ClipVizCandidate] = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        past_batch = []
        future_batch = []
        for idx in range(start, end):
            past, future = test_ds[idx]
            past_batch.append(past)
            future_batch.append(future)
        past_z = torch.stack(past_batch, dim=0)
        future_z = torch.stack(future_batch, dim=0)
        segment_z = torch.cat([past_z, future_z], dim=-1)
        center, std, _past_wn = _window_norm_params(
            past_z,
            std_floor=cfg.window_norm_std_floor,
            center_mode=cfg.window_norm_center,
        )
        wn = (segment_z - center) / std
        horizon = future_z.shape[-1]
        lookback = past_z.shape[-1]
        wn_horizon = wn[..., -horizon:]
        std_horizon = std.expand_as(wn_horizon)

        wn_hat = _roundtrip_dual(
            encoder, wn,
            coarse_h=cfg.coarse_height, fine_h=cfg.fine_height,
        )
        wn_hat_horizon = wn_hat[..., -horizon:]
        segment_z_hat = wn_hat * std + center

        wn_acc.merge(_accumulate_errors(wn, wn_hat, max_scale=cfg.max_scale))
        z_acc.merge(
            _accumulate_errors(wn, wn_hat, max_scale=cfg.max_scale, scale_to_z=std.expand_as(wn)),
        )
        wn_horizon_acc.merge(
            _accumulate_errors(wn_horizon, wn_hat_horizon, max_scale=cfg.max_scale),
        )
        z_horizon_acc.merge(
            _accumulate_errors(
                wn_horizon, wn_hat_horizon,
                max_scale=cfg.max_scale, scale_to_z=std_horizon,
            ),
        )
        z_horizon_decomp.merge(
            _accumulate_decomposition(
                wn_horizon, wn_hat_horizon,
                max_scale=cfg.max_scale, scale_to_z=std_horizon,
            ),
        )

        if viz_dir is not None and dataset in VIZ_DATASETS:
            clipped_h = wn_horizon.abs() > cfg.max_scale
            for bi in range(wn_horizon.shape[0]):
                window_idx = start + bi
                for vi in range(wn_horizon.shape[1]):
                    n_clip = int(clipped_h[bi, vi].sum().item())
                    if n_clip == 0:
                        continue
                    max_abs = float(wn_horizon[bi, vi].abs().max().item())
                    viz_candidates.append(
                        ClipVizCandidate(
                            dataset=dataset,
                            window_idx=window_idx,
                            variate=vi,
                            n_clipped_horizon=n_clip,
                            max_abs_wn=max_abs,
                            past_std=float(std[bi, vi, 0].item()),
                            segment_z=segment_z[bi].detach().cpu(),
                            segment_z_hat=segment_z_hat[bi].detach().cpu(),
                            wn_horizon=wn_horizon[bi].detach().cpu(),
                            wn_hat_horizon=wn_hat_horizon[bi].detach().cpu(),
                            clipped_horizon=clipped_h[bi].detach().cpu(),
                            lookback=lookback,
                        )
                    )

    row: Dict[str, Any] = {
        "dataset": dataset,
        "max_scale": cfg.max_scale,
        "epsilon_max": cfg.epsilon_max,
        "n_test_windows": float(n_windows),
        "n_values_horizon": wn_horizon_acc.n_values,
    }
    row.update(_finalize(wn_acc, epsilon_max=cfg.epsilon_max, space_label="wn_segment"))
    row.update(_finalize(z_acc, epsilon_max=cfg.epsilon_max, space_label="z_segment"))
    row.update(_finalize(wn_horizon_acc, epsilon_max=cfg.epsilon_max, space_label="wn_horizon"))
    row.update(_finalize(z_horizon_acc, epsilon_max=cfg.epsilon_max, space_label="z_horizon"))
    row.update(_finalize_decomposition(z_horizon_decomp, prefix="z_horizon"))

    if viz_dir is not None and viz_candidates:
        viz_candidates.sort(key=lambda c: (c.n_clipped_horizon, c.max_abs_wn), reverse=True)
        for rank, cand in enumerate(viz_candidates[:viz_per_dataset]):
            fname = f"{dataset}_win{cand.window_idx}_var{cand.variate}_rank{rank}.png"
            _plot_clipped_sample(cand, viz_dir / fname, max_scale=cfg.max_scale)

    return row


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )


def _write_markdown(rows: Sequence[Dict[str, Any]], path: Path, viz_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representation round-trip floor (test windows)",
        "",
        "Pipeline: train z-score → window mean-norm → dual 16×16 CDF round-trip (`mean` decoder).",
        f"Test stride={TEST_STRIDE}. Horizon metrics in global train z-score space.",
        "",
        "## Z-score horizon: clipping vs quantization",
        "",
        "Exact MSE split with `c = clamp(x)` before encode: "
        "`MSE_total = MSE_clip + MSE_quant + MSE_cross`, where "
        "`MSE_clip = E[(c-x)²]`, `MSE_quant = E[(x̂-c)²]` (in z-score units).",
        "MAE % uses component magnitudes `|c-x|` vs `|x̂-c|` (approximate attribution).",
        "",
        "| dataset | MSE | MAE | clip MSE (% ) | quant MSE (% ) | cross MSE (% ) | clip MAE (% ) | quant MAE (% ) | frac clipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['z_horizon_mse']:.6f} | {row['z_horizon_mae']:.6f} | "
            f"{row['z_horizon_mse_clip']:.6f} ({row['z_horizon_pct_mse_clip']:.1f}%) | "
            f"{row['z_horizon_mse_quant']:.6f} ({row['z_horizon_pct_mse_quant']:.1f}%) | "
            f"{row['z_horizon_mse_cross']:.6f} ({row['z_horizon_pct_mse_cross']:.1f}%) | "
            f"{row['z_horizon_pct_mae_clip']:.1f}% | {row['z_horizon_pct_mae_quant']:.1f}% | "
            f"{row['z_horizon_frac_clipped']:.4f} |"
        )

    lines.extend([
        "",
        "## Horizon errors in window-norm σ space",
        "",
        "| dataset | ε_max | MSE | MAE | RMSE | clip RMSE | quant RMSE (unclipped) | frac clipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['epsilon_max']:.4f} | "
            f"{row['wn_horizon_mse']:.6f} | {row['wn_horizon_mae']:.6f} | "
            f"{row['wn_horizon_rmse']:.6f} | {row['wn_horizon_rmse_clip_only']:.6f} | "
            f"{row['wn_horizon_rmse_quant_unclipped']:.6f} | "
            f"{row['wn_horizon_frac_clipped']:.4f} |"
        )

    lines.extend([
        "",
        "## Clipped sample plots",
        "",
        f"Saved under `{viz_dir.relative_to(REPO_ROOT)}/` for "
        f"{', '.join(VIZ_DATASETS)} (top windows by clip count).",
        "",
        "Theory: dual ε_max = max_scale / 256 (window-norm σ units).",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--test-stride", type=int, default=TEST_STRIDE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--viz-per-dataset", type=int, default=VIZ_PER_DATASET)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args(argv)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = list(DEFAULT_DATASETS)

    out_dir = args.output_dir
    viz_dir = None if args.no_viz else out_dir / "clipped_samples"

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for dataset in datasets:
        try:
            print(f"measuring {dataset}...", flush=True)
            row = measure_dataset(
                dataset,
                config_path=args.config,
                test_stride=args.test_stride,
                batch_size=args.batch_size,
                max_windows=args.max_windows,
                viz_dir=viz_dir,
                viz_per_dataset=args.viz_per_dataset,
            )
            rows.append(row)
            print(
                f"  z-horizon MSE={row['z_horizon_mse']:.6f} "
                f"(clip {row['z_horizon_pct_mse_clip']:.1f}% / "
                f"quant {row['z_horizon_pct_mse_quant']:.1f}% / "
                f"cross {row['z_horizon_pct_mse_cross']:.1f}%) "
                f"MAE={row['z_horizon_mae']:.6f} "
                f"frac_clip={row['z_horizon_frac_clipped']:.4f}",
                flush=True,
            )
        except Exception as exc:
            failures.append(f"{dataset}: {exc}")
            print(f"  SKIP {dataset}: {exc}", flush=True)

    _write_csv(rows, out_dir / "summary.csv")
    _write_markdown(rows, out_dir / "summary.md", viz_dir or out_dir / "clipped_samples")
    if failures:
        (out_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'}")
    if viz_dir is not None:
        print(f"plots in {viz_dir}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
