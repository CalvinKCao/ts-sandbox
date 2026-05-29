"""
MMPD probabilistic forecast plots from matrix eval NPZ cache.

Same layout as visualize_probabilistic.py: one test window, all variates,
context + GT + iTransformer + stochastic futures (or point-only mode).

Usage:
    python -m models.diffusion_tsf.visualize_mmpd_probabilistic \\
        --matrix-dir results/datasets/05-27-804460-mmpd-anchor-matrix \\
        --dataset ETTh1 --mode samples
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    load_itransformer_from_checkpoint,
)
from utils.eval_mmpd_gaussian_anchor import load_tsf_pipeline  # noqa: E402


def localize_path(path: str) -> Path:
    return Path(path.replace("/scratch/ccao87/ts-sandbox", project_root))


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def denorm_series(x: np.ndarray, mean_np: np.ndarray, std_np: np.ndarray) -> np.ndarray:
    m = np.asarray(mean_np, dtype=np.float64).reshape(-1, 1)
    s = np.asarray(std_np, dtype=np.float64).reshape(-1, 1)
    return np.asarray(x, dtype=np.float64) * s + m


def load_itrans_path(matrix_dir: Path, dataset: str) -> Path:
    manifest_path = matrix_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    gauss = manifest.get("anchor_runs", {}).get("gaussian", {}).get(dataset)
    if not gauss:
        raise KeyError(f"No gaussian anchor entry for {dataset} in manifest")
    itrans = localize_path(gauss["itrans_pt"])
    if not itrans.exists():
        raise FileNotFoundError(f"Missing iTrans checkpoint: {itrans}")
    return itrans


def resolve_pack_row(
    eval_indices: List[int],
    n_test: int,
    sample_index: Optional[int],
    random_seed: int,
) -> tuple[int, int]:
    """Return (pack_row, test_index)."""
    n_pack = len(eval_indices)
    if sample_index is not None:
        test_idx = int(sample_index)
        if test_idx not in eval_indices:
            raise ValueError(
                f"sample_index {test_idx} not in matrix eval subset ({n_pack} windows)"
            )
        return eval_indices.index(test_idx), test_idx

    rng = random.Random(random_seed)
    test_idx = rng.randint(0, n_test - 1)
    if test_idx in eval_indices:
        return eval_indices.index(test_idx), test_idx
    pack_row = rng.randint(0, n_pack - 1)
    return pack_row, eval_indices[pack_row]


def run_mmpd_visualization(
    matrix_dir: Path,
    dataset: str,
    output_dir: str,
    mode: str = "samples",
    num_futures: int = 5,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    sample_index: Optional[int] = None,
    random_seed: int = 42,
    name_suffix: str = "mmpd",
) -> str:
    npz_path = matrix_dir / "raw" / f"mmpd_{dataset}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    pack = np.load(npz_path)
    y_true = pack["y_true"]
    det = pack["deterministic"]
    samples = pack["samples"]
    eval_indices = [int(i) for i in pack["indices"].tolist()]
    subset_id = dataset

    manifest_path = matrix_dir / "run_manifest.json"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    meta = manifest["anchor_runs"]["gaussian"][dataset]["metadata"]
    variate_indices = list(meta["variate_indices"])
    var_names = meta.get("variate_names") or []
    n_vars = len(variate_indices)

    pipeline = load_tsf_pipeline()
    _, _, test_ds, norm_stats = pipeline.load_dataset(
        dataset,
        variate_indices,
        lookback=lookback_length,
        horizon=forecast_length,
        stride=1,
    )
    n_test = len(test_ds)
    pack_row, test_index = resolve_pack_row(
        eval_indices, n_test, sample_index, random_seed
    )

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)
    mean_np = np.asarray(norm_stats["mean"], dtype=np.float64)
    std_np = np.asarray(norm_stats["std"], dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    itrans_path = load_itrans_path(matrix_dir, dataset)
    itrans_model = load_itransformer_from_checkpoint(str(itrans_path), n_vars, device)

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)
    with torch.no_grad():
        B, C, L = past_t.shape
        x_enc = past_t.permute(0, 2, 1)
        seq_sl = getattr(itrans_model, "seq_len", L)
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        x_dec = torch.zeros(B, forecast_length, C, device=device)
        itrans_out = itrans_model(x_enc, None, x_dec, None)
        if isinstance(itrans_out, tuple):
            itrans_out = itrans_out[0]
        itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]

    past_dn = denorm(past, mean, std)
    future_dn = denorm(future[:, -forecast_length:], mean, std)
    itrans_dn = denorm(itrans_pred, mean, std)

    gt = denorm_series(y_true[pack_row], mean_np, std_np)
    point = denorm_series(det[pack_row], mean_np, std_np)
    n_avail = samples.shape[2]
    n_plot = min(num_futures, n_avail) if mode == "samples" else 0

    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        constrained_layout=True,
    )
    axes = axes.flatten() if n_vars > 1 else np.array([axes])

    context_len = min(forecast_length * 2, lookback_length)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, forecast_length)

    for col in range(n_vars):
        ax = axes[col]
        ax.plot(
            t_past,
            past_dn[col, -context_len:].numpy(),
            color="#757575",
            alpha=0.6,
            linewidth=1.0,
            label="Context" if col == 0 else "",
        )
        ax.plot(
            t_future,
            gt[col],
            color="#2196F3",
            linewidth=2.0,
            label="Ground Truth" if col == 0 else "",
        )
        ax.plot(
            t_future,
            itrans_dn[col].numpy(),
            color="#FF9800",
            linewidth=1.6,
            linestyle="--",
            label="iTransformer" if col == 0 else "",
        )
        if mode == "point":
            ax.plot(
                t_future,
                point[col],
                color="#E91E63",
                linewidth=1.6,
                linestyle="-",
                label="MMPD point" if col == 0 else "",
            )
        else:
            for i in range(n_plot):
                sample_dn = denorm_series(samples[pack_row, :, i, :], mean_np, std_np)
                ax.plot(
                    t_future,
                    sample_dn[col],
                    color="#E91E63",
                    linewidth=0.9,
                    alpha=0.45,
                    label="MMPD futures" if (col == 0 and i == 0) else "",
                )

        ax.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        ax.grid(True, alpha=0.2)
        vname = var_names[col] if col < len(var_names) else f"Var {col}"
        ax.set_title(vname, fontsize=11, fontweight="semibold")
        ax.tick_params(labelsize=8)

    for col in range(n_vars, len(axes)):
        fig.delaxes(axes[col])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, 1.03),
        )

    mode_label = "Point forecast" if mode == "point" else f"Stochastic futures (N={n_plot})"
    fig.suptitle(
        f"{dataset} • MMPD • Sample {test_index} • {mode_label}",
        fontsize=13,
        fontweight="bold",
        y=0.98 if n_rows > 1 else 0.95,
    )

    os.makedirs(output_dir, exist_ok=True)
    tag = f"{name_suffix}_point" if mode == "point" else name_suffix
    out_path = os.path.join(
        output_dir,
        f"probabilistic_forecast_{dataset}_{subset_id}_{tag}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probabilistic forecast plot to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--mode",
        choices=("samples", "point"),
        default="samples",
        help="samples=5 stochastic draws from NPZ; point=MMPD deterministic path",
    )
    parser.add_argument("--num-futures", type=int, default=5)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-suffix", type=str, default="mmpd")
    args = parser.parse_args()

    run_mmpd_visualization(
        matrix_dir=args.matrix_dir.resolve(),
        dataset=args.dataset,
        output_dir=args.output_dir,
        mode=args.mode,
        num_futures=args.num_futures,
        sample_index=args.index,
        random_seed=args.seed,
        name_suffix=args.name_suffix,
    )


if __name__ == "__main__":
    main()
