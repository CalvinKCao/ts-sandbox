#!/usr/bin/env python3
"""Top test windows by per-window anchor_mse / CRPS delta (binary grad_accum − fair MMPD).

Uses saved eval npz only (no GPU inference). Skips datasets without a finished
binary grad_accum_150_lr_lo staged eval unless --allow-fallback-binary is set.

Example:
  python utils/visualize_fair_mmpd_vs_binary_delta.py
  python utils/visualize_fair_mmpd_vs_binary_delta.py --datasets ETTh1,traffic --top-k 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import generate_dataset_job, load_dataset
from models.diffusion_tsf.visualize_comparison import denorm
from utils.eval_mmpd_gaussian_anchor import (
    _load_data_subset_policy,
    crps_gr,
    resolve_subset_meta_for_dataset,
)

DEFAULT_MMPD_RUN = REPO_ROOT / "results" / "datasets" / "06-16-mmpd-maskae-fair-13d"
SUBSET_CONFIG = REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "fair_mmpd_vs_ema099_grad_accum_150"
BINARY_CONFIG = "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo"
FALLBACK_BINARY_CONFIG = "binary_anchor_stationary_flat_subsets_ema099"
EVAL_TEST_STRIDE = 4
TRAIN_STRIDE = 1
PROB_COLORS = ["#E91E63", "#FF9800", "#4CAF50"]
ALL_DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "illness",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "PeMS",
    "solar_Alabama",
    "dalia",
    "dynamic",
)
# Binary grad_accum_150_lr_lo staged eval completed for all 13 flat-subset datasets.
FINISHED_GRAD_ACCUM_DATASETS = ALL_DATASETS


@dataclass
class BinaryRun:
    results_dir: Path
    ckpt_dir: Path
    metrics: Dict[str, float]
    config_suffix: str


@dataclass
class AlignedPack:
    indices: np.ndarray
    y_true: np.ndarray
    binary_det: np.ndarray
    binary_samples: np.ndarray
    mmpd_det: np.ndarray
    mmpd_samples: np.ndarray


def _read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _finished_binary_run(
    datasets_root: Path,
    ckpt_root: Path,
    dataset: str,
    config_suffix: str,
) -> Optional[BinaryRun]:
    pattern = f"*-{dataset}-{config_suffix}"
    candidates: List[Tuple[float, Path, Dict[str, float]]] = []
    for results_dir in datasets_root.glob(pattern):
        if not results_dir.is_dir():
            continue
        partial = results_dir / "partials" / f"{dataset}_staged_anchor.json"
        anchor_npz = results_dir / "raw" / f"staged_anchor_{dataset}.npz"
        samples_npz = results_dir / "raw" / f"staged_dpmpp_samples_{dataset}.npz"
        if not (partial.is_file() and anchor_npz.is_file() and samples_npz.is_file()):
            continue
        metrics = _read_json(partial)
        if "anchor_mse" not in metrics or "crps" not in metrics:
            continue
        candidates.append((results_dir.stat().st_mtime, results_dir, metrics))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    results_dir = candidates[0][1]
    metrics = candidates[0][2]
    ckpt_matches = sorted(
        ckpt_root.glob(f"*-{dataset}-{config_suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    ckpt_dir = ckpt_matches[0] if ckpt_matches else results_dir
    return BinaryRun(
        results_dir=results_dir,
        ckpt_dir=ckpt_dir,
        metrics=metrics,
        config_suffix=config_suffix,
    )


def discover_binary_run(
    datasets_root: Path,
    ckpt_root: Path,
    dataset: str,
    *,
    allow_fallback: bool,
) -> BinaryRun:
    run = _finished_binary_run(datasets_root, ckpt_root, dataset, BINARY_CONFIG)
    if run is not None:
        return run
    if allow_fallback:
        fb = _finished_binary_run(datasets_root, ckpt_root, dataset, FALLBACK_BINARY_CONFIG)
        if fb is not None:
            return fb
    raise FileNotFoundError(
        f"No finished binary staged eval for {dataset} "
        f"({BINARY_CONFIG}); rerun with --allow-fallback-binary for ema099."
    )


def load_mmpd_pack(mmpd_run: Path, dataset: str) -> Dict[str, np.ndarray]:
    path = mmpd_run / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MMPD eval npz: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_binary_pack(binary_run: BinaryRun, dataset: str) -> Dict[str, np.ndarray]:
    anchor_path = binary_run.results_dir / "raw" / f"staged_anchor_{dataset}.npz"
    samples_path = binary_run.results_dir / "raw" / f"staged_dpmpp_samples_{dataset}.npz"
    with np.load(anchor_path) as anchor:
        det = anchor["deterministic"]
        y_true_anchor = anchor["y_true"]
    with np.load(samples_path) as samples:
        y_true = samples["y_true"]
        sample_arr = samples["samples"]
    if not np.allclose(y_true_anchor, y_true, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"{dataset}: staged anchor vs dpmpp y_true mismatch")
    return {
        "y_true": y_true,
        "deterministic": det,
        "samples": sample_arr,
    }


def align_packs(
    mmpd: Dict[str, np.ndarray],
    binary: Dict[str, np.ndarray],
    dataset: str,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> AlignedPack:
    indices = np.asarray(mmpd["indices"], dtype=np.int64)
    n_bin = binary["y_true"].shape[0]
    if indices.max(initial=0) >= n_bin:
        raise RuntimeError(
            f"{dataset}: MMPD index {int(indices.max())} out of binary range {n_bin}"
        )
    bin_y = binary["y_true"][indices]
    bin_det = binary["deterministic"][indices]
    bin_samples = binary["samples"][indices]
    m_y = mmpd["y_true"]
    if not np.allclose(bin_y, m_y, rtol=rtol, atol=atol):
        bad = int(np.argmax(np.abs(bin_y - m_y).reshape(len(indices), -1).mean(axis=1)))
        raise RuntimeError(
            f"{dataset}: y_true mismatch at window idx={int(indices[bad])} "
            f"(row {bad}); check eval_test_stride alignment."
        )
    return AlignedPack(
        indices=indices,
        y_true=m_y,
        binary_det=bin_det,
        binary_samples=bin_samples,
        mmpd_det=mmpd["deterministic"],
        mmpd_samples=mmpd["samples"],
    )


def per_window_anchor_mse(y_true: np.ndarray, det: np.ndarray) -> np.ndarray:
    return ((y_true - det) ** 2).mean(axis=(1, 2))


def per_window_crps(y_true: np.ndarray, samples: np.ndarray, *, chunk: int = 32) -> np.ndarray:
    batch = y_true.shape[0]
    out = np.empty(batch, dtype=np.float64)
    for start in range(0, batch, chunk):
        end = min(start + chunk, batch)
        yt = y_true[start:end]
        ss = samples[start:end].astype(np.float64)
        term1 = np.abs(ss - yt[:, :, None, :]).mean(axis=2)
        term2 = np.abs(ss[:, :, :, None, :] - ss[:, :, None, :, :]).mean(axis=(2, 3))
        out[start:end] = (term1 - 0.5 * term2).mean(axis=(1, 2))
    return out


def rank_top_k(delta: np.ndarray, top_k: int) -> np.ndarray:
    """Indices into delta array, descending delta (binary − mmpd)."""
    k = min(top_k, delta.size)
    if k <= 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-delta)
    return order[:k]


def _variate_names(dataset: str, n_vars: int) -> List[str]:
    job = generate_dataset_job(dataset)
    policy = _load_data_subset_policy(SUBSET_CONFIG)
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    indices = [int(i) for i in subset["variate_indices"][:n_vars]]
    all_names = job.get("variate_names") or []
    if all_names and max(indices, default=0) < len(all_names):
        return [str(all_names[i]) for i in indices]
    return [f"v{i}" for i in range(n_vars)]


def _load_test_context(
    dataset: str,
    window_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    policy = _load_data_subset_policy(SUBSET_CONFIG)
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
    )
    past_list = []
    future_list = []
    for idx in window_indices:
        past, future = test_ds[int(idx)]
        past_list.append(past)
        future_list.append(future)
    past_batch = torch.stack(past_list, dim=0)
    future_batch = torch.stack(future_list, dim=0)
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)
    return past_batch, future_batch, {"mean": mean, "std": std}


def _plot_prob_lines(ax, t_future: np.ndarray, prob_lines: Sequence[torch.Tensor], col: int) -> None:
    for pi, prob in enumerate(prob_lines):
        color = PROB_COLORS[pi % len(PROB_COLORS)]
        ax.plot(
            t_future,
            prob[col].numpy(),
            color=color,
            lw=1.0,
            alpha=0.75,
            label=f"sample {pi + 1}" if col == 0 else "",
        )


def plot_window_panel(
    *,
    dataset: str,
    window_idx: int,
    rank: int,
    metric: str,
    delta: float,
    binary_mse: float,
    mmpd_mse: float,
    binary_crps: float,
    mmpd_crps: float,
    past: torch.Tensor,
    future: torch.Tensor,
    norm: Dict[str, torch.Tensor],
    pack_row: int,
    aligned: AlignedPack,
    output_path: Path,
    prob_draws: int,
    context_len: int,
) -> None:
    mean, std = norm["mean"], norm["std"]
    n_vars = aligned.y_true.shape[1]
    horizon = aligned.y_true.shape[2]
    var_names = _variate_names(dataset, n_vars)

    past_dn = denorm(past, mean, std)
    gt_t = torch.from_numpy(aligned.y_true[pack_row]).to(dtype=torch.float32)
    gt_dn = denorm(gt_t, mean, std)

    bin_det_dn = denorm(torch.from_numpy(aligned.binary_det[pack_row]), mean, std)
    mmpd_det_dn = denorm(torch.from_numpy(aligned.mmpd_det[pack_row]), mean, std)

    n_s = min(prob_draws, aligned.binary_samples.shape[2])
    bin_probs = [
        denorm(torch.from_numpy(aligned.binary_samples[pack_row, :, si, :]), mean, std)
        for si in range(n_s)
    ]
    mmpd_probs = [
        denorm(torch.from_numpy(aligned.mmpd_samples[pack_row, :, si, :]), mean, std)
        for si in range(n_s)
    ]

    t_past = np.arange(-context_len, 0)
    t_future = np.arange(horizon)
    row_labels = ["Binary grad_accum", "Fair MMPD"]

    fig, axes = plt.subplots(
        2,
        n_vars,
        figsize=(4.8 * n_vars, 5.2),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{dataset} | test window {window_idx} | rank {rank} by {metric} Δ\n"
        f"Δ={delta:+.5f}  anchor_mse: bin={binary_mse:.5f} mmpd={mmpd_mse:.5f}  "
        f"crps: bin={binary_crps:.5f} mmpd={mmpd_crps:.5f}",
        fontsize=11,
    )

    for row, (label, det_dn, prob_dns) in enumerate(
        [
            (row_labels[0], bin_det_dn, bin_probs),
            (row_labels[1], mmpd_det_dn, mmpd_probs),
        ]
    ):
        for col in range(n_vars):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#424242",
                lw=1.1,
                alpha=0.85,
            )
            ax.plot(t_future, gt_dn[col].numpy(), color="#2196F3", lw=1.8, label="GT")
            ax.plot(t_future, det_dn[col].numpy(), color="#6A1B9A", lw=1.6, label="anchor")
            _plot_prob_lines(ax, t_future, prob_dns, col)
            if row == 0:
                ax.set_title(var_names[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def process_dataset(
    dataset: str,
    *,
    mmpd_run: Path,
    datasets_root: Path,
    ckpt_root: Path,
    output_dir: Path,
    top_k: int,
    prob_draws: int,
    allow_fallback: bool,
    skip_plots: bool,
) -> Dict[str, object]:
    binary_run = discover_binary_run(
        datasets_root, ckpt_root, dataset, allow_fallback=allow_fallback
    )
    mmpd = load_mmpd_pack(mmpd_run, dataset)
    binary = load_binary_pack(binary_run, dataset)
    aligned = align_packs(mmpd, binary, dataset)

    mse_bin = per_window_anchor_mse(aligned.y_true, aligned.binary_det)
    mse_mmpd = per_window_anchor_mse(aligned.y_true, aligned.mmpd_det)
    crps_bin = per_window_crps(aligned.y_true, aligned.binary_samples)
    crps_mmpd = per_window_crps(aligned.y_true, aligned.mmpd_samples)

    mse_delta = mse_bin - mse_mmpd
    crps_delta = crps_bin - crps_mmpd

    # Sanity: aggregate should match partials roughly.
    agg_mse_delta = float(mse_delta.mean())
    agg_crps_delta = float(crps_delta.mean())
    print(
        f"[{dataset}] windows={len(aligned.indices)} "
        f"binary={binary_run.results_dir.name} "
        f"mean Δmse={agg_mse_delta:+.6f} mean Δcrps={agg_crps_delta:+.6f} "
        f"(partial anchor_mse bin={binary_run.metrics.get('anchor_mse')} "
        f"crps={binary_run.metrics.get('crps')})"
    )

    mmpd_partial = mmpd_run / "partials" / f"{dataset}_mmpd.json"
    if mmpd_partial.is_file():
        mmpd_metrics = _read_json(mmpd_partial)
        print(
            f"  mmpd partial anchor_mse={mmpd_metrics.get('anchor_mse')} "
            f"crps={mmpd_metrics.get('crps')}"
        )

    rankings: Dict[str, List[Dict[str, float]]] = {}
    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    for metric_name, delta, b_vals, m_vals in (
        ("anchor_mse", mse_delta, mse_bin, mse_mmpd),
        ("crps", crps_delta, crps_bin, crps_mmpd),
    ):
        top_rows = rank_top_k(delta, top_k)
        rows_meta: List[Dict[str, float]] = []
        for rank, pack_row in enumerate(top_rows, start=1):
            win_idx = int(aligned.indices[pack_row])
            row = {
                "rank": rank,
                "test_window_index": win_idx,
                "pack_row": int(pack_row),
                "delta": float(delta[pack_row]),
                "binary": float(b_vals[pack_row]),
                "mmpd": float(m_vals[pack_row]),
            }
            rows_meta.append(row)
            if skip_plots:
                continue
            past_batch, future_batch, norm = _load_test_context(dataset, [win_idx])
            plot_window_panel(
                dataset=dataset,
                window_idx=win_idx,
                rank=rank,
                metric=metric_name,
                delta=float(delta[pack_row]),
                binary_mse=float(mse_bin[pack_row]),
                mmpd_mse=float(mse_mmpd[pack_row]),
                binary_crps=float(crps_bin[pack_row]),
                mmpd_crps=float(crps_mmpd[pack_row]),
                past=past_batch[0],
                future=future_batch[0],
                norm=norm,
                pack_row=int(pack_row),
                aligned=aligned,
                output_path=ds_out / f"{metric_name}_delta_rank{rank:02d}_win{win_idx}.png",
                prob_draws=prob_draws,
                context_len=min(horizon := aligned.y_true.shape[2], 96 * 2),
            )
        rankings[metric_name] = rows_meta

    meta = {
        "dataset": dataset,
        "binary_results_dir": str(binary_run.results_dir),
        "binary_config": binary_run.config_suffix,
        "mmpd_run": str(mmpd_run),
        "n_windows": int(len(aligned.indices)),
        "mean_delta_anchor_mse": agg_mse_delta,
        "mean_delta_crps": agg_crps_delta,
        "aggregate_check": {
            "binary_anchor_mse": float(mse_bin.mean()),
            "mmpd_anchor_mse": float(mse_mmpd.mean()),
            "binary_crps": float(crps_gr(aligned.y_true, aligned.binary_samples)),
            "mmpd_crps": float(crps_gr(aligned.y_true, aligned.mmpd_samples)),
        },
        "rankings": rankings,
    }
    with (ds_out / "delta_rankings.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_summary_csv(output_dir: Path, metas: Sequence[Dict[str, object]]) -> Path:
    path = output_dir / "delta_top_summary.csv"
    fields = [
        "dataset",
        "metric",
        "rank",
        "test_window_index",
        "delta",
        "binary",
        "mmpd",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for meta in metas:
            ds = meta["dataset"]
            for metric, rows in meta["rankings"].items():
                for row in rows:
                    writer.writerow(
                        {
                            "dataset": ds,
                            "metric": metric,
                            "rank": row["rank"],
                            "test_window_index": row["test_window_index"],
                            "delta": f"{row['delta']:.8f}",
                            "binary": f"{row['binary']:.8f}",
                            "mmpd": f"{row['mmpd']:.8f}",
                        }
                    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(FINISHED_GRAD_ACCUM_DATASETS),
        help="Comma-separated datasets (default: 8 with finished grad_accum eval).",
    )
    parser.add_argument(
        "--mmpd-run",
        type=Path,
        default=DEFAULT_MMPD_RUN,
        help="Fair MMPD results root (partials + raw/mmpd_*.npz).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "results" / "datasets",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--prob-draws",
        type=int,
        default=3,
        help="Probabilistic sample lines per panel (from saved npz).",
    )
    parser.add_argument(
        "--allow-fallback-binary",
        action="store_true",
        help=f"Use {FALLBACK_BINARY_CONFIG} when grad_accum eval missing.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Only write rankings JSON/CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metas: List[Dict[str, object]] = []
    skipped: List[str] = []
    for dataset in datasets:
        try:
            meta = process_dataset(
                dataset,
                mmpd_run=args.mmpd_run,
                datasets_root=args.datasets_root,
                ckpt_root=args.ckpt_root,
                output_dir=args.output_dir,
                top_k=args.top_k,
                prob_draws=args.prob_draws,
                allow_fallback=args.allow_fallback_binary,
                skip_plots=args.skip_plots,
            )
            metas.append(meta)
        except FileNotFoundError as exc:
            print(f"[skip] {dataset}: {exc}")
            skipped.append(dataset)

    if metas:
        summary = write_summary_csv(args.output_dir, metas)
        print(f"Wrote {len(metas)} datasets -> {args.output_dir}")
        print(f"Summary CSV: {summary}")
    else:
        print("No datasets processed.", file=sys.stderr)
        sys.exit(1)
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
