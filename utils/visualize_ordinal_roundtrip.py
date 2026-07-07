#!/usr/bin/env python3
"""Quick before/after plot: global z-score -> ordinal encode -> decode on real windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.ordinal_window_norm import ordinal_decode, ordinal_encode
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset


def _window_z_scores(ds, window_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Past/future from z-scored ``ds.data``, not precomputed ordinal ranks."""
    start = window_idx * ds.stride
    past = ds.data[start : start + ds.lookback].T
    target_start = start + ds.lookback - ds.lookback_overlap
    target_end = start + ds.lookback + ds.horizon
    future = ds.data[target_start:target_end].T
    return past, future


def _pick_window(ds, *, prefer_ties: bool, variate: int, tie_atol: float):
    best_idx, best_score = 0, -1.0
    for idx in range(min(len(ds), 512)):
        past, future = ds[idx]
        x = torch.cat([past[variate : variate + 1], future[variate : variate + 1]], dim=-1)
        xs = torch.sort(x.reshape(-1)).values
        n_unique = 1
        last = xs[0].item()
        for v in xs[1:].tolist():
            if abs(v - last) > tie_atol:
                n_unique += 1
            last = v
        score = (len(xs) - n_unique) if prefer_ties else float(torch.std(x))
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def plot_roundtrip(
    *,
    dataset: str,
    config_path: Path,
    out_dir: Path,
    window_idx: int | None,
    variate: int,
    prefer_ties: bool,
    split: str = "test",
) -> Path:
    cfg = load_experiment_config(str(config_path))
    exp = cfg["experiment"]
    lookback = int(exp["lookback_length"])
    horizon = int(exp["forecast_length"])
    overlap = int(exp.get("lookback_overlap", 0))
    tie_atol = float(exp.get("ordinal_tie_atol", 1e-6))
    use_ord = bool(exp.get("use_ordinal_window_norm", True))

    train_ds, _, test_ds, norm_stats = load_dataset(
        dataset,
        lookback=lookback,
        horizon=horizon,
        lookback_overlap=overlap,
        stride=1,
        ordinal_tie_atol=tie_atol,
        use_ordinal_window_norm=use_ord,
    )
    ladder = norm_stats["ordinal_ladder"]
    ds = test_ds if split == "test" else train_ds
    if window_idx is None:
        window_idx = _pick_window(
            ds, prefer_ties=prefer_ties, variate=variate, tie_atol=tie_atol,
        )

    past, future = _window_z_scores(ds, window_idx)
    past_b = past.unsqueeze(0)
    fut_b = future.unsqueeze(0)

    past_ord, fut_ord, ladder, _ood_shift = ordinal_encode(
        past_b, fut_b, ladder=ladder,
    )
    past_rec, fut_rec = ordinal_decode(past_ord, fut_ord, ladder)

    k = int(ladder.n_unique[0, variate].item())
    n_past = past.shape[-1]
    n_fut = future.shape[-1]

    t_past = np.arange(n_past)
    t_fut = np.arange(n_past - overlap, n_past - overlap + n_fut)

    orig_p = past[variate].numpy()
    orig_f = future[variate].numpy()
    ord_p = past_ord[0, variate].numpy()
    ord_f = fut_ord[0, variate].numpy()
    rec_p = past_rec[0, variate].numpy()
    rec_f = fut_rec[0, variate].numpy()

    err_p = orig_p - rec_p
    err_f = orig_f - rec_f
    mae = float(np.mean(np.abs(np.concatenate([err_p, err_f]))))

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"{dataset} win={window_idx} var={variate} | "
        f"global z → ordinal (K={k} train-unique ranks [0,{int(ladder.rank_max_per_variate()[variate])}]) → decode | MAE={mae:.4g}",
        fontsize=11,
    )

    axes[0].plot(t_past, orig_p, color="C0", lw=1.2, label="past")
    axes[0].plot(t_fut, orig_f, color="C1", lw=1.2, label="future (+overlap)")
    axes[0].axvline(n_past - 0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[0].set_ylabel("global z")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Before ({split}-split global z-score)")

    axes[1].plot(t_past, ord_p, color="C2", lw=1.2)
    axes[1].plot(t_fut, ord_f, color="C3", lw=1.2)
    axes[1].axvline(n_past - 0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[1].set_ylabel("ordinal")
    rank_max = float(ladder.rank_max_per_variate()[variate].item())
    axes[1].set_ylim(-0.5, max(rank_max, 1.0) + 0.5)
    axes[1].set_title("After ordinal encode (global train ladder; ties share rank)")

    axes[2].plot(t_past, rec_p, color="C0", lw=1.2, ls="--", label="past recon")
    axes[2].plot(t_fut, rec_f, color="C1", lw=1.2, ls="--", label="future recon")
    axes[2].plot(t_past, orig_p, color="C0", lw=0.6, alpha=0.35)
    axes[2].plot(t_fut, orig_f, color="C1", lw=0.6, alpha=0.35)
    axes[2].axvline(n_past - 0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[2].set_ylabel("global z")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("After ordinal decode (dashed) vs original (faint)")

    axes[3].plot(t_past, err_p, color="C4", lw=1.0)
    axes[3].plot(t_fut, err_f, color="C5", lw=1.0)
    axes[3].axhline(0.0, color="gray", lw=0.6)
    axes[3].axvline(n_past - 0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[3].set_ylabel("orig − recon")
    axes[3].set_xlabel("timestep")
    axes[3].set_title("Reconstruction error (quantization to past-unique ladder)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_v{variate}_win{window_idx}_roundtrip.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml",
    )
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--variate", type=int, default=4, help="ETTh1 HUFL=4")
    p.add_argument("--window-idx", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "reports/ordinal_roundtrip_example",
    )
    p.add_argument(
        "--prefer-ties",
        action="store_true",
        help="Pick a window with many tied values (flat segments).",
    )
    args = p.parse_args()

    path = plot_roundtrip(
        dataset=args.dataset,
        config_path=args.config,
        out_dir=args.out_dir,
        window_idx=args.window_idx,
        variate=args.variate,
        prefer_ties=args.prefer_ties,
    )
    print(path)


if __name__ == "__main__":
    main()
