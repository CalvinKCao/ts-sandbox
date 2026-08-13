#!/usr/bin/env python3
"""Backfill compatible electricity point-gap panels and binary redboxes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MMPD_ROOT = REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
OUT = REPO / "results/fullvar-probabilistic-redbox-backfill/datasets"

ELECTRICITY = (
    {
        "name": "electricity_v000_159",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_msdefault_fixed.yaml",
        "ckpt": "results/ckpts/08-12-4730293-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_msdefault_fixed",
        "start": 0,
        "stop": 160,
    },
    {
        "name": "electricity_v160_320",
        "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_v160_320_msdefault_fixed.yaml",
        "ckpt": "results/ckpts/08-12-4730294-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity_v160_320_msdefault_fixed",
        "start": 160,
        "stop": 321,
    },
)
DYNAMIC = {
    "name": "dynamic_allv",
    "config": "configs/binary_window_norm_patch_refine_canvas128_p64x6_dynamic_allv_msdefault_fixed.yaml",
    "ckpt": "results/ckpts/08-12-4730295-dynamic-binary_window_norm_patch_refine_canvas128_p64x6_dynamic_allv_msdefault_fixed",
}


def _redbox(spec: dict[str, object], picks: list[int]) -> None:
    out = OUT / str(spec["name"]) / "redbox"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "temp/scripts/viz_ablation_staged_eval_samples.py",
        "--dataset", "electricity" if str(spec["name"]).startswith("electricity") else "dynamic",
        "--output-root", str(out), "--lookback", "336", "--horizon", "96",
        "--pack-test-stride", "4", "--pack-splits", "test", "--pool-indices",
        *map(str, [p for p in picks for _ in range(3)]), "--n-samples", str(len(picks) * 3),
        "--variables-to-plot", "3", "--sampler", "quad_t", "--num-sampling-steps", "20",
        "--draw-seed-step", "1000003", "--device", "cuda",
        "--runs", f"{spec['name']}:{spec['ckpt']}:{spec['config']}",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def _electricity_gap(spec: dict[str, object]) -> list[int]:
    from utils.compare_binary_mmpd_staged_diag import run_binary_staged_eval

    with np.load(MMPD_ROOT / "raw/mmpd_electricity.npz") as z:
        mmpd = {k: z[k] for k in ("y_true", "deterministic", "indices")}
    candidates = np.asarray(mmpd["indices"], dtype=np.int64)
    # Twelve evenly spread exact MMPD lattice windows keep this backfill quick.
    select = np.unique(np.linspace(0, len(candidates) - 1, 12, dtype=np.int64))
    windows = candidates[select].tolist()
    _, binary = run_binary_staged_eval(
        checkpoint_dir=(REPO / str(spec["ckpt"])), dataset="electricity",
        config_path=str(spec["config"]), window_indices=windows, test_stride=4,
        device=torch.device("cuda"), prob_samples=1, prob_steps=1,
    )
    start, stop = int(spec["start"]), int(spec["stop"])
    m_rows = {int(w): i for i, w in enumerate(candidates)}
    rows = np.asarray([m_rows[int(w)] for w in windows], dtype=np.int64)
    gt = mmpd["y_true"][rows, start:stop]
    mp = mmpd["deterministic"][rows, start:stop]
    bp = binary["final_anchor"]
    if gt.shape != bp.shape or not np.allclose(gt, binary["y_true"], rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"{spec['name']}: incompatible GT after split selection: {gt.shape} vs {bp.shape}")
    bm = ((bp - gt) ** 2).mean(axis=(1, 2))
    mm = ((mp - gt) ** 2).mean(axis=(1, 2))
    rank = np.argsort(np.abs(bm - mm))[::-1][:3]
    out = OUT / str(spec["name"]) / "point_gap"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for r, i in enumerate(rank):
        fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
        for ax, v in zip(axes, (0, min(1, stop - start - 1), min(2, stop - start - 1))):
            ax.plot(gt[i, v], color="black", lw=1.4, label="GT")
            ax.plot(bp[i, v], color="tab:blue", lw=1.1, label="binary anchor")
            ax.plot(mp[i, v], color="tab:orange", lw=1.1, label="MMPD deterministic")
            ax.set_ylabel(f"raw ch {start + v}")
        axes[0].legend(ncol=3, fontsize=8)
        fig.suptitle(f"{spec['name']} | window {windows[i]} | binary MSE {bm[i]:.4g}, MMPD MSE {mm[i]:.4g}")
        fig.tight_layout()
        path = out / f"top_gap_{r:02d}_window{windows[i]}.jpg"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        rows_out.append({"window_index": int(windows[i]), "binary_anchor_mse": float(bm[i]), "mmpd_anchor_mse": float(mm[i]), "abs_gap": float(abs(bm[i] - mm[i]))})
    (out / "top_windows.json").write_text(json.dumps(rows_out, indent=2) + "\n")
    return [int(x["window_index"]) for x in rows_out]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "electricity": "point-gap skipped: archived MMPD pack has 4 variates, not the 160/161 full-attention subsets",
        "dynamic": "point-gap skipped: archived MMPD variate subset differs",
        "probabilistic_redboxes": {},
    }
    for spec in ELECTRICITY:
        picks = [0, 1, 2]
        _redbox(spec, picks)
        summary["probabilistic_redboxes"][str(spec["name"])] = {
            "windows": picks, "draws_per_window": 3, "sampler": "quad_t", "steps": 20
        }
    _redbox(DYNAMIC, [0, 1, 2])
    summary["probabilistic_redboxes"]["dynamic_allv"] = {
        "windows": [0, 1, 2], "draws_per_window": 3, "sampler": "quad_t", "steps": 20
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
