#!/usr/bin/env python3
"""Diagnose low probabilistic sample diversity (items 2/3/5/6). Temp probe only."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.diffusion import BinaryDiffusionScheduler
from models.diffusion_tsf.pipeline.visualize_utils import _load_staged_diffusion_from_ckpt
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset


RUNS = [
    {
        "tag": "ETTh1_aug_fixed20",
        "ckpt_dir": (
            "results/ckpts/07-14-4241374-ETTh1-binary_anchor_ar_patch_decoder_ctx_"
            "lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_per_ds_best_g_aug_fixed20"
        ),
        "dataset": "ETTh1",
        "n_vars": 7,
    },
    {
        "tag": "ETTh2_joint_s30r20",
        "ckpt_dir": (
            "results/ckpts/07-15-4263255-ETTh2-binary_anchor_ar_patch_decoder_ctx_"
            "lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20"
        ),
        "dataset": "ETTh2",
        "n_vars": 7,
    },
]

N_WINDOWS = 4
N_SAMPLES = 20
NUM_INFER_STEPS = 20
SEED = 42
OUT_DIR = REPO / "temp" / "diversity_diag"


def _bin_indices_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """(B,V,H,W) binary CDF -> discrete bin index (B,V,W) in [0, H-1]."""
    h = cdf.shape[2]
    column_sum = cdf.float().sum(dim=2).clamp(1.0, float(h))
    return (column_sum - 1.0).clamp(0.0, float(h - 1)).long()


def _binary_entropy(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _dpmpp_step_indices(num_train_steps: int, num_infer: int, device: torch.device) -> torch.Tensor:
    ramp = torch.linspace(1.0, 0.0, num_infer, device=device)
    return torch.round((ramp ** 2) * (num_train_steps - 1)).long()


@dataclass
class StepStats:
    t_idx: int
    beta_next: float
    mean_p: float
    mean_entropy: float
    frac_uncertain: float  # |p-0.5| < 0.1
    frac_confident: float  # |p-0.5| > 0.4
    mean_x0_hamming_vs_sample0: float


def instrumented_sample(
    scheduler: BinaryDiffusionScheduler,
    model_fn,
    shape: Tuple[int, ...],
    *,
    num_steps: int,
    device: torch.device,
    sampler: str = "dpmpp",
) -> Tuple[torch.Tensor, List[StepStats], List[torch.Tensor]]:
    """Copy of BinaryDiffusionScheduler.sample with per-step entropy logging."""
    if sampler == "dpmpp":
        ramp = torch.linspace(1.0, 0.0, num_steps, device=device)
        step_indices = torch.round((ramp ** 2) * (scheduler.num_steps - 1)).long()
    else:
        step_indices = torch.linspace(
            scheduler.num_steps - 1, 0, num_steps, device=device, dtype=torch.long,
        )

    xt = torch.bernoulli(torch.full(shape, 0.5, device=device))
    stats: List[StepStats] = []
    x0_traj: List[torch.Tensor] = []

    for i, t_val in enumerate(step_indices):
        t_idx = int(t_val.item())
        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        x0_logits, _zt_logits = model_fn(xt, t_batch)
        p = torch.sigmoid(x0_logits)
        x0_hat = (p > 0.5).float()
        x0_traj.append(x0_hat.detach().cpu())

        beta_next = float("nan")
        if i < len(step_indices) - 1:
            t_next = int(step_indices[i + 1].item())
            beta_next = float(scheduler.betas[t_next].item())
            zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
            xt = (x0_hat.bool() ^ zt_new.bool()).float()
        else:
            xt = x0_hat

        ent = _binary_entropy(p)
        stats.append(
            StepStats(
                t_idx=t_idx,
                beta_next=beta_next,
                mean_p=float(p.mean().item()),
                mean_entropy=float(ent.mean().item()),
                frac_uncertain=float(((p - 0.5).abs() < 0.1).float().mean().item()),
                frac_confident=float(((p - 0.5).abs() > 0.4).float().mean().item()),
                mean_x0_hamming_vs_sample0=float("nan"),  # filled later across samples
            )
        )
    return xt, stats, x0_traj


def load_model(run: Dict[str, Any], device: torch.device):
    ckpt_dir = REPO / run["ckpt_dir"]
    ds = run["dataset"]
    best = ckpt_dir / ds / "vertical_dual" / "best.pt"
    guide = ckpt_dir / f"{ds}_patch_guidance.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    if not guide.is_file():
        raise FileNotFoundError(guide)
    model, ckpt = _load_staged_diffusion_from_ckpt(
        ckpt_path=str(best),
        stage="vertical_dual",
        itrans_ckpt_path=str(guide),
        n_vars=int(run["n_vars"]),
        device=device,
        guidance_type="patch_decoder",
    )
    meta_path = ckpt_dir / ds / "vertical_dual" / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return model, ckpt, meta


def make_windows(dataset: str, n_vars: int, n_windows: int, device: torch.device):
    # Match pipeline lookback/horizon for these configs.
    _, val_ds, test_ds, _ = load_dataset(
        dataset,
        variate_indices=list(range(n_vars)),
        lookback=336,
        horizon=720,
        stride=1,
        test_stride=4,
        use_ordinal_window_norm=True,
    )
    # Prefer test windows for eval-like conditions.
    src = test_ds if len(test_ds) >= n_windows else val_ds
    idxs = list(range(min(n_windows, len(src))))
    pasts = []
    for i in idxs:
        past, _future = src[i]
        pasts.append(past)
    past = torch.stack(pasts, dim=0).to(device)
    return past, idxs


def pairwise_mean_hamming(maps: List[torch.Tensor]) -> float:
    """maps: list of (B,V,H,W) binary tensors."""
    if len(maps) < 2:
        return 0.0
    stack = torch.stack(maps, dim=0)  # (S,B,V,H,W)
    s = stack.shape[0]
    total = 0.0
    n = 0
    for i in range(s):
        for j in range(i + 1, s):
            total += float((stack[i] != stack[j]).float().mean().item())
            n += 1
    return total / max(n, 1)


def unique_bins_stats(coarse_bins: np.ndarray) -> Dict[str, float]:
    """coarse_bins: (S, B, V, W) int."""
    s, b, v, w = coarse_bins.shape
    # per (b,v,w) how many unique among S
    uniques = []
    for bi in range(b):
        for vi in range(v):
            for wi in range(w):
                uniques.append(len(np.unique(coarse_bins[:, bi, vi, wi])))
    arr = np.asarray(uniques, dtype=np.float64)
    # also: fraction of columns with only 1 unique bin
    return {
        "mean_unique_bins": float(arr.mean()),
        "median_unique_bins": float(np.median(arr)),
        "p10_unique_bins": float(np.percentile(arr, 10)),
        "p90_unique_bins": float(np.percentile(arr, 90)),
        "frac_columns_unique1": float((arr == 1).mean()),
        "frac_columns_unique_le2": float((arr <= 2).mean()),
        "max_unique_bins": float(arr.max()),
        "n_columns": float(arr.size),
        "n_samples": float(s),
        "Hc": float(16),  # filled by caller if needed
    }


def diagnose_run(run: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    print(f"\n======== {run['tag']} ========", flush=True)
    model, ckpt, meta = load_model(run, device)
    past, win_idxs = make_windows(run["dataset"], run["n_vars"], N_WINDOWS, device)
    Hc = int(model.config.coarse_image_height)
    Hf = int(model.config.fine_image_height)
    sched: BinaryDiffusionScheduler = model.binary_scheduler
    step_idx = _dpmpp_step_indices(sched.num_steps, NUM_INFER_STEPS, device)
    betas_at_steps = [float(sched.betas[int(t)].item()) for t in step_idx.tolist()]

    print(
        f"stage={model.config.diffusion_stage} Hc={Hc} Hf={Hf} "
        f"schedule={sched.schedule_type} length_mode={sched.length_mode} "
        f"g={sched.length_g} T={sched.num_steps} beta0={float(sched.betas[0]):.3e} "
        f"betaT={float(sched.betas[-1]):.4f}",
        flush=True,
    )
    print(f"meta tuned g={meta.get('tuned_params', {}).get('binary_length_g')}", flush=True)
    print(
        "DPM++ step_indices (first10/last5):",
        step_idx[:10].tolist(),
        "...",
        step_idx[-5:].tolist(),
        flush=True,
    )
    print(
        "betas at those steps (first10/last5):",
        [f"{x:.4f}" for x in betas_at_steps[:10]],
        "...",
        [f"{x:.4e}" for x in betas_at_steps[-5:]],
        flush=True,
    )

    # --- Seed hygiene smoke: two generates with same seed must match; different seeds may differ ---
    torch.manual_seed(SEED)
    a = model.generate(past[:1], sampler="dpmpp", num_inference_steps=NUM_INFER_STEPS)
    torch.manual_seed(SEED)
    b = model.generate(past[:1], sampler="dpmpp", num_inference_steps=NUM_INFER_STEPS)
    same_seed_equal = bool(torch.equal(a["future_2d"], b["future_2d"]))
    torch.manual_seed(SEED + 17)
    c = model.generate(past[:1], sampler="dpmpp", num_inference_steps=NUM_INFER_STEPS)
    diff_seed_equal = bool(torch.equal(a["future_2d"], c["future_2d"]))
    print(
        f"seed hygiene: same_seed_repro={same_seed_equal} "
        f"diff_seed_identical_canvas={diff_seed_equal}",
        flush=True,
    )

    # --- Multi-sample probe with instrumented loop (first window only for step stats) ---
    # Rebuild one generate call's model_fn path by monkeypatching scheduler.sample briefly.
    all_coarse_bins = []  # list of (B,V,W)
    all_fine_bins = []
    all_pred_1d = []
    per_sample_step_stats: List[List[StepStats]] = []
    final_maps = []

    # Capture instrumented sample for window 0 only across samples; full generate for all windows.
    orig_sample = sched.sample

    def make_wrapped(collect_stats: bool):
        def wrapped(*args, **kwargs):
            # Always use our instrumented path for consistency.
            out, st, x0_traj = instrumented_sample(
                sched,
                kwargs.get("model_fn") or args[0],
                kwargs.get("shape") or args[1],
                num_steps=int(kwargs.get("num_steps", NUM_INFER_STEPS)),
                device=torch.device(kwargs.get("device", device)),
                sampler=str(kwargs.get("sampler", "dpmpp")),
            )
            if collect_stats:
                per_sample_step_stats.append(st)
            if kwargs.get("yield_intermediates"):
                return out, []
            return out
        return wrapped

    for s_idx in range(N_SAMPLES):
        seed = SEED + 0 * 1009 + s_idx * 17  # mirror staged_eval formula for batch 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Instrument only for first few samples to keep cost down.
        collect = s_idx < 3
        sched.sample = make_wrapped(collect)
        try:
            out = model.generate(past, sampler="dpmpp", num_inference_steps=NUM_INFER_STEPS)
        finally:
            sched.sample = orig_sample

        fut = out["future_2d"]  # (B,V,Hc+Hf,W)
        coarse = fut[:, :, :Hc]
        fine = fut[:, :, Hc:]
        all_coarse_bins.append(_bin_indices_from_cdf(coarse).cpu().numpy())
        all_fine_bins.append(_bin_indices_from_cdf(fine).cpu().numpy())
        pred = out.get("prediction_global_norm", out["prediction"]).detach().cpu().numpy()
        all_pred_1d.append(pred)
        final_maps.append(fut.detach().cpu())

        if s_idx == 0 or s_idx == N_SAMPLES - 1:
            print(f"  sample {s_idx}: future_2d={tuple(fut.shape)} pred={tuple(pred.shape)}", flush=True)

    coarse_bins = np.stack(all_coarse_bins, axis=0)  # (S,B,V,W)
    fine_bins = np.stack(all_fine_bins, axis=0)
    preds = np.stack(all_pred_1d, axis=0)  # (S,B,V,T)

    coarse_stats = unique_bins_stats(coarse_bins)
    coarse_stats["Hc"] = float(Hc)
    fine_stats = unique_bins_stats(fine_bins)
    fine_stats["Hc"] = float(Hf)

    # 1D forecast pairwise MSE / mean abs pairwise diff
    pairwise_mse = []
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            pairwise_mse.append(float(np.mean((preds[i] - preds[j]) ** 2)))
    mean_pw_mse = float(np.mean(pairwise_mse)) if pairwise_mse else 0.0
    # within-window std across samples, averaged
    sample_std = float(np.std(preds, axis=0).mean())
    canvas_hamming = pairwise_mean_hamming(final_maps)

    print("COARSE unique-bin stats:", json.dumps(coarse_stats, indent=2), flush=True)
    print("FINE unique-bin stats:", json.dumps(fine_stats, indent=2), flush=True)
    print(
        f"1D sample diversity: mean_pairwise_mse={mean_pw_mse:.6f} "
        f"mean_across_sample_std={sample_std:.6f} "
        f"canvas_pairwise_hamming={canvas_hamming:.6f}",
        flush=True,
    )

    # Aggregate step entropy across instrumented samples (window batch)
    step_summary = []
    if per_sample_step_stats:
        n_steps = len(per_sample_step_stats[0])
        for i in range(n_steps):
            ents = [ps[i].mean_entropy for ps in per_sample_step_stats]
            unc = [ps[i].frac_uncertain for ps in per_sample_step_stats]
            conf = [ps[i].frac_confident for ps in per_sample_step_stats]
            step_summary.append(
                {
                    "step_i": i,
                    "t_idx": per_sample_step_stats[0][i].t_idx,
                    "beta_next": per_sample_step_stats[0][i].beta_next,
                    "mean_entropy": float(np.mean(ents)),
                    "frac_uncertain": float(np.mean(unc)),
                    "frac_confident": float(np.mean(conf)),
                }
            )
        print("STEP entropy (avg over instrumented samples):", flush=True)
        for row in step_summary:
            print(
                f"  i={row['step_i']:02d} t={row['t_idx']:4d} "
                f"beta_next={row['beta_next']:.4e} "
                f"H={row['mean_entropy']:.4f} "
                f"unc={row['frac_uncertain']:.3f} conf={row['frac_confident']:.3f}",
                flush=True,
            )

    # Cross-sample x0_hat agreement at early/mid/late steps (from instrumented runs)
    # We only stored per-sample stats, not full x0 maps across all samples in instrumented path.
    # Re-run a cheap last-step-only diversity check: Hamming of final x0 across all samples already in canvas_hamming.

    return {
        "tag": run["tag"],
        "dataset": run["dataset"],
        "ckpt_ok": True,
        "same_seed_repro": same_seed_equal,
        "diff_seed_identical_canvas": diff_seed_equal,
        "schedule_type": sched.schedule_type,
        "length_mode": sched.length_mode,
        "length_g": float(sched.length_g),
        "beta0": float(sched.betas[0]),
        "betaT": float(sched.betas[-1]),
        "dpmpp_step_indices": step_idx.cpu().tolist(),
        "dpmpp_betas": betas_at_steps,
        "coarse_unique": coarse_stats,
        "fine_unique": fine_stats,
        "mean_pairwise_1d_mse": mean_pw_mse,
        "mean_across_sample_std": sample_std,
        "canvas_pairwise_hamming": canvas_hamming,
        "step_summary": step_summary,
        "window_indices": win_idxs,
        "Hc": Hc,
        "Hf": Hf,
        "meta_g": meta.get("tuned_params", {}).get("binary_length_g"),
    }


def schedule_static_report():
    """Schedule bounds without a model."""
    print("\n======== static schedule (no model) ========", flush=True)
    for g, mode in [(1.0, "none"), (1.0, "power"), (1.171, "power"), (3.0, "power")]:
        sch = BinaryDiffusionScheduler(
            num_steps=1000,
            beta_start=1e-5,
            beta_end=0.5,
            schedule_type="sqrt_linear" if mode == "none" else "linear",
            # joint configs use binary_noise_schedule: linear + length_mode power
            device="cpu",
            length_mode=mode if mode != "none" else "none",
            length_g=g,
        )
        # Also print pure sqrt_linear defaults
        idx = _dpmpp_step_indices(1000, 20, torch.device("cpu"))
        betas = [float(sch.betas[int(t)]) for t in idx.tolist()]
        print(
            f"mode={mode} g={g} schedule={sch.schedule_type} "
            f"beta[0]={float(sch.betas[0]):.3e} beta[-1]={float(sch.betas[-1]):.4f} "
            f"dpmpp_beta_max={max(betas):.4f} dpmpp_beta_min={min(betas):.3e} "
            f"n_steps_beta>0.1={sum(b > 0.1 for b in betas)} "
            f"n_steps_beta>0.01={sum(b > 0.01 for b in betas)}",
            flush=True,
        )

    sch_default = BinaryDiffusionScheduler(
        num_steps=1000, beta_start=1e-5, beta_end=0.5, schedule_type="sqrt_linear", device="cpu",
    )
    idx = _dpmpp_step_indices(1000, 20, torch.device("cpu"))
    betas = [float(sch_default.betas[int(t)]) for t in idx.tolist()]
    print("default sqrt_linear DPM++ betas:", [f"{b:.4e}" for b in betas], flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Verify ckpts first
    for run in RUNS:
        ckpt_dir = REPO / run["ckpt_dir"]
        best = ckpt_dir / run["dataset"] / "vertical_dual" / "best.pt"
        guide = ckpt_dir / f"{run['dataset']}_patch_guidance.pt"
        print(
            f"ckpt check {run['tag']}: best={best.is_file()} "
            f"({best.stat().st_size/1e6:.1f}MB)" if best.is_file() else
            f"ckpt check {run['tag']}: MISSING {best}",
            flush=True,
        )
        print(f"  guidance={guide.is_file()} ({guide.stat().st_size/1e6:.1f}MB)" if guide.is_file() else f"  MISSING {guide}", flush=True)

    schedule_static_report()

    results = []
    for run in RUNS:
        try:
            results.append(diagnose_run(run, device))
        except Exception as exc:
            print(f"FAILED {run['tag']}: {exc}", flush=True)
            raise

    out_path = OUT_DIR / "diversity_diag_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
