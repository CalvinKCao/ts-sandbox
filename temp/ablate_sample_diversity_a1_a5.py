#!/usr/bin/env python3
"""Inference-only diversity ablations (A1–A5) on ETTh1 / ETTh2 vertical_dual ckpts.

Tweaks (each vs baseline, one factor at a time):
  A1  x0 ~ Bernoulli(p) instead of hard threshold p>0.5
  A2  stochastic last step (Bernoulli x0; no silent freeze)
  A3  beta_floor on reflip (default 0.02)
  A4  more inference steps (50 vs 20)
  A5  linear timestep spacing (ddim) vs quadratic (quad_t)
  A1A2 combo (optional / --full): Bernoulli x0 + stochastic last step

Also reports CRPS + sample-mean MSE/MAE vs GT, and plots sample overlays.

Usage (from repo root, with .venv):
  python temp/ablate_sample_diversity_a1_a5.py --quick
  python temp/ablate_sample_diversity_a1_a5.py --full
  # Killarney: ./temp/submit_ablate_sample_diversity_a1_a5_killarney.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import models.diffusion_tsf.train_multivariate_pipeline as pipe
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _model_kwargs_from_tuned,
)
from models.diffusion_tsf.metrics import crps_ensemble
from models.diffusion_tsf.pipeline.visualize_utils import per_window_crps
from models.diffusion_tsf.train_multivariate_pipeline import (
    anchor_kwargs_from_params,
    create_diffusion_model,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_wrapped_guidance,
)

OUT_ROOT = REPO / "temp" / "diversity_ablation_a1_a5"

RUNS = [
    {
        "tag": "ETTh1_aug_fixed20",
        "dataset": "ETTh1",
        "ckpt_dir": (
            "results/ckpts/07-14-4241374-ETTh1-binary_anchor_ar_patch_decoder_ctx_"
            "lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_per_ds_best_g_aug_fixed20"
        ),
        "n_vars": 7,
    },
    {
        "tag": "ETTh2_joint_s30r20",
        "dataset": "ETTh2",
        "ckpt_dir": (
            "results/ckpts/07-15-4263255-ETTh2-binary_anchor_ar_patch_decoder_ctx_"
            "lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20"
        ),
        "n_vars": 7,
    },
]


@dataclass(frozen=True)
class SampleCfg:
    name: str
    x0_mode: str = "threshold"  # threshold | bernoulli
    last_step_stochastic: bool = False
    beta_floor: float = 0.0
    num_steps: int = 20
    spacing: str = "quad_t"  # quad_t | ddim


VARIANTS: List[SampleCfg] = [
    SampleCfg(name="baseline"),
    SampleCfg(name="A1_bernoulli_x0", x0_mode="bernoulli"),
    SampleCfg(name="A2_last_step_stoch", last_step_stochastic=True),
    SampleCfg(name="A3_beta_floor_0p02", beta_floor=0.02),
    SampleCfg(name="A4_steps_50", num_steps=50),
    SampleCfg(name="A5_linear_spacing", spacing="ddim"),
]

# Extra combo for comprehensive CRPS runs (A1+A2 won on ETTh2 diversity).
COMBO_VARIANT = SampleCfg(
    name="A1A2_bern_last",
    x0_mode="bernoulli",
    last_step_stochastic=True,
)


def _bin_indices_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    h = int(cdf.shape[2])
    column_sum = cdf.float().sum(dim=2).clamp(1.0, float(h))
    return (column_sum - 1.0).clamp(0.0, float(h - 1)).long()


def _unique_bin_stats(coarse_bins: np.ndarray) -> Dict[str, float]:
    """coarse_bins: (S, B, V, W)."""
    s, b, v, w = coarse_bins.shape
    uniques = [
        len(np.unique(coarse_bins[:, bi, vi, wi]))
        for bi in range(b)
        for vi in range(v)
        for wi in range(w)
    ]
    arr = np.asarray(uniques, dtype=np.float64)
    return {
        "mean_unique_bins": float(arr.mean()),
        "median_unique_bins": float(np.median(arr)),
        "frac_unique1": float((arr == 1).mean()),
        "frac_unique_le2": float((arr <= 2).mean()),
        "max_unique_bins": float(arr.max()),
        "n_columns": float(arr.size),
        "n_samples": float(s),
    }


@torch.no_grad()
def ablate_sample(
    scheduler,
    model_fn,
    shape: Tuple[int, ...],
    *,
    cfg: SampleCfg,
    device: torch.device,
) -> torch.Tensor:
    """Binary reverse sample with A1–A5 knobs (inference-only)."""
    num_steps = int(cfg.num_steps)
    if cfg.spacing in {"quad_t", "ddim_quad"}:
        ramp = torch.linspace(1.0, 0.0, num_steps, device=device)
        step_indices = torch.round((ramp ** 2) * (scheduler.num_steps - 1)).long()
    elif cfg.spacing == "ddim":
        step_indices = torch.linspace(
            scheduler.num_steps - 1, 0, num_steps, device=device, dtype=torch.long,
        )
    else:
        raise ValueError(f"unknown spacing {cfg.spacing!r}")

    xt = torch.bernoulli(torch.full(shape, 0.5, device=device))
    for i, t_val in enumerate(step_indices):
        t_idx = int(t_val.item())
        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        x0_logits, _zt_logits = model_fn(xt, t_batch)
        p = torch.sigmoid(x0_logits)
        is_last = i >= len(step_indices) - 1
        # A2: force Bernoulli on the final step even if mid-loop uses threshold.
        use_bern = cfg.x0_mode == "bernoulli" or (is_last and cfg.last_step_stochastic)
        if use_bern:
            x0_hat = torch.bernoulli(p)
        elif cfg.x0_mode == "threshold":
            x0_hat = (p > 0.5).float()
        else:
            raise ValueError(f"unknown x0_mode {cfg.x0_mode!r}")

        if is_last:
            xt = x0_hat
            continue

        t_next = int(step_indices[i + 1].item())
        beta_next = max(float(scheduler.betas[t_next].item()), float(cfg.beta_floor))
        zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
        xt = (x0_hat.bool() ^ zt_new.bool()).float()
    return xt


def load_model(run: Dict[str, Any], device: torch.device):
    ckpt_dir = REPO / run["ckpt_dir"]
    ds = run["dataset"]
    meta = json.loads((ckpt_dir / ds / "vertical_dual" / "metadata.json").read_text())
    tuned = dict(meta["tuned_params"])

    pipe.IMAGE_HEIGHT = 32
    pipe.COARSE_IMAGE_HEIGHT = 16
    pipe.FINE_IMAGE_HEIGHT = 16
    pipe.USE_GUIDANCE_CHANNEL = True
    pipe.LOOKBACK_LENGTH = 336
    pipe.FORECAST_LENGTH = 720
    # Match ordinal_norm leaf configs these ckpts were trained with.
    pipe.USE_ORDINAL_WINDOW_NORM = True
    pipe.USE_WINDOW_NORMALIZATION = False
    pipe.BINARY_NOISE_SCHEDULE = tuned.get("binary_noise_schedule", "linear")
    pipe.BINARY_LENGTH_MODE = tuned.get("binary_length_mode", "power")
    pipe.BINARY_LENGTH_G = float(tuned.get("binary_length_g", 1.0))
    pipe.BINARY_LENGTH_SCALE = float(tuned.get("binary_length_scale", 1.0))

    n_iv = int(run["n_vars"])
    # Build ordinal ladder before guidance load (patch guidance validates it).
    _, _, test_ds, _ = load_dataset(
        ds,
        variate_indices=list(range(n_iv)),
        lookback=336,
        horizon=720,
        stride=1,
        test_stride=4,
        use_ordinal_window_norm=True,
    )
    guidance = load_wrapped_guidance(
        str(ckpt_dir / f"{ds}_patch_guidance.pt"),
        n_iv,
        device,
        guidance_type="patch_decoder",
    )
    mk = anchor_kwargs_from_params(tuned)
    mk.update(_model_kwargs_from_tuned(tuned))
    model = create_diffusion_model(
        n_variates=n_iv,
        lookback=336,
        horizon=720,
        guidance_model=guidance,
        diffusion_stage="vertical_dual",
        ordinal_ladder=pipe.GLOBAL_ORDINAL_LADDER,
        use_ordinal_window_norm=True,
        **mk,
    ).to(device)
    ckpt = torch.load(
        ckpt_dir / ds / "vertical_dual" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
    model.eval()
    assert bool(model.config.use_ordinal_window_norm), "ordinal flag not set on model"
    return model, test_ds, meta


def patch_scheduler(model, cfg: SampleCfg):
    """Replace binary_scheduler.sample with ablate_sample bound to cfg."""
    sched = model.binary_scheduler
    orig = sched.sample

    def wrapped(
        model_fn,
        shape,
        num_steps=20,
        device="cpu",
        verbose=False,
        sampler="quad_t",
        yield_intermediates=False,
        reverse_step_indices=None,
        snapshot_timesteps=None,
    ):
        del verbose, sampler, reverse_step_indices, snapshot_timesteps
        # Honor caller num_steps only if cfg didn't override via variant; variants set cfg.num_steps.
        local = SampleCfg(
            name=cfg.name,
            x0_mode=cfg.x0_mode,
            last_step_stochastic=cfg.last_step_stochastic,
            beta_floor=cfg.beta_floor,
            num_steps=int(cfg.num_steps),
            spacing=cfg.spacing,
        )
        out = ablate_sample(
            sched, model_fn, shape, cfg=local, device=torch.device(device),
        )
        if yield_intermediates:
            return out, []
        return out

    sched.sample = wrapped
    return orig


def gather_windows(test_ds, n_windows: int, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    n = len(test_ds)
    idxs = sorted(rng.choice(n, size=min(n_windows, n), replace=False).tolist())
    pasts, futures = [], []
    for i in idxs:
        past, future = test_ds[i]
        pasts.append(past)
        futures.append(future)
    return (
        torch.stack(pasts, dim=0).to(device),
        torch.stack(futures, dim=0).to(device),
        idxs,
    )


def run_variant(
    model,
    past: torch.Tensor,
    cfg: SampleCfg,
    *,
    n_samples: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns coarse_bins (S,B,V,W), preds (S,B,V,T), canvases list on CPU."""
    Hc = int(model.config.coarse_image_height)
    orig = patch_scheduler(model, cfg)
    coarse_list, pred_list, canvases = [], [], []
    try:
        for s in range(n_samples):
            seed = base_seed + s * 17
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # sampler name ignored by patch; still pass quad_t for generate() plumbing
            out = model.generate(
                past,
                sampler="quad_t",
                num_inference_steps=int(cfg.num_steps),
            )
            fut = out["future_2d"]
            coarse_list.append(_bin_indices_from_cdf(fut[:, :, :Hc]).cpu().numpy())
            pred = out.get("prediction_global_norm", out["prediction"])
            pred_list.append(pred.detach().cpu().numpy())
            canvases.append(fut.detach().cpu().numpy())
    finally:
        model.binary_scheduler.sample = orig

    return (
        np.stack(coarse_list, axis=0),
        np.stack(pred_list, axis=0),
        np.stack(canvases, axis=0),
    )


def plot_window_samples(
    *,
    out_path: Path,
    past: np.ndarray,
    future: np.ndarray,
    preds_by_variant: Dict[str, np.ndarray],
    window_idx: int,
    variate: int,
    title: str,
):
    """past (T_lb,), future (T_hz,), preds (S, T)."""
    n_var = len(preds_by_variant)
    fig, axes = plt.subplots(n_var, 1, figsize=(12, 2.4 * n_var), sharex=True)
    if n_var == 1:
        axes = [axes]
    t_past = np.arange(past.shape[-1])
    # Align future length to pred length if overlap/subsample differs.
    for ax, (name, preds) in zip(axes, preds_by_variant.items()):
        ax.plot(t_past, past, color="black", lw=1.2, label="past")
        # GT: plot on forecast axis starting after lookback
        t0 = past.shape[-1]
        fut = future
        pred_t = preds.shape[-1]
        if fut.shape[-1] >= pred_t:
            fut_plot = fut[-pred_t:]
        else:
            fut_plot = fut
        t_fut = np.arange(t0, t0 + fut_plot.shape[-1])
        ax.plot(t_fut, fut_plot, color="green", lw=1.4, label="gt")
        t_pred = np.arange(t0, t0 + pred_t)
        for s in range(preds.shape[0]):
            ax.plot(
                t_pred,
                preds[s],
                color="C0",
                alpha=max(0.15, 0.7 / preds.shape[0]),
                lw=0.9,
                label="samples" if s == 0 else None,
            )
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _align_future_to_pred(future: np.ndarray, pred_len: int, lookback_overlap: int) -> np.ndarray:
    """Strip overlap prefix then match pred length (B,V,T)."""
    k = int(lookback_overlap)
    yt = future[..., k:] if k > 0 else future
    if yt.shape[-1] == pred_len:
        return yt
    if yt.shape[-1] > pred_len:
        return yt[..., :pred_len]
    raise RuntimeError(
        f"future length {yt.shape[-1]} < pred_len {pred_len} (k={k}, raw={future.shape[-1]})"
    )


def _forecast_metrics(
    y_true: np.ndarray,
    preds_sbt: np.ndarray,
) -> Dict[str, float]:
    """y_true (B,V,T), preds (S,B,V,T) → CRPS / sample-mean MSE/MAE."""
    # crps_ensemble expects samples (B,V,S,T)
    samples = np.transpose(preds_sbt, (1, 2, 0, 3))
    crps = float(crps_ensemble(y_true, samples))
    per_win = per_window_crps(y_true, samples)
    mean_pred = preds_sbt.mean(axis=0)
    mse = float(np.mean((mean_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(mean_pred - y_true)))
    # also mean-over-samples of per-sample MSE (not just MSE of mean)
    per_sample_mse = float(np.mean((preds_sbt - y_true[None]) ** 2))
    return {
        "crps": crps,
        "crps_per_window_mean": float(per_win.mean()),
        "sample_mean_mse": mse,
        "sample_mean_mae": mae,
        "mean_per_sample_mse": per_sample_mse,
    }


def diagnose_run(
    run: Dict[str, Any],
    *,
    device: torch.device,
    n_windows: int,
    n_samples: int,
    n_plot_windows: int,
    seed: int,
    batch_size: int,
    variants: List[SampleCfg],
    out_root: Path,
    plot_variates: Sequence[int] = (0, 1),
) -> Dict[str, Any]:
    print(f"\n======== {run['tag']} ========", flush=True)
    model, test_ds, meta = load_model(run, device)
    past_all, future_all, win_idxs = gather_windows(test_ds, n_windows, seed, device)
    print(
        f"windows={len(win_idxs)} idxs_head={win_idxs[:8]} "
        f"past={tuple(past_all.shape)} future={tuple(future_all.shape)} "
        f"g={meta.get('tuned_params', {}).get('binary_length_g')} "
        f"batch_size={batch_size}",
        flush=True,
    )

    results: Dict[str, Any] = {
        "tag": run["tag"],
        "dataset": run["dataset"],
        "window_indices": win_idxs,
        "meta_g": meta.get("tuned_params", {}).get("binary_length_g"),
        "n_windows": len(win_idxs),
        "n_samples": n_samples,
        "variants": {},
    }
    preds_all: Dict[str, np.ndarray] = {}
    K = int(getattr(model.config, "lookback_overlap", 0) or 0)
    n_win = past_all.shape[0]

    for cfg in variants:
        print(f"  → {cfg.name} …", flush=True)
        coarse_chunks, pred_chunks = [], []
        for start in range(0, n_win, batch_size):
            end = min(start + batch_size, n_win)
            past = past_all[start:end]
            # Offset seed by window-block so batches don't share identical noise schedules.
            block_seed = seed + start * 1009
            coarse, preds, _canvas = run_variant(
                model, past, cfg, n_samples=n_samples, base_seed=block_seed,
            )
            coarse_chunks.append(coarse)
            pred_chunks.append(preds)
            print(
                f"     batch windows[{start}:{end}] done",
                flush=True,
            )
        # Concatenate along batch dim: each chunk is (S,B_i,V,...)
        coarse = np.concatenate(coarse_chunks, axis=1)
        preds = np.concatenate(pred_chunks, axis=1)
        preds_all[cfg.name] = preds

        stats = _unique_bin_stats(coarse)
        pw = [
            float(np.mean((preds[i] - preds[j]) ** 2))
            for i in range(n_samples)
            for j in range(i + 1, n_samples)
        ]
        mean_pw = float(np.mean(pw)) if pw else 0.0
        std_s = float(np.std(preds, axis=0).mean())

        y_true = _align_future_to_pred(
            future_all.detach().cpu().numpy(),
            pred_len=int(preds.shape[-1]),
            lookback_overlap=K,
        )
        metrics = _forecast_metrics(y_true, preds)
        row = {
            **asdict(cfg),
            **stats,
            "mean_pairwise_1d_mse": mean_pw,
            "mean_across_sample_std": std_s,
            **metrics,
        }
        results["variants"][cfg.name] = row
        print(
            f"     uniq={stats['mean_unique_bins']:.3f} frac1={stats['frac_unique1']:.3f} "
            f"crps={metrics['crps']:.4f} mean_mse={metrics['sample_mean_mse']:.4f} "
            f"pw_mse={mean_pw:.4f}",
            flush=True,
        )

    plot_dir = out_root / run["tag"] / "plots"
    n_plot = min(n_plot_windows, past_all.shape[0])
    for wi in range(n_plot):
        for v in plot_variates:
            if v >= past_all.shape[1]:
                continue
            past_np = past_all[wi, v].detach().cpu().numpy()
            fut_np = future_all[wi, v].detach().cpu().numpy()
            if K > 0 and fut_np.shape[-1] > K:
                fut_np = fut_np[K:]
            var_preds = {name: arr[:, wi, v, :] for name, arr in preds_all.items()}
            plot_window_samples(
                out_path=plot_dir / f"win{win_idxs[wi]}_var{v}.png",
                past=past_np,
                future=fut_np,
                preds_by_variant=var_preds,
                window_idx=win_idxs[wi],
                variate=v,
                title=f"{run['tag']} window={win_idxs[wi]} var={v}",
            )
    print(f"  plots → {plot_dir}", flush=True)

    # Delta vs baseline for quick reading
    base = results["variants"].get("baseline")
    if base is not None:
        deltas = {}
        for name, row in results["variants"].items():
            if name == "baseline":
                continue
            deltas[name] = {
                "d_crps": float(row["crps"] - base["crps"]),
                "d_sample_mean_mse": float(row["sample_mean_mse"] - base["sample_mean_mse"]),
                "d_unique_bins": float(row["mean_unique_bins"] - base["mean_unique_bins"]),
            }
        results["delta_vs_baseline"] = deltas
        print("  Δ vs baseline (negative CRPS/MSE is better):", flush=True)
        for name, d in deltas.items():
            print(
                f"     {name}: d_crps={d['d_crps']:+.4f} "
                f"d_mse={d['d_sample_mean_mse']:+.4f} "
                f"d_uniq={d['d_unique_bins']:+.3f}",
                flush=True,
            )
    return results


def print_summary_table(all_results: List[Dict[str, Any]]) -> None:
    print(
        "\n======== SUMMARY (uniq / CRPS / sample-mean MSE; ΔCRPS vs baseline) ========",
        flush=True,
    )
    header = (
        f"{'run':<22} {'variant':<22} {'uniq':>7} {'frac1':>7} "
        f"{'crps':>8} {'mse':>8} {'d_crps':>8}"
    )
    print(header, flush=True)
    for res in all_results:
        base_crps = res["variants"].get("baseline", {}).get("crps")
        for name, row in res["variants"].items():
            d_crps = ""
            if base_crps is not None and name != "baseline":
                d_crps = f"{row['crps'] - base_crps:+8.4f}"
            else:
                d_crps = f"{'—':>8}"
            print(
                f"{res['tag']:<22} {name:<22} "
                f"{row['mean_unique_bins']:7.3f} {row['frac_unique1']:7.3f} "
                f"{row.get('crps', float('nan')):8.4f} "
                f"{row.get('sample_mean_mse', float('nan')):8.4f} "
                f"{d_crps}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 windows, 8 samples")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Comprehensive: 48 windows, 20 samples, batch 4, include A1+A2 combo",
    )
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-plot-windows", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-combo", action="store_true", help="Add A1+A2 Bernoulli combo")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output root (default: temp/diversity_ablation_a1_a5)",
    )
    ap.add_argument(
        "--runs",
        default="all",
        help="comma tags or 'all'",
    )
    args = ap.parse_args()
    if args.quick and args.full:
        raise SystemExit("pass only one of --quick / --full")
    if args.quick:
        args.n_windows = 2
        args.n_samples = 8
        args.n_plot_windows = 2
        args.batch_size = 2
    if args.full:
        args.n_windows = 48
        args.n_samples = 20
        args.n_plot_windows = 4
        args.batch_size = 4
        args.include_combo = True

    out_root = Path(args.out_dir) if args.out_dir else OUT_ROOT
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS)
    if args.include_combo:
        variants.append(COMBO_VARIANT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} out={out_root} windows={args.n_windows} "
        f"samples={args.n_samples} batch={args.batch_size} "
        f"variants={[v.name for v in variants]}",
        flush=True,
    )

    selected = RUNS
    if args.runs != "all":
        want = {x.strip() for x in args.runs.split(",")}
        selected = [r for r in RUNS if r["tag"] in want or r["dataset"] in want]
        if not selected:
            raise SystemExit(f"no runs matched {args.runs!r}")

    all_results = []
    for run in selected:
        # Fail fast if ckpt missing (common on fresh login pull without results).
        ckpt = REPO / run["ckpt_dir"] / run["dataset"] / "vertical_dual" / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
        all_results.append(
            diagnose_run(
                run,
                device=device,
                n_windows=args.n_windows,
                n_samples=args.n_samples,
                n_plot_windows=args.n_plot_windows,
                seed=args.seed,
                batch_size=args.batch_size,
                variants=variants,
                out_root=out_root,
            )
        )

    print_summary_table(all_results)
    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_json}", flush=True)
    print(f"Plots under {out_root}/<tag>/plots/", flush=True)


if __name__ == "__main__":
    main()
