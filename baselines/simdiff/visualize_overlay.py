#!/usr/bin/env python3
"""Overlay forecasts: ts-sandbox Gaussian diffusion (A+B) vs SimDiff on the same test windows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMDIFF_ROOT = REPO_ROOT / "SimDiff"


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def _load_ts_diffusion(
    ckpt_path: Path,
    meta_path: Path,
    scenario: str,
    n_vars: int,
    lookback: int,
    horizon: int,
    overlap: int,
    device: torch.device,
):
    import models.diffusion_tsf.train_multivariate_pipeline as pipe

    pipe.EXPERIMENT = scenario
    pipe.N_VARIATES = n_vars
    from models.diffusion_tsf.guidance import iTransformerGuidance
    from models.diffusion_tsf.train_multivariate_pipeline import (
        create_diffusion_model,
        load_itransformer_from_checkpoint,
    )

    ckpt_dir = ckpt_path.parent.parent
    subset_id = json.loads(meta_path.read_text())["subset_id"]
    itrans_path = ckpt_dir / f"{subset_id}_itransformer_finetuned.pt"
    itrans = load_itransformer_from_checkpoint(str(itrans_path), n_vars, device)
    model = create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        lookback_overlap=overlap,
    ).to(device)
    model.set_guidance_model(iTransformerGuidance(itrans))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _build_simdiff_args(sample_times: int) -> argparse.Namespace:
    """Minimal SimDiff args for exchange 96→96 (mirrors eval_comparable)."""
    return argparse.Namespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id="exchange_96_96",
        model="SimDiff",
        data="custom",
        root_path=str(SIMDIFF_ROOT / "dataset"),
        data_path="exchange_rate.csv",
        features="M",
        target="OT",
        freq="d",
        checkpoints=str(SIMDIFF_ROOT / "checkpoints"),
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=8,
        dec_in=8,
        c_out=8,
        d_model=128,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.0,
        skip_dropout=0.5,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        stride=1,
        patch_len=2,
        coss=5.0,
        rmom=13,
        n_b=3,
        sample_times=sample_times,
        vs_times=6,
        is_diff=1,
        s_steps=3,
        skip_type="time_quadratic",
        method="multistep",
        lower_order_final="true",
        order=2,
        diff_steps=100,
        use_mom=1,
        new_norm=1,
        use_shuffle=1,
        use_first=0,
        num_workers=0,
        itr=1,
        train_epochs=100,
        batch_size=1,
        patience=20,
        learning_rate=0.001,
        des="Exp",
        loss="MSE",
        loss_type="MAE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        extra_tag="",
        num_heads=2,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        seasonal_patterns="Monthly",
        mask_rate=0.25,
        anomaly_ratio=0.25,
        inverse=False,
    )


def _purge_modules(prefixes=("models", "utils", "exp", "data_provider", "layers")):
    for key in list(sys.modules):
        if key in prefixes or any(key.startswith(p + ".") for p in prefixes):
            del sys.modules[key]


def _load_simdiff(ckpt_path: Path, device: torch.device, sample_times: int = 20):
    _purge_modules()
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(SIMDIFF_ROOT))

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    ns = _build_simdiff_args(sample_times)
    exp = Exp_Long_Term_Forecast(ns)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=device))
    exp.model.to(device)
    exp.model.eval()
    return exp.model, ns


@torch.no_grad()
def predict_diffusion(model, past: torch.Tensor, horizon: int, overlap: int, device) -> torch.Tensor:
    past_b = past.unsqueeze(0).to(device)
    out = model.generate(past_b)
    pred = out["prediction"].cpu()[0]
    if pred.shape[-1] > horizon:
        pred = pred[:, -horizon:]
    return pred


@torch.no_grad()
def predict_simdiff(model, batch_x, batch_y, batch_x_mark, batch_y_mark, args, device) -> torch.Tensor:
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
    outputs, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=args.sample_times)
    return outputs[0, -args.pred_len :, :].detach().cpu().permute(1, 0)


def run_overlay(
    run_dir: Path,
    simdiff_ckpt: Path,
    output_dir: Path,
    num_samples: int = 3,
    n_vars_plot: int = 3,
    lookback: int = 104,
    horizon: int = 96,
    overlap: int = 8,
    scenario: str = "A+B",
    simdiff_sample_times: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    ckpt_dir = run_dir / "ckpts"
    meta_path = ckpt_dir / "exp_A+B" / "metadata.json"
    diff_ckpt = ckpt_dir / "exp_A+B" / "best.pt"
    if not diff_ckpt.is_file():
        raise FileNotFoundError(diff_ckpt)

    sys.path.insert(0, str(REPO_ROOT))
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset

    n_vars = 8
    _, _, test_ds, norm_stats = load_dataset(
        "exchange_rate",
        lookback=lookback,
        horizon=horizon,
        lookback_overlap=overlap,
    )
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    print("Loading SimDiff...", flush=True)
    sim_model, sim_args = _load_simdiff(simdiff_ckpt, device, simdiff_sample_times)
    from data_provider.data_factory import data_provider

    sim_args.batch_size = 1
    sim_args.num_workers = 0
    sim_test, _ = data_provider(sim_args, "test")

    _purge_modules()
    if str(SIMDIFF_ROOT) in sys.path:
        sys.path.remove(str(SIMDIFF_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    print("Loading Gaussian A+B diffusion...", flush=True)
    diff_model = _load_ts_diffusion(
        diff_ckpt, meta_path, scenario, n_vars, lookback, horizon, overlap, device
    )

    seq_sim = sim_args.seq_len
    offset = lookback - seq_sim
    n_test = len(test_ds)
    max_idx = min(n_test - 1, len(sim_test) - 1 - offset)
    if max_idx < 0:
        raise RuntimeError("No overlapping test windows between loaders")
    sample_indices = np.linspace(0, max_idx, min(num_samples, max_idx + 1), dtype=int)

    output_dir.mkdir(parents=True, exist_ok=True)
    n_cols = min(n_vars_plot, n_vars)

    col_names = [str(i) for i in range(n_cols)]
    col_names[-1] = "OT" if n_cols == 8 else col_names[-1]

    cache = []
    for idx in sample_indices:
        past, future = test_ds[idx]
        sim_idx = idx + offset
        if sim_idx < 0 or sim_idx >= len(sim_test):
            continue
        print(f"  sample {idx}: diffusion...", flush=True)
        diff_pred = predict_diffusion(diff_model, past, horizon, overlap, device)
        print(f"  sample {idx}: simdiff...", flush=True)
        sx, sy, sxm, sym = sim_test[sim_idx]
        sim_pred = predict_simdiff(
            sim_model,
            torch.tensor(sx, dtype=torch.float32).unsqueeze(0),
            torch.tensor(sy, dtype=torch.float32).unsqueeze(0),
            torch.tensor(sxm, dtype=torch.float32).unsqueeze(0),
            torch.tensor(sym, dtype=torch.float32).unsqueeze(0),
            sim_args,
            device,
        )
        cache.append((idx, past, future, diff_pred, sim_pred))

    if not cache:
        raise RuntimeError("No aligned test windows for overlay plot")

    n_rows = len(cache)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 3.4 * n_rows), squeeze=False
    )

    for row, (idx, past, future, diff_pred, sim_pred) in enumerate(cache):
        future_h = future[:, -horizon:]
        past_dn = denorm(past, mean, std)
        gt_dn = denorm(future_h, mean, std)
        diff_dn = denorm(diff_pred, mean, std)
        sim_dn = denorm(sim_pred, mean, std)

        context_len = min(horizon * 2, lookback)
        t_past = np.arange(-context_len, 0)
        t_fut = np.arange(0, horizon)

        for col in range(n_cols):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#9E9E9E",
                alpha=0.45,
                lw=0.8,
            )
            ax.plot(t_fut, gt_dn[col].numpy(), color="#2196F3", lw=1.7, label="Ground truth")
            ax.plot(
                t_fut,
                diff_dn[col].numpy(),
                color="#E91E63",
                lw=1.2,
                ls="--",
                label="Gaussian A+B",
            )
            ax.plot(
                t_fut,
                sim_dn[col].numpy(),
                color="#4CAF50",
                lw=1.2,
                ls="-.",
                label="SimDiff",
            )
            ax.axvline(0, color="k", ls=":", alpha=0.25)
            d_mae = np.mean(np.abs(diff_dn[col].numpy() - gt_dn[col].numpy()))
            s_mae = np.mean(np.abs(sim_dn[col].numpy() - gt_dn[col].numpy()))
            ax.text(
                0.97,
                0.97,
                f"A+B {d_mae:.3f}\nSim {s_mae:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
            )
            if row == 0:
                ax.set_title(col_names[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"test idx {idx}", fontsize=9)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "exchange_rate — Gaussian diffusion (A+B) vs SimDiff\n"
        f"aligned forecast window | L={lookback} H={horizon} (A+B) · SimDiff 96→96",
        fontsize=13,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    out1 = output_dir / "overlay_exchange_rate_multivar.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # Single-variate zoom (OT / last channel)
    v = n_vars - 1
    fig2, axes2 = plt.subplots(len(cache), 1, figsize=(10, 2.8 * len(cache)), squeeze=False)
    for row, (idx, past, future, diff_pred, sim_pred) in enumerate(cache):
        ax = axes2[row, 0]
        future_h = future[:, -horizon:]
        gt_dn = denorm(future_h, mean, std)[v].numpy()
        diff_dn = denorm(diff_pred, mean, std)[v].numpy()
        sim_dn = denorm(sim_pred, mean, std)[v].numpy()
        past_dn = denorm(past, mean, std)[v, -context_len:].numpy()

        ax.plot(t_past, past_dn, color="#9E9E9E", alpha=0.5, lw=0.9)
        ax.plot(t_fut, gt_dn, color="#2196F3", lw=2.0, label="Ground truth")
        ax.plot(t_fut, diff_dn, color="#E91E63", lw=1.4, ls="--", label="Gaussian A+B")
        ax.plot(t_fut, sim_dn, color="#4CAF50", lw=1.4, ls="-.", label="SimDiff")
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        ax.set_ylabel("OT (denorm)")
        ax.set_title(f"Test window {idx}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)

    fig2.suptitle("exchange_rate OT — overlay comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out2 = output_dir / "overlay_exchange_rate_OT.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "results/runs/05-19-3662573-exp_A_B_exchange-rate",
    )
    p.add_argument(
        "--simdiff-ckpt",
        type=Path,
        default=SIMDIFF_ROOT
        / "checkpoints/long_term_forecast_exchange_96_96_SimDiff_custom_ftM_sl96_ll48_pl96_dm128_nh8_el1_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--vars", type=int, default=3)
    p.add_argument("--lookback", type=int, default=104)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--overlap", type=int, default=8)
    p.add_argument("--simdiff-samples", type=int, default=20)
    args = p.parse_args()

    out = args.output_dir or (args.run_dir / "viz")
    run_overlay(
        args.run_dir,
        args.simdiff_ckpt,
        out,
        num_samples=args.num_samples,
        n_vars_plot=args.vars,
        lookback=args.lookback,
        horizon=args.horizon,
        overlap=args.overlap,
        simdiff_sample_times=args.simdiff_samples,
    )


if __name__ == "__main__":
    main()
