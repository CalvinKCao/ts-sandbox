"""
ETTh2 CI latent: plot test windows — ground truth vs finetuned iTransformer vs finetuned diffusion.

Shaded band + markers: last K=lookback_overlap steps where diffusion is trained to reconstruct
the known past tail (compare grey past line vs diffusion (overlap recon) on the same x range).

Uses the same splits / stride as training (stride=24 on test). Denormalizes with the global
mean/std from load_dataset so the y-axis matches real units.

Pull artifacts from the cluster (adjust user, host, SCRATCH path), then run locally:

  REMOTE=ccao87@killarney.alliancecan.ca
  RROOT=/scratch/ccao87/ts-sandbox/models/diffusion_tsf
  LROOT=/path/to/ts-sandbox/models/diffusion_tsf

  mkdir -p "$LROOT/results" "$LROOT/checkpoints_ci_runs/ETTh2" "$LROOT/checkpoints_latent"
  rsync -avz "$REMOTE:$RROOT/results/ci_latent_ETTh2_H128.json" "$LROOT/results/"
  rsync -avz "$REMOTE:$RROOT/checkpoints_ci_runs/ETTh2/" "$LROOT/checkpoints_ci_runs/ETTh2/"
  rsync -avz "$REMOTE:$RROOT/checkpoints_latent/vae_dim1_H128.pt" "$LROOT/checkpoints_latent/"

  python -m models.diffusion_tsf.visualize_ci_latent_etth2 \\
      --dataset ETTh2 --output-dir models/diffusion_tsf/results/ci_latent_viz_etth2

Optional: --run-tag exchange_rate_s42  --variate-indices 0,1,...  --diffusion-ckpt path.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.latent_diffusion_model import LatentDiffusionTSF
from models.diffusion_tsf.latent_experiment_common import (
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    create_itransformer,
    dataset_registry_row,
    get_device,
    load_dataset,
)
from models.diffusion_tsf.train_ci_latent_etth2 import (
    CKPT_LATENT,
    CIiTransformerGuidance,
    N_VARIATES,
    build_config,
)
from models.diffusion_tsf.vae import TimeSeriesVAE


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """x: (B, V, T); mean, std: (1, V). Broadcast per variate along time."""
    # (1, V, 1) * (B, V, T) — do not insert an extra batch dim on mean/std
    m = mean.unsqueeze(-1)
    s = std.unsqueeze(-1)
    return x * s + m


def main():
    p = argparse.ArgumentParser(description="Visualize CI latent vs iTransformer (denorm = global load_dataset stats)")
    p.add_argument("--dataset", type=str, default="ETTh2")
    p.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Subdir under --run-ckpt-dir (default: same as --dataset; use e.g. exchange_rate_s42 for exchange)",
    )
    p.add_argument(
        "--run-ckpt-dir",
        type=str,
        default="",
        help="Parent of per-run finetune dirs (default: models/diffusion_tsf/checkpoints_ci_runs)",
    )
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/ci_latent_viz_etth2")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--num-samples", type=int, default=4, help="test windows to plot")
    p.add_argument("--max-vars", type=int, default=7, help="variates per figure (columns)")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument(
        "--diffusion-ckpt",
        type=str,
        default="",
        help="override path to diffusion_finetuned_*.pt",
    )
    p.add_argument(
        "--variate-indices",
        type=str,
        default="",
        help="comma-separated 7 ints for load_dataset (exchange_rate subset); empty = all columns",
    )
    args = p.parse_args()

    device = get_device()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, seasonal, embed_freq = dataset_registry_row(args.dataset)
    cfg = build_config(args.image_height, seasonal)
    run_tag = args.run_tag or args.dataset
    run_parent = Path(args.run_ckpt_dir) if args.run_ckpt_dir else (_SCRIPT_DIR / "checkpoints_ci_runs")
    if not run_parent.is_absolute():
        run_parent = _PROJECT_ROOT / run_parent
    run_dir = run_parent / run_tag

    vae_path = CKPT_LATENT / f"vae_dim1_H{args.image_height}.pt"
    itrans_path = run_dir / f"itransformer_finetuned_{run_tag}.pt"
    if args.diffusion_ckpt:
        diff_path = Path(args.diffusion_ckpt)
        if not diff_path.is_absolute():
            diff_path = _PROJECT_ROOT / diff_path
    else:
        diff_path = run_dir / f"diffusion_finetuned_{run_tag}_H{args.image_height}.pt"

    for path, label in [
        (vae_path, "VAE"),
        (itrans_path, "finetuned iTransformer"),
        (diff_path, "finetuned diffusion"),
    ]:
        if not path.is_file():
            raise SystemExit(f"Missing {label} checkpoint: {path}")

    vidx = None
    if args.variate_indices.strip():
        vidx = [int(x.strip()) for x in args.variate_indices.split(",") if x.strip()]
    _, _, test_ds, meta = load_dataset(
        args.dataset,
        variate_indices=vidx,
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=24,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    n_test = len(test_ds)
    if n_test == 0:
        raise SystemExit("Empty test set")

    mean = torch.tensor(meta["mean"], dtype=torch.float32)
    std = torch.tensor(meta["std"], dtype=torch.float32)

    vae_ck = torch.load(vae_path, map_location=device, weights_only=False)
    vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
    vae.load_state_dict(vae_ck["vae_state_dict"])
    if "scale_factor" in vae_ck:
        vae.scale_factor.copy_(vae_ck["scale_factor"].to(device).view_as(vae.scale_factor))
    vae.eval()

    it_g = create_itransformer(
        num_vars=N_VARIATES, pred_len=FORECAST_LENGTH, freq=embed_freq,
    ).to(device)
    it_g.load_state_dict(torch.load(itrans_path, map_location=device, weights_only=False)["model_state_dict"])
    guidance = CIiTransformerGuidance(
        it_g, num_variates=N_VARIATES, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH,
    )
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)
    dck = torch.load(diff_path, map_location=device, weights_only=False)
    model.load_state_dict(dck["model_state_dict"], strict=False)
    model.eval()

    itrans = create_itransformer(
        num_vars=N_VARIATES, pred_len=FORECAST_LENGTH, freq=embed_freq,
    ).to(device)
    itc = torch.load(itrans_path, map_location=device, weights_only=False)
    itrans.load_state_dict(itc["model_state_dict"])
    itrans.eval()

    Vplot = min(args.max_vars, N_VARIATES)
    step = max(1, n_test // max(args.num_samples, 1))
    sample_indices = [min(i * step, n_test - 1) for i in range(args.num_samples)]

    for si, idx in enumerate(sample_indices):
        past, future = test_ds[idx]
        past = past.unsqueeze(0).to(device)
        future = future.unsqueeze(0).to(device)
        B = 1
        past_flat = past.reshape(B * N_VARIATES, LOOKBACK_LENGTH)

        K = LOOKBACK_OVERLAP
        with torch.no_grad():
            gen = model.generate(
                past_flat,
                num_ddim_steps=args.ddim_steps,
                trim_lookback_overlap=False if K > 0 else True,
            )
            pred_flat = gen["prediction"].squeeze(1).reshape(B, N_VARIATES, -1)
            if K > 0:
                pred_diff_overlap = pred_flat[:, :, :K]
                pred_diff = pred_flat[:, :, K:]
            else:
                pred_diff_overlap = None
                pred_diff = pred_flat

            x_enc = past.permute(0, 2, 1)
            y_it = itrans(x_enc, None, None, None)
            if isinstance(y_it, tuple):
                y_it = y_it[0]
            pred_it = y_it.permute(0, 2, 1)

        gt_fore = future[:, :, K:]

        past_d = _denorm(past.cpu(), mean, std)
        gt_d = _denorm(gt_fore.cpu(), mean, std)
        it_d = _denorm(pred_it.cpu(), mean, std)
        df_d = _denorm(pred_diff.cpu(), mean, std)
        if K > 0:
            dfo_d = _denorm(pred_diff_overlap.cpu(), mean, std)
            past_tail_d = past_d[:, :, -K:]

        fig, axes = plt.subplots(1, Vplot, figsize=(3.2 * Vplot, 3.8), sharey=False)
        if Vplot == 1:
            axes = [axes]
        t_past = torch.arange(LOOKBACK_LENGTH)
        t_overlap = torch.arange(LOOKBACK_LENGTH - K, LOOKBACK_LENGTH) if K > 0 else None
        t_future = torch.arange(LOOKBACK_LENGTH, LOOKBACK_LENGTH + FORECAST_LENGTH)

        for v in range(Vplot):
            ax = axes[v]
            ax.plot(t_past, past_d[0, v].numpy(), color="0.35", lw=1.2, label="past (GT)")
            if K > 0 and t_overlap is not None:
                ax.axvspan(
                    LOOKBACK_LENGTH - K - 0.5,
                    LOOKBACK_LENGTH - 0.5,
                    color="0.85",
                    alpha=0.35,
                    zorder=0,
                )
                ax.plot(
                    t_overlap,
                    dfo_d[0, v].numpy(),
                    color="C3",
                    lw=1.6,
                    ls="-",
                    marker="o",
                    ms=3,
                    label="diffusion (overlap recon)",
                )
            ax.plot(t_future, gt_d[0, v].numpy(), color="0.1", lw=1.8, label="future GT")
            ax.plot(t_future, it_d[0, v].numpy(), color="C0", lw=1.4, ls="--", label="iTransformer")
            ax.plot(t_future, df_d[0, v].numpy(), color="C3", lw=1.4, label="CI latent diffusion")
            ax.axvline(LOOKBACK_LENGTH - 0.5, color="0.5", lw=0.9, ls=":")
            ax.set_title(f"var {v}")
            ax.set_xlabel("time step")
            if v == 0:
                ax.set_ylabel("value (denorm)")
            ax.grid(True, alpha=0.25)
        axes[-1].legend(loc="upper left", fontsize=7)
        fig.suptitle(f"{args.dataset} ({run_tag}) test idx={idx} stride=24", fontsize=11)
        fig.tight_layout()
        safe_tag = run_tag.replace("/", "_")
        png_path = out_dir / f"ci_{safe_tag}_win{idx}_H{args.image_height}.png"
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png_path}")
        if K > 0:
            mae_ov = (dfo_d[0] - past_tail_d[0]).abs().mean().item()
            print(f"  overlap MAE vs past tail (all {N_VARIATES} vars, K={K}): {mae_ov:.5f}")


if __name__ == "__main__":
    main()
