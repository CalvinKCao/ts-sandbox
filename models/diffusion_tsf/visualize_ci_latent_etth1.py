"""
Plot ETTh1 test windows: **ground truth** vs **iTransformer** vs **CI latent diffusion**.

Matches the eval protocol in `train_ci_latent_etth1.py` (global normalization from
`load_dataset`, same stride / overlap). Uses the **finetuned** checkpoint written at
the end of stage 3 (`guided_finetuned_H{H}.pt` or `unguided_finetuned_H{H}.pt`).

Similar layout to `visualize_7var_etth1.py` (rows = windows, cols = variates).

Repo root, venv on:

  python -m models.diffusion_tsf.visualize_ci_latent_etth1 \\
    --output-dir models/diffusion_tsf/results/latent_viz

Guided (default):

  python -m models.diffusion_tsf.visualize_ci_latent_etth1 \\
    --latent-ckpt models/diffusion_tsf/checkpoints_ci_latent/guided_finetuned_H128.pt

Unguided:

  python -m models.diffusion_tsf.visualize_ci_latent_etth1 --no-guidance \\
    --latent-ckpt models/diffusion_tsf/checkpoints_ci_latent/unguided_finetuned_H128.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

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
    get_device,
    load_dataset,
)
from models.diffusion_tsf.train_ci_latent_etth1 import (
    N_VARIATES,
    _build_ci_guidance,
    build_ci_config,
)
from models.diffusion_tsf.vae import TimeSeriesVAE

DEFAULT_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """x: (C, T), mean/std (1, C)."""
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def main():
    p = argparse.ArgumentParser(description="CI latent vs iTransformer on ETTh1")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--no-guidance", action="store_true", help="unguided CI model")
    p.add_argument(
        "--latent-ckpt",
        type=str,
        default=None,
        help="Finetuned LatentDiffusionTSF state (default: guided/unguided_finetuned_H*.pt)",
    )
    p.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="TimeSeriesVAE checkpoint (default: checkpoints_latent/vae_dim1_H{H}.pt)",
    )
    p.add_argument(
        "--itrans-ckpt",
        type=str,
        default=None,
        help="7-var ETTh1 iTransformer (default: checkpoints_ci_latent/itransformer_etth1_7var.pt)",
    )
    p.add_argument(
        "--allow-pretrained-fallback",
        action="store_true",
        help="If finetuned .pt missing, load guided/unguided_pretrained_H*.pt (not fine-tuned on ETTh1)",
    )
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/latent_viz")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=500)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--variables", type=int, default=7, help="first N variates as columns")
    args = p.parse_args()

    use_guidance = not args.no_guidance
    tag = "unguided" if args.no_guidance else "guided"
    ih = args.image_height

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    latent_dir = _PROJECT_ROOT / "models" / "diffusion_tsf" / "checkpoints_latent"
    ci_dir = _PROJECT_ROOT / "models" / "diffusion_tsf" / "checkpoints_ci_latent"
    if args.latent_ckpt:
        latent_path = Path(args.latent_ckpt)
        if not latent_path.is_file():
            latent_path = _PROJECT_ROOT / args.latent_ckpt
    else:
        latent_path = ci_dir / f"{tag}_finetuned_H{ih}.pt"
        if not latent_path.is_file() and args.allow_pretrained_fallback:
            latent_path = ci_dir / f"{tag}_pretrained_H{ih}.pt"
            print(f"WARNING: using pretrained (not finetuned): {latent_path}")

    if not latent_path.is_file():
        raise SystemExit(
            f"Missing latent checkpoint: {latent_path}\n"
            f"Run train_ci_latent_etth1 stage 3, or pass --latent-ckpt, or --allow-pretrained-fallback."
        )

    vae_path = Path(args.vae_ckpt) if args.vae_ckpt else latent_dir / f"vae_dim1_H{ih}.pt"
    if not vae_path.is_file():
        vae_path = _PROJECT_ROOT / vae_path
    if not vae_path.is_file():
        raise SystemExit(f"Missing VAE: {vae_path}")

    itrans_path = Path(args.itrans_ckpt) if args.itrans_ckpt else ci_dir / "itransformer_etth1_7var.pt"
    if not itrans_path.is_file():
        itrans_path = _PROJECT_ROOT / itrans_path
    if not itrans_path.is_file():
        raise SystemExit(f"Missing iTransformer: {itrans_path}")

    seven_ix = list(range(N_VARIATES))
    _, _, test_ds, norm_stats = load_dataset(
        args.dataset,
        seven_ix,
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=args.stride,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    n_test = len(test_ds)
    if n_test == 0:
        raise SystemExit("Empty test set.")
    n_cap = min(n_test, args.max_test_samples)
    if n_cap < n_test:
        test_ds = Subset(test_ds, list(range(n_cap)))

    n_plot = min(args.num_samples, len(test_ds))
    idxs = np.linspace(0, len(test_ds) - 1, n_plot, dtype=int)
    n_vars_plot = min(args.variables, N_VARIATES)
    k = LOOKBACK_OVERLAP
    context_len = min(FORECAST_LENGTH * 2, LOOKBACK_LENGTH)
    t_past = np.arange(-context_len, 0)
    t_fut = np.arange(0, FORECAST_LENGTH)

    cfg = build_ci_config(ih, use_guidance)
    vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
    vae_ck = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_ck["vae_state_dict"])
    if "scale_factor" in vae_ck:
        vae.scale_factor.copy_(vae_ck["scale_factor"].to(device).view_as(vae.scale_factor))
    vae.eval()

    guidance = _build_ci_guidance(itrans_path, device) if use_guidance else None
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)
    ldk = torch.load(latent_path, map_location=device, weights_only=False)
    res = model.load_state_dict(ldk["model_state_dict"], strict=False)
    if res.missing_keys:
        print("WARNING: missing keys when loading latent ckpt:", res.missing_keys[:8])
    if res.unexpected_keys:
        print("WARNING: unexpected keys:", res.unexpected_keys[:8])
    model.eval()

    itrans = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    itk = torch.load(itrans_path, map_location=device, weights_only=False)
    itrans.load_state_dict(itk["model_state_dict"])
    itrans.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_names = list(DEFAULT_NAMES)

    fig, axes = plt.subplots(
        n_plot,
        n_vars_plot,
        figsize=(2.9 * n_vars_plot, 2.7 * n_plot),
        squeeze=False,
    )

    for row, idx in enumerate(idxs):
        past, future = test_ds[idx]
        past_b = past.unsqueeze(0).to(device)

        with torch.no_grad():
            x_enc = past_b.permute(0, 2, 1)
            x_dec = torch.zeros(1, FORECAST_LENGTH, N_VARIATES, device=device)
            itrans_out = itrans(x_enc, None, x_dec, None)
            if isinstance(itrans_out, tuple):
                itrans_out = itrans_out[0]
            itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]

            past_flat = past_b.reshape(N_VARIATES, LOOKBACK_LENGTH)
            res_g = model.generate(past_flat, num_ddim_steps=args.ddim_steps, verbose=False)
            ci_raw = res_g["prediction"].squeeze(1).cpu()

        past_dn = _denorm(past, mean, std)
        future_dn = _denorm(future, mean, std)
        fut_trim = future_dn[:, k:]
        itrans_dn = _denorm(itrans_pred, mean, std)
        ci_dn = _denorm(ci_raw, mean, std)

        for col in range(n_vars_plot):
            ax = axes[row, col]
            gt = fut_trim[col].numpy()
            it = itrans_dn[col].numpy()
            ci = ci_dn[col].numpy()

            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#9E9E9E",
                alpha=0.5,
                linewidth=0.8,
            )
            ax.plot(t_fut, gt, color="#2196F3", linewidth=1.4, label="GT" if row == 0 and col == 0 else "")
            ax.plot(
                t_fut,
                it,
                color="#FF9800",
                linewidth=1.1,
                linestyle="--",
                alpha=0.9,
                label="iTrans" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_fut,
                ci,
                color="#4CAF50",
                linewidth=1.15,
                label="CI latent" if row == 0 and col == 0 else "",
            )
            ax.axvline(0, color="black", linestyle=":", alpha=0.25)
            mae_it = float(np.mean(np.abs(it - gt)))
            mae_ci = float(np.mean(np.abs(ci - gt)))
            ax.text(
                0.97,
                0.97,
                f"iT {mae_it:.2f}\nCI {mae_ci:.2f}",
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.72),
            )
            vname = var_names[col] if col < len(var_names) else f"v{col}"
            if row == 0:
                ax.set_title(vname, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"idx {idx}", fontsize=8)
            ax.tick_params(labelsize=6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"{args.dataset} CI latent ({tag}) H={ih} stride={args.stride} DDIM={args.ddim_steps}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_png = out_dir / f"ci_latent_{tag}_{args.dataset}_H{ih}_stride{args.stride}_ddim{args.ddim_steps}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png.resolve()}")
    print(f"Sample indices (test_ds): {idxs.tolist()}")
    print(f"Latent ckpt: {latent_path}")


if __name__ == "__main__":
    main()
