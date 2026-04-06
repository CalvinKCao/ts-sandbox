"""
Visualize TimeSeriesVAE reconstruction (2D occupancy images: past | future).

The CI-latent VAE is trained on **univariate** synthetic data; here each variate is encoded
the same way as in training (per-window local norm + PixelEncoder + VAE).

Pull checkpoint from cluster (example):

  REMOTE=ccao87@killarney.alliancecan.ca
  RROOT=/scratch/ccao87/ts-sandbox/models/diffusion_tsf
  mkdir -p models/diffusion_tsf/checkpoints_latent
  rsync -avz "$REMOTE:$RROOT/checkpoints_latent/vae_dim1_H128.pt" models/diffusion_tsf/checkpoints_latent/

  python -m models.diffusion_tsf.visualize_vae_reconstruction \\
      --dataset ETTh2 --image-height 128 --num-samples 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.latent_experiment_common import (
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    dataset_registry_row,
    get_device,
    load_dataset,
)
from models.diffusion_tsf.train_ci_latent_etth2 import (
    CKPT_LATENT,
    PixelEncoder,
    _unified_2d,
    build_config,
)
from models.diffusion_tsf.vae import TimeSeriesVAE


def main():
    p = argparse.ArgumentParser(
        description="Plot VAE input 2D image vs deterministic reconstruction (encode mu → decode)",
    )
    p.add_argument("--dataset", type=str, default="ETTh2")
    p.add_argument("--vae-path", type=str, default="", help="default: checkpoints_latent/vae_dim1_H{height}.pt")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--split", type=str, choices=("train", "val", "test"), default="test")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--max-vars", type=int, default=4, help="variates to plot per window")
    p.add_argument(
        "--variate-indices",
        type=str,
        default="",
        help="comma-separated ints for load_dataset (e.g. exchange_rate subset)",
    )
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/vae_recon_viz")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="use reparameterized z instead of mu (noisier recon)",
    )
    args = p.parse_args()

    device = get_device()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_path = Path(args.vae_path) if args.vae_path else CKPT_LATENT / f"vae_dim1_H{args.image_height}.pt"
    if not vae_path.is_absolute():
        vae_path = _PROJECT_ROOT / vae_path
    if not vae_path.is_file():
        raise SystemExit(f"Missing VAE checkpoint: {vae_path}")

    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    latent_ch = int(ckpt.get("latent_channels", 4))
    ckpt_h = ckpt.get("image_height")
    if ckpt_h is not None and int(ckpt_h) != args.image_height:
        raise SystemExit(
            f"VAE was trained with image_height={ckpt_h} but you passed --image-height {args.image_height}",
        )

    _, _, seasonal, _ = dataset_registry_row(args.dataset)
    cfg = build_config(args.image_height, seasonal)

    vidx = None
    if args.variate_indices.strip():
        vidx = [int(x.strip()) for x in args.variate_indices.split(",") if x.strip()]

    train_ds, val_ds, test_ds, _ = load_dataset(
        args.dataset,
        variate_indices=vidx,
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=24,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    n = len(ds)
    if n == 0:
        raise SystemExit("Empty split")

    pixel_enc = PixelEncoder(cfg).to(device).eval()
    vae = TimeSeriesVAE(latent_channels=latent_ch).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    if "scale_factor" in ckpt:
        vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
    vae.eval()

    n_vars = train_ds[0][0].shape[0]
    Vplot = min(args.max_vars, n_vars)
    step = max(1, n // max(args.num_samples, 1))
    sample_indices = [min(i * step, n - 1) for i in range(args.num_samples)]

    for si, idx in enumerate(sample_indices):
        past, future = ds[idx]
        past = past.unsqueeze(0).to(device)
        future = future.unsqueeze(0).to(device)

        fig, axes = plt.subplots(Vplot, 2, figsize=(10, 2.2 * Vplot), squeeze=False)
        for v in range(Vplot):
            pv = past[:, v : v + 1, :]
            fv = future[:, v : v + 1, :]
            with torch.no_grad():
                x = _unified_2d(pixel_enc, pv, fv)
                if args.stochastic:
                    z = vae.encode(x, sample=True)
                else:
                    mu, _ = vae.encode_mu_logvar(x)
                    z = mu
                recon = vae.decode(z)
            mse = F.mse_loss(recon, x).item()

            for j, (title, im) in enumerate((("input", x), ("recon", recon))):
                ax = axes[v, j]
                ax.imshow(im[0, 0].cpu().numpy(), aspect="auto", cmap="gray", vmin=-1, vmax=1)
                ax.set_title(f"v{v} {title}  mse={mse:.5f}" if j == 1 else f"v{v} {title}")
                ax.set_xticks([])
                ax.set_yticks([])

        mode = "z_sample" if args.stochastic else "mu"
        fig.suptitle(
            f"{args.dataset} {args.split} idx={idx}  VAE {vae_path.name}  ({mode})",
            fontsize=10,
        )
        fig.tight_layout()
        png = out_dir / f"vae_recon_{args.dataset}_{args.split}_idx{idx}_H{args.image_height}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(png)


if __name__ == "__main__":
    main()
