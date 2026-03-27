"""
Load latent-diffusion checkpoints (VAE + iTransformer + latent U-Net) and plot
test windows: context + ground-truth future vs iTransformer vs latent diffusion.

Designed for ETTh1 univariate runs from `train_latent_experiment.py` (default variate 0 = HUFL; use `--variate-index 6` for OT).

**Pull checkpoints from Killarney** (example — adjust user/host/paths):

```bash
rsync -avz ccao87@killarney.alliancecan.ca:/scratch/ccao87/ts-sandbox/models/diffusion_tsf/checkpoints_latent/ \\
  ./models/diffusion_tsf/checkpoints_latent/
```

**Run** (from repo root, venv activated):

```bash
python -m models.diffusion_tsf.visualize_latent_inference \\
  --ckpt-dir models/diffusion_tsf/checkpoints_latent \\
  --image-height 128 \\
  --output-dir models/diffusion_tsf/results/latent_viz \\
  --num-samples 4
```

Optional **univariate** pixel `DiffusionTSF` overlay (same `image_height`, `n_vars=1`).
Do **not** pass the 7-variate `ETTh1/best.pt` — shapes will not match.

```bash
python -m models.diffusion_tsf.visualize_latent_inference \\
  --pixel-ckpt path/to/univariate_diffusion.pt
```
"""

from __future__ import annotations

import argparse
import os
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

from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.latent_diffusion_model import LatentDiffusionTSF
from models.diffusion_tsf.latent_experiment_common import (
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    create_itransformer,
    get_device,
    load_dataset,
)
from models.diffusion_tsf.train_latent_experiment import build_itrans_guidance, build_latent_config
from models.diffusion_tsf.vae import TimeSeriesVAE


def _denorm_univariate(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """x: (T,) or (1, T); mean/std from load_dataset, shape (1,1)."""
    if x.dim() == 2:
        x = x.squeeze(0)
    m = float(mean.squeeze())
    s = float(std.squeeze())
    return (x.detach().cpu().numpy() * s + m).astype(np.float64)


def _itrans_forward_raw(model: torch.nn.Module, past_bvc: torch.Tensor, device: torch.device) -> torch.Tensor:
    """past_bvc: (B, 1, L) -> (B, 1, FORECAST_LENGTH) in same normalized space as dataset."""
    b, c, l = past_bvc.shape
    x_enc = past_bvc.permute(0, 2, 1)
    x_dec = torch.zeros(b, FORECAST_LENGTH, c, device=device)
    out = model(x_enc, None, x_dec, None)
    if isinstance(out, tuple):
        out = out[0]
    return out.permute(0, 2, 1)


def main():
    p = argparse.ArgumentParser(description="Visualize latent diffusion inference on ETTh1 test")
    p.add_argument("--ckpt-dir", type=str, default="models/diffusion_tsf/checkpoints_latent")
    p.add_argument("--vae-name", type=str, default=None, help="default: vae_dim1_H{image_height}.pt")
    p.add_argument("--itrans-name", type=str, default="itransformer_dim1.pt")
    p.add_argument("--latent-name", type=str, default=None, help="default: latent_diffusion_dim1_H{image_height}.pt")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument(
        "--variate-index",
        type=int,
        default=0,
        help="ETTh1 column index for univariate path (0=HUFL, 6=OT)",
    )
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/latent_viz")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--stride", type=int, default=24, help="test set stride (match train_latent stage3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pixel-ckpt",
        type=str,
        default=None,
        help="Optional univariate (1-ch) pixel DiffusionTSF ckpt; not 7-var ETTh1 best.pt",
    )
    p.add_argument("--max-test-samples", type=int, default=500, help="cap test set size for index selection")
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir.resolve()}")

    ih = args.image_height
    vae_name = args.vae_name or f"vae_dim1_H{ih}.pt"
    latent_name = args.latent_name or f"latent_diffusion_dim1_H{ih}.pt"
    vae_path = ckpt_dir / vae_name
    itrans_path = ckpt_dir / args.itrans_name
    latent_path = ckpt_dir / latent_name

    for path, label in [(vae_path, "VAE"), (itrans_path, "iTransformer"), (latent_path, "latent diffusion")]:
        if not path.is_file():
            raise SystemExit(f"Missing {label} checkpoint: {path}")

    cfg = build_latent_config(ih)
    vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
    vck = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vck["vae_state_dict"])
    if "scale_factor" in vck:
        vae.scale_factor.copy_(vck["scale_factor"].to(device).view_as(vae.scale_factor))
    vae.eval()

    guidance = build_itrans_guidance(itrans_path, device)
    latent_model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)
    lck = torch.load(latent_path, map_location=device, weights_only=False)
    latent_model.load_state_dict(lck["model_state_dict"], strict=True)
    latent_model.eval()

    itrans_model = create_itransformer(num_vars=1).to(device)
    itk = torch.load(itrans_path, map_location=device, weights_only=False)
    itrans_model.load_state_dict(itk["model_state_dict"])
    itrans_model.eval()

    pixel_model = None
    if args.pixel_ckpt:
        ck = Path(args.pixel_ckpt)
        if not ck.is_file():
            raise SystemExit(f"--pixel-ckpt not found: {ck}")
        from models.diffusion_tsf.latent_experiment_common import create_pixel_diffusion_baseline

        pixel_model = create_pixel_diffusion_baseline(ih).to(device)
        pck = torch.load(ck, map_location=device, weights_only=False)
        sd = pck.get("model_state_dict", pck)
        try:
            pixel_model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise SystemExit(
                "Pixel ckpt did not load (wrong architecture?). "
                "Use a univariate DiffusionTSF checkpoint with matching image_height; "
                "7-variate ETTh1 best.pt will not work.\n"
                f"Original error: {e}"
            ) from e
        pixel_model.set_guidance_model(guidance)
        pixel_model.eval()

    train_ds, val_ds, test_ds, norm_stats = load_dataset(
        args.dataset,
        [args.variate_index],
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=args.stride,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    n_test = len(test_ds)
    if n_test == 0:
        raise SystemExit("Empty test set (check stride / dataset).")
    n_cap = min(n_test, args.max_test_samples)
    if n_cap < n_test:
        test_ds = Subset(test_ds, list(range(n_cap)))

    n_plot = min(args.num_samples, len(test_ds))
    idxs = np.linspace(0, len(test_ds) - 1, n_plot, dtype=int)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_len = min(FORECAST_LENGTH * 2, LOOKBACK_LENGTH)
    t_past = np.arange(-context_len, 0)
    t_fut = np.arange(0, FORECAST_LENGTH)

    fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.8 * n_plot), squeeze=False)
    var_label = f"var{args.variate_index}"

    for row, idx in enumerate(idxs):
        past, future = test_ds[idx]
        past_t = past.unsqueeze(0).to(device)
        future_t = future.unsqueeze(0)

        with torch.no_grad():
            it_pred = _itrans_forward_raw(itrans_model, past_t, device).cpu()[0]
            lat = latent_model.generate(past_t, num_ddim_steps=args.ddim_steps, verbose=False)
            lat_pred = lat["prediction"].cpu()[0]

            pix_pred = None
            if pixel_model is not None:
                px = pixel_model.generate(past_t, num_ddim_steps=args.ddim_steps, verbose=False)
                pix_pred = px["prediction"].cpu()[0]

        gt = future_t[0, :, LOOKBACK_OVERLAP:]
        past_dn = _denorm_univariate(past[0], mean, std)
        gt_dn = _denorm_univariate(gt, mean, std)
        it_dn = _denorm_univariate(it_pred[0, : FORECAST_LENGTH], mean, std)
        lat_dn = _denorm_univariate(lat_pred, mean, std)
        pix_dn = _denorm_univariate(pix_pred[0], mean, std) if pix_pred is not None else None

        ax = axes[row, 0]
        ax.plot(t_past, past_dn[-context_len:], color="#9E9E9E", alpha=0.55, linewidth=0.9, label="context")
        ax.plot(t_fut, gt_dn, color="#2196F3", linewidth=1.6, label="ground truth")
        ax.plot(t_fut, it_dn, color="#FF9800", linewidth=1.2, linestyle="--", alpha=0.9, label="iTransformer")
        ax.plot(t_fut, lat_dn, color="#E91E63", linewidth=1.3, label="latent diffusion")
        if pix_dn is not None:
            ax.plot(t_fut, pix_dn, color="#4CAF50", linewidth=1.2, alpha=0.85, label="pixel U-Net (ckpt)")

        ax.axvline(0, color="black", linestyle=":", alpha=0.25)
        mae_it = np.mean(np.abs(it_dn - gt_dn))
        mae_lat = np.mean(np.abs(lat_dn - gt_dn))
        txt = f"iTrans MAE: {mae_it:.3f}\nLatent MAE: {mae_lat:.3f}"
        if pix_dn is not None:
            txt += f"\nPixel MAE: {np.mean(np.abs(pix_dn - gt_dn)):.3f}"
        ax.text(
            0.97,
            0.97,
            txt,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.78),
        )
        ax.set_ylabel(f"sample idx {idx}")
        if row == 0:
            ax.set_title(f"{args.dataset} {var_label} (H={ih}, DDIM {args.ddim_steps})")
        ax.legend(loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    out_png = out_dir / f"latent_{args.dataset}_{var_label}_H{ih}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png.resolve()}")


if __name__ == "__main__":
    main()
