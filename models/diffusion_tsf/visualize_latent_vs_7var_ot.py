"""
Overlay ETTh1 **OT** (7-var joint model) vs **univariate latent** on the same test windows.

7-var checkpoint predicts all channels; we plot only the OT channel (index 6 in ETTh1.csv).
Latent checkpoints from `train_latent_experiment.py` default to **variate 0 (HUFL)** unless you
trained with another column — set `--latent-variate-index` to match your latent ckpt.

Usage (repo root, venv on):

  python -m models.diffusion_tsf.visualize_latent_vs_7var_ot \\
    --seven-var-ckpt models/diffusion_tsf/checkpoints_7var/ETTh1/best.pt \\
    --latent-ckpt-dir models/diffusion_tsf/checkpoints_latent \\
    --image-height 128 \\
    --output-dir models/diffusion_tsf/results/latent_viz
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

from models.diffusion_tsf.config import DiffusionTSFConfig
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

# Must match `train_7var_pipeline.create_diffusion_model` / ETTh1/best.pt
MV_IMAGE_HEIGHT = 128


def _denorm_univariate(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if x.dim() == 2:
        x = x.squeeze(0)
    m = float(mean.squeeze())
    s = float(std.squeeze())
    return (x.detach().cpu().numpy() * s + m).astype(np.float64)


def _denorm_one_ch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, ch: int) -> np.ndarray:
    """x: (T,) forecast slice; mean/std (1, C)."""
    m = float(mean.squeeze()[ch])
    s = float(std.squeeze()[ch])
    return (x.detach().cpu().numpy() * s + m).astype(np.float64)


def _itrans_forward_raw(model: torch.nn.Module, past_bvc: torch.Tensor, device: torch.device) -> torch.Tensor:
    b, c, l = past_bvc.shape
    x_enc = past_bvc.permute(0, 2, 1)
    x_dec = torch.zeros(b, FORECAST_LENGTH, c, device=device)
    out = model(x_enc, None, x_dec, None)
    if isinstance(out, tuple):
        out = out[0]
    return out.permute(0, 2, 1)


def _pretrain_dir_for_dim(dim: int, base_dir: str) -> str:
    return os.path.join(base_dir, f"pretrained_dim{dim}")


def _create_7var_diffusion() -> DiffusionTSF:
    cfg = DiffusionTSFConfig(
        num_variables=7,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH + LOOKBACK_OVERLAP,
        lookback_overlap=LOOKBACK_OVERLAP,
        past_loss_weight=0.3,
        image_height=MV_IMAGE_HEIGHT,
        representation_mode="cdf",
        use_coordinate_channel=True,
        use_guidance_channel=True,
        num_diffusion_steps=1000,
        model_type="unet",
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
        use_hybrid_condition=True,
        use_gradient_checkpointing=False,
        use_amp=False,
    )
    return DiffusionTSF(cfg)


def main():
    p = argparse.ArgumentParser(description="7-var OT vs latent on same ETTh1 test windows")
    p.add_argument(
        "--seven-var-ckpt",
        type=str,
        default="models/diffusion_tsf/checkpoints_7var/ETTh1/best.pt",
    )
    p.add_argument(
        "--checkpoints-7var-root",
        type=str,
        default="models/diffusion_tsf/checkpoints_7var",
        help="Parent dir containing pretrained_dim7/itransformer.pt",
    )
    p.add_argument("--latent-ckpt-dir", type=str, default="models/diffusion_tsf/checkpoints_latent")
    p.add_argument("--vae-name", type=str, default=None)
    p.add_argument("--itrans-name", type=str, default="itransformer_dim1.pt")
    p.add_argument("--latent-name", type=str, default=None)
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument(
        "--ot-channel-index",
        type=int,
        default=6,
        help="ETTh1 column index for OT (0..6)",
    )
    p.add_argument(
        "--latent-variate-index",
        type=int,
        default=0,
        help="Column used for univariate latent ckpt (default 0 = HUFL, matches train_latent_experiment)",
    )
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/latent_viz")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=500)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seven_ckpt = Path(args.seven_var_ckpt)
    if not seven_ckpt.is_file():
        raise SystemExit(f"7-var checkpoint not found: {seven_ckpt.resolve()}")

    latent_dir = Path(args.latent_ckpt_dir)
    if not latent_dir.is_dir():
        raise SystemExit(f"Latent checkpoint dir not found: {latent_dir.resolve()}")

    ih = args.image_height
    vae_name = args.vae_name or f"vae_dim1_H{ih}.pt"
    latent_name = args.latent_name or f"latent_diffusion_dim1_H{ih}.pt"
    vae_path = latent_dir / vae_name
    itrans_path = latent_dir / args.itrans_name
    latent_path = latent_dir / latent_name

    if args.image_height != MV_IMAGE_HEIGHT:
        raise SystemExit(
            f"This script loads 7-var checkpoints trained at image_height={MV_IMAGE_HEIGHT}; "
            f"got --image-height {args.image_height}. Use {MV_IMAGE_HEIGHT} or extend the loader."
        )

    for path, label in [(vae_path, "VAE"), (itrans_path, "iTransformer dim1"), (latent_path, "latent diffusion")]:
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    # --- 7-var test (full channels) + latent test (single column); same stride / split ---
    seven_indices = list(range(7))
    _, _, test_mv, norm_mv = load_dataset(
        args.dataset,
        seven_indices,
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=args.stride,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    _, _, test_1v, norm_1v = load_dataset(
        args.dataset,
        [args.latent_variate_index],
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=args.stride,
        lookback_overlap=LOOKBACK_OVERLAP,
    )

    if len(test_mv) != len(test_1v):
        raise SystemExit(f"Test len mismatch: 7-var {len(test_mv)} vs 1-var {len(test_1v)}")

    mean_mv = torch.tensor(norm_mv["mean"], dtype=torch.float32)
    std_mv = torch.tensor(norm_mv["std"], dtype=torch.float32)
    mean_1v = torch.tensor(norm_1v["mean"], dtype=torch.float32)
    std_1v = torch.tensor(norm_1v["std"], dtype=torch.float32)

    n_test = len(test_mv)
    if n_test == 0:
        raise SystemExit("Empty test set.")
    n_cap = min(n_test, args.max_test_samples)
    if n_cap < n_test:
        test_mv = Subset(test_mv, list(range(n_cap)))
        test_1v = Subset(test_1v, list(range(n_cap)))

    n_plot = min(args.num_samples, len(test_mv))
    idxs = np.linspace(0, len(test_mv) - 1, n_plot, dtype=int)

    # --- Latent stack ---
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

    itrans1 = create_itransformer(num_vars=1).to(device)
    itk = torch.load(itrans_path, map_location=device, weights_only=False)
    itrans1.load_state_dict(itk["model_state_dict"])
    itrans1.eval()

    # --- 7-var diffusion + dim7 iTransformer ---
    root_7 = os.path.abspath(args.checkpoints_7var_root)
    dim7_dir = _pretrain_dir_for_dim(7, root_7)
    it7_path = os.path.join(dim7_dir, "itransformer.pt")
    if not os.path.isfile(it7_path):
        raise SystemExit(f"Missing dim7 iTransformer: {it7_path}")

    itrans7 = create_itransformer(num_vars=7).to(device)
    it7k = torch.load(it7_path, map_location=device, weights_only=False)
    itrans7.load_state_dict(it7k["model_state_dict"])
    itrans7.eval()

    diff7 = _create_7var_diffusion().to(device)
    g7 = iTransformerGuidance(
        itrans7, use_norm=True, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH
    )
    diff7.set_guidance_model(g7)
    d7k = torch.load(seven_ckpt, map_location=device, weights_only=False)
    diff7.load_state_dict(d7k["model_state_dict"])
    diff7.eval()

    ot_i = args.ot_channel_index
    if ot_i < 0 or ot_i > 6:
        raise SystemExit("--ot-channel-index must be in 0..6 for ETTh1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_len = min(FORECAST_LENGTH * 2, LOOKBACK_LENGTH)
    t_past = np.arange(-context_len, 0)
    t_fut = np.arange(0, FORECAST_LENGTH)

    fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.9 * n_plot), squeeze=False)

    note = ""
    if args.latent_variate_index != ot_i:
        note = (
            f"latent trained on col {args.latent_variate_index} "
            f"(7-var line is OT col {ot_i})"
        )

    for row, idx in enumerate(idxs):
        past_mv, future_mv = test_mv[idx]
        past_1v, future_1v = test_1v[idx]

        past_mv_t = past_mv.unsqueeze(0).to(device)
        past_1v_t = past_1v.unsqueeze(0).to(device)

        with torch.no_grad():
            it1_pred = _itrans_forward_raw(itrans1, past_1v_t, device).cpu()[0]
            it7_pred = _itrans_forward_raw(itrans7, past_mv_t, device).cpu()[0]

            lat = latent_model.generate(past_1v_t, num_ddim_steps=args.ddim_steps, verbose=False)
            lat_pred = lat["prediction"].cpu()[0]

            g7_res = diff7.generate(past_mv_t, num_ddim_steps=args.ddim_steps, verbose=False)
            diff7_pred = g7_res["prediction"].cpu()[0]

        # GT / preds on original scale — OT channel from 7-var norm
        gt_mv = future_mv[:, LOOKBACK_OVERLAP:]
        gt_ot_dn = _denorm_one_ch(gt_mv[ot_i], mean_mv, std_mv, ot_i)

        it7_ot_dn = _denorm_one_ch(it7_pred[ot_i, :FORECAST_LENGTH], mean_mv, std_mv, ot_i)
        d7_ot_dn = _denorm_one_ch(diff7_pred[ot_i, :FORECAST_LENGTH], mean_mv, std_mv, ot_i)

        past_ot_dn = _denorm_one_ch(past_mv[ot_i], mean_mv, std_mv, ot_i)
        gt_1 = future_1v[:, LOOKBACK_OVERLAP:]
        gt_1_dn = _denorm_univariate(gt_1[0], mean_1v, std_1v)
        it1_dn = _denorm_univariate(it1_pred[0, :FORECAST_LENGTH], mean_1v, std_1v)
        lat_dn = _denorm_univariate(lat_pred, mean_1v, std_1v)
        past_1_dn = _denorm_univariate(past_1v[0], mean_1v, std_1v)

        ax = axes[row, 0]
        ax.plot(t_past, past_ot_dn[-context_len:], color="#BDBDBD", alpha=0.55, linewidth=0.9, label="context (OT)")
        ax.plot(t_fut, gt_ot_dn, color="#2196F3", linewidth=1.6, label="GT (OT, 7-var norm)")
        ax.plot(t_fut, it7_ot_dn, color="#FF9800", linewidth=1.1, linestyle="--", alpha=0.9, label="iTrans dim7 (OT)")
        ax.plot(t_fut, d7_ot_dn, color="#4CAF50", linewidth=1.35, label="7-var diffusion (OT)")
        ax.plot(t_fut, lat_dn, color="#E91E63", linewidth=1.3, label=f"latent (var {args.latent_variate_index})")
        ax.plot(t_fut, it1_dn, color="#795548", linewidth=0.95, linestyle=":", alpha=0.85, label="iTrans dim1 (latent var)")

        ax.axvline(0, color="black", linestyle=":", alpha=0.25)
        mae_d7 = float(np.mean(np.abs(d7_ot_dn - gt_ot_dn)))
        mae_lat = float(np.mean(np.abs(lat_dn - gt_1_dn)))
        txt = f"MAE 7-var OT: {mae_d7:.3f}\nMAE latent vs GT(var{args.latent_variate_index}): {mae_lat:.3f}"
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
        ax.set_ylabel(f"idx {idx}")
        if row == 0:
            title = f"{args.dataset} OT vs latent H={ih} (DDIM {args.ddim_steps})"
            if note:
                title += f" — {note}"
            ax.set_title(title)
        ax.legend(loc="upper left", fontsize=6.5, ncol=2)

    plt.tight_layout()
    out_png = out_dir / f"latent_vs_7var_{args.dataset}_OT_latentvar{args.latent_variate_index}_H{ih}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png.resolve()}")


if __name__ == "__main__":
    main()
