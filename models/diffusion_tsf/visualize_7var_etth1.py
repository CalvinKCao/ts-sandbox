"""
7-var ETTh1 diffusion + iTransformer plots on the **same test protocol** as
`visualize_latent_inference.py`: stride, max_test_samples cap, linspace sample indices, seed.

Outputs a grid: rows = test windows, columns = variates (default all 7).

  python -m models.diffusion_tsf.visualize_7var_etth1 \\
    --output-dir models/diffusion_tsf/results/latent_viz
"""

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
from torch.utils.data import Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.latent_experiment_common import (
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    create_itransformer,
    get_device,
    load_dataset,
)

MV_IMAGE_HEIGHT = 128

DEFAULT_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


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


def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """x: (C, T), mean/std (1, C)."""
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument(
        "--seven-var-ckpt",
        type=str,
        default="models/diffusion_tsf/checkpoints_7var/ETTh1/best.pt",
    )
    p.add_argument(
        "--checkpoints-7var-root",
        type=str,
        default="models/diffusion_tsf/checkpoints_7var",
    )
    p.add_argument("--output-dir", type=str, default="models/diffusion_tsf/results/latent_viz")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=500)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument(
        "--variables",
        type=int,
        default=7,
        help="how many variates to plot per row (first N columns)",
    )
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.seven_var_ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path.resolve()}")

    meta_path = ckpt_path.parent / "metadata.json"
    var_names = list(DEFAULT_NAMES)
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        var_names = meta.get("variate_names") or var_names

    root_7 = os.path.abspath(args.checkpoints_7var_root)
    it7_path = os.path.join(_pretrain_dir_for_dim(7, root_7), "itransformer.pt")
    if not os.path.isfile(it7_path):
        raise SystemExit(f"Missing dim7 iTransformer: {it7_path}")

    seven_ix = list(range(7))
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
    n_vars_plot = min(args.variables, 7)

    itrans7 = create_itransformer(num_vars=7).to(device)
    it7k = torch.load(it7_path, map_location=device, weights_only=False)
    itrans7.load_state_dict(it7k["model_state_dict"])
    itrans7.eval()

    diff7 = _create_7var_diffusion().to(device)
    g7 = iTransformerGuidance(
        itrans7, use_norm=True, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH
    )
    diff7.set_guidance_model(g7)
    d7k = torch.load(ckpt_path, map_location=device, weights_only=False)
    diff7.load_state_dict(d7k["model_state_dict"])
    diff7.eval()

    k = LOOKBACK_OVERLAP
    context_len = min(FORECAST_LENGTH * 2, LOOKBACK_LENGTH)
    t_past = np.arange(-context_len, 0)
    t_fut = np.arange(0, FORECAST_LENGTH)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_plot,
        n_vars_plot,
        figsize=(2.9 * n_vars_plot, 2.7 * n_plot),
        squeeze=False,
    )

    for row, idx in enumerate(idxs):
        past, future = test_ds[idx]
        past_t = past.unsqueeze(0).to(device)

        with torch.no_grad():
            x_enc = past_t.permute(0, 2, 1)
            x_dec = torch.zeros(1, FORECAST_LENGTH, 7, device=device)
            itrans_out = itrans7(x_enc, None, x_dec, None)
            if isinstance(itrans_out, tuple):
                itrans_out = itrans_out[0]
            itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]

            res = diff7.generate(past_t, num_ddim_steps=args.ddim_steps, verbose=False)
            diff_pred = res["prediction"].cpu()[0]

        past_dn = _denorm(past, mean, std)
        future_dn = _denorm(future, mean, std)
        fut_trim = future_dn[:, k:]
        itrans_dn = _denorm(itrans_pred, mean, std)
        diff_dn = _denorm(diff_pred, mean, std)

        for col in range(n_vars_plot):
            ax = axes[row, col]
            gt = fut_trim[col].numpy()
            it = itrans_dn[col].numpy()
            df = diff_dn[col].numpy()

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
                df,
                color="#E91E63",
                linewidth=1.15,
                label="7-var diff" if row == 0 and col == 0 else "",
            )
            ax.axvline(0, color="black", linestyle=":", alpha=0.25)
            mae_it = float(np.mean(np.abs(it - gt)))
            mae_df = float(np.mean(np.abs(df - gt)))
            ax.text(
                0.97,
                0.97,
                f"iT {mae_it:.2f}\nD {mae_df:.2f}",
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
        f"{args.dataset} 7-var (H={MV_IMAGE_HEIGHT}, stride={args.stride}, DDIM={args.ddim_steps})",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_png = out_dir / f"7var_{args.dataset}_H{MV_IMAGE_HEIGHT}_stride{args.stride}_ddim{args.ddim_steps}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png.resolve()}")
    print(f"Sample indices (test_ds): {idxs.tolist()}")


if __name__ == "__main__":
    main()
