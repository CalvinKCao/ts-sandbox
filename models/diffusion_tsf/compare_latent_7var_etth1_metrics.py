"""
Compare ETTh1 **latent** (1-var) vs **7-var diffusion** test metrics in **normalized** space.

`results_7var/ETTh1/results.json` reports joint MSE/MAE over all 7 channels with test **stride=1024**
(the 7-var training pipeline). `train_latent_experiment` stage-3 JSON uses **stride=24** and one
variate (default column 0 = HUFL). Those numbers are **not** directly comparable until you align
protocol; this script uses one **stride** for both (default 24, matching latent stage 3).

Usage (repo root):

  python -m models.diffusion_tsf.compare_latent_7var_etth1_metrics \\
    --stride 24 --ddim-steps 50

Optional: `--max-samples 200` to cap eval time on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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

MV_IMAGE_HEIGHT = 128


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


def _trend_acc_1d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred, target (N, T)."""
    pd = pred[:, 1:] - pred[:, :-1]
    td = target[:, 1:] - target[:, :-1]
    return ((pd > 0) == (td > 0)).float().mean().item()


def _trend_acc_mv(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred, target (B, C, T); matches train_7var evaluate_model."""
    pd = pred[:, :, 1:] - pred[:, :, :-1]
    td = target[:, :, 1:] - target[:, :, :-1]
    return ((pd > 0) == (td > 0)).float().mean().item()


def _print_json_snapshots() -> None:
    seven = _PROJECT_ROOT / "models/diffusion_tsf/results_7var/ETTh1/results.json"
    latent_globs = list((_PROJECT_ROOT / "models/diffusion_tsf/results").glob("latent_experiment_H*.json"))
    print("\n--- On-disk JSON (different protocols; see script docstring) ---")
    if seven.is_file():
        with open(seven) as f:
            d = json.load(f)
        em = d.get("eval_metrics", {}).get("averaged") or d.get("eval_metrics", {}).get("single")
        if em:
            print(f"7-var results_7var/ETTh1/results.json: MSE={em['mse']:.6f} MAE={em['mae']:.6f} "
                  f"trend={em.get('trend_accuracy', float('nan')):.4f}  (joint 7-ch, stride=1024 in pipeline)")
    else:
        print("(no results_7var/ETTh1/results.json)")
    if latent_globs:
        for p in sorted(latent_globs):
            with open(p) as f:
                d = json.load(f)
            print(
                f"Latent {p.name}: latent_mse={d.get('latent_mse')} latent_mae={d.get('latent_mae')}  "
                f"(1-var, stride=24 in stage3; variate index matches train script, default 0=HUFL)"
            )
    else:
        print("(no models/diffusion_tsf/results/latent_experiment_H*.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seven-var-ckpt", type=str, default="models/diffusion_tsf/checkpoints_7var/ETTh1/best.pt")
    ap.add_argument("--checkpoints-7var-root", type=str, default="models/diffusion_tsf/checkpoints_7var")
    ap.add_argument("--latent-ckpt-dir", type=str, default="models/diffusion_tsf/checkpoints_latent")
    ap.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    ap.add_argument("--latent-variate-index", type=int, default=0)
    ap.add_argument("--ot-channel-index", type=int, default=6)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    if args.image_height != MV_IMAGE_HEIGHT:
        raise SystemExit(f"7-var ETTh1 best.pt expects image_height={MV_IMAGE_HEIGHT}.")

    device = get_device()
    ot = args.ot_channel_index
    lv = args.latent_variate_index

    seven_ckpt = Path(args.seven_var_ckpt)
    latent_dir = Path(args.latent_ckpt_dir)
    if not seven_ckpt.is_file():
        raise SystemExit(f"Missing {seven_ckpt}")
    if not latent_dir.is_dir():
        raise SystemExit(f"Missing {latent_dir}")

    ih = args.image_height
    vae_path = latent_dir / f"vae_dim1_H{ih}.pt"
    itrans_path = latent_dir / "itransformer_dim1.pt"
    latent_path = latent_dir / f"latent_diffusion_dim1_H{ih}.pt"
    for p in (vae_path, itrans_path, latent_path):
        if not p.is_file():
            raise SystemExit(f"Missing {p}")

    # Data: same stride for both; 7-var and 1-var windows align index-by-index.
    seven_ix = list(range(7))
    _, _, test_mv, _ = load_dataset(
        "ETTh1", seven_ix,
        lookback=LOOKBACK_LENGTH, horizon=FORECAST_LENGTH,
        stride=args.stride, lookback_overlap=LOOKBACK_OVERLAP,
    )
    _, _, test_1v, _ = load_dataset(
        "ETTh1", [lv],
        lookback=LOOKBACK_LENGTH, horizon=FORECAST_LENGTH,
        stride=args.stride, lookback_overlap=LOOKBACK_OVERLAP,
    )
    assert len(test_mv) == len(test_1v)
    n = len(test_mv)
    if args.max_samples > 0:
        n = min(n, args.max_samples)
        test_mv = Subset(test_mv, list(range(n)))
        test_1v = Subset(test_1v, list(range(n)))

    loader_mv = DataLoader(test_mv, batch_size=args.batch_size, shuffle=False)
    loader_1v = DataLoader(test_1v, batch_size=args.batch_size, shuffle=False)

    # --- Latent ---
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

    # --- 7-var ---
    root_7 = os.path.abspath(args.checkpoints_7var_root)
    it7_path = os.path.join(_pretrain_dir_for_dim(7, root_7), "itransformer.pt")
    if not os.path.isfile(it7_path):
        raise SystemExit(f"Missing {it7_path}")

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

    k = LOOKBACK_OVERLAP
    sse_lat = sse_7_joint = sse_7_ot = sse_7_lv = 0.0
    sae_lat = sae_7_joint = sae_7_ot = sae_7_lv = 0.0
    n_el_lat = n_el_7_joint = n_el_7_ot = n_el_7_lv = 0
    sum_trend_lat = sum_trend_7j = sum_trend_7ot = sum_trend_7lv = 0.0
    n_trend_batches = 0

    with torch.no_grad():
        for (past_mv, fut_mv), (past_1v, fut_1v) in zip(loader_mv, loader_1v):
            past_mv = past_mv.to(device)
            past_1v = past_1v.to(device)
            true_mv = fut_mv[:, :, k:].to(device)
            true_1v = fut_1v[:, :, k:].to(device)

            gl = latent_model.generate(past_1v, num_ddim_steps=args.ddim_steps, verbose=False)
            pred_lat = gl["prediction"]

            g7res = diff7.generate(past_mv, num_ddim_steps=args.ddim_steps, verbose=False)
            pred_7 = g7res["prediction"]

            sse_lat += torch.nn.functional.mse_loss(pred_lat, true_1v, reduction="sum").item()
            sae_lat += torch.nn.functional.l1_loss(pred_lat, true_1v, reduction="sum").item()
            n_el_lat += pred_lat.numel()
            sum_trend_lat += _trend_acc_1d(pred_lat.squeeze(1), true_1v.squeeze(1))

            sse_7_joint += torch.nn.functional.mse_loss(pred_7, true_mv, reduction="sum").item()
            sae_7_joint += torch.nn.functional.l1_loss(pred_7, true_mv, reduction="sum").item()
            n_el_7_joint += pred_7.numel()
            sum_trend_7j += _trend_acc_mv(pred_7, true_mv)

            sse_7_ot += torch.nn.functional.mse_loss(
                pred_7[:, ot], true_mv[:, ot], reduction="sum"
            ).item()
            sae_7_ot += torch.nn.functional.l1_loss(
                pred_7[:, ot], true_mv[:, ot], reduction="sum"
            ).item()
            n_el_7_ot += pred_7[:, ot].numel()
            sum_trend_7ot += _trend_acc_1d(pred_7[:, ot], true_mv[:, ot])

            sse_7_lv += torch.nn.functional.mse_loss(
                pred_7[:, lv], true_mv[:, lv], reduction="sum"
            ).item()
            sae_7_lv += torch.nn.functional.l1_loss(
                pred_7[:, lv], true_mv[:, lv], reduction="sum"
            ).item()
            n_el_7_lv += pred_7[:, lv].numel()
            sum_trend_7lv += _trend_acc_1d(pred_7[:, lv], true_mv[:, lv])

            n_trend_batches += 1

    mse_lat = sse_lat / max(1, n_el_lat)
    mae_lat = sae_lat / max(1, n_el_lat)
    mse_7_joint = sse_7_joint / max(1, n_el_7_joint)
    mae_7_joint = sae_7_joint / max(1, n_el_7_joint)
    mse_7_ot = sse_7_ot / max(1, n_el_7_ot)
    mae_7_ot = sae_7_ot / max(1, n_el_7_ot)
    mse_7_lv = sse_7_lv / max(1, n_el_7_lv)
    mae_7_lv = sae_7_lv / max(1, n_el_7_lv)

    _print_json_snapshots()

    print("\n--- Aligned re-eval (normalized space, same stride, DDIM steps) ---")
    print(
        f"Protocol: stride={args.stride}, ddim_steps={args.ddim_steps}, "
        f"test_windows={n}, latent_variate={lv}, ot_channel={ot}"
    )
    print(
        f"Latent (1-var, var {lv}):     MSE={mse_lat:.6f}  MAE={mae_lat:.6f}  "
        f"trend_acc(batch-mean)={sum_trend_lat / max(1, n_trend_batches):.4f}"
    )
    print(
        f"7-var joint (7×T):            MSE={mse_7_joint:.6f}  MAE={mae_7_joint:.6f}  "
        f"trend={sum_trend_7j / max(1, n_trend_batches):.4f}"
    )
    print(
        f"7-var OT channel only:        MSE={mse_7_ot:.6f}  MAE={mae_7_ot:.6f}  "
        f"trend={sum_trend_7ot / max(1, n_trend_batches):.4f}"
    )
    print(
        f"7-var same column as latent:  MSE={mse_7_lv:.6f}  MAE={mae_7_lv:.6f}  "
        f"trend={sum_trend_7lv / max(1, n_trend_batches):.4f}  (channel {lv})"
    )
    print(
        "\nFair latent vs 7-var diffusion: compare **Latent** line to **7-var same column** "
        f"(channel {lv}). Use **OT** row only when latent-variate-index is 6."
    )


if __name__ == "__main__":
    main()
