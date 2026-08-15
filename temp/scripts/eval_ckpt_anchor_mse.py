#!/usr/bin/env python3
"""Compute one-shot val anchor_mse (+ pixel_acc) for a vertical_dual finetune ckpt.

Why this exists: job 4279968 used hp_objective=anchor_pixel_acc, so Optuna never
logged anchor_mse (val_loss_history.json has anchor_mse=null). Trial winner
weights (trial_21, best at ep1) were also overwritten by refit → best.pt.

Run on a machine that has the .pt (Killarney scratch or a synced local copy):

  source .venv/bin/activate
  python temp/scripts/eval_ckpt_anchor_mse.py \\
    --run-dir results/ckpts/07-16-4279968-ETTh2-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_pixel_acc

If you still have the Optuna winner on disk somewhere:
  python temp/scripts/eval_ckpt_anchor_mse.py --ckpt /path/to/trial_21_best.pt --run-dir <same run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN = (
    "results/ckpts/07-16-4279968-ETTh2-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_"
    "ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_pixel_acc"
)
DEFAULT_CFG = (
    "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_"
    "bs_mid_vertical_dual_joint_g_lr_batch_s30r20_pixel_acc.yaml"
)


def _resolve_guidance(run_dir: Path, subset: str) -> Path:
    for name in (
        f"{subset}_patch_guidance.pt",
        f"{subset}_patch_guidance_hp_best.pt",
        "patch_guidance_synthetic.pt",
    ):
        p = run_dir / name
        if p.is_file():
            return p
    hits = sorted(run_dir.glob("*_patch_guidance*.pt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no patch guidance under {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--ckpt", default="", help="defaults to <run>/<subset>/vertical_dual/best.pt")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--subset", default="")
    ap.add_argument("--val-fraction", type=float, default=0.5, help="same as hp_anchor_eval_val_fraction")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-pixel-acc", action="store_true")
    args = ap.parse_args()

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
        _anchor_mse_on_loader,
        _anchor_pixel_error_on_loader,
        _fraction_subset,
    )
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.visualize_utils import _load_staged_diffusion_from_ckpt
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (REPO / run_dir).resolve()
    subset = args.subset or args.dataset
    ckpt = Path(args.ckpt).expanduser() if args.ckpt else (
        run_dir / subset / "vertical_dual" / "best.pt"
    )
    if not ckpt.is_absolute():
        ckpt = (REPO / ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"missing ckpt {ckpt}\n"
            "Optuna trial_21_best.pt is deleted after refit; only refit best.pt remains "
            "(and this WSL tree may not have synced the .pt at all). Point --ckpt at a "
            "cluster path if you have one."
        )

    meta_path = ckpt.parent / "metadata.json"
    tuned = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        tuned = dict(meta.get("tuned_params") or {})
        print(
            f"meta: selection={meta.get('selection_metric')} "
            f"hp_best_sel={meta.get('hp_best_val_loss')} "
            f"refit_best_sel={meta.get('best_selection_score')} "
            f"best_trial={meta.get('best_trial')} "
            f"g={tuned.get('binary_length_g')} lr={tuned.get('learning_rate')}"
        )

    guide = _resolve_guidance(run_dir, subset)
    cfg = load_experiment_config(
        str(REPO / args.config),
        cli_overrides={
            "dataset": args.dataset,
            "checkpoint_dir": str(run_dir),
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg)
    state.dataset = args.dataset
    if args.device:
        state.device = args.device
    device = state.resolve_device()

    job = generate_dataset_job(state.dataset)
    variate_indices = list(state.variate_indices or job["variate_indices"])
    state.variate_indices = variate_indices
    subset_meta = state.data_subset_resolved or {}
    train_ds, val_ds, _test_ds, norm_stats = load_dataset(
        state.dataset,
        variate_indices,
        stride=int(subset_meta.get("train_stride", state.window_stride)),
        test_stride=int(subset_meta.get("test_stride", 4)),
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    anchor_ds = _fraction_subset(val_ds, float(args.val_fraction), int(args.seed))
    loader = DataLoader(
        anchor_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0,
    )
    print(
        f"ckpt={ckpt}\nguide={guide}\n"
        f"anchor_windows={len(anchor_ds)}/{len(val_ds)} frac={args.val_fraction} "
        f"device={device}"
    )

    model, _ = _load_staged_diffusion_from_ckpt(
        ckpt_path=str(ckpt),
        stage="vertical_dual",
        itrans_ckpt_path=str(guide),
        n_vars=len(variate_indices),
        device=device,
        tuned_params=tuned or None,
        guidance_type=state.guidance_type,
    )
    model.eval()

    mse = _anchor_mse_on_loader(model, loader, device)
    print(f"anchor_mse={mse:.6f}")
    if not args.skip_pixel_acc:
        err, acc = _anchor_pixel_error_on_loader(model, loader, device)
        print(f"anchor_pixel_acc={acc:.6f}  sel_err={err:.6f}")


if __name__ == "__main__":
    main()
