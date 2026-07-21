"""512x512 / 32x32-patch counterpart of the oracle-coarse smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.ordinal_patch_refinement_killtest import smoke
from models.diffusion_tsf.dit import FactorizedDiT
from models.diffusion_tsf.ordinal_window_norm import ordinal_decode, ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool


HIGH = 512
PATCH = 32
HORIZON = 16


class _HorizonView:
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        past, future = self.base[index]
        return past, future[..., :HORIZON]


def _patch_batch(upscaled, target, past_canvas, coarse_bins):
    xs, conds, ys, coords = [], [], [], []
    for t in range(HORIZON):
        row0, col0 = int(coarse_bins[t].item()) * PATCH, t * PATCH
        x, valid = smoke._extract_block(upscaled, row0, col0, PATCH)
        y, _ = smoke._extract_block(target, row0, col0, PATCH)
        hist, _ = smoke._extract_block(past_canvas, row0, 0, PATCH)
        rows = torch.arange(PATCH, device=x.device).view(1, PATCH, 1)
        boundary = (rows == 0).to(x.dtype).expand(1, PATCH, PATCH)
        time_pos = torch.full_like(x, float(t) / (HORIZON - 1))
        vertical_pos = torch.linspace(row0 / HIGH, (row0 + PATCH - 1) / HIGH, PATCH, device=x.device).view(1, PATCH, 1).expand_as(x)
        xs.append(torch.cat([x, boundary, valid, time_pos, vertical_pos], dim=0))
        conds.append(hist); ys.append(y); coords.append((row0, col0))
    return torch.stack(xs), torch.stack(conds), torch.stack(ys), coords


def _collapse(ranks):
    return ranks.reshape(*ranks.shape[:-1], HORIZON, PATCH).mean(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--variate", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=smoke.DEFAULT_OUTPUT.parent / "smoke-512")
    args = parser.parse_args()
    smoke.set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    variates = list(range(7 if args.dataset == "ETTh1" else 1))
    _train, _val, _test, norm_stats = load_dataset(args.dataset, variates, lookback=args.lookback, horizon=HORIZON, stride=1, test_stride=4, use_ordinal_window_norm=True)
    ladder = norm_stats["ordinal_ladder"]
    pool, *_ = load_tsf_pack_pool(args.dataset, variates, lookback=args.lookback, horizon=HORIZON, train_stride=1, test_stride=4, pack_splits=["test"])
    past, future = _HorizonView(pool)[args.window_index]
    past, future = past.unsqueeze(0).to(device), future.unsqueeze(0).to(device)
    past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(past, future, ladder=ladder, apply_ood_shift=True, causal_only=True)
    assert future_ord is not None
    rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=torch.float32)
    target_all = smoke._cdf_from_values(future_ord, rank_max, HIGH)
    coarse_all = smoke._cdf_from_values(future_ord, rank_max, 16)
    past_all = smoke._cdf_from_values(past_ord[..., -16:], rank_max, HIGH)
    vi = args.variate
    target = target_all[:, vi:vi + 1].repeat_interleave(PATCH, dim=-1)
    coarse = coarse_all[:, vi:vi + 1]
    upscaled = F.interpolate(coarse, size=(HIGH, HIGH), mode="nearest")
    past_canvas = past_all[:, vi:vi + 1].repeat_interleave(PATCH, dim=-1)
    coarse_bins = TimeSeriesTo2D.bin_indices_from_cdf(coarse)[0, 0].long()
    inputs, cond, target_patches, coords = _patch_batch(upscaled[0], target[0], past_canvas[0], coarse_bins)
    model = FactorizedDiT(in_channels=5, cond_channels=1, out_channels=1, image_height=PATCH, patch_size=(8, 8), embed_dim=384, depth=8, num_heads=6, context_dim=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    inputs, cond, target_patches = inputs.to(device), cond.to(device), target_patches.to(device)
    valid, losses = inputs[:, 2:3], []
    for _ in range(args.steps):
        logits = model(inputs, torch.zeros(len(inputs), device=device), cond)
        loss = (F.binary_cross_entropy_with_logits(logits, target_patches, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.item()))
    with torch.no_grad():
        refined_patches, monotonic = smoke._project_monotone(torch.sigmoid(model(inputs, torch.zeros(len(inputs), device=device), cond)))
        refined = upscaled.clone()
        for i, (row0, col0) in enumerate(coords): refined[..., row0:row0 + PATCH, col0:col0 + PATCH] = refined_patches[i:i + 1]
    rank_max_one = rank_max[vi:vi + 1]
    gt_rank = future_ord[:, vi:vi + 1]
    naive_rank = _collapse(smoke._decode_ranks(upscaled, rank_max_one))
    refined_rank = _collapse(smoke._decode_ranks(refined, rank_max_one))
    ladder_one, shift_one = smoke._subset_ladder(ladder_b, vi), ood_shift[:, vi:vi + 1]
    _, gt = ordinal_decode(past_ord[:, vi:vi + 1], gt_rank, ladder_one, ood_shift=shift_one)
    _, naive = ordinal_decode(past_ord[:, vi:vi + 1], naive_rank, ladder_one, ood_shift=shift_one)
    _, refined_value = ordinal_decode(past_ord[:, vi:vi + 1], refined_rank, ladder_one, ood_shift=shift_one)
    metrics = {"initial_bce": losses[0], "final_bce": losses[-1], "steps": args.steps, "patch_monotonic_violation_pre_projection": monotonic, "naive_rank_mae": float((naive_rank-gt_rank).abs().mean()), "refined_rank_mae": float((refined_rank-gt_rank).abs().mean()), "naive_snapped_z_mae": float((naive-gt).abs().mean()), "refined_snapped_z_mae": float((refined_value-gt).abs().mean()), "exact_snapped_bin_accuracy": float((refined_value==gt).float().mean())}
    meta = smoke.SmokeMetadata(args.dataset, vi, args.window_index, args.lookback, HORIZON, 16, HIGH, PATCH, "train-split z-score; ordinal encode with causal OOD shift; no instance normalization", "decode_with_ladder rounds GT/naive/refined ranks to the same global ladder", str(device), sum(p.numel() for p in model.parameters()))
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {"past": past[0, vi].cpu().numpy(), "future_raw": future[0, vi].cpu().numpy(), "gt_snapped": gt[0,0].cpu().numpy(), "naive": naive[0,0].cpu().numpy(), "refined": refined_value[0,0].cpu().numpy(), "coarse_cdf_16": coarse[0,0].cpu().numpy(), "target_cdf_256": target[0,0].cpu().numpy(), "upscaled_coarse_256": upscaled[0,0].cpu().numpy(), "refined_cdf_256": refined[0,0].cpu().numpy(), "input_patches": inputs.cpu().numpy(), "target_patches": target_patches.cpu().numpy(), "refined_patches": refined_patches.cpu().numpy(), "patch_coords": np.asarray(coords), "losses": np.asarray(losses)}
    np.savez_compressed(args.output / "smoke_arrays.npz", **arrays)
    (args.output / "metrics.json").write_text(json.dumps({"metadata": asdict(meta), "metrics": metrics}, indent=2), encoding="utf-8")
    paths = smoke._plot(args.output, arrays, meta)
    print(json.dumps({"output": str(args.output), "metrics": metrics, "figures": [str(p) for p in paths]}, indent=2))


if __name__ == "__main__": main()
