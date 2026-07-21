#!/usr/bin/env python3
"""One-window oracle-coarse smoke test for ordinal patch refinement.

The test intentionally supplies ground-truth 16-bin future CDFs.  It checks
only whether a patch refiner can restore the missing within-bin detail, while
reusing the repository's train-z-score -> ordinal-ladder -> bounded-CDF path.
It is not a production forecaster and never launches cluster work.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.diffusion_tsf.dit import FactorizedDiT
from models.diffusion_tsf.ordinal_window_norm import OrdinalLadder, ordinal_decode, ordinal_encode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ordinal_patch_refinement_killtest" / "smoke"


@dataclass(frozen=True)
class SmokeMetadata:
    dataset: str
    variate: int
    source_window_index: int
    lookback: int
    horizon: int
    coarse_height: int
    fine_height: int
    patch_size: int
    normalization: str
    ordinal_snapping: str
    device: str
    parameters: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset_ladder(ladder: OrdinalLadder, variate: int) -> OrdinalLadder:
    return OrdinalLadder(
        values=ladder.values[:, variate : variate + 1].clone(),
        n_unique=ladder.n_unique[:, variate : variate + 1].clone(),
        tie_atol=ladder.tie_atol,
    )


def _cdf_from_values(
    values: torch.Tensor,
    rank_max: torch.Tensor,
    height: int,
) -> torch.Tensor:
    """Direct bounded CDF encoding, via the repository's dual-height encoder."""
    to_2d = TimeSeriesTo2D(height=height, max_scale=1.0).to(values.device)
    cdf, _unused = to_2d.encode_dual_heights_bounded(
        values,
        coarse_height=height,
        fine_height=1,
        value_min=0.0,
        value_max_per_variate=rank_max,
    )
    return cdf


def _decode_ranks(cdf: torch.Tensor, rank_max: torch.Tensor) -> torch.Tensor:
    height = int(cdf.shape[-2])
    bins = TimeSeriesTo2D.bin_indices_from_cdf(cdf)
    return (bins + 0.5) / float(height) * rank_max.view(1, -1, 1)


def _project_monotone(prob: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Project local probability maps to hard, bottom-up CDF staircases."""
    raw = (prob >= 0.5).to(prob.dtype)
    violation = float((raw[..., 1:, :] > raw[..., :-1, :]).float().mean().item())
    occupancy = raw.sum(dim=-2, keepdim=True).round().clamp(0, raw.shape[-2])
    rows = torch.arange(raw.shape[-2], device=raw.device).view(1, 1, -1, 1)
    return (rows < occupancy).to(raw.dtype), violation


def _extract_block(canvas: torch.Tensor, row0: int, col0: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a vertically padded square block and its valid-pixel mask."""
    h, w = canvas.shape[-2:]
    out = torch.zeros((*canvas.shape[:-2], size, size), device=canvas.device, dtype=canvas.dtype)
    valid = torch.zeros_like(out)
    r0, r1 = max(0, row0), min(h, row0 + size)
    c0, c1 = max(0, col0), min(w, col0 + size)
    if r0 < r1 and c0 < c1:
        dr0, dc0 = r0 - row0, c0 - col0
        out[..., dr0 : dr0 + (r1 - r0), dc0 : dc0 + (c1 - c0)] = canvas[..., r0:r1, c0:c1]
        valid[..., dr0 : dr0 + (r1 - r0), dc0 : dc0 + (c1 - c0)] = 1.0
    return out, valid


def _patch_batch(
    upscaled: torch.Tensor,
    target: torch.Tensor,
    past_canvas: torch.Tensor,
    coarse_bins: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Build one non-overlapping 16x16 patch per future step from input coordinates."""
    inputs, conds, targets, coords = [], [], [], []
    horizon = int(coarse_bins.shape[-1])
    high = int(upscaled.shape[-2])
    for t in range(horizon):
        # Coarse bin k occupies high rows [k*16, (k+1)*16); this is input-derived.
        row0 = int(coarse_bins[t].item()) * patch_size
        col0 = t * patch_size
        inp, valid = _extract_block(upscaled, row0, col0, patch_size)
        tgt, _ = _extract_block(target, row0, col0, patch_size)
        hist, _ = _extract_block(past_canvas, row0, 0, patch_size)
        rows = torch.arange(patch_size, device=upscaled.device).view(1, patch_size, 1)
        boundary = (rows == 0).to(upscaled.dtype).expand(1, patch_size, patch_size)
        time_pos = torch.full_like(inp, float(t) / max(1, horizon - 1))
        vertical_pos = torch.linspace(row0 / high, (row0 + patch_size - 1) / high, patch_size, device=upscaled.device).view(1, patch_size, 1).expand_as(inp)
        inputs.append(torch.cat([inp, boundary, valid, time_pos, vertical_pos], dim=0))
        conds.append(hist)
        targets.append(tgt)
        coords.append((row0, col0))
    return torch.stack(inputs), torch.stack(conds), torch.stack(targets), coords


def _plot(output: Path, arrays: dict[str, np.ndarray], meta: SmokeMetadata) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    title = f"{meta.dataset} v{meta.variate} window={meta.source_window_index}"
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(-len(arrays["past"]), 0)
    ax.plot(x, arrays["past"], label="history", color="0.4")
    ax.plot(np.arange(16), arrays["gt_snapped"], label="GT snapped", marker="o")
    ax.plot(np.arange(16), arrays["naive"], label="naive upscale", marker="x")
    ax.plot(np.arange(16), arrays["refined"], label="refined", marker="s")
    ax.set(title=f"Decoded normalized trace — {title}", xlabel="relative time", ylabel="train-z-score")
    ax.legend(ncol=2)
    fig.tight_layout(); path = output / "decoded_trace.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    panels = [("coarse_cdf_16", arrays["coarse_cdf_16"]), ("target_cdf_256", arrays["target_cdf_256"]), ("upscaled_coarse_256", arrays["upscaled_coarse_256"]), ("refined_cdf_256", arrays["refined_cdf_256"]), ("target_minus_upscaled", arrays["target_cdf_256"] - arrays["upscaled_coarse_256"]), ("refined_minus_target", arrays["refined_cdf_256"] - arrays["target_cdf_256"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, image) in zip(axes.flat, panels):
        ax.imshow(image, origin="lower", aspect="auto", cmap="coolwarm" if "minus" in name else "viridis")
        ax.set_title(name); ax.set_xlabel("future column"); ax.set_ylabel("value row")
    fig.suptitle(title); fig.tight_layout(); path = output / "cdf_canvases.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(arrays["upscaled_coarse_256"], origin="lower", aspect="auto", cmap="viridis")
    for row0, col0 in arrays["patch_coords"]:
        ax.add_patch(plt.Rectangle((col0, row0), 16, 16, fill=False, edgecolor="white", linewidth=0.8))
    ax.set(title=f"Input-coordinate patch selection (white; padding would be clipped) — {title}", xlabel="high-res time column", ylabel="high-res value row")
    fig.tight_layout(); path = output / "patch_selection.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, i in zip(axes.flat, range(8)):
        ax.imshow(arrays["input_patches"][i, 0], origin="lower", cmap="viridis")
        ax.contour(arrays["target_patches"][i, 0], levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(f"t={i}"); ax.axis("off")
    fig.suptitle(f"Input patches with target boundary overlays — {title}"); fig.tight_layout(); path = output / "patches_before.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, i in zip(axes.flat, range(8)):
        ax.imshow(arrays["refined_patches"][i, 0], origin="lower", cmap="viridis")
        ax.contour(arrays["target_patches"][i, 0], levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(f"t={i}"); ax.axis("off")
    fig.suptitle(f"Refined patches with target boundary overlays — {title}"); fig.tight_layout(); path = output / "patches_after.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--variate", type=int, default=0, help="Configured subset convention is first variate.")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.horizon != 16:
        raise ValueError("This gated smoke test is intentionally fixed to the requested 16-step horizon.")
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    n_variates = 7 if args.dataset == "ETTh1" else 1
    variates = list(range(n_variates))

    # This obtains the same train-set z-score and global ordinal ladder as the binary path.
    _train, _val, _test, norm_stats = load_dataset(
        args.dataset, variates, lookback=args.lookback, horizon=args.horizon,
        stride=1, test_stride=4, use_ordinal_window_norm=True,
    )
    ladder: OrdinalLadder = norm_stats["ordinal_ladder"]
    pool, _starts, _splits, _lens, _stats = load_tsf_pack_pool(
        args.dataset, variates, lookback=args.lookback, horizon=args.horizon,
        train_stride=1, test_stride=4, pack_splits=["test"],
    )
    past, future = pool[args.window_index]
    past = past.unsqueeze(0).to(device=device, dtype=torch.float32)
    future = future.unsqueeze(0).to(device=device, dtype=torch.float32)
    if not 0 <= args.variate < n_variates:
        raise ValueError(f"variate {args.variate} outside 0..{n_variates - 1}")

    # Causal OOD shift is identical to the ordinal discriminator filter; no target leaks into it.
    past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
        past, future, ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    assert future_ord is not None
    rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=torch.float32)
    target_all = _cdf_from_values(future_ord, rank_max, 256)
    coarse_all = _cdf_from_values(future_ord, rank_max, 16)
    past_all = _cdf_from_values(past_ord[..., -16:], rank_max, 256)
    vi = args.variate
    target = target_all[:, vi : vi + 1].repeat_interleave(16, dim=-1)
    coarse = coarse_all[:, vi : vi + 1]
    upscaled = F.interpolate(coarse, size=(256, 256), mode="nearest")
    past_canvas = past_all[:, vi : vi + 1].repeat_interleave(16, dim=-1)
    coarse_bins = TimeSeriesTo2D.bin_indices_from_cdf(coarse)[0, 0].long()
    inputs, cond, target_patches, coords = _patch_batch(upscaled[0], target[0], past_canvas[0], coarse_bins, 16)

    model = FactorizedDiT(
        in_channels=5, cond_channels=1, out_channels=1, image_height=16,
        patch_size=(8, 8), embed_dim=384, depth=8, num_heads=6, context_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    inputs, cond, target_patches = inputs.to(device), cond.to(device), target_patches.to(device)
    valid = inputs[:, 2:3]
    losses: list[float] = []
    for _step in range(args.steps):
        logits = model(inputs, torch.zeros(inputs.shape[0], device=device), cond)
        per_pixel = F.binary_cross_entropy_with_logits(logits, target_patches, reduction="none")
        loss = (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        prob = torch.sigmoid(model(inputs, torch.zeros(inputs.shape[0], device=device), cond))
        refined_patches, monotonic_violation = _project_monotone(prob)
        refined = upscaled.clone()
        for i, (row0, col0) in enumerate(coords):
            refined[..., row0 : row0 + 16, col0 : col0 + 16] = refined_patches[i : i + 1]

    selected_rank_max = rank_max[vi : vi + 1]
    ladder_one = _subset_ladder(ladder_b, vi)
    selected_shift = ood_shift[:, vi : vi + 1]
    gt_rank = future_ord[:, vi : vi + 1]
    naive_rank = _decode_ranks(upscaled, selected_rank_max)
    refined_rank = _decode_ranks(refined, selected_rank_max)
    _unused, gt_snapped = ordinal_decode(past_ord[:, vi : vi + 1], gt_rank, ladder_one, ood_shift=selected_shift)
    _unused, naive = ordinal_decode(past_ord[:, vi : vi + 1], naive_rank, ladder_one, ood_shift=selected_shift)
    _unused, refined_value = ordinal_decode(past_ord[:, vi : vi + 1], refined_rank, ladder_one, ood_shift=selected_shift)
    assert gt_snapped is not None and naive is not None and refined_value is not None
    metrics = {
        "initial_bce": losses[0], "final_bce": losses[-1], "steps": args.steps,
        "patch_monotonic_violation_pre_projection": monotonic_violation,
        "naive_rank_mae": float((naive_rank - gt_rank).abs().mean().item()),
        "refined_rank_mae": float((refined_rank - gt_rank).abs().mean().item()),
        "naive_snapped_z_mae": float((naive - gt_snapped).abs().mean().item()),
        "refined_snapped_z_mae": float((refined_value - gt_snapped).abs().mean().item()),
        "exact_snapped_bin_accuracy": float((refined_value == gt_snapped).float().mean().item()),
    }
    parameters = sum(p.numel() for p in model.parameters())
    meta = SmokeMetadata(args.dataset, vi, args.window_index, args.lookback, args.horizon, 16, 256, 16,
                         "train-split z-score; ordinal encode with causal OOD shift; no instance normalization",
                         "decode_with_ladder rounds all GT/naive/refined ranks to the same global ladder", str(device), parameters)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "past": past[0, vi].detach().cpu().numpy(), "future_raw": future[0, vi].detach().cpu().numpy(),
        "gt_snapped": gt_snapped[0, 0].detach().cpu().numpy(), "naive": naive[0, 0].detach().cpu().numpy(),
        "refined": refined_value[0, 0].detach().cpu().numpy(), "coarse_cdf_16": coarse[0, 0].detach().cpu().numpy(),
        "target_cdf_256": target[0, 0].detach().cpu().numpy(), "upscaled_coarse_256": upscaled[0, 0].detach().cpu().numpy(),
        "refined_cdf_256": refined[0, 0].detach().cpu().numpy(), "input_patches": inputs.detach().cpu().numpy(),
        "target_patches": target_patches.detach().cpu().numpy(), "refined_patches": refined_patches.detach().cpu().numpy(),
        "patch_coords": np.asarray(coords, dtype=np.int64), "losses": np.asarray(losses, dtype=np.float32),
    }
    np.savez_compressed(args.output / "smoke_arrays.npz", **arrays)
    (args.output / "metrics.json").write_text(json.dumps({"metadata": asdict(meta), "metrics": metrics}, indent=2), encoding="utf-8")
    figures = _plot(args.output, arrays, meta)
    print(json.dumps({"output": str(args.output), "metrics": metrics, "figures": [str(p) for p in figures]}, indent=2))


if __name__ == "__main__":
    main()
