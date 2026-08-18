#!/usr/bin/env python
"""Fast end-to-end smoke for patch_refine (geometry + 1 train step + generate)."""

from __future__ import annotations

import time

import torch

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.patch_refine_geometry import (
    PatchLayout,
    blend_patch_bins,
    blend_patch_bins_layout,
    coarse_edges_from_cdf,
    patch_layout_for_fixed_col0,
    primary_stride_col0s,
    select_patch_locations,
    subsample_unique_seg_layout,
)
from models.diffusion_tsf.patch_refine_segments import (
    coverage_gap_layout,
    select_coverage_gap_locations,
    select_primary_ar_locations,
)
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D


def _test_geometry() -> None:
    coarse = torch.zeros(1, 1, 16, 16)
    # Transition near mid-height for every column.
    coarse[:, :, :8, :] = 1.0
    edges = coarse_edges_from_cdf(coarse, canvas_height=256)
    locs = select_patch_locations(
        edges,
        canvas_height=256,
        patch_height=32,
        patch_width=8,
        col_stride=6,
    )
    assert locs, "expected at least one crop"
    covered = set()
    for loc in locs:
        for t in range(loc.col0, loc.col0 + 8):
            if loc.row0 <= int(edges[0, 0, t]) < loc.row0 + 32:
                covered.add(t)
    assert covered == set(range(16)), f"incomplete coverage: {sorted(covered)}"

    # Average bins 1 and 3 -> absolute mid with shared row0.
    patches = torch.zeros(2, 1, 32, 8)
    patches[0, 0, :2, :] = 1.0  # local bin 1
    patches[1, 0, :4, :] = 1.0  # local bin 3
    Fake = type(locs[0])
    fake_locs = [
        Fake(flat_index=0, batch_index=0, variate_index=0, row0=10, col0=0),
        Fake(flat_index=0, batch_index=0, variate_index=0, row0=10, col0=0),
    ]
    edge8 = torch.full((1, 1, 8), 16, dtype=torch.long)  # inside [10, 42)
    hard, counts = blend_patch_bins(
        patches,
        fake_locs,
        edge8,
        canvas_height=256,
        patch_height=32,
        patch_width=8,
    )
    bins = TimeSeriesTo2D.bin_indices_from_cdf(hard)
    assert int(bins[0, 0, 0].item()) == 12, bins[0, 0, 0]  # 10 + round((1+3)/2)
    assert float(counts.min()) >= 1.0
    print("geometry ok")


def _loc_key(loc) -> tuple:
    return (loc.batch_index, loc.variate_index, loc.row0, loc.col0)


def _test_layout_equivalence() -> None:
    torch.manual_seed(0)
    coarse = torch.zeros(2, 3, 16, 24)
    # Varied boundary heights so some crops miss and need gap fills.
    for b in range(2):
        for v in range(3):
            row = 4 + (b + 2 * v) % 10
            coarse[b, v, :row, :] = 1.0
            coarse[b, v, :, 10 + v] = 0.0
            coarse[b, v, :15, 10 + v] = 1.0
    edges = coarse_edges_from_cdf(coarse, canvas_height=256)
    canvas_h, patch_h, patch_w, stride = 256, 32, 8, 6
    locs = select_patch_locations(
        edges,
        canvas_height=canvas_h,
        patch_height=patch_h,
        patch_width=patch_w,
        col_stride=stride,
    )
    rng = torch.Generator().manual_seed(1)
    patches = torch.rand(len(locs), 1, patch_h, patch_w, generator=rng)
    patches = (patches > 0.4).float()
    # Force a couple fully empty / full columns so abstain matches the old path.
    patches[0, 0, :, 0] = 0
    patches[1, 0, :, 1] = 1
    hard_list, counts_list = blend_patch_bins(
        patches, locs, edges,
        canvas_height=canvas_h, patch_height=patch_h, patch_width=patch_w,
    )
    layout = PatchLayout.from_locations(locs, device=patches.device)
    hard_t, counts_t = blend_patch_bins_layout(
        patches, layout, edges,
        canvas_height=canvas_h, patch_height=patch_h, patch_width=patch_w,
    )
    assert torch.equal(hard_list, hard_t), "blend layout mismatch"
    assert torch.equal(counts_list, counts_t), "blend vote-count mismatch"

    primary = select_primary_ar_locations(
        edges,
        canvas_height=canvas_h,
        patch_height=patch_h,
        patch_width=patch_w,
        col_stride=stride,
    )
    primary_keys = {_loc_key(loc) for loc in primary}
    layout_keys = set()
    layouts = []
    for col0 in primary_stride_col0s(int(edges.shape[-1]), patch_w, stride):
        step = patch_layout_for_fixed_col0(
            edges,
            torch.full((edges.shape[0],), col0, dtype=torch.long),
            canvas_height=canvas_h,
            patch_height=patch_h,
            patch_width=patch_w,
        )
        layouts.append(step)
        layout_keys.update(_loc_key(loc) for loc in step.to_locations())
    assert primary_keys == layout_keys, "primary AR layout key mismatch"

    gap_list = select_coverage_gap_locations(
        edges, primary,
        canvas_height=canvas_h, patch_height=patch_h, patch_width=patch_w,
    )
    gap_layout = coverage_gap_layout(
        edges, PatchLayout.cat(layouts),
        canvas_height=canvas_h, patch_height=patch_h, patch_width=patch_w,
    )
    gap_list_keys = {_loc_key(loc) for loc in gap_list}
    gap_t_keys = set() if gap_layout is None else {
        _loc_key(loc) for loc in gap_layout.to_locations()
    }
    assert gap_list_keys == gap_t_keys, f"gap keys {gap_list_keys} vs {gap_t_keys}"
    print("layout equivalence ok")


def _test_train_generate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DiffusionTSFConfig(
        num_variables=1,
        lookback_length=336,
        forecast_length=104,  # 96 + 8 overlap
        dataset_forecast_length=96,
        lookback_overlap=8,
        diffusion_lookback_cap=336,
        image_height=32,
        coarse_image_height=16,
        fine_image_height=16,
        patch_refine_canvas_height=256,
        patch_refine_patch_height=32,
        patch_refine_patch_width=8,
        patch_refine_col_stride=6,
        diffusion_stage="patch_refine",
        dit_patch_size=(4, 4),
        dit_cond_patch_size=(8, 8),
        dit_embed_dim=64,
        dit_depth=2,
        dit_num_heads=2,
        disable_cross_attention=True,
        use_coordinate_channel=True,
        use_deterministic_anchor_loss=True,
        deterministic_anchor_lambda=0.99,
        prediction_target="x0",
        loss_weighting="min_snr",
        min_snr_gamma=2.0,
        binary_noise_schedule="linear",
        use_window_normalization=True,
        use_variate_embedding=False,
        past_cond_resize_to_horizon=False,
        patch_refine_unique_segments=True,
    )
    model = DiffusionTSF(cfg).to(device)
    model.train()
    past = torch.randn(1, 1, 336, device=device)
    future = torch.randn(1, 1, 104, device=device)
    t0 = time.time()
    out = model.forward(past, future)
    loss = out["loss"]
    loss.backward()
    print(
        f"train_step ok loss={float(loss.detach()):.4f} "
        f"n_patches={float(out['n_patches'])} {time.time()-t0:.1f}s"
    )

    model.eval()
    coarse = torch.zeros(1, 1, 16, 104, device=device)
    coarse[:, :, :8, :] = 1.0
    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(
            past,
            sampler="anchor",
            future_coarse_2d=coarse,
        )
    assert gen["prediction"].shape[-1] == 96, gen["prediction"].shape
    print(f"generate ok pred={tuple(gen['prediction'].shape)} {time.time()-t1:.1f}s")

    model.config.patch_refine_unique_segments = False
    with torch.no_grad():
        gen_dense = model.generate(
            past,
            sampler="anchor",
            future_coarse_2d=coarse,
        )
    assert gen_dense["prediction"].shape[-1] == 96, gen_dense["prediction"].shape
    print(f"generate dense ok n_patches={len(gen_dense['patch_locations'])}")


def _test_patch_fraction() -> None:
    torch.manual_seed(0)
    coarse = torch.zeros(2, 7, 16, 24)
    coarse[:, :, :8, :] = 1.0
    edges = coarse_edges_from_cdf(coarse, canvas_height=128)
    col0 = torch.zeros(2, dtype=torch.long)
    layout = patch_layout_for_fixed_col0(
        edges, col0, canvas_height=128, patch_height=64, patch_width=6,
    )
    assert layout.n_patches == 14
    kept = subsample_unique_seg_layout(
        layout, 0.5, unique_segments=True, training=True,
    )
    assert kept.n_patches == 8, kept.n_patches  # even variates 0,2,4,6 × B=2
    assert set(kept.variate_index.tolist()) == {0, 2, 4, 6}
    eval_kept = subsample_unique_seg_layout(
        layout, 0.5, unique_segments=True, training=False,
    )
    assert eval_kept.n_patches == 14
    try:
        subsample_unique_seg_layout(layout, 0.5, unique_segments=False, training=True)
    except ValueError as exc:
        assert "unique_segments" in str(exc)
    else:
        raise AssertionError("expected fail-fast without unique_segments")
    print("patch fraction ok")


if __name__ == "__main__":
    _test_geometry()
    _test_layout_equivalence()
    _test_patch_fraction()
    _test_train_generate()
    print("patch_refine smoke passed")
