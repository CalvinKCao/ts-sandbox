"""Staged-eval sample panels: 1d forecast, 2d coarse|fine, patch-box refine.

Render-only adapters over ``generate_staged_forecast`` outputs. Used by
``StagedEvalPhase`` (``viz_patch_boxes``) and ablation ``write_redbox_forecast_viz``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, Rectangle

MAX_UNBLENDED_PATCHES = 24


def upsample_coarse_to_canvas(coarse_2d: np.ndarray, canvas_h: int) -> np.ndarray:
    """``(V, Hc, W)`` → ``(V, canvas_h, W)`` nearest."""
    t = torch.from_numpy(np.asarray(coarse_2d, dtype=np.float32))[None]
    up = torch.nn.functional.interpolate(
        t, size=(int(canvas_h), int(coarse_2d.shape[-1])), mode="nearest",
    )
    return up[0].numpy()


def save_jpg(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format="jpg", bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)


def plot_1d(
    *,
    path: Path,
    past: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    n_vars: int,
    dpi: int,
    guidance: Optional[np.ndarray] = None,
) -> None:
    v_show = min(int(n_vars), past.shape[0])
    fig, axes = plt.subplots(v_show, 1, figsize=(11, 2.2 * v_show), sharex=True)
    if v_show == 1:
        axes = [axes]
    lb = past.shape[-1]
    x_p = np.arange(-lb, 0)
    x_f = np.arange(gt.shape[-1])
    for v, ax in enumerate(axes):
        ax.plot(x_p, past[v], color="0.55", lw=1.0, label="lookback")
        ax.plot(x_f, gt[v], color="black", lw=1.2, label="GT")
        ax.plot(x_f, pred[v], color="#1f77b4", lw=1.2, alpha=0.9, label="refine/pred")
        if guidance is not None:
            g = np.asarray(guidance)
            if g.ndim != 2 or g.shape[0] <= v:
                raise ValueError(
                    f"guidance must be (V,H) covering plotted variates; got {g.shape}"
                )
            if g.shape[-1] != gt.shape[-1]:
                raise ValueError(
                    f"guidance H={g.shape[-1]} != GT H={gt.shape[-1]}"
                )
            ax.plot(
                x_f,
                g[v],
                color="#FF9800",
                lw=1.3,
                linestyle="--",
                alpha=0.95,
                label="guidance",
            )
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_ylabel(f"v{v}")
        ax.grid(alpha=0.15)
        if v == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=4)
            ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("t (horizon starts at 0)")
    fig.tight_layout()
    save_jpg(fig, path, dpi)


def plot_2d_coarse_fine(
    *,
    path: Path,
    coarse: np.ndarray,
    second: np.ndarray,
    second_name: str,
    title: str,
    n_vars: int,
    dpi: int,
) -> None:
    v_show = min(int(n_vars), coarse.shape[0])
    fig, axes = plt.subplots(v_show, 2, figsize=(10, 2.0 * v_show), sharex=False)
    if v_show == 1:
        axes = np.asarray([axes])
    for v in range(v_show):
        for col, (arr, name) in enumerate(((coarse, "coarse"), (second, second_name))):
            ax = axes[v, col]
            h, w = arr[v].shape[-2], arr[v].shape[-1]
            ax.imshow(
                arr[v], aspect="auto", origin="lower", extent=[0, w, 0, h],
                cmap="plasma", vmin=0.0, vmax=1.0,
            )
            ax.set_title(f"v{v} {name} ({h}x{w})", fontsize=9)
            ax.set_ylabel("row")
            if v == v_show - 1:
                ax.set_xlabel("t")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_jpg(fig, path, dpi)


def add_patch_boxes(ax, locs, patch_h: int, patch_w: int) -> None:
    for loc in locs:
        ax.add_patch(
            Rectangle(
                (float(loc.col0), float(loc.row0)),
                float(patch_w),
                float(patch_h),
                fill=False,
                edgecolor="red",
                linewidth=0.8,
            )
        )


def plot_refine_boxes(
    *,
    path: Path,
    coarse_up_v: np.ndarray,
    blended_v: np.ndarray,
    patch_cdf: torch.Tensor,
    locations: Sequence[Any],
    variate: int,
    patch_h: int,
    patch_w: int,
    title: str,
    dpi: int,
    max_patches: int = MAX_UNBLENDED_PATCHES,
) -> None:
    """3-row panel: upsampled coarse | unblended grid | blended (+ red boxes)."""
    locs_v = [
        loc for loc in locations
        if int(loc.batch_index) == 0 and int(loc.variate_index) == int(variate)
    ]
    pair_idx = [
        i for i, loc in enumerate(locations)
        if int(loc.batch_index) == 0 and int(loc.variate_index) == int(variate)
    ]
    if len(pair_idx) > max_patches:
        pair_idx = pair_idx[:max_patches]

    n = len(pair_idx)
    ncols = min(8, max(n, 1))
    nrows = int(math.ceil(max(n, 1) / ncols))

    fig = plt.figure(figsize=(12, 3.2 + 1.35 * nrows + 3.2))
    gs = fig.add_gridspec(
        2 + nrows, ncols,
        height_ratios=[3.0] + [1.2] * nrows + [3.0],
        hspace=0.55,
        wspace=0.2,
    )

    ax_a = fig.add_subplot(gs[0, :])
    h, w = coarse_up_v.shape[-2], coarse_up_v.shape[-1]
    ax_a.imshow(
        coarse_up_v, aspect="auto", origin="lower", extent=[0, w, 0, h],
        cmap="plasma", vmin=0.0, vmax=1.0,
    )
    add_patch_boxes(ax_a, locs_v, patch_h, patch_w)
    ax_a.set_title(
        f"(A) coarse CDF nearest-up -> {h}x{w}  |  {len(locs_v)} boxes  v{variate}",
        fontsize=10,
    )
    ax_a.set_ylabel("row")
    ax_a.set_xlabel("t")

    ax_b_list: List[Tuple[Any, Any, int, int]] = []
    if n == 0:
        ax_b = fig.add_subplot(gs[1, :])
        ax_b.set_axis_off()
        ax_b.text(0.5, 0.5, "no patches for this variate", ha="center", va="center")
        ax_b.set_title("(B) unblended refine patches", fontsize=10)
    else:
        first_ax = None
        for j, pi in enumerate(pair_idx):
            r, c = divmod(j, ncols)
            ax = fig.add_subplot(gs[1 + r, c])
            if first_ax is None:
                first_ax = ax
            patch = patch_cdf[pi, 0].detach().cpu().numpy().astype(np.float32)
            ph, pw = patch.shape[-2], patch.shape[-1]
            ax.imshow(
                patch, aspect="auto", origin="lower", extent=[0, pw, 0, ph],
                cmap="plasma", vmin=0.0, vmax=1.0,
            )
            loc = locations[pi]
            ax.set_title(f"r{int(loc.row0)},c{int(loc.col0)}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax_b_list.append((loc, ax, ph, pw))
        if first_ax is not None:
            first_ax.set_ylabel(f"(B) unblended n={n}/{max_patches}", fontsize=9)

    ax_c = fig.add_subplot(gs[1 + nrows, :])
    hb, wb = blended_v.shape[-2], blended_v.shape[-1]
    ax_c.imshow(
        blended_v, aspect="auto", origin="lower", extent=[0, wb, 0, hb],
        cmap="plasma", vmin=0.0, vmax=1.0,
    )
    add_patch_boxes(ax_c, locs_v, patch_h, patch_w)
    ax_c.set_title(f"(C) blended future_2d_fine v{variate} ({hb}x{wb})", fontsize=10)
    ax_c.set_ylabel("row")
    ax_c.set_xlabel("t")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for loc, ax_b_j, ph, pw in ax_b_list:
        xy_a = (float(loc.col0) + float(patch_w) / 2.0, float(loc.row0) + float(patch_h) / 2.0)
        xy_b = (float(pw) / 2.0, float(ph) / 2.0)
        con = ConnectionPatch(
            xyA=xy_a,
            xyB=xy_b,
            coordsA=ax_a.transData,
            coordsB=ax_b_j.transData,
            arrowstyle="->",
            color="red",
            linewidth=0.7,
            alpha=0.75,
            mutation_scale=10,
        )
        fig.add_artist(con)
    save_jpg(fig, path, dpi)


def pick_indices(
    n_pool: int,
    n_samples: int,
    seed: int,
    explicit: Optional[Sequence[int]] = None,
) -> List[int]:
    if explicit is not None:
        picks = [int(i) for i in explicit]
        for i in picks:
            if i < 0 or i >= n_pool:
                raise ValueError(f"pool index {i} out of range n={n_pool}")
        return picks
    if n_pool < 1:
        raise RuntimeError("empty pool")
    k = min(int(n_samples), n_pool)
    rng = np.random.default_rng(int(seed))
    return sorted(int(x) for x in rng.choice(n_pool, size=k, replace=False))


@torch.no_grad()
def write_staged_sample_panels(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    kind: str,
    coarse_model: Any,
    fine_model: Any,
    pool: Sequence[Any],
    picks: Sequence[int],
    device: torch.device,
    sampler: str = "quad_t",
    num_sampling_steps: int = 20,
    seed: int = 42,
    variables_to_plot: int = 0,
    jpeg_dpi: int = 120,
) -> List[Path]:
    """Generate staged forecasts for ``picks`` and write 1d / 2d / refine_boxes.

    Walkthrough (hook site for staged_eval viz_patch_boxes):
      pool[i] → generate_staged_forecast → prediction_global_norm + 2d maps
      (+ patch_cdf_unblended / patch_locations when kind=patch_refine).
    """
    from utils.staged_binary_forecast import generate_staged_forecast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    second_name = "blended refine" if kind == "patch_refine" else "fine"
    written: List[Path] = []

    canvas_h = int(getattr(fine_model.config, "patch_refine_canvas_height", 256) or 256)
    patch_h = int(getattr(fine_model.config, "patch_refine_patch_height", 16) or 16)
    patch_w = int(getattr(fine_model.config, "patch_refine_patch_width", 8) or 8)
    n_vars_run = int(getattr(coarse_model.config, "n_variates", 0) or 0)
    if n_vars_run <= 0:
        # Fall back from first pool sample.
        past0, _ = pool[int(picks[0])]
        n_vars_run = int(past0.shape[0])
    n_plot = int(variables_to_plot)
    if n_plot <= 0:
        n_plot = n_vars_run
    else:
        n_plot = min(n_plot, n_vars_run)

    for s_i, pool_i in enumerate(picks):
        past_t, future_t = pool[int(pool_i)]
        past = past_t.unsqueeze(0).to(device)
        future = future_t.unsqueeze(0).to(device)
        torch.manual_seed(int(seed) + int(pool_i))
        result = generate_staged_forecast(
            coarse_model,
            fine_model,
            past,
            vertical_dual=False,
            sampler=sampler,
            num_inference_steps=int(num_sampling_steps),
        )
        overlap = int(getattr(fine_model.config, "lookback_overlap", 0) or 0)
        gt = future[0, :, overlap:] if overlap else future[0]
        pred = result["prediction_global_norm"][0]
        if pred.shape != gt.shape:
            h = min(int(pred.shape[-1]), int(gt.shape[-1]))
            pred = pred[..., -h:]
            gt = gt[..., -h:]

        past_np = past[0].detach().cpu().numpy().astype(np.float32)
        gt_np = gt.detach().cpu().numpy().astype(np.float32)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        if "guidance_prediction_global_norm" not in result:
            raise RuntimeError(
                f"{run_name}: missing guidance_prediction_global_norm from "
                "generate_staged_forecast (need patch-guidance overlay on redbox 1d)"
            )
        guide_t = result["guidance_prediction_global_norm"][0]
        if guide_t.shape != pred.shape:
            h = min(int(guide_t.shape[-1]), int(pred.shape[-1]))
            guide_t = guide_t[..., -h:]
        guide_np = guide_t.detach().cpu().numpy().astype(np.float32)
        coarse_2d = result["future_2d_coarse"][0].detach().cpu().numpy().astype(np.float32)
        second_2d = result["future_2d_fine"][0].detach().cpu().numpy().astype(np.float32)

        p1 = out_dir / f"sample{s_i:02d}_pool{pool_i}_1d.jpg"
        plot_1d(
            path=p1,
            past=past_np,
            gt=gt_np,
            pred=pred_np,
            guidance=guide_np,
            title=(
                f"{run_name}/{dataset} pool={pool_i} kind={kind} sampler={sampler} "
                f"(GT + refine + guidance)"
            ),
            n_vars=n_plot,
            dpi=jpeg_dpi,
        )
        p2 = out_dir / f"sample{s_i:02d}_pool{pool_i}_2d_coarse_fine.jpg"
        plot_2d_coarse_fine(
            path=p2,
            coarse=coarse_2d,
            second=second_2d,
            second_name=second_name,
            title=f"{run_name}/{dataset} pool={pool_i} coarse | {second_name}",
            n_vars=n_plot,
            dpi=jpeg_dpi,
        )
        written.extend([p1, p2])

        if kind == "patch_refine":
            if "patch_cdf_unblended" not in result or "patch_locations" not in result:
                raise RuntimeError(
                    f"{run_name}: patch_refine missing patch_cdf_unblended/patch_locations"
                )
            patch_cdf = result["patch_cdf_unblended"]
            locations = result["patch_locations"]
            coarse_up = upsample_coarse_to_canvas(coarse_2d, canvas_h)
            v_show = min(n_plot, coarse_2d.shape[0])
            for v in range(v_show):
                pv = out_dir / f"sample{s_i:02d}_pool{pool_i}_v{v}_refine_boxes.jpg"
                plot_refine_boxes(
                    path=pv,
                    coarse_up_v=coarse_up[v],
                    blended_v=second_2d[v],
                    patch_cdf=patch_cdf,
                    locations=locations,
                    variate=v,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    title=(
                        f"{run_name}/{dataset} pool={pool_i} v{v} "
                        f"patch_h={patch_h} patch_w={patch_w}"
                    ),
                    dpi=jpeg_dpi,
                )
                written.append(pv)

    (out_dir / "manifest.txt").write_text(
        "\n".join(
            [
                f"run={run_name}",
                f"kind={kind}",
                f"dataset={dataset}",
                f"sampler={sampler}",
                f"steps={num_sampling_steps}",
                f"picks={list(picks)}",
                *[str(p) for p in written],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return written
