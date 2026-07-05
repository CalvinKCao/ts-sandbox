"""Pipeline visualization helpers (compressed JPEG, wandb-friendly)."""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.diffusion_tsf.pipeline.config import visualization_settings

logger = logging.getLogger(__name__)


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize (C, T) tensor using (C,) or (1, C) stats."""
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def pick_sample_indices(n_dataset: int, n_samples: int, seed: int = 42) -> List[int]:
    if n_dataset <= 0:
        return []
    n = min(n_samples, n_dataset)
    if n >= n_dataset:
        return list(range(n_dataset))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_dataset), n))


def save_figure_jpg(fig: plt.Figure, path: str, dpi: int = 100) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    return path


def _format_scale_banner(
    *,
    raw_range: Optional[Tuple[float, float]] = None,
    norm_range: Optional[Tuple[float, float]] = None,
    space_label: str = "",
    extra: str = "",
) -> str:
    parts = []
    if space_label:
        parts.append(space_label)
    if raw_range is not None:
        parts.append(f"raw [{raw_range[0]:.3f}, {raw_range[1]:.3f}]")
    if norm_range is not None:
        parts.append(f"current [{norm_range[0]:.3f}, {norm_range[1]:.3f}]")
    if extra:
        parts.append(extra)
    return " | ".join(parts) if parts else ""


def _tensor_value_range(t: torch.Tensor) -> Tuple[float, float]:
    arr = t.detach().cpu().float().numpy()
    return float(arr.min()), float(arr.max())


def _viz_cfg(state: Any) -> Dict[str, Any]:
    cfg = getattr(state, "merged_config", None) or {}
    if "visualization" not in cfg:
        return {"enabled": True, "n_samples": 3, "n_dual_scale_vars": 3, "jpeg_dpi": 100}
    return visualization_settings(cfg)


def generate_itrans_prediction_viz(
    itrans_model: torch.nn.Module,
    dataset,
    stats: Tuple[Any, Any],
    device: torch.device,
    output_dir: str,
    tag: str,
    *,
    n_samples: int = 3,
    forecast_length: int = 96,
    lookback_length: int = 96,
    seed: int = 42,
    jpeg_dpi: int = 100,
) -> List[str]:
    """1D forecast plots for iTransformer at a pipeline checkpoint."""
    mean, std = stats
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)
    os.makedirs(output_dir, exist_ok=True)

    indices = pick_sample_indices(len(dataset), n_samples, seed=seed)
    saved: List[str] = []
    pred_len = int(getattr(itrans_model, "pred_len", forecast_length))
    plot_horizon = min(int(forecast_length), pred_len)

    for row, idx in enumerate(indices):
        past, future = dataset[idx]
        past_t = past.unsqueeze(0).to(device)
        B, C, L = past_t.shape
        x_enc = past_t.permute(0, 2, 1)
        seq_sl = getattr(itrans_model, "seq_len", L)
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        x_dec = torch.zeros(B, pred_len, C, device=device)

        with torch.no_grad():
            out = itrans_model(x_enc, None, x_dec, None)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.permute(0, 2, 1).cpu()[0]

        past_dn = denorm(past, mean_t, std_t)
        future_sliced = future[:, -plot_horizon:]
        future_dn = denorm(future_sliced, mean_t, std_t)
        pred_dn = denorm(pred, mean_t, std_t)

        n_vars_plot = min(3, C)
        fig, axes = plt.subplots(
            1, n_vars_plot, figsize=(4.5 * n_vars_plot, 3.0), squeeze=False, constrained_layout=True
        )
        t_past = np.arange(-lookback_length, 0)
        t_future = np.arange(0, plot_horizon)

        for col in range(n_vars_plot):
            ax = axes[0, col]
            gt = future_dn[col].numpy()
            pr = pred_dn[col].numpy()
            ax.plot(t_past, past_dn[col, -lookback_length:].numpy(), color="#9E9E9E", alpha=0.5, linewidth=1.0)
            ax.plot(t_future, gt, color="#2196F3", linewidth=1.5, label="GT" if col == 0 else "")
            ax.plot(t_future, pr, color="#FF9800", linewidth=1.2, linestyle="--", label="iTrans" if col == 0 else "")
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
            mae = float(np.mean(np.abs(pr - gt)))
            ax.set_title(f"Var {col} | MAE {mae:.3f}", fontsize=9)

        fig.suptitle(f"iTransformer {tag} | sample {idx}", fontsize=11, fontweight="semibold")
        if n_vars_plot:
            axes[0, 0].legend(loc="upper left", fontsize=7)

        path = os.path.join(output_dir, f"itrans_{tag}_sample{row:02d}_idx{idx}.jpg")
        saved.append(save_figure_jpg(fig, path, dpi=jpeg_dpi))

    return saved


def _load_staged_diffusion_from_ckpt(
    *,
    ckpt_path: str,
    stage: str,
    itrans_ckpt_path: str,
    n_vars: int,
    device: torch.device,
    tuned_params: Optional[Dict[str, Any]] = None,
    guidance_type: Optional[str] = None,
):
    from models.diffusion_tsf.train_multivariate_pipeline import (
        create_diffusion_model,
        load_wrapped_guidance,
        load_diffusion_state_keep_attached_guidance,
    )
    from models.diffusion_tsf.visualize_comparison import (
        apply_checkpoint_architecture,
        infer_anchor_kwargs,
        infer_diffusion_type,
        infer_model_type,
    )

    guidance = load_wrapped_guidance(
        str(itrans_ckpt_path),
        n_vars,
        device,
        guidance_type=guidance_type,
    )
    diff_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = diff_ckpt.get("config") or tuned_params or {}
    if isinstance(meta, dict) and "diffusion_params" in meta:
        meta = {**meta, **(meta.get("diffusion_params") or {})}
    diff_type = infer_diffusion_type(diff_ckpt, meta.get("diffusion_type"))
    backbone = infer_model_type(diff_ckpt)
    apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta if isinstance(meta, dict) else {})

    model = create_diffusion_model(
        n_variates=n_vars,
        diffusion_type=diff_type,
        model_type=backbone,
        diffusion_stage=stage,
        guidance_model=guidance,
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(model, diff_ckpt["model_state_dict"])
    model.eval()
    return model, diff_ckpt


def _as_bv1hw_cdf(cdf_map: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Reshape staged CDF maps to (B*V, 1, H, W) for occupancy decode."""
    if cdf_map.dim() == 3:
        cdf_map = cdf_map.unsqueeze(0)
    if cdf_map.dim() != 4:
        raise ValueError(f"expected 3D/4D CDF map, got shape {tuple(cdf_map.shape)}")
    b, v, h, w = cdf_map.shape
    return cdf_map.reshape(b * v, 1, h, w), b, v


def _decode_staged_cdf_1d(
    cdf_map: torch.Tensor,
    *,
    value_range: float,
    to_2d,
    cdf_decoder: str = "mean",
) -> torch.Tensor:
    """Decode (B,V,H,W) or (V,H,W) CDF maps to (B,V,W) normalized 1D values."""
    flat, b, v = _as_bv1hw_cdf(cdf_map)
    decoded = to_2d._decode_occupancy_in_range(
        flat,
        value_range=float(value_range),
        cdf_decoder=cdf_decoder,
    )
    return decoded.reshape(b, v, -1)


def _future_core_slice(
    future_norm: torch.Tensor,
    *,
    width: int,
    lookback_overlap: int,
) -> torch.Tensor:
    """Align window-norm future GT with staged map width (overlap prefix + core)."""
    k = int(lookback_overlap)
    total = int(future_norm.shape[-1])
    if total == width:
        return future_norm[..., k:] if k > 0 and width > k else future_norm
    if total == width + k and k > 0:
        return future_norm[..., k : k + width]
    if total >= width:
        return future_norm[..., -width:]
    raise ValueError(
        f"cannot align future_norm width {total} to map width {width} with overlap {k}"
    )


def _staged_fine_value_range(diff_model) -> float:
    cfg = getattr(diff_model, "config", None)
    if getattr(cfg, "staged_representation", "value_precision") == "haar_frequency":
        return float(getattr(cfg, "haar_fine_max_scale", 0.0) or diff_model.to_2d.max_scale)
    if getattr(cfg, "staged_representation", "value_precision") == "fourier_frequency":
        per_var = getattr(cfg, "fourier_fine_max_scale_per_variate", None)
        if per_var:
            return float(max(per_var))
        fine_scale = float(getattr(cfg, "fourier_fine_max_scale", 0.0) or 0.0)
        if fine_scale > 0.0:
            return fine_scale
        coarse_h = int(getattr(cfg, "coarse_image_height", diff_model.to_2d.height))
        return 2.0 * float(cfg.max_scale) / float(coarse_h)
    return float(diff_model.to_2d.max_scale) / float(diff_model.to_2d.height)


def _window_stats_for_past(diff_model, past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-window center/std for dataset-global-z past (V, T)."""
    _, _, (center, wstd) = diff_model._normalize_sequence(past.unsqueeze(0), None)
    return center.squeeze(0), wstd.squeeze(0)


def _staged_window_norm_to_raw(
    x: torch.Tensor,
    *,
    win_center: torch.Tensor,
    win_std: torch.Tensor,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
) -> torch.Tensor:
    """Staged 1D decode lives in window-norm space; map to dataset raw scale."""
    from models.diffusion_tsf.visualize_comparison import denorm as denorm_cmp

    global_z = x * win_std + win_center
    return denorm_cmp(global_z, global_mean, global_std)


def _staged_fine_residual_to_raw(
    fine: torch.Tensor,
    *,
    win_std: torch.Tensor,
    global_std: torch.Tensor,
) -> torch.Tensor:
    """Fine residual is additive in window-norm space (no center shift)."""
    return fine * win_std * global_std


def _plot_dual_scale_sample(
    *,
    res: Dict[str, torch.Tensor],
    diff_model,
    past: torch.Tensor,
    future: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    dataset_name: str,
    sample_index: int,
    output_dir: str,
    tag: str,
    variables_to_plot: int,
    applied_h: int,
    jpeg_dpi: int,
) -> str:
    from models.diffusion_tsf.visualize_comparison import denorm as denorm_cmp

    past_coarse = res["past_2d_coarse"].cpu()
    past_fine = res["past_2d_fine"].cpu()
    future_coarse = res["future_2d_coarse"].cpu()
    future_fine = res["future_2d_fine"].cpu()

    coarse_map_full = torch.cat([past_coarse, future_coarse], dim=-1)
    fine_map_full = torch.cat([past_fine, future_fine], dim=-1)

    to_2d = diff_model.to_2d
    coarse_1d = to_2d._decode_occupancy_in_range(
        coarse_map_full, value_range=to_2d.max_scale, cdf_decoder="mean"
    ).cpu()
    fine_1d = to_2d._decode_occupancy_in_range(
        fine_map_full, value_range=_staged_fine_value_range(diff_model), cdf_decoder="mean"
    ).cpu()
    combined_1d = coarse_1d + fine_1d

    W_past = past_coarse.shape[-1]
    W_fut = future_coarse.shape[-1]
    t_axis = np.arange(-W_past, W_fut)
    gt_past = past[:, -W_past:]
    gt_future = future[:, -W_fut:]
    gt_aligned = torch.cat([gt_past, gt_future], dim=-1)

    win_center, win_std = _window_stats_for_past(diff_model, past)
    gt_full_dn = denorm_cmp(gt_aligned, mean, std)
    coarse_dn = _staged_window_norm_to_raw(
        coarse_1d[0], win_center=win_center, win_std=win_std,
        global_mean=mean, global_std=std,
    )
    fine_dn = _staged_fine_residual_to_raw(fine_1d[0], win_std=win_std, global_std=std)
    combined_dn = _staged_window_norm_to_raw(
        combined_1d[0], win_center=win_center, win_std=win_std,
        global_mean=mean, global_std=std,
    )

    n_vars = past.shape[0]
    n_vars_to_plot = min(variables_to_plot, n_vars)
    fig, axes = plt.subplots(
        4, n_vars_to_plot,
        figsize=(4.0 * n_vars_to_plot, 8.5),
        sharex="row",
        constrained_layout=True,
    )
    if n_vars_to_plot == 1:
        axes = axes.reshape(4, 1)

    for col in range(n_vars_to_plot):
        ax1 = axes[0, col]
        ax1.plot(t_axis, gt_full_dn[col].numpy(), color="#2196F3", linewidth=1.6, label="GT (dataset scale)")
        ax1.plot(
            t_axis, coarse_dn[col].numpy(), color="#FF9800", linewidth=1.2,
            drawstyle="steps-mid", alpha=0.85, label="Coarse (2D cols)",
        )
        ax1.plot(t_axis, combined_dn[col].numpy(), color="#E91E63", linewidth=1.2, label="Combined (2D cols)")
        ax1.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        ax1.grid(True, alpha=0.12)
        ax1.set_title(f"Var {col} 1D", fontsize=9)
        if col == 0:
            ax1.legend(loc="lower left", fontsize=6)
            ax1.set_ylabel("denorm value")

        ax2 = axes[1, col]
        ax2.plot(t_axis, fine_dn[col].numpy(), color="#4CAF50", linewidth=1.2)
        ax2.axhline(y=0, color="grey", linestyle="--", alpha=0.4)
        ax2.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        ax2.grid(True, alpha=0.12)
        ax2.set_title("Fine residual (2D cols)", fontsize=9)

        ax3 = axes[2, col]
        im3 = ax3.imshow(
            coarse_map_full[0, col].numpy(), aspect="auto", origin="lower",
            extent=[-W_past, W_fut, 0, applied_h], cmap="plasma",
        )
        ax3.axvline(x=0, color="white", linestyle="--", alpha=0.5)
        ax3.set_title("Coarse 2D CDF map", fontsize=9)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = axes[3, col]
        im4 = ax4.imshow(
            fine_map_full[0, col].numpy(), aspect="auto", origin="lower",
            extent=[-W_past, W_fut, 0, applied_h], cmap="plasma",
        )
        ax4.axvline(x=0, color="white", linestyle="--", alpha=0.5)
        ax4.set_title("Fine 2D CDF map", fontsize=9)
        xlabel = (
            "2D CDF column index (past cols < 0, forecast cols >= 0; "
            "NOT raw series timestep — negative region is lookback conditioning width)"
        )
        ax4.set_xlabel(xlabel, fontsize=7)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    scale_note = _format_scale_banner(
        raw_range=_tensor_value_range(gt_full_dn),
        norm_range=_tensor_value_range(gt_aligned),
        space_label="1D rows: dataset-denorm (model via window-norm inverse)",
        extra="2D maps: occupancy in [0,1] after max_scale binning",
    )
    fig.suptitle(
        f"{dataset_name} sample {sample_index} | {tag}\n{scale_note}",
        fontsize=10, fontweight="bold",
    )
    out_path = os.path.join(
        output_dir, f"{tag}_{dataset_name}_sample{sample_index}.jpg"
    )
    return save_figure_jpg(fig, out_path, dpi=jpeg_dpi)


def generate_staged_dual_scale_comparisons(
    *,
    coarse_ckpt_path: str,
    fine_ckpt_path: str,
    itrans_ckpt_path: str,
    dataset_name: str,
    variate_indices: Sequence[int],
    output_dir: str,
    device: torch.device,
    tuned_params: Optional[Dict[str, Any]] = None,
    lookback_length: int = 96,
    forecast_length: int = 96,
    diffusion_sampler: str = "anchor",
    num_inference_steps: int = 20,
    variables_to_plot: int = 3,
    sample_indices: Optional[Sequence[int]] = None,
    n_samples: int = 3,
    random_seed: int = 42,
    jpeg_dpi: int = 100,
    tag: str = "staged_dual_scale",
) -> List[str]:
    """Dual-scale viz for separate coarse/fine staged checkpoints (chains generation)."""
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
    from models.diffusion_tsf.visualize_comparison import apply_checkpoint_architecture, infer_diffusion_type

    n_vars = len(variate_indices)
    _, _, test_ds, norm_stats = load_dataset(
        dataset_name, list(variate_indices), stride=1,
        lookback=lookback_length, horizon=forecast_length,
    )
    if len(test_ds) == 0:
        raise ValueError(f"No test samples for {dataset_name}")

    if sample_indices is None:
        sample_indices = pick_sample_indices(len(test_ds), n_samples, seed=random_seed)

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    coarse_model, coarse_ckpt = _load_staged_diffusion_from_ckpt(
        ckpt_path=coarse_ckpt_path,
        stage="coarse",
        itrans_ckpt_path=itrans_ckpt_path,
        n_vars=n_vars,
        device=device,
        tuned_params=tuned_params,
    )
    fine_model, fine_ckpt = _load_staged_diffusion_from_ckpt(
        ckpt_path=fine_ckpt_path,
        stage="fine",
        itrans_ckpt_path=itrans_ckpt_path,
        n_vars=n_vars,
        device=device,
        tuned_params=tuned_params,
    )
    diff_type = infer_diffusion_type(fine_ckpt, None)
    applied_h = apply_checkpoint_architecture(fine_ckpt, diff_type)

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for sample_index in sample_indices:
        past, future = test_ds[sample_index]
        past_t = past.unsqueeze(0).to(device)

        with torch.no_grad():
            coarse_res = coarse_model.generate(
                past_t,
                sampler=diffusion_sampler,
                num_inference_steps=num_inference_steps,
            )
            fine_res = fine_model.generate(
                past_t,
                sampler=diffusion_sampler,
                num_inference_steps=num_inference_steps,
                future_coarse_2d=coarse_res["future_2d_coarse"],
            )

        res = {
            "past_2d_coarse": coarse_res["past_2d_coarse"],
            "future_2d_coarse": coarse_res["future_2d_coarse"],
            "past_2d_fine": fine_res["past_2d_fine"],
            "future_2d_fine": fine_res["future_2d_fine"],
        }
        saved.append(_plot_dual_scale_sample(
            res=res,
            diff_model=fine_model,
            past=past,
            future=future,
            mean=mean,
            std=std,
            dataset_name=dataset_name,
            sample_index=sample_index,
            output_dir=output_dir,
            tag=tag,
            variables_to_plot=variables_to_plot,
            applied_h=applied_h,
            jpeg_dpi=jpeg_dpi,
        ))

    return saved


def generate_pipeline_visualizations(
    model: torch.nn.Module,
    itrans_model: torch.nn.Module,
    dataset,
    stats: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    output_dir: str,
    subset_id: str,
    n_samples: int = 1,
    forecast_length: int = 96,
    lookback_length: int = 96,
    jpeg_dpi: int = 100,
    seed: int = 42,
) -> List[str]:
    """DDIM denoising path + 1D comparison plots (eval phase)."""
    os.makedirs(output_dir, exist_ok=True)
    mean, std = stats
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    saved_paths: List[str] = []
    indices = pick_sample_indices(len(dataset), n_samples, seed=seed)
    file_idx = 1

    for row, idx in enumerate(indices):
        past, future = dataset[idx]
        past_t = past.unsqueeze(0).to(device)

        with torch.no_grad():
            B, C, L = past_t.shape
            x_enc = past_t.permute(0, 2, 1)
            seq_sl = getattr(itrans_model, "seq_len", L)
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            x_dec = torch.zeros(B, forecast_length, C, device=device)
            itrans_out = itrans_model(x_enc, None, x_dec, None)
            if isinstance(itrans_out, tuple):
                itrans_out = itrans_out[0]
            itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]

            torch.manual_seed(42 + idx)
            result = model.generate(
                past_t,
                sampler="ddim",
                num_inference_steps=20,
                yield_intermediates=True,
            )

            diff_pred = result.get("prediction_global_norm", result["prediction"]).cpu()[0]
            intermediates = result.get("intermediates", [])

        if row == 0 and intermediates:
            for t_step, img_tensor in intermediates:
                img = img_tensor[0, 0].cpu().numpy()
                if img.ndim == 3 and img.shape[0] == 2:
                    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
                    for si, ax in enumerate(axes):
                        ax.imshow(img[si], aspect="auto", cmap="viridis")
                        ax.set_title(f"{'coarse' if si == 0 else 'fine'} t={t_step}", fontsize=8)
                        ax.axis("off")
                else:
                    if img.ndim == 3:
                        img = img[0]
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))
                    ax.imshow(img, aspect="auto", cmap="viridis")
                    ax.set_title(f"t={t_step}", fontsize=8)
                    ax.axis("off")

                path = os.path.join(
                    output_dir, f"{file_idx:03d}_2D_denoising_sample{row}_step{t_step:04d}.jpg"
                )
                saved_paths.append(save_figure_jpg(fig, path, dpi=jpeg_dpi))
                file_idx += 1

        past_dn = denorm(past, mean, std)
        future_sliced = future[:, -forecast_length:]
        future_dn = denorm(future_sliced, mean, std)
        itrans_dn = denorm(itrans_pred, mean, std)
        diff_pred_sliced = diff_pred[:, -forecast_length:] if diff_pred.shape[-1] > forecast_length else diff_pred
        diff_dn = denorm(diff_pred_sliced, mean, std)

        n_vars_plot = min(3, C)
        fig, axes = plt.subplots(
            1, n_vars_plot, figsize=(4.5 * n_vars_plot, 3.0), squeeze=False, constrained_layout=True
        )
        t_past = np.arange(-lookback_length, 0)
        t_future = np.arange(0, forecast_length)

        for col in range(n_vars_plot):
            ax = axes[0, col]
            gt = future_dn[col].numpy()
            it = itrans_dn[col].numpy()
            df = diff_dn[col].numpy()
            ax.plot(t_past, past_dn[col, -lookback_length:].numpy(), color="#9E9E9E", alpha=0.5, linewidth=1.0)
            ax.plot(t_future, gt, color="#2196F3", linewidth=1.4, label="GT" if col == 0 else "")
            ax.plot(t_future, it, color="#FF9800", linewidth=1.1, linestyle="--", alpha=0.85, label="iTrans" if col == 0 else "")
            ax.plot(t_future, df, color="#E91E63", linewidth=1.1, label="Diffusion" if col == 0 else "")
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
            ax.set_title(
                f"Var {col} | iT {np.mean(np.abs(it - gt)):.3f} | D {np.mean(np.abs(df - gt)):.3f}",
                fontsize=9,
            )

        fig.suptitle(f"{subset_id} sample {idx}", fontsize=11, fontweight="semibold")
        path = os.path.join(output_dir, f"{file_idx:03d}_1D_comparison_sample{row}.jpg")
        saved_paths.append(save_figure_jpg(fig, path, dpi=jpeg_dpi))
        file_idx += 1

    return saved_paths


def run_pretrain_diffusion_visualizations(
    state: Any,
    *,
    diff_ckpt_path: Optional[str] = None,
    coarse_ckpt_path: Optional[str] = None,
    fine_ckpt_path: Optional[str] = None,
    itrans_ckpt_path: str,
    tuned_params: Optional[Dict[str, Any]] = None,
    tag: str = "pretrain_synthetic",
) -> List[str]:
    """Visualize non-finetuned diffusion after synthetic pretrain."""
    from models.diffusion_tsf.train_multivariate_pipeline import generate_dataset_job

    viz = _viz_cfg(state)
    if not viz.get("enabled", True):
        return []

    variate_indices = state.variate_indices
    if variate_indices is None:
        variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

    output_dir = os.path.join(state.results_dir, "viz", tag)
    device = state.resolve_device()

    coarse_ckpt = coarse_ckpt_path or getattr(state, "diffusion_coarse_pretrain_ckpt", None)
    fine_ckpt = fine_ckpt_path or diff_ckpt_path or getattr(state, "diffusion_fine_pretrain_ckpt", None)
    if coarse_ckpt and fine_ckpt:
        return generate_staged_dual_scale_comparisons(
            coarse_ckpt_path=coarse_ckpt,
            fine_ckpt_path=fine_ckpt,
            itrans_ckpt_path=itrans_ckpt_path,
            dataset_name=state.dataset,
            variate_indices=variate_indices,
            output_dir=output_dir,
            device=device,
            tuned_params=tuned_params,
            lookback_length=state.lookback_length,
            forecast_length=state.forecast_length,
            diffusion_sampler=viz.get("dual_scale_sampler", "anchor"),
            num_inference_steps=int(viz.get("dual_scale_inference_steps", 20)),
            variables_to_plot=int(viz.get("n_dual_scale_vars", 3)),
            n_samples=1 if state.smoke_test else int(viz.get("n_samples", 3)),
            random_seed=state.seed,
            jpeg_dpi=int(viz.get("jpeg_dpi", 100)),
            tag=tag,
        )

    return []


def run_staged_finetune_visualizations(
    state: Any,
    *,
    coarse_ckpt_path: str,
    fine_ckpt_path: str,
    itrans_ckpt_path: str,
    tuned_params: Optional[Dict[str, Any]] = None,
    tag: str = "staged_diffusion_finetuned",
) -> List[str]:
    """Dual-scale viz for finetuned coarse/fine checkpoints."""
    from models.diffusion_tsf.train_multivariate_pipeline import generate_dataset_job

    viz = _viz_cfg(state)
    if not viz.get("enabled", True):
        return []

    variate_indices = state.variate_indices
    if variate_indices is None:
        variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

    output_dir = os.path.join(state.results_dir, "viz", tag)
    device = state.resolve_device()
    return generate_staged_dual_scale_comparisons(
        coarse_ckpt_path=coarse_ckpt_path,
        fine_ckpt_path=fine_ckpt_path,
        itrans_ckpt_path=itrans_ckpt_path,
        dataset_name=state.dataset,
        variate_indices=variate_indices,
        output_dir=output_dir,
        device=device,
        tuned_params=tuned_params,
        lookback_length=state.lookback_length,
        forecast_length=state.forecast_length,
        diffusion_sampler=viz.get("dual_scale_sampler", "anchor"),
        num_inference_steps=int(viz.get("dual_scale_inference_steps", 20)),
        variables_to_plot=int(viz.get("n_dual_scale_vars", 3)),
        n_samples=1 if state.smoke_test else int(viz.get("n_samples", 3)),
        random_seed=state.seed,
        jpeg_dpi=int(viz.get("jpeg_dpi", 100)),
        tag=tag,
    )


def run_itrans_checkpoint_visualizations(
    state: Any,
    itrans_ckpt_path: str,
    tag: str,
) -> List[str]:
    """iTransformer forecast plots at synthetic-pretrain or finetune checkpoint."""
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
        load_itransformer_from_checkpoint,
    )

    viz = _viz_cfg(state)
    if not viz.get("enabled", True):
        return []

    variate_indices = state.variate_indices
    if variate_indices is None:
        variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = int(subset_meta.get("test_stride", 1))

    _, _, test_ds, norm_stats = load_dataset(
        state.dataset, variate_indices, stride=train_stride, test_stride=test_stride,
    )
    device = state.resolve_device()
    n_iv = len(variate_indices)
    model = load_itransformer_from_checkpoint(itrans_ckpt_path, n_iv, device)
    output_dir = os.path.join(state.results_dir, "viz", tag)

    return generate_itrans_prediction_viz(
        model,
        test_ds,
        (norm_stats["mean"], norm_stats["std"]),
        device,
        output_dir,
        tag=tag,
        n_samples=1 if state.smoke_test else int(viz.get("n_samples", 3)),
        forecast_length=state.forecast_length,
        lookback_length=state.lookback_length,
        seed=state.seed,
        jpeg_dpi=int(viz.get("jpeg_dpi", 100)),
    )


def _as_channel_first(past: torch.Tensor, future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if past.dim() == 1:
        past = past.unsqueeze(0)
    if future.dim() == 1:
        future = future.unsqueeze(0)
    return past, future


def itrans_checkpoint_metadata(path: str, num_vars: int, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt
    weight_key = "enc_embedding.value_embedding.weight"
    if weight_key not in state:
        raise RuntimeError(f"iTransformer checkpoint missing {weight_key!r}: {path}")
    seq_len = int(state[weight_key].shape[1])
    pred_len = None
    if "projector.weight" in state:
        pred_len = int(state["projector.weight"].shape[0])
    return {
        "itrans_checkpoint_path": os.path.abspath(path),
        "itrans_loaded_from_checkpoint": True,
        "itrans_num_variates": int(num_vars),
        "itrans_seq_len": seq_len,
        "itrans_pred_len": pred_len,
        "itrans_lookback_length": seq_len,
        "itrans_horizon_length": pred_len,
    }


def compute_synthetic_dataset_stats(
    dataset,
    *,
    n_probe: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mean/std/min/max/quartiles per variate on pre-normalized RealTS windows."""
    from models.diffusion_tsf.pipeline.phase_diagnostics import compute_dataset_stats

    stats = compute_dataset_stats(dataset, prefix="pretrain", n_probe=n_probe, seed=seed)
    # backward-compatible aliases
    if "pretrain_n_variates" in stats:
        return stats
    return {"pretrain_n_variates": 0, "pretrain_variate_means": [], "pretrain_variate_stds": []}


def _build_staged_encoding_model(
    state: Any,
    itrans_ckpt_path: str,
    *,
    stage: str,
    tuned_params: Optional[Dict[str, Any]],
):
    from models.diffusion_tsf.train_multivariate_pipeline import (
        anchor_kwargs_from_params,
        create_diffusion_model,
        load_itransformer_from_checkpoint,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from models.diffusion_tsf.guidance import iTransformerGuidance
    from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals

    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=False)
    device = state.resolve_device()
    n_vars = int(state.n_variates)
    itrans = load_itransformer_from_checkpoint(itrans_ckpt_path, n_vars, device)
    guidance = iTransformerGuidance(itrans)
    model = create_diffusion_model(
        guidance_model=guidance,
        **anchor_kwargs_from_params(tuned_params),
    ).to(device)
    model.eval()
    return model, itrans, device


def _plot_realts_1d_pre_post_norm(
    *,
    past: torch.Tensor,
    future: torch.Tensor,
    past_norm: torch.Tensor,
    future_norm: torch.Tensor,
    sample_index: int,
    output_dir: str,
    lookback_length: int,
    jpeg_dpi: int,
    variables_to_plot: int,
) -> str:
    past, future = _as_channel_first(past, future)
    past_norm = past_norm[0] if past_norm.dim() == 3 else past_norm
    future_norm = future_norm[0] if future_norm.dim() == 3 else future_norm
    n_vars = past.shape[0]
    n_cols = min(variables_to_plot, n_vars)

    full_pre = torch.cat([past, future], dim=-1)
    full_post = torch.cat([past_norm, future_norm], dim=-1)
    t_axis = np.arange(-lookback_length, future.shape[-1])

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(4.2 * n_cols, 5.0),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    row_labels = ("pre-normalized", "post-instance-norm")

    for col in range(n_cols):
        for row, series in enumerate((full_pre, full_post)):
            ax = axes[row, col]
            ax.plot(t_axis, series[col].numpy(), color="#2196F3", linewidth=1.2)
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.3)
            ax.grid(True, alpha=0.12)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(f"var {col}", fontsize=9)
            if row == 1:
                ax.set_xlabel("raw series timestep (t=0 = forecast start)")

    scale_note = _format_scale_banner(
        raw_range=_tensor_value_range(full_pre),
        norm_range=_tensor_value_range(full_post),
        space_label="top=pre-norm (dataset z-score space)",
        extra="bottom=post instance-norm (window norm)",
    )
    fig.suptitle(f"RealTS sample {sample_index}\n{scale_note}", fontsize=10, fontweight="semibold")
    path = os.path.join(output_dir, f"realts_sample_1d_idx{sample_index}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def _plot_staged_2d_maps_native(
    *,
    maps: Dict[str, torch.Tensor],
    sample_index: int,
    output_dir: str,
    tag: str,
    variables_to_plot: int,
    jpeg_dpi: int,
) -> str:
    coarse = maps["coarse"][0].cpu().numpy()
    fine = maps["fine"][0].cpu().numpy()
    n_vars = min(variables_to_plot, coarse.shape[0])
    fig, axes = plt.subplots(
        2, n_vars,
        figsize=(4.0 * n_vars, 5.0),
        constrained_layout=True,
        squeeze=False,
    )
    row_labels = ("coarse 2D", "fine 2D")

    for col in range(n_vars):
        for row, data in enumerate((coarse, fine)):
            ax = axes[row, col]
            h, w = data[col].shape
            im = ax.imshow(
                data[col],
                aspect="auto",
                origin="lower",
                extent=[0, w, 0, h],
                cmap="plasma",
            )
            ax.set_title(f"{row_labels[row]} | var {col} ({h}x{w})", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{tag} sample {sample_index} (native 2D fed to model, no rescale/pad)\n"
        + _format_scale_banner(
            norm_range=(0.0, 1.0),
            space_label="occupancy CDF",
            extra=f"extent=[0, W] columns x [0, H] bins",
        ),
        fontsize=10,
        fontweight="semibold",
    )
    path = os.path.join(output_dir, f"{tag}_2d_idx{sample_index}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def _plot_itrans_window_norm_1d(
    *,
    past_norm: torch.Tensor,
    future_norm: torch.Tensor,
    guidance_norm: torch.Tensor,
    sample_index: int,
    output_dir: str,
    lookback_length: int,
    jpeg_dpi: int,
    variables_to_plot: int,
) -> str:
    past_norm = past_norm[0] if past_norm.dim() == 3 else past_norm
    future_norm = future_norm[0] if future_norm.dim() == 3 else future_norm
    guidance_norm = guidance_norm[0] if guidance_norm.dim() == 3 else guidance_norm
    n_vars = past_norm.shape[0]
    n_cols = min(variables_to_plot, n_vars)
    K = guidance_norm.shape[-1] - future_norm.shape[-1]
    if K < 0:
        K = 0

    t_past = np.arange(-lookback_length, 0)
    t_future = np.arange(0, future_norm.shape[-1])

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.5 * n_cols, 3.2),
        squeeze=False,
        constrained_layout=True,
    )
    for col in range(n_cols):
        ax = axes[0, col]
        ax.plot(t_past, past_norm[col, -lookback_length:].numpy(), color="#9E9E9E", alpha=0.6, linewidth=1.0)
        ax.plot(t_future, future_norm[col].numpy(), color="#2196F3", linewidth=1.4, label="GT (window norm)")
        ax.plot(
            np.arange(-K, guidance_norm.shape[-1] - K),
            guidance_norm[col].numpy(),
            color="#FF9800",
            linewidth=1.2,
            linestyle="--",
            label="iTrans",
        )
        ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
        mae = float(np.mean(np.abs(guidance_norm[col, K:].numpy() - future_norm[col].numpy())))
        ax.set_title(f"var {col} | MAE {mae:.3f}", fontsize=9)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"iTransformer prediction | sample {sample_index}", fontsize=11, fontweight="semibold")
    path = os.path.join(output_dir, f"itrans_pred_1d_idx{sample_index}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def run_staged_synthetic_pretrain_diagnostics(
    state: Any,
    *,
    itrans_ckpt_path: str,
    itrans_meta: Dict[str, Any],
    tuned_params: Optional[Dict[str, Any]] = None,
    n_samples: int = 10000,
    stage: str = "coarse",
    diffusion_ckpt_path: Optional[str] = None,
    include_dataset_stats: bool = True,
    include_phase_start: bool = True,
) -> Dict[str, Any]:
    """RealTS + iTransformer diagnostics for staged synthetic pretrain (architecture.md §Phase 1)."""
    from models.diffusion_tsf.realts import get_synthetic_dataloader
    from models.diffusion_tsf.train_multivariate_pipeline import (
        get_synth_cache_dir,
        synthetic_epoch_capacity_pretrain_diffusion,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals

    viz = _viz_cfg(state)
    patch_globals(pipeline_mod, state, honor_dataset_windows=False)

    device = state.resolve_device()
    n_val = 0 if state.smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if state.smoke_test else synthetic_epoch_capacity_pretrain_diffusion()
    synth_cache = get_synth_cache_dir(checkpoint_dir=state.checkpoint_dir, smoke_test=state.smoke_test)
    loader = get_synthetic_dataloader(
        batch_size=2,
        lookback_length=state.lookback_length,
        forecast_length=state.forecast_length,
        num_variables=int(state.n_variates),
        num_samples=n_samples,
        num_workers=0,
        seed=state.seed,
        lookback_overlap=state.lookback_overlap,
        cache_dir=synth_cache,
        skip_cross_var_aug=(int(state.n_variates) > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )
    dataset = loader.dataset

    try:
        ckpt_meta = itrans_checkpoint_metadata(itrans_ckpt_path, int(state.n_variates), device)
    except Exception as exc:
        logger.warning("Could not read iTransformer checkpoint metadata (%s): %s", itrans_ckpt_path, exc)
        ckpt_meta = {
            "itrans_checkpoint_path": os.path.abspath(itrans_ckpt_path),
            "itrans_num_variates": int(state.n_variates),
            "itrans_seq_len": int(state.itrans_lookback_length or state.lookback_length),
            "itrans_pred_len": int(state.forecast_length),
            "itrans_lookback_length": int(state.itrans_lookback_length or state.lookback_length),
            "itrans_horizon_length": int(state.forecast_length),
        }
    ckpt_meta["itrans_loaded_from_checkpoint"] = bool(itrans_meta.get("loaded", True))
    ckpt_meta["itrans_source"] = str(itrans_meta.get("source", "checkpoint"))

    stats = (
        compute_synthetic_dataset_stats(
            dataset,
            n_probe=32 if state.smoke_test else 256,
            seed=state.seed,
        )
        if include_dataset_stats
        else {}
    )

    summary = {
        "pretrain/itrans_loaded": int(ckpt_meta["itrans_loaded_from_checkpoint"]),
        "pretrain/itrans_seq_len": ckpt_meta["itrans_seq_len"],
        "pretrain/itrans_pred_len": ckpt_meta["itrans_pred_len"],
    }
    if include_dataset_stats and stats:
        summary.update({
            "pretrain/n_variates": stats["pretrain_n_variates"],
            "pretrain/stats_n_probe": stats["pretrain_stats_n_probe"],
        })
        for i, (m, s) in enumerate(zip(stats["pretrain_variate_means"], stats["pretrain_variate_stds"])):
            summary[f"pretrain/variate_{i}_mean"] = m
            summary[f"pretrain/variate_{i}_std"] = s

    config_updates = {**stats, **ckpt_meta}
    viz_paths: Dict[str, List[str]] = {}
    if not viz.get("enabled", True):
        return {"summary": summary, "config": config_updates, "viz": viz_paths}

    output_dir = os.path.join(state.results_dir, "viz", "staged_synthetic_pretrain_diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    if state.smoke_test:
        return {"summary": summary, "config": config_updates, "viz": viz_paths}

    if diffusion_ckpt_path and os.path.exists(diffusion_ckpt_path):
        model, _ = _load_staged_diffusion_from_ckpt(
            ckpt_path=diffusion_ckpt_path,
            stage=stage,
            itrans_ckpt_path=itrans_ckpt_path,
            n_vars=int(state.n_variates),
            device=device,
            tuned_params=tuned_params,
        )
    else:
        model, _, device = _build_staged_encoding_model(
            state,
            itrans_ckpt_path,
            stage=stage,
            tuned_params=tuned_params,
        )
    coarse_model = None
    if stage == "fine":
        coarse_ckpt = getattr(state, "diffusion_coarse_pretrain_ckpt", None)
        if not coarse_ckpt:
            coarse_ckpt = os.path.join(
                state.checkpoint_dir, "pretrained_coarse", "pretrained_diffusion.pt",
            )
        if os.path.exists(coarse_ckpt):
            coarse_model, _ = _load_staged_diffusion_from_ckpt(
                ckpt_path=coarse_ckpt,
                stage="coarse",
                itrans_ckpt_path=itrans_ckpt_path,
                n_vars=int(state.n_variates),
                device=device,
                tuned_params=tuned_params,
            )
        else:
            logger.warning(
                "staged pretrain diagnostics: no coarse ckpt for fine stage at %s",
                coarse_ckpt,
            )
    from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics

    if include_phase_start:
        run_phase_start_diagnostics(
            state,
            phase_name=f"staged_diffusion_pretrain/{stage}",
            models=[model],
            model_labels=[f"diffusion_{stage}"],
            datasets=[dataset] if include_dataset_stats else None,
            dataset_prefixes=["pretrain"] if include_dataset_stats else None,
            ckpt_info=[{
                "kind": "itrans",
                "path": itrans_ckpt_path,
                "n_variates": int(state.n_variates),
                "lookback": int(state.lookback_length),
                "horizon": int(state.forecast_length),
                "extra": {"source": itrans_meta.get("source")},
            }],
        )
    variables_to_plot = int(viz.get("n_dual_scale_vars", 3))
    jpeg_dpi = int(viz.get("jpeg_dpi", 100))
    sample_index = pick_sample_indices(len(dataset), 1, seed=state.seed)[0]

    past, future = dataset[sample_index]
    past_cf, future_cf = _as_channel_first(past, future)
    past_b = past_cf.unsqueeze(0).to(device)
    future_b = future_cf.unsqueeze(0).to(device)

    with torch.no_grad():
        past_norm, future_norm, norm_stats = model._normalize_sequence(past_b, future_b)
        future_maps = model._encode_staged_maps(future_norm)
        W_fut = future_maps[stage].shape[-1]
        guidance_norm = model._get_guidance_forecast_norm(
            past_b, past_norm, norm_stats, W_fut,
        )
        guidance_maps = model._encode_staged_maps(guidance_norm)

    sample_paths = [
        _plot_realts_1d_pre_post_norm(
            past=past_cf,
            future=future_cf,
            past_norm=past_norm[0].cpu(),
            future_norm=future_norm[0].cpu(),
            sample_index=sample_index,
            output_dir=output_dir,
            lookback_length=state.lookback_length,
            jpeg_dpi=jpeg_dpi,
            variables_to_plot=variables_to_plot,
        ),
        _plot_staged_2d_maps_native(
            maps={k: v.detach().cpu() for k, v in future_maps.items() if k in {"coarse", "fine"}},
            sample_index=sample_index,
            output_dir=output_dir,
            tag="realts_sample",
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
        ),
        _plot_itrans_window_norm_1d(
            past_norm=past_norm[0].cpu(),
            future_norm=future_norm[0].cpu(),
            guidance_norm=guidance_norm[0].cpu(),
            sample_index=sample_index,
            output_dir=output_dir,
            lookback_length=state.lookback_length,
            jpeg_dpi=jpeg_dpi,
            variables_to_plot=variables_to_plot,
        ),
        _plot_staged_2d_maps_native(
            maps={k: v.detach().cpu() for k, v in guidance_maps.items() if k in {"coarse", "fine"}},
            sample_index=sample_index,
            output_dir=output_dir,
            tag="itrans_pred",
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
        ),
    ]
    viz_paths["viz/staged_synthetic_pretrain/realts_1d"] = [sample_paths[0]]
    viz_paths["viz/staged_synthetic_pretrain/realts_2d"] = [sample_paths[1]]
    viz_paths["viz/staged_synthetic_pretrain/itrans_1d"] = [sample_paths[2]]
    viz_paths["viz/staged_synthetic_pretrain/itrans_2d"] = [sample_paths[3]]

    with torch.no_grad():
        gen_out = _staged_diag_generate(
            model, past_b, coarse_model=coarse_model, sampler="anchor",
        )
        diff_paths = _plot_diffusion_model_space_prediction(
            gen_out=gen_out,
            past_norm=past_norm[0].cpu(),
            future_norm=future_norm[0].cpu(),
            model=model,
            sample_index=sample_index,
            output_dir=output_dir,
            lookback_length=state.lookback_length,
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
            tag=f"diffusion_{stage}",
        )
        viz_paths["viz/staged_synthetic_pretrain/diffusion_pred"] = diff_paths

        cond_paths = run_diffusion_conditioning_diagnostics(
            state=state,
            model=model,
            past=past_b,
            future=future_b,
            stage=stage,
            output_dir=os.path.join(output_dir, "conditioning"),
            sample_index=sample_index,
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
            coarse_model=coarse_model,
        )
        for k, v in cond_paths.items():
            viz_paths.setdefault(k, []).extend(v)

    return {"summary": summary, "config": config_updates, "viz": viz_paths}


def _staged_diag_generate(
    model,
    past: torch.Tensor,
    *,
    coarse_model=None,
    sampler: str = "anchor",
) -> Dict[str, torch.Tensor]:
    stage = getattr(getattr(model, "config", None), "diffusion_stage", None)
    if stage == "fine":
        if coarse_model is None:
            raise ValueError("fine-stage diagnostics require coarse_model")
        coarse_out = coarse_model.generate(past, sampler=sampler)
        return model.generate(
            past,
            sampler=sampler,
            future_coarse_2d=coarse_out["future_2d_coarse"],
        )
    return model.generate(past, sampler=sampler)


def _plot_diffusion_model_space_prediction(
    *,
    gen_out: Dict[str, torch.Tensor],
    past_norm: torch.Tensor,
    future_norm: torch.Tensor,
    model,
    sample_index: int,
    output_dir: str,
    lookback_length: int,
    variables_to_plot: int,
    jpeg_dpi: int,
    tag: str,
) -> List[str]:
    """1D + 2D diffusion prediction in window-norm / model space (no denorm)."""
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []

    to_2d = model.to_2d
    future_coarse = gen_out["future_2d_coarse"].cpu()
    coarse_1d = _decode_staged_cdf_1d(
        future_coarse, value_range=to_2d.max_scale, to_2d=to_2d,
    )[0]
    fine_raw = gen_out.get("future_2d_fine")
    if fine_raw is not None:
        future_fine = fine_raw.cpu()
        fine_1d = _decode_staged_cdf_1d(
            future_fine,
            value_range=_staged_fine_value_range(model),
            to_2d=to_2d,
        )[0]
        combined_1d = coarse_1d + fine_1d
        pred_maps = {
            "coarse": future_coarse if future_coarse.dim() == 4 else future_coarse.unsqueeze(0),
            "fine": future_fine if future_fine.dim() == 4 else future_fine.unsqueeze(0),
        }
    else:
        future_fine = None
        fine_1d = None
        combined_1d = coarse_1d
        pred_maps = {
            "coarse": future_coarse if future_coarse.dim() == 4 else future_coarse.unsqueeze(0),
        }
    k_overlap = int(getattr(getattr(model, "config", None), "lookback_overlap", 0))
    w_map = int(future_coarse.shape[-1])
    if k_overlap > 0 and w_map > int(getattr(model.config, "forecast_length", w_map)):
        plot_coarse = coarse_1d[..., k_overlap:]
        plot_combined = combined_1d[..., k_overlap:]
        plot_fine = fine_1d[..., k_overlap:] if fine_1d is not None else None
        w_plot = plot_coarse.shape[-1]
    else:
        plot_coarse = coarse_1d
        plot_combined = combined_1d
        plot_fine = fine_1d
        w_plot = w_map
    t_fut = np.arange(0, w_plot)
    n_cols = min(variables_to_plot, plot_coarse.shape[0])

    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.2), squeeze=False, constrained_layout=True)
    gt_fut = _future_core_slice(
        future_norm, width=w_map, lookback_overlap=k_overlap,
    )
    for col in range(n_cols):
        ax = axes[0, col]
        ax.plot(t_fut, gt_fut[col].numpy(), color="#2196F3", linewidth=1.4, label="GT fut")
        ax.plot(t_fut, plot_coarse[col].numpy(), color="#FF9800", linewidth=1.1, label="Coarse pred")
        ax.plot(t_fut, plot_combined[col].numpy(), color="#E91E63", linewidth=1.1, label="Combined")
        ax.grid(True, alpha=0.12)
        ax.set_title(f"var {col}", fontsize=9)
        if col == 0:
            ax.legend(fontsize=7)
    fig.suptitle(
        f"{tag} 1D model-space sample {sample_index}\n"
        + _format_scale_banner(norm_range=_tensor_value_range(plot_combined), space_label="window-norm"),
        fontsize=10,
    )
    paths.append(save_figure_jpg(fig, os.path.join(output_dir, f"{tag}_pred_1d_idx{sample_index}.jpg"), dpi=jpeg_dpi))

    paths.append(
        _plot_staged_2d_maps_native(
            maps=pred_maps,
            sample_index=sample_index,
            output_dir=output_dir,
            tag=f"{tag}_pred",
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
        )
    )
    return paths


def _plot_cond_2d_native(
    tensor: torch.Tensor,
    *,
    sample_index: int,
    output_dir: str,
    tag: str,
    variables_to_plot: int,
    jpeg_dpi: int,
    title: str,
) -> str:
    """Plot a single (B,V,H,W) or (V,H,W) cond tensor as native 2D."""
    os.makedirs(output_dir, exist_ok=True)
    if tensor.dim() == 4:
        data = tensor[0].cpu().numpy()
    else:
        data = tensor.cpu().numpy()
    n_vars = min(variables_to_plot, data.shape[0])
    fig, axes = plt.subplots(1, n_vars, figsize=(4.0 * n_vars, 3.5), squeeze=False, constrained_layout=True)
    for col in range(n_vars):
        h, w = data[col].shape
        im = axes[0, col].imshow(data[col], aspect="auto", origin="lower", extent=[0, w, 0, h], cmap="viridis")
        axes[0, col].set_title(f"var {col} ({h}x{w})", fontsize=8)
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title}\n"
        + _format_scale_banner(norm_range=(float(data.min()), float(data.max())), space_label="model cond"),
        fontsize=10,
    )
    path = os.path.join(output_dir, f"{tag}_2d_idx{sample_index}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def plot_cross_attn_topk(
    attn_weights: torch.Tensor,
    *,
    target_variate: int,
    output_dir: str,
    tag: str,
    top_k: int = 5,
    jpeg_dpi: int = 100,
) -> str:
    """Bar chart of top attended variates (excluding self)."""
    os.makedirs(output_dir, exist_ok=True)
    w = attn_weights.detach().cpu().float()
    if w.dim() == 2:
        w = w[0]
    n_vars = w.numel()
    scores = w.clone()
    if 0 <= target_variate < n_vars:
        scores[target_variate] = -1.0
    top_k = min(top_k, n_vars - 1)
    vals, idxs = torch.topk(scores, k=top_k)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    ax.bar([f"v{int(i)}" for i in idxs], vals.numpy(), color="#673AB7")
    ax.set_ylabel("mean attention weight")
    ax.set_title(f"cross-attn top-{top_k} (target var {target_variate})")
    path = os.path.join(output_dir, f"cross_attn_topk_{tag}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def run_diffusion_conditioning_diagnostics(
    *,
    state: Any,
    model,
    past: torch.Tensor,
    future: torch.Tensor,
    stage: str,
    output_dir: str,
    sample_index: int = 0,
    variables_to_plot: int = 3,
    jpeg_dpi: int = 100,
    coarse_model=None,
) -> Dict[str, List[str]]:
    """Visualize lookback 2D cond, iTrans 2D pred, and cross-attn top variates."""
    from models.diffusion_tsf.pipeline.logging_utils import get_diagnostic_logger

    diag_logger = get_diagnostic_logger()
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, List[str]] = {}

    cap = model.diagnostic_capture_staged(past, future, capture_cross_attn=True)
    cond = cap["cond_for_unet"]
    paths.setdefault(f"viz/conditioning/{stage}/lookback_2d", []).append(
        _plot_cond_2d_native(
            cond[:, :1],
            sample_index=sample_index,
            output_dir=output_dir,
            tag=f"lookback_cond_{stage}",
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
            title=f"2D lookback conditioning ({stage})",
        )
    )

    guidance_maps = cap.get("guidance_maps")
    if guidance_maps is not None and stage in guidance_maps:
        paths.setdefault(f"viz/conditioning/{stage}/itrans_2d", []).append(
            _plot_staged_2d_maps_native(
                maps={k: v.detach().cpu() for k, v in guidance_maps.items() if k in {"coarse", "fine"}},
                sample_index=sample_index,
                output_dir=output_dir,
                tag=f"itrans_pred_{stage}",
                variables_to_plot=variables_to_plot,
                jpeg_dpi=jpeg_dpi,
            )
        )

    attn_w = cap.get("cross_attn_weights")
    if attn_w is not None:
        target_v = 0
        p = plot_cross_attn_topk(
            attn_w, target_variate=target_v, output_dir=output_dir,
            tag=f"{stage}_var{target_v}", jpeg_dpi=jpeg_dpi,
        )
        paths.setdefault(f"viz/conditioning/{stage}/cross_attn", []).append(p)
        w = attn_w[0].detach().cpu()
        top_vals, top_idx = torch.topk(w, k=min(5, w.numel()))
        for rank, (vi, wt) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
            diag_logger.info(
                "cross-attn %s target_v=%d rank=%d variate=%d weight=%.4f",
                stage, target_v, rank + 1, vi, wt,
            )

    gen_out = _staged_diag_generate(model, past, coarse_model=coarse_model, sampler="anchor")
    paths.setdefault(f"viz/conditioning/{stage}/diffusion_pred", []).extend(
        _plot_diffusion_model_space_prediction(
            gen_out=gen_out,
            past_norm=cap["past_norm"][0].cpu(),
            future_norm=cap["future_norm"][0].cpu(),
            model=model,
            sample_index=sample_index,
            output_dir=output_dir,
            lookback_length=state.lookback_length,
            variables_to_plot=variables_to_plot,
            jpeg_dpi=jpeg_dpi,
            tag=f"diffusion_{stage}",
        )
    )
    return paths


def run_real_dataset_phase_diagnostics(
    state: Any,
    *,
    train_ds,
    model,
    itrans_ckpt_path: str,
    stage: str,
    diffusion_ckpt_path: Optional[str] = None,
    coarse_ckpt_path: Optional[str] = None,
    tag: str = "real_dataset",
    include_phase_start: bool = True,
) -> Dict[str, Any]:
    """Shared diagnostics for iTrans / finetune / eval phases on real data."""
    from models.diffusion_tsf.pipeline.phase_diagnostics import (
        compute_dataset_stats,
        run_phase_start_diagnostics,
    )

    viz = _viz_cfg(state)
    if not viz.get("enabled", True) or state.smoke_test:
        stats = compute_dataset_stats(
            train_ds, prefix="dataset",
            n_probe=32 if state.smoke_test else 256,
            seed=state.seed,
        )
        return {"summary": stats, "viz": {}}

    output_dir = os.path.join(state.results_dir, "viz", tag, stage)
    os.makedirs(output_dir, exist_ok=True)
    sample_index = pick_sample_indices(len(train_ds), 1, seed=state.seed)[0]
    past, future = train_ds[sample_index]
    past_cf, future_cf = _as_channel_first(past, future)
    past_b = past_cf.unsqueeze(0).to(state.resolve_device())
    future_b = future_cf.unsqueeze(0).to(state.resolve_device())

    ckpt_info = [{
        "kind": "itrans",
        "path": itrans_ckpt_path,
        "n_variates": int(state.n_variates),
        "lookback": int(state.lookback_length),
        "horizon": int(state.forecast_length),
    }]
    if diffusion_ckpt_path:
        ckpt_info.append({
            "kind": f"diffusion_{stage}",
            "path": diffusion_ckpt_path,
            "n_variates": int(state.n_variates),
            "lookback": int(state.lookback_length),
            "horizon": int(state.forecast_length),
        })

    summary: Dict[str, Any] = {}
    if include_phase_start:
        summary = run_phase_start_diagnostics(
            state,
            phase_name=f"{tag}/{stage}",
            models=[model],
            model_labels=[f"diffusion_{stage}"],
            datasets=[train_ds],
            dataset_prefixes=["dataset"],
            ckpt_info=ckpt_info,
        )

    variables_to_plot = int(viz.get("n_dual_scale_vars", 3))
    jpeg_dpi = int(viz.get("jpeg_dpi", 100))
    viz_paths: Dict[str, List[str]] = {}

    with torch.no_grad():
        coarse_model = None
        if stage == "fine" and coarse_ckpt_path:
            coarse_model, _ = _load_staged_diffusion_from_ckpt(
                ckpt_path=coarse_ckpt_path,
                stage="coarse",
                itrans_ckpt_path=itrans_ckpt_path,
                n_vars=int(state.n_variates),
                device=state.resolve_device(),
            )
        past_norm, future_norm, norm_stats = model._normalize_sequence(past_b, future_b)
        p1 = _plot_realts_1d_pre_post_norm(
            past=past_cf, future=future_cf,
            past_norm=past_norm[0].cpu(), future_norm=future_norm[0].cpu(),
            sample_index=sample_index, output_dir=output_dir,
            lookback_length=state.lookback_length, jpeg_dpi=jpeg_dpi,
            variables_to_plot=variables_to_plot,
        )
        future_maps = model._encode_staged_maps(future_norm)
        p2 = _plot_staged_2d_maps_native(
            maps={k: v.detach().cpu() for k, v in future_maps.items() if k in {"coarse", "fine"}},
            sample_index=sample_index, output_dir=output_dir, tag="dataset_sample",
            variables_to_plot=variables_to_plot, jpeg_dpi=jpeg_dpi,
        )
        W_fut = future_maps[stage].shape[-1]
        guidance_norm = model._get_guidance_forecast_norm(past_b, past_norm, norm_stats, W_fut)
        guidance_maps = model._encode_staged_maps(guidance_norm)
        p3 = _plot_itrans_window_norm_1d(
            past_norm=past_norm[0].cpu(), future_norm=future_norm[0].cpu(),
            guidance_norm=guidance_norm[0].cpu(),
            sample_index=sample_index, output_dir=output_dir,
            lookback_length=state.lookback_length, jpeg_dpi=jpeg_dpi,
            variables_to_plot=variables_to_plot,
        )
        p4 = _plot_staged_2d_maps_native(
            maps={k: v.detach().cpu() for k, v in guidance_maps.items() if k in {"coarse", "fine"}},
            sample_index=sample_index, output_dir=output_dir, tag="itrans_pred",
            variables_to_plot=variables_to_plot, jpeg_dpi=jpeg_dpi,
        )
        cond_paths = run_diffusion_conditioning_diagnostics(
            state=state, model=model, past=past_b, future=future_b, stage=stage,
            output_dir=os.path.join(output_dir, "conditioning"),
            sample_index=sample_index, variables_to_plot=variables_to_plot, jpeg_dpi=jpeg_dpi,
            coarse_model=coarse_model,
        )

    viz_paths[f"viz/{tag}/dataset_1d"] = [p1]
    viz_paths[f"viz/{tag}/dataset_2d"] = [p2]
    viz_paths[f"viz/{tag}/itrans_1d"] = [p3]
    viz_paths[f"viz/{tag}/itrans_2d"] = [p4]
    for k, v in cond_paths.items():
        viz_paths.setdefault(k, []).extend(v)

    return {"summary": summary, "viz": viz_paths}


def run_itrans_finetune_diagnostics(
    state: Any,
    *,
    ckpt_path: str,
    train_ds,
) -> Dict[str, Any]:
    """iTrans phase: real dataset sample + iTrans prediction viz."""
    from models.diffusion_tsf.train_multivariate_pipeline import load_itransformer_from_checkpoint

    viz = _viz_cfg(state)
    if not viz.get("enabled", True):
        return {"viz": {}}

    from models.diffusion_tsf.pipeline.phase_diagnostics import compute_dataset_stats, run_phase_start_diagnostics

    stats = compute_dataset_stats(
        train_ds, prefix="dataset",
        n_probe=32 if state.smoke_test else 256,
        seed=state.seed,
    )
    if state.smoke_test:
        return {"summary": stats, "viz": {}}

    device = state.resolve_device()
    n_iv = int(state.n_variates)
    model = load_itransformer_from_checkpoint(ckpt_path, n_iv, device)
    phase_summary = run_phase_start_diagnostics(
        state,
        phase_name="itrans_finetune_hp",
        models=[model],
        model_labels=["itrans"],
        datasets=[train_ds],
        dataset_prefixes=["dataset"],
        ckpt_info=[{
            "kind": "itrans",
            "path": ckpt_path,
            "n_variates": n_iv,
            "lookback": int(state.lookback_length),
            "horizon": int(state.forecast_length),
        }],
    )
    summary = {**stats, **phase_summary}

    output_dir = os.path.join(state.results_dir, "viz", "itrans_finetune_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    sample_index = pick_sample_indices(len(train_ds), 1, seed=state.seed)[0]
    past, future = train_ds[sample_index]
    past_cf, future_cf = _as_channel_first(past, future)
    variables_to_plot = int(viz.get("n_dual_scale_vars", 3))
    jpeg_dpi = int(viz.get("jpeg_dpi", 100))

    viz_out: Dict[str, Any] = {"summary": summary, "viz": {}}
    try:
        from models.diffusion_tsf.guidance import iTransformerGuidance
        from models.diffusion_tsf.train_multivariate_pipeline import create_diffusion_model
        guidance = iTransformerGuidance(model)
        enc_model = create_diffusion_model(guidance_model=guidance, diffusion_stage="coarse").to(device)
        enc_model.eval()
        past_b = past_cf.unsqueeze(0).to(device)
        future_b = future_cf.unsqueeze(0).to(device)
        with torch.no_grad():
            past_norm, future_norm, norm_stats = enc_model._normalize_sequence(past_b, future_b)
            future_maps = enc_model._encode_staged_maps(future_norm)
            W_fut = future_norm.shape[-1]
            g_norm = enc_model._get_guidance_forecast_norm(past_b, past_norm, norm_stats, W_fut)
            g_maps = enc_model._encode_staged_maps(g_norm)
        viz_out["viz"]["viz/itrans_finetune/dataset_1d"] = [
            _plot_realts_1d_pre_post_norm(
                past=past_cf, future=future_cf,
                past_norm=past_norm[0].cpu(), future_norm=future_norm[0].cpu(),
                sample_index=sample_index, output_dir=output_dir,
                lookback_length=state.lookback_length, jpeg_dpi=jpeg_dpi,
                variables_to_plot=variables_to_plot,
            )
        ]
        viz_out["viz"]["viz/itrans_finetune/dataset_2d"] = [
            _plot_staged_2d_maps_native(
                maps={k: v.detach().cpu() for k, v in future_maps.items() if k in {"coarse", "fine"}},
                sample_index=sample_index, output_dir=output_dir, tag="dataset_sample",
                variables_to_plot=variables_to_plot, jpeg_dpi=jpeg_dpi,
            )
        ]
        viz_out["viz"]["viz/itrans_finetune/itrans_2d"] = [
            _plot_staged_2d_maps_native(
                maps={k: v.detach().cpu() for k, v in g_maps.items() if k in {"coarse", "fine"}},
                sample_index=sample_index, output_dir=output_dir, tag="itrans_pred",
                variables_to_plot=variables_to_plot, jpeg_dpi=jpeg_dpi,
            )
        ]
        viz_out["viz"]["viz/itrans_finetune/itrans_1d"] = [
            _plot_itrans_window_norm_1d(
                past_norm=past_norm[0].cpu(), future_norm=future_norm[0].cpu(),
                guidance_norm=g_norm[0].cpu(),
                sample_index=sample_index, output_dir=output_dir,
                lookback_length=state.lookback_length, jpeg_dpi=jpeg_dpi,
                variables_to_plot=variables_to_plot,
            )
        ]
    except Exception as exc:
        logger.warning("iTrans finetune diagnostic maps skipped: %s", exc)

    return viz_out


def decode_staged_anchor_components(
    fine_model,
    coarse_out: Dict[str, Any],
    fine_out: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode coarse, fine residual, and combined anchor predictions to numpy."""
    from models.diffusion_tsf.pipeline.phases.staged_eval import _staged_anchor_global_norm

    final = _staged_anchor_global_norm(fine_model, coarse_out, fine_out)
    coarse_2d = coarse_out["future_2d_coarse"]
    fine_2d = fine_out["future_2d_fine"]
    B, V = coarse_2d.shape[:2]
    BV = B * V
    to_2d = fine_model.to_2d
    coarse_flat = coarse_2d.reshape(BV, 1, coarse_2d.shape[-2], coarse_2d.shape[-1])
    fine_flat = fine_2d.reshape(BV, 1, fine_2d.shape[-2], fine_2d.shape[-1])
    coarse_1d = to_2d._decode_occupancy_in_range(
        coarse_flat, value_range=to_2d.max_scale, cdf_decoder="mean",
    )
    fine_1d = to_2d._decode_occupancy_in_range(
        fine_flat, value_range=_staged_fine_value_range(fine_model), cdf_decoder="mean",
    )
    coarse_np = coarse_1d.reshape(B, V, -1).detach().cpu().numpy()
    fine_np = fine_1d.reshape(B, V, -1).detach().cpu().numpy()
    if getattr(fine_model.config, "coarse_flatline_blur_fine_target", False):
        combined = torch.from_numpy(coarse_np + fine_np).to(device=coarse_1d.device, dtype=coarse_1d.dtype)
        coarse_t = coarse_1d.reshape(B, V, -1)
        blurred = fine_model._blur_coarse_1d(coarse_t, flatline_source=combined)
        coarse_np = blurred.detach().cpu().numpy()
    k = fine_model._overlap_repr_cols()
    if k > 0:
        coarse_np = coarse_np[..., k:]
        fine_np = fine_np[..., k:]
    if int(getattr(fine_model.config, "representation_time_stride", 1)) > 1:
        import torch
        coarse_np = fine_model._upsample_repr_to_raw_horizon(torch.from_numpy(coarse_np)).numpy()
        fine_np = fine_model._upsample_repr_to_raw_horizon(torch.from_numpy(fine_np)).numpy()
    return coarse_np, fine_np, final


def per_window_anchor_mse(y_true: np.ndarray, det: np.ndarray) -> np.ndarray:
    return ((y_true - det) ** 2).mean(axis=(1, 2))


def per_window_crps(y_true: np.ndarray, samples: np.ndarray, *, chunk: int = 32) -> np.ndarray:
    batch = y_true.shape[0]
    out = np.empty(batch, dtype=np.float64)
    for start in range(0, batch, chunk):
        end = min(start + chunk, batch)
        yt = y_true[start:end]
        ss = samples[start:end].astype(np.float64)
        term1 = np.abs(ss - yt[:, :, None, :]).mean(axis=2)
        term2 = np.abs(ss[:, :, :, None, :] - ss[:, :, None, :, :]).mean(axis=(2, 3))
        out[start:end] = (term1 - 0.5 * term2).mean(axis=(1, 2))
    return out


def plot_worst_window_panel(
    *,
    past: torch.Tensor,
    future: torch.Tensor,
    coarse_pred: np.ndarray,
    fine_pred: np.ndarray,
    final_pred: np.ndarray,
    metric: str,
    rank: int,
    window_index: int,
    score: float,
    output_dir: str,
    jpeg_dpi: int = 100,
) -> str:
    """GT vs coarse/fine/final for a worst-error eval window."""
    os.makedirs(output_dir, exist_ok=True)
    past_cf, future_cf = _as_channel_first(past, future)
    gt = future_cf.numpy()
    common_len = min(
        gt.shape[-1],
        coarse_pred.shape[-1],
        fine_pred.shape[-1],
        final_pred.shape[-1],
    )
    gt = gt[..., -common_len:]
    coarse_pred = coarse_pred[..., -common_len:]
    fine_pred = fine_pred[..., -common_len:]
    final_pred = final_pred[..., -common_len:]
    H = common_len
    t_axis = np.arange(0, H)
    n_vars = min(3, gt.shape[0])

    fig, axes = plt.subplots(n_vars, 1, figsize=(6.5, 2.4 * n_vars), squeeze=False, constrained_layout=True)
    for v in range(n_vars):
        ax = axes[v, 0]
        ax.plot(t_axis, gt[v], color="#2196F3", linewidth=1.5, label="GT")
        ax.plot(t_axis, coarse_pred[v], color="#FF9800", linewidth=1.1, alpha=0.85, label="Coarse")
        ax.plot(t_axis, fine_pred[v], color="#4CAF50", linewidth=1.0, alpha=0.85, label="Fine")
        ax.plot(t_axis, final_pred[v], color="#E91E63", linewidth=1.2, label="Final")
        ax.grid(True, alpha=0.12)
        ax.set_title(f"var {v}", fontsize=9)
        if v == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"worst {metric} rank {rank} | window {window_index} | score={score:.5f}\n"
        + _format_scale_banner(norm_range=(float(gt.min()), float(gt.max())), space_label="window-norm"),
        fontsize=10,
    )
    path = os.path.join(output_dir, f"{metric}_rank{rank:02d}_win{window_index}.jpg")
    return save_figure_jpg(fig, path, dpi=jpeg_dpi)


def run_eval_worst_window_visualizations(
    state: Any,
    *,
    test_ds,
    pack: Dict[str, np.ndarray],
    worst_manifest: List[Dict[str, Any]],
) -> List[str]:
    """Generate GT vs pred panels for worst-window manifest entries."""
    viz = _viz_cfg(state)
    if not viz.get("enabled", True) or not worst_manifest:
        return []

    output_dir = os.path.join(state.results_dir, "viz", "eval_worst")
    paths: List[str] = []
    coarse = pack.get("coarse_anchor")
    fine = pack.get("fine_anchor")
    final = pack.get("final_anchor", pack.get("deterministic"))
    if coarse is None or fine is None or final is None:
        return []

    for entry in worst_manifest:
        wi = int(entry["window_index"])
        past, future = test_ds[wi]
        paths.append(
            plot_worst_window_panel(
                past=past,
                future=future,
                coarse_pred=coarse[wi],
                fine_pred=fine[wi],
                final_pred=final[wi],
                metric=str(entry["metric"]),
                rank=int(entry["rank"]),
                window_index=wi,
                score=float(entry["score"]),
                output_dir=output_dir,
                jpeg_dpi=int(viz.get("jpeg_dpi", 100)),
            )
        )
    return paths
