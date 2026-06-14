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


def _viz_cfg(state: Any) -> Dict[str, Any]:
    return visualization_settings(getattr(state, "merged_config", None))


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

    for row, idx in enumerate(indices):
        past, future = dataset[idx]
        past_t = past.unsqueeze(0).to(device)
        B, C, L = past_t.shape
        x_enc = past_t.permute(0, 2, 1)
        seq_sl = getattr(itrans_model, "seq_len", L)
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        x_dec = torch.zeros(B, forecast_length, C, device=device)

        with torch.no_grad():
            out = itrans_model(x_enc, None, x_dec, None)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.permute(0, 2, 1).cpu()[0]

        past_dn = denorm(past, mean_t, std_t)
        future_sliced = future[:, -forecast_length:]
        future_dn = denorm(future_sliced, mean_t, std_t)
        pred_dn = denorm(pred, mean_t, std_t)

        n_vars_plot = min(3, C)
        fig, axes = plt.subplots(
            1, n_vars_plot, figsize=(4.5 * n_vars_plot, 3.0), squeeze=False, constrained_layout=True
        )
        t_past = np.arange(-lookback_length, 0)
        t_future = np.arange(0, forecast_length)

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
):
    from models.diffusion_tsf.train_multivariate_pipeline import (
        create_diffusion_model,
        load_itransformer_from_checkpoint,
        load_diffusion_state_keep_attached_guidance,
    )
    from models.diffusion_tsf.guidance import iTransformerGuidance
    from models.diffusion_tsf.visualize_comparison import (
        apply_checkpoint_architecture,
        infer_anchor_kwargs,
        infer_diffusion_type,
        infer_model_type,
    )

    itrans_guidance_model = load_itransformer_from_checkpoint(str(itrans_ckpt_path), n_vars, device)
    diff_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = diff_ckpt.get("config") or tuned_params or {}
    if isinstance(meta, dict) and "diffusion_params" in meta:
        meta = {**meta, **(meta.get("diffusion_params") or {})}
    diff_type = infer_diffusion_type(diff_ckpt, meta.get("diffusion_type"))
    backbone = infer_model_type(diff_ckpt)
    apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta if isinstance(meta, dict) else {})

    itrans_guidance = iTransformerGuidance(itrans_guidance_model)
    model = create_diffusion_model(
        n_variates=n_vars,
        diffusion_type=diff_type,
        model_type=backbone,
        diffusion_stage=stage,
        guidance_model=itrans_guidance,
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(model, diff_ckpt["model_state_dict"])
    model.eval()
    return model, diff_ckpt


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
        fine_map_full, value_range=to_2d.max_scale / to_2d.height, cdf_decoder="mean"
    ).cpu()
    combined_1d = coarse_1d + fine_1d

    W_past = past_coarse.shape[-1]
    W_fut = future_coarse.shape[-1]
    t_axis = np.arange(-W_past, W_fut)
    gt_full_norm = torch.cat([past, future[:, -W_fut:]], dim=-1)

    gt_full_dn = denorm_cmp(gt_full_norm, mean, std)
    coarse_dn = denorm_cmp(coarse_1d[0], mean, std)
    fine_dn = fine_1d[0] * std.view(-1, 1)
    combined_dn = denorm_cmp(combined_1d[0], mean, std)

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
        ax1.plot(t_axis, gt_full_dn[col].numpy(), color="#2196F3", linewidth=1.6, label="GT")
        ax1.plot(t_axis, coarse_dn[col].numpy(), color="#FF9800", linewidth=1.2, drawstyle="steps-mid", alpha=0.85, label="Coarse")
        ax1.plot(t_axis, combined_dn[col].numpy(), color="#E91E63", linewidth=1.2, label="Combined")
        ax1.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        ax1.grid(True, alpha=0.12)
        ax1.set_title(f"Var {col} 1D", fontsize=9)
        if col == 0:
            ax1.legend(loc="lower left", fontsize=6)

        ax2 = axes[1, col]
        ax2.plot(t_axis, fine_dn[col].numpy(), color="#4CAF50", linewidth=1.2)
        ax2.axhline(y=0, color="grey", linestyle="--", alpha=0.4)
        ax2.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        ax2.grid(True, alpha=0.12)
        ax2.set_title("Fine residual", fontsize=9)

        ax3 = axes[2, col]
        im3 = ax3.imshow(
            coarse_map_full[0, col].numpy(), aspect="auto", origin="lower",
            extent=[-W_past, W_fut, 0, applied_h], cmap="plasma",
        )
        ax3.axvline(x=0, color="white", linestyle="--", alpha=0.5)
        ax3.set_title("Coarse 2D", fontsize=9)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = axes[3, col]
        im4 = ax4.imshow(
            fine_map_full[0, col].numpy(), aspect="auto", origin="lower",
            extent=[-W_past, W_fut, 0, applied_h], cmap="plasma",
        )
        ax4.axvline(x=0, color="white", linestyle="--", alpha=0.5)
        ax4.set_title("Fine 2D", fontsize=9)
        ax4.set_xlabel("t")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{dataset_name} sample {sample_index} | {tag}",
        fontsize=11, fontweight="bold",
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


def generate_dual_scale_comparisons(
    *,
    diff_ckpt_path: str,
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
    tag: str = "dual_scale",
) -> List[str]:
    """Dual-scale CDF visualization (coarse/fine/combined + 2D maps)."""
    from models.diffusion_tsf.train_multivariate_pipeline import (
        create_diffusion_model,
        load_dataset,
        load_itransformer_from_checkpoint,
        load_diffusion_state_keep_attached_guidance,
    )
    from models.diffusion_tsf.guidance import iTransformerGuidance
    from models.diffusion_tsf.visualize_comparison import (
        apply_checkpoint_architecture,
        infer_anchor_kwargs,
        infer_diffusion_type,
        infer_model_type,
    )

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

    itrans_guidance_model = load_itransformer_from_checkpoint(str(itrans_ckpt_path), n_vars, device)
    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    meta = diff_ckpt.get("config") or tuned_params or {}
    diff_type = infer_diffusion_type(diff_ckpt, meta.get("diffusion_type"))
    backbone = infer_model_type(diff_ckpt)
    applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta if isinstance(meta, dict) else {})

    itrans_guidance = iTransformerGuidance(itrans_guidance_model)
    diff_model = create_diffusion_model(
        n_variates=n_vars,
        diffusion_type=diff_type,
        model_type=backbone,
        guidance_model=itrans_guidance,
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(diff_model, diff_ckpt["model_state_dict"])
    diff_model.eval()

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for sample_index in sample_indices:
        past, future = test_ds[sample_index]
        past_t = past.unsqueeze(0).to(device)

        with torch.no_grad():
            res = diff_model.generate(
                past_t,
                sampler=diffusion_sampler,
                num_inference_steps=num_inference_steps,
            )

        if "past_2d_coarse" not in res:
            logger.warning("Model is not dual-scale; skipping dual-scale viz for sample %s", sample_index)
            continue

        saved.append(_plot_dual_scale_sample(
            res=res,
            diff_model=diff_model,
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

    if not diff_ckpt_path and not fine_ckpt:
        return []

    return generate_dual_scale_comparisons(
        diff_ckpt_path=diff_ckpt_path or fine_ckpt,
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
