"""Phase-start diagnostics: architecture, loss config, dataset stats, checkpoint metadata."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from models.diffusion_tsf.pipeline.logging_utils import get_diagnostic_logger

logger = get_diagnostic_logger()


def _layer_description(module: nn.Module) -> str:
    name = module.__class__.__name__
    if isinstance(module, nn.Linear):
        return f"Linear(in={module.in_features}, out={module.out_features})"
    if isinstance(module, nn.Conv2d):
        return (
            f"Conv2d({module.in_channels}->{module.out_channels}, "
            f"k={module.kernel_size}, s={module.stride})"
        )
    if isinstance(module, nn.LayerNorm):
        return f"LayerNorm({module.normalized_shape})"
    if isinstance(module, nn.Embedding):
        return f"Embedding({module.num_embeddings}, {module.embedding_dim})"
    if isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.Sigmoid)):
        return name
    if isinstance(module, nn.Dropout):
        return f"Dropout(p={module.p})"
    return name


def _denoiser_root(model: Any) -> nn.Module:
    if hasattr(model, "noise_predictor"):
        return model.noise_predictor
    if hasattr(model, "model"):
        return model.model
    return model


def architecture_summary_string(model: Any, *, max_leaf_modules: int = 40) -> str:
    """Compact layer chain for wandb summary / config."""
    root = _denoiser_root(model)
    leaf_mods = [
        m for _, m in root.named_modules()
        if _ != "" and len(list(m.children())) == 0
    ]
    if not leaf_mods:
        return root.__class__.__name__
    chain = " > ".join(_layer_description(m) for m in leaf_mods[:max_leaf_modules])
    if len(leaf_mods) > max_leaf_modules:
        chain += f" ... (+{len(leaf_mods) - max_leaf_modules} more)"
    return chain


def log_model_architecture(model: Any, *, label: str = "model") -> str:
    """Emit a layer-by-layer description of the denoiser subtree."""
    root = _denoiser_root(model)

    lines: List[str] = []
    for mod_name, module in root.named_modules():
        if mod_name == "":
            continue
        if len(list(module.children())) > 0:
            continue
        desc = _layer_description(module)
        lines.append(f"{mod_name}: {desc}")

    logger.info("[%s] PyTorch architecture (%d leaf modules):", label, len(lines))
    for line in lines:
        logger.debug("  %s", line)
    chain = architecture_summary_string(model)
    if lines:
        logger.info("[%s] layer chain: %s", label, chain)
    return chain


def training_loss_config_dict(model: Any) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {"training_loss": "unknown (no model.config)"}

    diffusion_type = getattr(cfg, "diffusion_type", "binary")
    loss_weighting = getattr(cfg, "loss_weighting", "none")
    pred_target = getattr(cfg, "prediction_target", "x0")
    stage = getattr(cfg, "diffusion_stage", None)

    out: Dict[str, Any] = {
        "diffusion_type": str(diffusion_type),
        "loss_weighting": str(loss_weighting),
        "prediction_target": str(pred_target),
        "regular_loss": "loss_x0 + loss_zt (BCE on clean-bit + flip-mask heads)",
    }
    if stage:
        out["diffusion_stage"] = str(stage)

    if getattr(cfg, "use_deterministic_anchor_loss", False):
        lam = float(getattr(cfg, "deterministic_anchor_lambda", 0.99))
        out["deterministic_anchor_lambda"] = lam
        out["anchor_loss_weight"] = 1.0 - lam
        out["combined_loss"] = (
            f"{lam:.4f} * regular_loss + {1.0 - lam:.4f} * anchor_BCE "
            f"(stationary_flat anchor at t=T-1)"
        )
    else:
        out["combined_loss"] = "regular_loss (anchor disabled)"

    return out


def log_training_loss_config(model: Any, state: Any = None) -> Dict[str, Any]:
    info = training_loss_config_dict(model)
    if info.get("training_loss"):
        logger.info("training loss: %s", info["training_loss"])
        return info

    parts = [
        f"diffusion_type={info['diffusion_type']}",
        f"loss_weighting={info['loss_weighting']}",
        info["regular_loss"],
        f"combined = {info['combined_loss']}",
        f"prediction_target={info['prediction_target']}",
    ]
    if "diffusion_stage" in info:
        parts.append(f"diffusion_stage={info['diffusion_stage']}")
    logger.info("training loss: %s", " | ".join(parts))
    return info


def log_checkpoint_metadata(
    kind: str,
    path: Optional[str],
    *,
    loaded: bool = True,
    n_variates: Optional[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    exists = bool(path and os.path.exists(path))
    logger.info(
        "[%s] checkpoint loaded=%s exists=%s path=%s",
        kind,
        loaded and exists,
        exists,
        os.path.abspath(path) if path else None,
    )
    dims = []
    if n_variates is not None:
        dims.append(f"n_variates={n_variates}")
    if lookback is not None:
        dims.append(f"lookback={lookback}")
    if horizon is not None:
        dims.append(f"horizon={horizon}")
    if dims:
        logger.info("[%s] dimensions: %s", kind, ", ".join(dims))
    if extra:
        for k, v in extra.items():
            logger.debug("[%s] %s=%s", kind, k, v)


def compute_dataset_stats(
    dataset,
    *,
    prefix: str = "dataset",
    n_probe: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mean/std/min/max/quartiles per variate on pre-normalized windows."""
    from models.diffusion_tsf.pipeline.visualize_utils import (
        _as_channel_first,
        _unpack_past_future,
        pick_sample_indices,
    )

    n = len(dataset)
    if n <= 0:
        return {f"{prefix}_n_variates": 0}

    indices = pick_sample_indices(n, n_probe, seed=seed)
    per_var_values: List[List[List[float]]] = []

    for idx in indices:
        past, future = _unpack_past_future(dataset[idx])
        past, future = _as_channel_first(past, future)
        seq = torch.cat([past, future], dim=-1)
        per_var_values.append(seq.numpy().tolist())

    n_vars = len(per_var_values[0])
    out: Dict[str, Any] = {
        f"{prefix}_n_variates": n_vars,
        f"{prefix}_stats_n_probe": len(indices),
        f"{prefix}_variate_means": [],
        f"{prefix}_variate_stds": [],
        f"{prefix}_variate_mins": [],
        f"{prefix}_variate_maxs": [],
        f"{prefix}_variate_q25": [],
        f"{prefix}_variate_q50": [],
        f"{prefix}_variate_q75": [],
    }

    for v in range(n_vars):
        vals = np.array([row[v] for row in per_var_values], dtype=np.float64).ravel()
        out[f"{prefix}_variate_means"].append(float(vals.mean()))
        out[f"{prefix}_variate_stds"].append(float(vals.std()))
        out[f"{prefix}_variate_mins"].append(float(vals.min()))
        out[f"{prefix}_variate_maxs"].append(float(vals.max()))
        out[f"{prefix}_variate_q25"].append(float(np.percentile(vals, 25)))
        out[f"{prefix}_variate_q50"].append(float(np.percentile(vals, 50)))
        out[f"{prefix}_variate_q75"].append(float(np.percentile(vals, 75)))

    logger.info(
        "[%s] stats from %d windows: n_variates=%d",
        prefix,
        len(indices),
        n_vars,
    )
    for i in range(n_vars):
        logger.debug(
            "[%s] var %d: mean=%.4f std=%.4f min=%.4f max=%.4f q25=%.4f q50=%.4f q75=%.4f",
            prefix,
            i,
            out[f"{prefix}_variate_means"][i],
            out[f"{prefix}_variate_stds"][i],
            out[f"{prefix}_variate_mins"][i],
            out[f"{prefix}_variate_maxs"][i],
            out[f"{prefix}_variate_q25"][i],
            out[f"{prefix}_variate_q50"][i],
            out[f"{prefix}_variate_q75"][i],
        )
    return out


def run_phase_start_diagnostics(
    state: Any,
    *,
    phase_name: str,
    models: Optional[Sequence[Any]] = None,
    model_labels: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[Any]] = None,
    dataset_prefixes: Optional[Sequence[str]] = None,
    ckpt_info: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Orchestrate architecture dump, loss config, dataset stats, and ckpt meta."""
    logger.info("=== phase diagnostics: %s ===", phase_name)
    summary: Dict[str, Any] = {"phase": phase_name}

    if models:
        labels = model_labels or [f"model_{i}" for i in range(len(models))]
        for model, label in zip(models, labels):
            if model is not None:
                arch_key = f"architecture/{label}"
                summary[arch_key] = log_model_architecture(model, label=f"{phase_name}/{label}")
                loss_info = log_training_loss_config(model, state)
                for lk, lv in loss_info.items():
                    summary[f"loss/{label}/{lk}"] = lv

    if datasets:
        prefixes = dataset_prefixes or [f"dataset_{i}" for i in range(len(datasets))]
        n_probe = 32 if getattr(state, "smoke_test", False) else 256
        for ds, prefix in zip(datasets, prefixes):
            if ds is not None and len(ds) > 0:
                stats = compute_dataset_stats(
                    ds,
                    prefix=prefix,
                    n_probe=n_probe,
                    seed=int(getattr(state, "seed", 42)),
                )
                summary.update(stats)

    if ckpt_info:
        for info in ckpt_info:
            log_checkpoint_metadata(
                info.get("kind", "checkpoint"),
                info.get("path"),
                loaded=bool(info.get("loaded", True)),
                n_variates=info.get("n_variates"),
                lookback=info.get("lookback"),
                horizon=info.get("horizon"),
                extra=info.get("extra"),
            )

    return summary


def select_spaced_top_k(
    scores: np.ndarray,
    series_starts: np.ndarray,
    *,
    k: int = 10,
    min_spacing: int = 48,
) -> List[int]:
    """Greedy top-k by score with minimum spacing on series start positions."""
    order = np.argsort(-scores)
    selected: List[int] = []
    for idx in order:
        start = int(series_starts[idx])
        if all(abs(start - int(series_starts[j])) >= min_spacing for j in selected):
            selected.append(int(idx))
        if len(selected) >= k:
            break
    return selected
