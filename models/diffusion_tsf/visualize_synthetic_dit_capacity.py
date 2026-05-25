"""
Visualize synthetic DiT capacity checkpoints on random synthetic windows.

This is the synthetic analogue of the repo's comparison scripts: it overlays
ground truth, iTransformer baseline (if available), and all discovered DiT
variants on several random forecast windows, plus a side panel with extra
windows from the same task family so the raw data shape is visible.

By default it discovers the newest per-variant checkpoints under `results/`.
Because the original pulled runs did not save checkpoints, this script is most
useful after rerunning the updated trainer with `--checkpoint-dir`.

Usage:
    python -m models.diffusion_tsf.visualize_synthetic_dit_capacity \
        --output results/viz/synthetic_dit_capacity/latest.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_synthetic_dit_capacity import (
    RunConfig,
    SyntheticTaskDataset,
    VARIANTS,
    VariantSpec,
    build_model,
    create_itransformer,
)


VARIANT_COLORS = {
    "dit_tiny_no_guidance": "#1f77b4",
    "dit_default_no_guidance": "#ff7f0e",
    "dit_large_no_guidance": "#2ca02c",
    "dit_default_with_guidance": "#d62728",
}


def parse_variant_spec(raw: Dict) -> VariantSpec:
    patch = tuple(raw["dit_patch_size"]) if isinstance(raw.get("dit_patch_size"), list) else raw["dit_patch_size"]
    return VariantSpec(
        name=raw["name"],
        dit_embed_dim=raw["dit_embed_dim"],
        dit_depth=raw["dit_depth"],
        dit_num_heads=raw["dit_num_heads"],
        use_guidance=raw["use_guidance"],
        dit_patch_size=patch,
        dit_mlp_ratio=raw.get("dit_mlp_ratio", 4.0),
    )


def parse_run_config(raw: Dict) -> RunConfig:
    return RunConfig(**raw)


def discover_latest_variant_checkpoints(results_root: Path, run_dirs: Optional[List[str]] = None) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    search_roots = [results_root / rd for rd in run_dirs] if run_dirs else [results_root]
    for variant in VARIANTS:
        candidates: List[Path] = []
        for root in search_roots:
            if root.is_dir() and (root / "ckpts").is_dir():
                p = root / "ckpts" / f"{variant}.pt"
                if p.exists():
                    candidates.append(p)
            if root.is_dir():
                candidates.extend(root.glob(f"*-synth-dit-*/ckpts/{variant}.pt"))
        if candidates:
            discovered[variant] = max(candidates, key=lambda p: p.stat().st_mtime)
    return discovered


def load_variant_models(
    checkpoint_paths: Dict[str, Path],
    device: torch.device,
) -> Tuple[Dict[str, torch.nn.Module], RunConfig, Optional[iTransformerGuidance]]:
    models: Dict[str, torch.nn.Module] = {}
    baseline_guidance: Optional[iTransformerGuidance] = None
    run_cfg: Optional[RunConfig] = None

    for variant, ckpt_path in checkpoint_paths.items():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        spec = parse_variant_spec(payload["spec"])
        local_run_cfg = parse_run_config(payload["run_config"])
        if run_cfg is None:
            run_cfg = local_run_cfg

        model = build_model(spec, local_run_cfg, device)
        if spec.use_guidance:
            itrans_ckpt = ckpt_path.with_name(f"{variant}_itransformer.pt")
            if not itrans_ckpt.exists():
                raise FileNotFoundError(
                    f"Guided variant checkpoint found but missing baseline iTransformer: {itrans_ckpt}"
                )
            itrans = create_itransformer(
                seq_len=local_run_cfg.lookback,
                pred_len=local_run_cfg.horizon,
                num_vars=1,
                dropout=0.1,
            ).to(device)
            itrans.load_state_dict(torch.load(itrans_ckpt, map_location=device, weights_only=False))
            guidance = iTransformerGuidance(itrans)
            model.set_guidance_model(guidance)
            baseline_guidance = guidance

        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models[variant] = model

    if run_cfg is None:
        raise RuntimeError("No checkpoints discovered.")
    return models, run_cfg, baseline_guidance


def sample_task_names(num_samples: int, task_mode: str, seed: int) -> List[str]:
    rng = random.Random(seed)
    if task_mode == "linear":
        return [SyntheticTaskDataset.TASK_LINEAR] * num_samples
    if task_mode == "periodic":
        return [SyntheticTaskDataset.TASK_PERIODIC] * num_samples
    return [
        rng.choice([SyntheticTaskDataset.TASK_LINEAR, SyntheticTaskDataset.TASK_PERIODIC])
        for _ in range(num_samples)
    ]


def make_window(task: str, run_cfg: RunConfig, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = SyntheticTaskDataset(
        n_samples=1,
        lookback=run_cfg.lookback,
        horizon=run_cfg.horizon,
        lookback_overlap=run_cfg.lookback_overlap,
        task=task,
        seed=seed,
    )
    return ds[0]


@torch.no_grad()
def run_baseline(
    baseline_guidance: Optional[iTransformerGuidance],
    past: torch.Tensor,
    horizon: int,
) -> Optional[torch.Tensor]:
    if baseline_guidance is None:
        return None
    out = baseline_guidance.get_forecast(past.unsqueeze(0), horizon)
    return out.squeeze(0).cpu()


@torch.no_grad()
def run_variant(model: torch.nn.Module, past: torch.Tensor, run_cfg: RunConfig, device: torch.device) -> torch.Tensor:
    out = model.generate(
        past.unsqueeze(0).to(device),
        use_ddim=True,
        num_ddim_steps=run_cfg.ddim_steps,
        cfg_scale=run_cfg.cfg_scale,
    )
    pred = out["prediction"].squeeze(0).cpu()
    return pred


def plot_forecast_row(
    ax,
    past: torch.Tensor,
    future: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    baseline_pred: Optional[torch.Tensor],
    task: str,
    run_cfg: RunConfig,
) -> None:
    t_past = np.arange(-run_cfg.lookback, 0)
    t_future = np.arange(0, run_cfg.horizon)
    gt = future[:, run_cfg.lookback_overlap :]

    ax.plot(t_past, past[0].numpy(), color="0.6", linewidth=1.0, label="Context")
    ax.plot(t_future, gt[0].numpy(), color="black", linewidth=1.8, label="Ground Truth")
    if baseline_pred is not None:
        ax.plot(
            t_future,
            baseline_pred[0].numpy(),
            color="#9467bd",
            linestyle="--",
            linewidth=1.3,
            label="iTransformer baseline",
        )
    for variant, pred in preds.items():
        ax.plot(
            t_future,
            pred[0].numpy(),
            color=VARIANT_COLORS.get(variant, None),
            linewidth=1.2,
            label=variant,
        )
    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.set_title(f"{task} sample")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Value")


def plot_extra_windows(ax, task: str, run_cfg: RunConfig, seed: int, n_extra: int) -> None:
    for i in range(n_extra):
        past, future = make_window(task, run_cfg, seed + 1000 + i)
        full = torch.cat([past[0], future[0, run_cfg.lookback_overlap :]], dim=-1)
        t = np.arange(full.shape[-1])
        ax.plot(t, full.numpy(), linewidth=1.0, alpha=0.85, label=f"window {i+1}")
    ax.axvline(run_cfg.lookback, color="black", linestyle=":", alpha=0.3)
    ax.set_title("Extra windows from same task")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")


def make_figure(
    models: Dict[str, torch.nn.Module],
    run_cfg: RunConfig,
    baseline_guidance: Optional[iTransformerGuidance],
    output_path: Path,
    num_samples: int,
    task_mode: str,
    seed: int,
    n_extra_windows: int,
    device: torch.device,
) -> None:
    task_names = sample_task_names(num_samples, task_mode, seed)
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4.5 * num_samples), squeeze=False)

    for row, task in enumerate(task_names):
        past, future = make_window(task, run_cfg, seed + row * 17)
        preds = {name: run_variant(model, past, run_cfg, device) for name, model in models.items()}
        baseline_pred = run_baseline(baseline_guidance, past, run_cfg.horizon)
        plot_forecast_row(axes[row, 0], past, future, preds, baseline_pred, task, run_cfg)
        plot_extra_windows(axes[row, 1], task, run_cfg, seed + row * 29, n_extra_windows)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic DiT capacity checkpoints.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-dirs", type=str, default=None, help="Comma-separated run dirs under results/")
    parser.add_argument("--task", type=str, default="mixed", choices=["mixed", "linear", "periodic"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--extra-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/viz/synthetic_dit_capacity/latest_synthetic_dit_capacity.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = Path(args.results_root)
    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()] if args.run_dirs else None
    checkpoint_paths = discover_latest_variant_checkpoints(results_root, run_dirs=run_dirs)
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No synthetic capacity checkpoints found. Re-run training with --checkpoint-dir to populate ckpts/."
        )
    print("Discovered checkpoints:")
    for variant, path in checkpoint_paths.items():
        print(f"  {variant}: {path}")

    models, run_cfg, baseline_guidance = load_variant_models(checkpoint_paths, device)
    make_figure(
        models=models,
        run_cfg=run_cfg,
        baseline_guidance=baseline_guidance,
        output_path=Path(args.output),
        num_samples=args.num_samples,
        task_mode=args.task,
        seed=args.seed,
        n_extra_windows=args.extra_windows,
        device=device,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
