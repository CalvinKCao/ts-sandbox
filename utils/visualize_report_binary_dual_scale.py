#!/usr/bin/env python3
"""Batch viz for CFG-ablation / matrix report: forecast panels + 2D denoise snapshots.

For each dataset (12 in the combined report), plots 1–2 test windows with:
  - GT, iTrans guidance, optional full baseline, anchor, N probabilistic dpmpp samples
  - 2D coarse/fine heatmaps at denoise timesteps (default t=999,500,250,100,50,0)

Example:
  python utils/visualize_report_binary_dual_scale.py --smoke-test
  python utils/visualize_report_binary_dual_scale.py \\
    --output-dir reports/06-01_cfg_ablation_mmpd_matrix_combined
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    dataset_window_lengths,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
)
from models.diffusion_tsf.visualize_comparison import (
    apply_checkpoint_architecture,
    choose_extra_indices,
    denorm,
    infer_anchor_kwargs,
    infer_diffusion_type,
    infer_model_type,
)
from utils.visualize_binary_dual_scale_forecast import (
    _itrans_forward,
    _load_subset_bundle,
    _resolve_itrans_paths,
)

REPORT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "PeMS",
    "dalia",
    "electricity",
    "exchange_rate",
    "illness",
    "solar_Alabama",
    "traffic",
    "weather",
]

CFG_ABLATION_DATASETS = [
    "ETTh1",
    "ETTh2",
    "exchange_rate",
    "weather",
    "traffic",
    "PeMS",
    "dalia",
]


def pick_ckpt_dir(ckpt_root: Path, dataset: str) -> Path:
    direct = ckpt_root / f"05-31-3828089-{dataset}-binary_dual_scale"
    if direct.is_dir() and any(direct.glob("*/best.pt")):
        return direct
    best_dir = ""
    best_mtime = 0
    for d in ckpt_root.glob(f"*-{dataset}-binary_dual_scale"):
        if not d.is_dir():
            continue
        m = d.stat().st_mtime
        if m > best_mtime and any(d.glob("*/best.pt")):
            best_mtime = m
            best_dir = str(d)
    if not best_dir:
        raise FileNotFoundError(f"No binary_dual_scale ckpt for {dataset} under {ckpt_root}")
    return Path(best_dir)


def clamp_timesteps(requested: Sequence[int], num_training_steps: int) -> List[int]:
    """Map user steps (e.g. 1000) to valid indices and ensure 0 is included last."""
    out: List[int] = []
    for t in requested:
        if t >= num_training_steps:
            out.append(num_training_steps - 1)
        else:
            out.append(max(0, int(t)))
    if 0 not in out:
        out.append(0)
    return sorted(set(out), reverse=True)


def build_reverse_schedule(
    requested: Sequence[int],
    num_training_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    steps = clamp_timesteps(requested, num_training_steps)
    reverse = torch.tensor(steps, dtype=torch.long, device=device)
    snapshots = tuple(t for t in steps if t > 0 or len(steps) == 1)
    if 0 not in snapshots:
        snapshots = (*snapshots, 0)
    return reverse, snapshots


def plot_2d_denoise_panel(
    intermediates: List[Tuple[int, torch.Tensor]],
    variate: int,
    out_path: Path,
    dataset: str,
    test_index: int,
    cfg_label: str = "",
) -> None:
    """intermediates: list of (t, tensor B,V,2,H,W)."""
    if not intermediates:
        return
    n_steps = len(intermediates)
    fig, axes = plt.subplots(
        n_steps,
        2,
        figsize=(7, 2.2 * n_steps),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (t_step, tensor) in enumerate(intermediates):
        img = tensor[0, variate].cpu().numpy()
        for col, scale_name in enumerate(("coarse", "fine")):
            ax = axes[row, col]
            ax.imshow(img[col], aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"t={t_step} {scale_name}", fontsize=9)
            ax.axis("off")
    title = f"{dataset} — 2D denoise (var {variate}, test idx {test_index})"
    if cfg_label:
        title += f" | {cfg_label}"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def cfg_scale_label(cfg_scale: Optional[float]) -> str:
    from models.diffusion_tsf.cfg_inference import cfg_mix_applies

    if cfg_scale is None or not cfg_mix_applies(float(cfg_scale)):
        return "CFG off"
    return f"CFG w={float(cfg_scale):g}"


def cfg_viz_subdir(cfg_scale: Optional[float]) -> str:
    from models.diffusion_tsf.cfg_inference import cfg_mix_applies

    if cfg_scale is None or not cfg_mix_applies(float(cfg_scale)):
        return "viz"
    s = f"{float(cfg_scale):g}".replace(".", "p")
    return f"viz_cfg{s}"


def cfg_generate_kwargs(cfg_scale: Optional[float]) -> dict:
    from models.diffusion_tsf.cfg_inference import cfg_mix_applies

    if cfg_scale is not None and cfg_mix_applies(float(cfg_scale)):
        return {"cfg_scale": float(cfg_scale)}
    return {}


def load_diffusion_bundle(
    checkpoint_dir: Path,
    dataset: str,
    device: torch.device,
    cfg_scale: Optional[float] = None,
) -> dict:
    sub = _load_subset_bundle(checkpoint_dir, dataset)
    subset_id = sub["subset_id"]
    n_vars = len(sub["variate_indices"])
    lookback, horizon = dataset_window_lengths(dataset)

    guidance_path, full_path = _resolve_itrans_paths(checkpoint_dir, subset_id)
    if guidance_path is None:
        raise FileNotFoundError(f"Missing {subset_id}_itransformer_finetuned.pt")

    guidance_model = load_itransformer_from_checkpoint(str(guidance_path), n_vars, device)
    full_model = None
    if full_path is not None:
        full_model = load_itransformer_from_checkpoint(str(full_path), n_vars, device)

    ckpt = torch.load(sub["best_pt"], map_location=device, weights_only=False)
    diff_type = infer_diffusion_type(ckpt, None)
    backbone = infer_model_type(ckpt, None)
    apply_checkpoint_architecture(ckpt, diff_type, None)
    anchor_kwargs = infer_anchor_kwargs(ckpt, sub["metadata"])
    diff_model = create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        diffusion_type=diff_type,
        model_type=backbone,
        guidance_model=iTransformerGuidance(guidance_model),
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(diff_model, ckpt["model_state_dict"])
    from models.diffusion_tsf.cfg_inference import cfg_mix_applies

    if cfg_scale is not None and cfg_mix_applies(float(cfg_scale)):
        diff_model.config.use_cfg_inference = True
        diff_model.config.cfg_scale = float(cfg_scale)
    diff_model.eval()

    return {
        "sub": sub,
        "subset_id": subset_id,
        "lookback": lookback,
        "horizon": horizon,
        "guidance_model": guidance_model,
        "full_model": full_model,
        "diff_model": diff_model,
    }


def visualize_sample(
    bundle: dict,
    dataset: str,
    test_ds,
    mean: torch.Tensor,
    std: torch.Tensor,
    test_index: int,
    output_dir: Path,
    prob_samples: int,
    prob_steps: int,
    denoise_timesteps: Sequence[int],
    seed: int,
    device: torch.device,
    plot_variate: int,
    cfg_scale: Optional[float] = None,
) -> List[Path]:
    saved: List[Path] = []
    sub = bundle["sub"]
    subset_id = bundle["subset_id"]
    horizon = bundle["horizon"]
    lookback = bundle["lookback"]
    n_vars = len(sub["variate_indices"])
    names = sub["variate_names"] or [f"v{i}" for i in range(n_vars)]
    diff_model = bundle["diff_model"]
    cfg_kw = cfg_generate_kwargs(cfg_scale)
    cfg_label = cfg_scale_label(cfg_scale)

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)
    future_slice = future[:, -horizon:]

    with torch.no_grad():
        guidance_pred = _itrans_forward(bundle["guidance_model"], past_t, horizon, device)
        full_pred = (
            _itrans_forward(bundle["full_model"], past_t, horizon, device)
            if bundle["full_model"] is not None
            else None
        )

        torch.manual_seed(seed + test_index)
        anchor_out = diff_model.generate(
            past_t, sampler="anchor", num_inference_steps=prob_steps, **cfg_kw,
        )
        anchor_pred = anchor_out.get(
            "prediction_global_norm", anchor_out["prediction"]
        ).cpu()[0]

        prob_preds: List[torch.Tensor] = []
        for k in range(prob_samples):
            torch.manual_seed(seed + 10_000 + test_index * prob_samples + k)
            out = diff_model.generate(
                past_t,
                sampler="dpmpp",
                num_inference_steps=prob_steps,
                **cfg_kw,
            )
            prob_preds.append(
                out.get("prediction_global_norm", out["prediction"]).cpu()[0]
            )

        T = int(diff_model.config.binary_num_steps)
        reverse_idx, snapshots = build_reverse_schedule(denoise_timesteps, T, device)
        torch.manual_seed(seed + 20_000 + test_index)
        denoise_out = diff_model.generate(
            past_t,
            sampler="dpmpp",
            num_inference_steps=max(1, len(reverse_idx) - 1),
            yield_intermediates=True,
            reverse_step_indices=reverse_idx,
            snapshot_timesteps=snapshots,
            **cfg_kw,
        )
        intermediates = denoise_out.get("intermediates", [])

    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, horizon)

    fig, axes = plt.subplots(
        n_vars,
        1,
        figsize=(11, 2.0 * n_vars),
        squeeze=False,
        constrained_layout=True,
    )
    for v in range(n_vars):
        ax = axes[v, 0]
        past_dn = denorm(past, mean, std)[v].numpy()
        gt = denorm(future_slice, mean, std)[v].numpy()
        gdn = denorm(guidance_pred, mean, std)[v].numpy()
        adn = denorm(anchor_pred[:, -horizon:], mean, std)[v].numpy()

        ax.plot(t_past, past_dn[-context_len:], color="#9E9E9E", lw=0.9, alpha=0.6)
        ax.plot(t_future, gt, color="#2196F3", lw=1.8, label="Ground truth")
        ax.plot(
            t_future,
            gdn,
            color="#FF9800",
            lw=1.2,
            ls="--",
            label="iTrans guidance",
        )
        if full_pred is not None:
            fdn = denorm(full_pred[:, -horizon:], mean, std)[v].numpy()
            ax.plot(
                t_future,
                fdn,
                color="#4CAF50",
                lw=1.2,
                ls="-.",
                label="iTrans full baseline",
            )
        ax.plot(
            t_future,
            adn,
            color="#E91E63",
            lw=1.4,
            label="Anchor",
        )
        for k, pp in enumerate(prob_preds):
            pdn = denorm(pp[:, -horizon:], mean, std)[v].numpy()
            ax.plot(
                t_future,
                pdn,
                color="#F48FB1",
                lw=0.9,
                alpha=0.55,
                label="Prob sample" if v == 0 and k == 0 else "",
            )
        ax.axvline(0, color="k", ls=":", alpha=0.25)
        ax.set_ylabel(names[v] if v < len(names) else f"var {v}", fontsize=9)
        if v == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{dataset} / {subset_id} — test idx {test_index} | {cfg_label} | "
        f"prob=dpmpp×{prob_samples} (steps={prob_steps})",
        fontsize=11,
        fontweight="bold",
    )
    ds_dir = output_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = ds_dir / f"forecast_idx{test_index}.png"
    fig.savefig(forecast_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(forecast_path)

    denoise_path = ds_dir / f"denoise_2d_idx{test_index}_var{plot_variate}.png"
    plot_2d_denoise_panel(
        intermediates,
        plot_variate,
        denoise_path,
        dataset,
        test_index,
        cfg_label=cfg_label,
    )
    if denoise_path.exists():
        saved.append(denoise_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "06-01_cfg_ablation_mmpd_matrix_combined",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(REPORT_DATASETS),
        help="Comma-separated dataset names (default: all 12 in combined report)",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument(
        "--denoise-timesteps",
        type=str,
        default="1000,500,250,100,50",
        help="Training-time indices to snapshot (1000 -> T-1)",
    )
    parser.add_argument("--plot-variate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Inference CFG scale (>1 enables CFG). Omit for CFG-off.",
    )
    parser.add_argument(
        "--cfg-scales",
        type=str,
        default=None,
        help="Comma-separated scales; writes viz_cfg{scale}/ per scale (e.g. 4,10).",
    )
    parser.add_argument(
        "--cfg-ablation",
        action="store_true",
        help="Shorthand: 7 ablation datasets + cfg scales 4,10 -> viz_cfg4/ and viz_cfg10/.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="ETTh1 only, 1 sample, 2 prob draws, fewer denoise steps",
    )
    args = parser.parse_args()

    if args.cfg_ablation:
        if args.cfg_scales is None and args.cfg_scale is None:
            args.cfg_scales = "4,10"
        if args.datasets == ",".join(REPORT_DATASETS):
            args.datasets = ",".join(CFG_ABLATION_DATASETS)

    if args.smoke_test:
        datasets = ["ETTh1"]
        args.samples_per_dataset = 1
        args.prob_samples = 2
        denoise_ts = [999, 250, 0]
        cfg_scale_list = [4.0] if args.cfg_ablation or args.cfg_scales else [args.cfg_scale]
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        denoise_ts = [int(x) for x in args.denoise_timesteps.split(",")]
        if args.cfg_scales:
            cfg_scale_list = [float(x) for x in args.cfg_scales.split(",") if x.strip()]
        elif args.cfg_scale is not None:
            cfg_scale_list = [float(args.cfg_scale)]
        else:
            cfg_scale_list = [None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = args.output_dir.resolve()

    for cfg_scale in cfg_scale_list:
        viz_root = output_base / cfg_viz_subdir(cfg_scale)
        manifest: Dict[str, List[str]] = {}
        cfg_tag = cfg_scale_label(cfg_scale).replace("CFG ", "")
        print(f"device={device}  out={viz_root}  {cfg_tag}  datasets={datasets}", flush=True)

        for dataset in datasets:
            try:
                ckpt_dir = pick_ckpt_dir(args.ckpt_root.resolve(), dataset)
            except FileNotFoundError as e:
                print(f"[skip] {dataset}: {e}", flush=True)
                continue

            bundle = load_diffusion_bundle(ckpt_dir, dataset, device, cfg_scale=cfg_scale)
            variate_indices = bundle["sub"]["variate_indices"]
            _, _, test_ds, norm_stats = load_dataset(
                dataset,
                variate_indices,
                stride=1,
                test_stride=1,
                lookback=bundle["lookback"],
                horizon=bundle["horizon"],
            )
            n_test = len(test_ds)
            if n_test == 0:
                print(f"[skip] {dataset}: empty test set", flush=True)
                continue

            ds_seed = args.seed + sum((i + 1) * ord(c) for i, c in enumerate(dataset))
            rng = random.Random(ds_seed)
            indices = rng.sample(range(n_test), min(args.samples_per_dataset, n_test))

            mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
            std = torch.tensor(norm_stats["std"], dtype=torch.float32)
            paths: List[str] = []

            for test_index in indices:
                print(f"[viz] {cfg_tag} {dataset} idx={test_index} ...", flush=True)
                out_paths = visualize_sample(
                    bundle,
                    dataset,
                    test_ds,
                    mean,
                    std,
                    test_index,
                    viz_root,
                    args.prob_samples,
                    args.prob_steps,
                    denoise_ts,
                    args.seed,
                    device,
                    args.plot_variate,
                    cfg_scale=cfg_scale,
                )
                paths.extend(str(p) for p in out_paths)

            manifest[dataset] = paths
            del bundle["diff_model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        manifest_path = viz_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
