#!/usr/bin/env python3
"""Plot coarse-band and fine-residual 1D components for staged binary diffusion.

Dual-scale encoding splits each normalized value into a full-range coarse bin plus
an in-bin residual. This script decodes those bands separately for GT and for
coarse/fine model anchor predictions (fine stage conditioned on predicted coarse).

Example:
  python utils/visualize_staged_coarse_fine_components.py \\
    --dataset ETTh1 \\
    --checkpoint-dir results/ckpts/06-02-3849018-ETTh1-binary_dual_scale_staged \\
    --test-index 1153

  python utils/visualize_staged_coarse_fine_components.py --all-datasets
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "reports" / "06-01_cfg_ablation_mmpd_matrix_combined" / "viz_coarse-fine-components"
)
MANIFEST_PATH = (
    REPO_ROOT / "reports" / "06-01_cfg_ablation_mmpd_matrix_combined" / "viz_manifest.json"
)

STAGED_DATASETS = [
    "ETTh1",
    "ETTh2",
    "PeMS",
    "dalia",
    "exchange_rate",
    "traffic",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.visualize_comparison import denorm
from utils.visualize_staged_forecast import (
    _build_pipeline_state,
    _load_staged_bundle,
    _load_staged_diffusion,
    _resolve_itrans_paths,
    _window_lengths,
    pick_staged_ckpt_dir,
)
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset, load_itransformer_from_checkpoint


def _overlap_slice(x: torch.Tensor, k: int) -> torch.Tensor:
    return x[..., k:] if k > 0 else x


def decode_coarse_component(model: torch.nn.Module, coarse_2d: torch.Tensor) -> torch.Tensor:
    """Decode full-range coarse CDF map to normalized 1D coarse band (C, T)."""
    if coarse_2d.dim() == 3:
        coarse_2d = coarse_2d.unsqueeze(0)
    out = model.decode_from_2d(coarse_2d, from_diffusion=False, decoder_method="mean")
    return out[0]


def decode_fine_component(model: torch.nn.Module, fine_2d: torch.Tensor) -> torch.Tensor:
    """Decode in-bin residual CDF map to normalized 1D fine band (C, T)."""
    if fine_2d.dim() == 3:
        fine_2d = fine_2d.unsqueeze(0)
    value_range = model.config.max_scale / model.config.image_height
    out = model.to_2d._decode_occupancy_in_range(
        fine_2d,
        value_range=value_range,
        cdf_decoder="mean",
    )
    if out.shape[1] == 1:
        out = out.squeeze(1)
    return out[0]


def _dit_patch_width(model: torch.nn.Module) -> int:
    ps = getattr(model.config, "dit_patch_size", (8, 8))
    if isinstance(ps, (list, tuple)):
        return int(ps[1])
    return int(ps)


def _forecast_patch_edges(
    forecast_length: int,
    lookback_overlap: int,
    plot_length: int,
    patch_w: int,
) -> np.ndarray:
    """Interior DiT patch edges along the forecast time axis (plot x >= 0)."""
    edges: List[float] = []
    for col in range(patch_w, forecast_length, patch_w):
        x = col - lookback_overlap
        if 0 < x < plot_length:
            edges.append(float(x))
    return np.asarray(edges, dtype=float)


def _add_patch_boundary_lines(ax: plt.Axes, edges: np.ndarray) -> None:
    for x in edges:
        ax.axvline(x, color="#3949AB", lw=1.0, ls="-", alpha=0.5, zorder=0)


def _staged_test_indices_from_manifest() -> Dict[str, int]:
    if not MANIFEST_PATH.is_file():
        return {}
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out: Dict[str, int] = {}
    for section in ("cfg_off", "coarse_fine_multiphase"):
        for dataset, paths in (data.get(section) or {}).items():
            if dataset in out or not paths:
                continue
            name = Path(paths[0]).stem
            if "_idx" not in name:
                continue
            try:
                out[dataset] = int(name.rsplit("_idx", 1)[-1])
            except ValueError:
                continue
    return out


def _future_canvas(future_norm: torch.Tensor, forecast_length: int) -> torch.Tensor:
    """Last *forecast_length* steps of the dataset future window (model 2D canvas)."""
    if future_norm.shape[-1] < forecast_length:
        raise ValueError(
            f"future window length {future_norm.shape[-1]} < model forecast_length {forecast_length}"
        )
    return future_norm[..., -forecast_length:]


def _align_to_ref(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    t = ref.shape[-1]
    if x.shape[-1] == t:
        return x
    if x.shape[-1] > t:
        return x[..., -t:]
    raise ValueError(f"length {x.shape[-1]} shorter than reference {t}")


def _denorm_window_band(
    model: torch.nn.Module,
    band: torch.Tensor,
    stats: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Map a per-window normalized band back to dataset z-score space."""
    if band.dim() == 2:
        band = band.unsqueeze(0)
    return model._denormalize(band, stats).squeeze(0)


@torch.no_grad()
def infer_staged_components(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    past: torch.Tensor,
    future_zscore: torch.Tensor,
    *,
    prob_steps: int,
    seed: int,
    test_index: int,
) -> Dict[str, torch.Tensor]:
    """Anchor coarse/fine bands in dataset z-score space (same as generate() output)."""
    device = past.device
    k = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
    w = int(coarse_model.config.forecast_length)
    past_b = past if past.dim() == 3 else past.unsqueeze(0)
    canvas_global = _future_canvas(future_zscore, w).unsqueeze(0).to(device)
    _, canvas_win, stats = coarse_model._normalize_sequence(past_b, canvas_global)

    torch.manual_seed(seed + test_index)
    coarse_out = coarse_model.generate(
        past_b, sampler="anchor", num_inference_steps=prob_steps
    )
    pred_coarse_2d = coarse_out["future_2d_coarse"]

    torch.manual_seed(seed + test_index)
    fine_out = fine_model.generate(
        past_b,
        sampler="anchor",
        num_inference_steps=prob_steps,
        future_coarse_2d=pred_coarse_2d,
    )
    pred_fine_2d = fine_out["future_2d_fine"]
    chained = fine_out.get("prediction_global_norm", fine_out["prediction"])[0].cpu()

    gt_coarse_2d, gt_fine_2d = coarse_model.encode_dual_to_2d_binary(canvas_win)

    def _bands(coarse_2d: torch.Tensor, fine_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c_win = _overlap_slice(decode_coarse_component(coarse_model, coarse_2d[0]), k)
        f_win = _overlap_slice(decode_fine_component(fine_model, fine_2d[0]), k)
        c_z = _denorm_window_band(coarse_model, c_win, stats)
        f_z = _denorm_window_band(fine_model, f_win, stats)
        return c_z.cpu(), f_z.cpu()

    gt_c, gt_f = _bands(gt_coarse_2d, gt_fine_2d)
    pred_c, pred_f = _bands(pred_coarse_2d, pred_fine_2d)
    recombined = pred_c + pred_f
    gt_dual = _overlap_slice(
        coarse_model.decode_dual_from_2d(
            gt_coarse_2d, gt_fine_2d, from_diffusion=False, decoder_method="mean"
        )[0],
        k,
    )
    gt_full = _denorm_window_band(coarse_model, gt_dual, stats).cpu()

    ref = chained
    return {
        "gt_coarse": _align_to_ref(gt_c, ref),
        "gt_fine": _align_to_ref(gt_f, ref),
        "pred_coarse": _align_to_ref(pred_c, ref),
        "pred_fine": _align_to_ref(pred_f, ref),
        "recombined": _align_to_ref(recombined, ref),
        "chained": ref,
        "gt_full": _align_to_ref(gt_full, ref),
    }


def plot_staged_coarse_fine_components_panel(
    checkpoint_dir: Path,
    dataset: str,
    output_dir: Path,
    test_index: Optional[int],
    prob_steps: int,
    seed: int,
    device: torch.device,
) -> Path:
    sub = _load_staged_bundle(checkpoint_dir, dataset)
    subset_id = sub["subset_id"]
    variate_indices = sub["variate_indices"]
    n_vars = len(variate_indices)
    state = _build_pipeline_state(checkpoint_dir, dataset, subset_id)
    lookback, horizon = _window_lengths(dataset, state)

    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=1,
        test_stride=1,
        lookback=lookback,
        horizon=horizon,
    )
    n_test = len(test_ds)
    if n_test == 0:
        raise ValueError(f"Empty test set for {dataset}")

    manifest_idx = _staged_test_indices_from_manifest().get(dataset)
    if test_index is None:
        test_index = manifest_idx if manifest_idx is not None else random.Random(seed).randrange(n_test)

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    guidance_path, _ = _resolve_itrans_paths(checkpoint_dir, subset_id)
    if guidance_path is None:
        raise FileNotFoundError(
            f"Missing guidance checkpoint: {subset_id}_itransformer_finetuned.pt under {checkpoint_dir}"
        )
    guidance_model = load_itransformer_from_checkpoint(
        str(guidance_path), n_vars, device
    )
    itrans_guidance = iTransformerGuidance(guidance_model)
    coarse_model = _load_staged_diffusion(
        state, "coarse", sub["coarse_pt"], itrans_guidance, n_vars, device
    )
    fine_model = _load_staged_diffusion(
        state, "fine", sub["fine_pt"], itrans_guidance, n_vars, device
    )

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)
    k = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
    comps = infer_staged_components(
        coarse_model,
        fine_model,
        past_t,
        future,
        prob_steps=prob_steps,
        seed=seed,
        test_index=test_index,
    )

    future_slice = future[:, -horizon:]
    if k > 0:
        future_slice = future_slice[..., k:]
    t_fut_len = int(future_slice.shape[-1])

    def _trim_to_eval(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == t_fut_len:
            return x
        if x.shape[-1] > t_fut_len:
            return x[..., -t_fut_len:]
        raise ValueError(f"series length {x.shape[-1]} < eval horizon {t_fut_len}")

    comps = {key: _trim_to_eval(val) for key, val in comps.items()}

    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, t_fut_len)
    patch_w = _dit_patch_width(coarse_model)
    patch_edges = _forecast_patch_edges(
        int(coarse_model.config.forecast_length),
        k,
        t_fut_len,
        patch_w,
    )
    patch_label_added = False

    fig, axes = plt.subplots(
        n_vars,
        3,
        figsize=(18, 2.4 * n_vars),
        squeeze=False,
        constrained_layout=True,
    )
    names = sub["variate_names"] or [f"v{i}" for i in range(n_vars)]

    for v in range(n_vars):
        past_dn = denorm(past, mean, std)[v].numpy()
        gt_series = denorm(future_slice, mean, std)[v].numpy()
        gt_c = denorm(comps["gt_coarse"], mean, std)[v].numpy()
        gt_f = denorm(comps["gt_fine"], mean, std)[v].numpy()
        pred_c = denorm(comps["pred_coarse"], mean, std)[v].numpy()
        pred_f = denorm(comps["pred_fine"], mean, std)[v].numpy()
        recombined = denorm(comps["recombined"], mean, std)[v].numpy()
        chained = denorm(comps["chained"], mean, std)[v].numpy()

        def _plot_context(ax: plt.Axes) -> None:
            ax.plot(t_past, past_dn[-context_len:], color="#9E9E9E", lw=0.9, alpha=0.55)
            ax.axvline(0, color="k", ls=":", alpha=0.25)
            _add_patch_boundary_lines(ax, patch_edges)

        col_specs = (
            (0, "Coarse band", gt_c, pred_c),
            (1, "Fine residual", gt_f, pred_f),
            (2, "Coarse + fine sum", gt_series, recombined),
        )
        for col, title, gt_line, pred_line in col_specs:
            ax = axes[v, col]
            _plot_context(ax)
            ax.plot(t_future, gt_line, color="#2196F3", lw=1.8, label="Ground truth")
            ax.plot(t_future, pred_line, color="#E91E63", lw=1.4, label="Pred (anchor)")
            if col == 2:
                ax.plot(
                    t_future,
                    chained,
                    color="#FF9800",
                    lw=1.0,
                    ls="--",
                    alpha=0.85,
                    label="2-stage decode",
                )
            ax.set_title(title if v == 0 else "", fontsize=9)
            ax.set_ylabel(names[v] if v < len(names) else f"var {v}", fontsize=9)
            if v == 0 and col == 2:
                handles, labels = ax.get_legend_handles_labels()
                if len(patch_edges) > 0 and not patch_label_added:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color="#3949AB",
                            lw=1.0,
                            alpha=0.5,
                            label=f"DiT patch edge ({patch_w}×{patch_w})",
                        )
                    )
                    labels.append(f"DiT patch edge ({patch_w}×{patch_w})")
                    patch_label_added = True
                ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=7, ncol=2)
            ax.grid(alpha=0.2)

    fig.suptitle(
        f"{dataset} / {subset_id} — test idx {test_index} | "
        f"staged coarse vs fine (anchor, steps={prob_steps}, dit_patch={patch_w}×{patch_w})",
        fontsize=11,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        output_dir
        / f"coarse_fine_components_{dataset}_{subset_id}_idx{test_index}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_all_datasets(
    ckpt_root: Path,
    output_base: Path,
    datasets: List[str],
    prob_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for dataset in datasets:
        try:
            ckpt_dir = pick_staged_ckpt_dir(ckpt_root, dataset)
        except FileNotFoundError as e:
            print(f"  [skip] {dataset}: {e}", flush=True)
            continue
        ds_seed = seed + sum((i + 1) * ord(c) for i, c in enumerate(dataset))
        test_index = _staged_test_indices_from_manifest().get(dataset)
        out = plot_staged_coarse_fine_components_panel(
            ckpt_dir,
            dataset,
            output_base / dataset,
            test_index=test_index,
            prob_steps=prob_steps,
            seed=ds_seed,
            device=device,
        )
        rel = str(out.relative_to(output_base.parent))
        written[dataset] = rel
        print(f"  wrote {rel}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help=f"Run for staged grid datasets: {', '.join(STAGED_DATASETS)}",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated override (default: --dataset or staged list)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = args.output_dir.resolve()

    if args.all_datasets or (args.dataset is None and args.checkpoint_dir is None):
        ds_list = (
            [d.strip() for d in args.datasets.split(",") if d.strip()]
            if args.datasets
            else STAGED_DATASETS
        )
        print(f"Coarse/fine components -> {output_base}  datasets={ds_list}", flush=True)
        written = run_all_datasets(
            args.ckpt_root.resolve(),
            output_base,
            ds_list,
            args.prob_steps,
            args.seed,
            device,
        )
        if MANIFEST_PATH.is_file():
            manifest: Dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        else:
            manifest = {}
        manifest["coarse_fine_components"] = {
            ds: [path] for ds, path in written.items()
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Updated manifest: {MANIFEST_PATH}", flush=True)
        return

    if args.dataset is None or args.checkpoint_dir is None:
        parser.error("Provide --checkpoint-dir and --dataset, or use --all-datasets")

    ds_seed = args.seed + sum((i + 1) * ord(c) for i, c in enumerate(args.dataset))
    out = plot_staged_coarse_fine_components_panel(
        args.checkpoint_dir.resolve(),
        args.dataset,
        output_base / args.dataset,
        args.test_index,
        args.prob_steps,
        ds_seed,
        device,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
