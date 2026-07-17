#!/usr/bin/env python3
"""Compare GT vs pred coarse/fine 2D maps across g=1..10 4ep finetune stubs.

Targets the ETTh2 noise-sched ablation series:
  results/ckpts/07-15-*-ETTh2-binary_noise_sched_ablation_vertical_dual_g{N}p0
  → {subset}/vertical_dual/best.pt  (fixed-HP ~4ep)

Same real ETTh2 windows for every g so differences are schedule-only.

Example (Killarney):
  source .venv/bin/activate   # or use submit wrapper
  python temp/viz_g_ablation_finetune_maps.py --dataset ETTh2 --n-windows 3
  ./temp/submit_viz_g_ablation_finetune_maps_killarney.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG_STEM_TMPL = "binary_noise_sched_ablation_vertical_dual_g{g}p0"
GUIDANCE_GLOBS = (
    "{subset}_patch_guidance.pt",
    "{subset}_patch_guidance_hp_best.pt",
    "patch_guidance_synthetic.pt",
    "pretrained_patch_guidance.pt",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("viz_g_ablation_finetune_maps")


def _parse_g_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _discover_run(ckpts_root: Path, dataset: str, g: int) -> Path:
    stem = CONFIG_STEM_TMPL.format(g=g)
    token = f"-{dataset}-{stem}"
    cands = sorted(
        (p for p in ckpts_root.iterdir() if p.is_dir() and p.name.endswith(token)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in cands:
        bests = list(run.glob("*/vertical_dual/best.pt"))
        if bests:
            return run
    raise FileNotFoundError(
        f"no *{token} run with */vertical_dual/best.pt under {ckpts_root}"
    )


def _resolve_best_and_subset(run_dir: Path, subset_hint: Optional[str]) -> Tuple[Path, str]:
    if subset_hint:
        cand = run_dir / subset_hint / "vertical_dual" / "best.pt"
        if cand.is_file():
            return cand, subset_hint
    bests = sorted(run_dir.glob("*/vertical_dual/best.pt"))
    if not bests:
        raise FileNotFoundError(f"missing */vertical_dual/best.pt under {run_dir}")
    best = bests[0]
    return best, best.parent.parent.name


def _resolve_guidance(run_dir: Path, subset_id: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
    for tmpl in GUIDANCE_GLOBS:
        cand = run_dir / tmpl.format(subset=subset_id)
        if cand.is_file():
            return cand
    alts = sorted(run_dir.glob("*_patch_guidance*.pt"))
    if alts:
        return alts[0]
    raise FileNotFoundError(f"no patch guidance under {run_dir}")


def _load_tuned_params(best_pt: Path) -> Dict[str, Any]:
    meta = best_pt.parent / "metadata.json"
    if not meta.is_file():
        return {}
    with open(meta, encoding="utf-8") as f:
        obj = json.load(f)
    return dict(obj.get("tuned_params") or {})


def _plot_g_compare_grid(
    *,
    panels: List[Tuple[int, np.ndarray, np.ndarray]],
    gt_coarse: np.ndarray,
    gt_fine: np.ndarray,
    var_idx: int,
    window_idx: int,
    out_path: Path,
    jpeg_dpi: int = 100,
) -> str:
    """One var: GT coarse/fine + pred coarse/fine for every g."""
    from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg

    n_g = len(panels)
    # rows: GT_c, GT_f once; then for each g: pred_c, pred_f — too tall.
    # Compact: row0 GT coarse | GT fine; then n_g rows of pred_c | pred_f with g label.
    fig, axes = plt.subplots(
        n_g + 1,
        2,
        figsize=(10.0, 2.0 * (n_g + 1)),
        constrained_layout=True,
        squeeze=False,
    )
    w = min(gt_coarse.shape[-1], gt_fine.shape[-1], *(p[1].shape[-1] for p in panels))
    gc = gt_coarse[var_idx, :, -w:]
    gf = gt_fine[var_idx, :, -w:]

    for col, (label, data, cmap) in enumerate((
        ("GT coarse", gc, "Blues"),
        ("GT fine", gf, "Blues"),
    )):
        ax = axes[0, col]
        h, ww = data.shape
        im = ax.imshow(
            data, aspect="auto", origin="lower", extent=[0, ww, 0, h],
            cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
        )
        ax.set_title(f"{label} ({h}x{ww})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r, (g, pc, pf) in enumerate(panels, start=1):
        for col, (label, data, cmap) in enumerate((
            (f"g={g} pred coarse", pc[var_idx, :, -w:], "Oranges"),
            (f"g={g} pred fine", pf[var_idx, :, -w:], "Oranges"),
        )):
            ax = axes[r, col]
            h, ww = data.shape
            im = ax.imshow(
                data, aspect="auto", origin="lower", extent=[0, ww, 0, h],
                cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
            )
            ax.set_title(label, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"g={g}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"ETTh2 4ep finetune g-ablation | window {window_idx} | var {var_idx}\n"
        "GT (top) vs pred coarse/fine per length_g",
        fontsize=11,
        fontweight="semibold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_figure_jpg(fig, str(out_path), dpi=jpeg_dpi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh2")
    p.add_argument("--g-values", default="1-10", help="e.g. 1-10 or 1,3,5,7")
    p.add_argument(
        "--config-tmpl",
        default="configs/binary_noise_sched_ablation_vertical_dual_g{g}p0.yaml",
    )
    p.add_argument("--ckpts-root", default="results/ckpts")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--n-vars-plot", type=int, default=3)
    p.add_argument("--windows", default="", help="comma indices into test set (optional)")
    p.add_argument("--sampler", default="anchor", choices=("anchor", "dpmpp", "ddim"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--out-dir", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--compare-var", type=int, default=0, help="var for g-grid summary panel")
    args = p.parse_args()

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.visualize_utils import (
        _load_staged_diffusion_from_ckpt,
        _plot_dual_concat_synth_panel,
        pick_sample_indices,
    )
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    g_values = _parse_g_list(args.g_values)
    ckpts_root = (REPO / args.ckpts_root).resolve()
    out_root = Path(args.out_dir).expanduser() if args.out_dir else (
        REPO / "results" / "viz" / f"g_ablation_finetune_maps_{args.dataset}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # Shared geometry/state from g1 config (dataset windows, ordinal, heights).
    base_cfg_path = str(REPO / args.config_tmpl.format(g=g_values[0]))
    cfg0 = load_experiment_config(
        base_cfg_path,
        cli_overrides={
            "dataset": args.dataset,
            "checkpoint_dir": str(ckpts_root),
            "results_dir": str(out_root),
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg0)
    state.dataset = args.dataset
    state.results_dir = str(out_root)
    if args.device:
        state.device = args.device
    device = state.resolve_device()
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    job = generate_dataset_job(state.dataset)
    variate_indices = list(state.variate_indices or job["variate_indices"])
    state.variate_indices = variate_indices
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    test_stride = int(subset_meta.get("test_stride", 4))
    train_ds, val_ds, test_ds, norm_stats = load_dataset(
        state.dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    ds = test_ds if args.split == "test" else val_ds
    if args.windows.strip():
        indices = [int(x) for x in args.windows.split(",") if x.strip()]
    else:
        indices = pick_sample_indices(len(ds), args.n_windows, seed=args.seed)
    logger.info("windows=%s split=%s n=%d", indices, args.split, len(ds))

    # Discover runs once
    runs: Dict[int, Tuple[Path, Path, Path, Dict[str, Any]]] = {}
    for g in g_values:
        run = _discover_run(ckpts_root, args.dataset, g)
        best, subset = _resolve_best_and_subset(run, args.dataset)
        guide = _resolve_guidance(run, subset)
        tuned = _load_tuned_params(best)
        # Ensure length_g matches this leaf even if metadata omitted it.
        tuned.setdefault("binary_length_g", float(g))
        tuned.setdefault("binary_length_mode", "power" if g != 1 else "none")
        # g1p0 yaml uses mode none; g>1 use power — read from leaf when possible.
        leaf = REPO / args.config_tmpl.format(g=g)
        if leaf.is_file():
            leaf_cfg = load_experiment_config(str(leaf))
            exp = leaf_cfg.get("experiment") or {}
            if "binary_length_mode" in exp:
                tuned["binary_length_mode"] = exp["binary_length_mode"]
            if "binary_length_g" in exp:
                tuned["binary_length_g"] = float(exp["binary_length_g"])
            if "binary_length_scale" in exp:
                tuned["binary_length_scale"] = float(exp["binary_length_scale"])
        runs[g] = (run, best, guide, tuned)
        logger.info(
            "g=%s run=%s best=%s guide=%s length_g=%s mode=%s",
            g, run.name, best, guide.name,
            tuned.get("binary_length_g"), tuned.get("binary_length_mode"),
        )

    lb = int(state.lookback_length)
    all_saved: List[str] = []
    # For summary grid: store per-window list of (g, pred_c, pred_f) + GT from first g
    for wi, idx in enumerate(indices):
        past, future = ds[idx]
        if not torch.is_tensor(past):
            past = torch.as_tensor(past, dtype=torch.float32)
        if not torch.is_tensor(future):
            future = torch.as_tensor(future, dtype=torch.float32)
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        past_np = past.detach().cpu().numpy()
        future_np = future.detach().cpu().numpy()

        gt_coarse = gt_fine = None
        compare_panels: List[Tuple[int, np.ndarray, np.ndarray]] = []

        for g in g_values:
            run, best, guide, tuned = runs[g]
            # Patch length schedule globals to match this g before create/load.
            pipeline_mod.BINARY_LENGTH_MODE = str(tuned.get("binary_length_mode", "none"))
            pipeline_mod.BINARY_LENGTH_G = float(tuned.get("binary_length_g", g))
            pipeline_mod.BINARY_LENGTH_SCALE = float(tuned.get("binary_length_scale", 1.0))
            state.binary_length_mode = pipeline_mod.BINARY_LENGTH_MODE
            state.binary_length_g = pipeline_mod.BINARY_LENGTH_G
            state.binary_length_scale = pipeline_mod.BINARY_LENGTH_SCALE

            model, _ = _load_staged_diffusion_from_ckpt(
                ckpt_path=str(best),
                stage="vertical_dual",
                itrans_ckpt_path=str(guide),
                n_vars=len(variate_indices),
                device=device,
                tuned_params=tuned,
                guidance_type=getattr(state, "guidance_type", None),
            )
            with torch.no_grad():
                steps = 1 if args.sampler == "anchor" else int(args.steps)
                out = model.generate(
                    past_b, sampler=args.sampler, num_inference_steps=steps,
                )
                _pn, future_norm, _ = model._normalize_sequence(past_b, future_b)
                gt_maps = model._encode_staged_maps(future_norm)

            if "future_2d_coarse" not in out or "future_2d_fine" not in out:
                raise KeyError(f"g={g} generate missing coarse/fine maps")
            pred = out.get("prediction", out.get("prediction_global_norm"))
            if pred is None:
                raise KeyError(f"g={g} generate missing prediction")
            pred = pred[0].detach().cpu().numpy()
            pred_c = out["future_2d_coarse"][0].detach().cpu().numpy()
            pred_f = out["future_2d_fine"][0].detach().cpu().numpy()
            gc = gt_maps["coarse"][0].detach().cpu().numpy()
            gf = gt_maps["fine"][0].detach().cpu().numpy()
            if gt_coarse is None:
                gt_coarse, gt_fine = gc, gf

            if pred.shape[-1] <= future_np.shape[-1]:
                future_core = future_np[..., -pred.shape[-1] :]
            else:
                future_core = future_np
            common = min(future_core.shape[-1], pred.shape[-1])
            future_core = future_core[..., -common:]
            pred = pred[..., -common:]

            g_dir = out_root / f"g{g}p0" / f"win{idx:04d}"
            g_dir.mkdir(parents=True, exist_ok=True)
            path = g_dir / f"g{g}_win{idx}_maps.jpg"
            all_saved.append(
                _plot_dual_concat_synth_panel(
                    past_np=past_np,
                    future_core=future_core,
                    pred=pred,
                    gt_coarse=gc,
                    gt_fine=gf,
                    pred_coarse=pred_c,
                    pred_fine=pred_f,
                    lookback=lb,
                    sample_idx=int(idx),
                    stage="vertical_dual",
                    sampler=args.sampler,
                    output_path=str(path),
                    variables_to_plot=int(args.n_vars_plot),
                    jpeg_dpi=100,
                    title=(
                        f"{args.dataset} 4ep finetune | g={g} | window {idx} | "
                        f"sampler={args.sampler}\n"
                        f"run={run.name}"
                    ),
                )
            )
            logger.info("wrote %s", path)
            compare_panels.append((g, pred_c, pred_f))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert gt_coarse is not None and gt_fine is not None
        summary = out_root / "compare" / f"win{idx:04d}_var{args.compare_var}_g1to{g_values[-1]}.jpg"
        all_saved.append(
            _plot_g_compare_grid(
                panels=compare_panels,
                gt_coarse=gt_coarse,
                gt_fine=gt_fine,
                var_idx=int(args.compare_var),
                window_idx=int(idx),
                out_path=summary,
            )
        )
        logger.info("wrote compare grid %s", summary)

    logger.info("done (%d files) → %s", len(all_saved), out_root)


if __name__ == "__main__":
    main()
