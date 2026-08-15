#!/usr/bin/env python3
"""
# Pipeline home: utils/staged_eval_sample_viz.py + StagedEvalPhase (visualization.viz_patch_boxes).
# Ablation --redbox-viz still calls viz_run below.
One-off: staged_eval sample panels for the three ETTh1 ablation ckpts.

Works around the cancelled in-pipeline KeyError('fine') on patch_refine runs
(pred_maps uses key 'refine' but the native 2D plotter hardcodes maps['fine']).

Also writes patch-box diagnostic panels for patch_refine jobs (guided_p8 /
window_norm_pr): nearest-upsampled coarse + unblended patch grid + blended
refine canvas, with red PatchLocation rectangles and ConnectionPatch arrows
from each coarse box (A) to its unblended refined patch panel (B).

Supports deterministic anchor (default, 1 step) and probabilistic sampling
(``quad_t``, typically 20 steps; also ``ddim`` / ``ddim_quad``).

For each run, writes under ``--output-root/<run_name>/``:
  - sample{ii:02d}_pool{pool_i}_1d.jpg
  - sample{ii:02d}_pool{pool_i}_2d_coarse_fine.jpg
  - sample{ii:02d}_pool{pool_i}_v{v}_refine_boxes.jpg  (patch_refine only)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]


def _peek_code_root() -> Optional[Path]:
    env = (os.environ.get("TS_SANDBOX_CODE_ROOT") or "").strip()
    raw = env or None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--code-root" and i + 1 < len(argv):
            raw = argv[i + 1]
            break
        if a.startswith("--code-root="):
            raw = a.split("=", 1)[1]
            break
    return Path(raw).resolve() if raw else None


_CODE_ROOT = _peek_code_root()
# Prefer worktree with native-past expand helper (ordinal-fine) over script tree.
# Insert REPO first, then CODE_ROOT at front so CODE wins.
for _root in (REPO_ROOT, _CODE_ROOT):
    if _root is None:
        continue
    s = str(_root)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_ablation_disc_l8_l16 import load_ablation_run, _load_models as _load_ablation_models
from utils.eval_mmpd_gaussian_anchor import (
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.staged_binary_forecast import generate_staged_forecast
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)

DEFAULT_RUNS = (
    "guided_p8:results/ckpts/08-01-4519745-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8:"
    "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_guided_p8.yaml",
    "window_norm_pr:results/ckpts/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm:"
    "configs/binary_window_norm_patch_refine_earlyjuly_norm.yaml",
    "ordinal_fine:results/ckpts/08-02-4525834-ETTh1-binary_ordinal_fine_finer_earlyjuly_hps:"
    "configs/binary_ordinal_fine_finer_earlyjuly_hps.yaml",
)

MAX_UNBLENDED_PATCHES = 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--pack-test-stride", type=int, default=16)
    p.add_argument("--pack-splits", default="test")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--draw-seed-step",
        type=int,
        default=0,
        help="Additional seed offset per emitted sample; use with repeated pool indices.",
    )
    p.add_argument("--pool-indices", type=int, nargs="+", default=None)
    p.add_argument(
        "--variables-to-plot",
        type=int,
        default=0,
        help="Variates to plot in 1d/2d/red-box panels. 0 = all (default).",
    )
    p.add_argument("--jpeg-dpi", type=int, default=120)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument(
        "--sampler",
        default="quad_t",
        choices=("anchor", "quad_t", "ddim", "ddim_quad"),
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="Import root with DiffusionTSF._expand_horizon_cond_to_past_width "
        "(e.g. $SCRATCH/ts-sandbox-ordinal-fine). Also via TS_SANDBOX_CODE_ROOT.",
    )
    p.add_argument(
        "--skip-existing-runs",
        action="store_true",
        help="Skip a run if output-root/<name>/manifest.txt already exists.",
    )
    return p.parse_args()


def _parse_runs(specs: Sequence[str]) -> List[Dict[str, str]]:
    out = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"bad --runs entry (name:ckpt:config): {spec}")
        out.append({"name": parts[0], "ckpt": parts[1], "config": parts[2]})
    return out


def _pick_indices(n_pool: int, n_samples: int, seed: int, explicit: Optional[List[int]]) -> List[int]:
    if explicit:
        bad = [i for i in explicit if i < 0 or i >= n_pool]
        if bad:
            raise ValueError(f"pool indices out of range for n={n_pool}: {bad}")
        return [int(i) for i in explicit]
    rng = np.random.default_rng(seed)
    k = min(int(n_samples), n_pool)
    return sorted(int(i) for i in rng.choice(n_pool, size=k, replace=False).tolist())



from utils.staged_eval_sample_viz import (  # noqa: E402
    add_patch_boxes as _add_patch_boxes,
    plot_1d as _plot_1d,
    plot_2d_coarse_fine as _plot_2d_coarse_fine,
    plot_refine_boxes as _plot_refine_boxes,
    save_jpg as _save_jpg,
    upsample_coarse_to_canvas as _upsample_coarse_to_canvas,
    write_staged_sample_panels,
)

# Local plotters live in utils.staged_eval_sample_viz; viz_run below still loads models.


def viz_run(
    args,
    *,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    device: torch.device,
    picks: List[int],
    pool: Any,
) -> List[Path]:
    dataset = str(args.dataset)
    out_dir = args.output_root / run_name
    if bool(getattr(args, "skip_existing_runs", False)) and (out_dir / "manifest.txt").is_file():
        print(
            f"\n=== {run_name} SKIP existing ({out_dir / 'manifest.txt'}) ===",
            flush=True,
        )
        return [Path(line) for line in (out_dir / "manifest.txt").read_text().splitlines()
                if line.endswith(".jpg")]

    print(f"\n=== {run_name} ({ckpt_root.name}) ===", flush=True)
    run, coarse, refine, _ladder, kind, _canvas_h = _load_ablation_models(
        dataset=dataset,
        ckpt_root=ckpt_root,
        config_path=config_path,
        lookback=args.lookback,
        horizon=args.horizon,
        device=device,
    )
    _state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
    print(f"[{run_name}] kind={kind} pool={len(pool)} picks={picks}", flush=True)
    # Native-past (past_cond_resize_to_horizon=false) needs horizon expand helper.
    if not bool(getattr(_state, "past_cond_resize_to_horizon", True)):
        if not hasattr(refine, "_expand_horizon_cond_to_past_width"):
            raise RuntimeError(
                f"{run_name}: past_cond_resize_to_horizon=false but "
                "DiffusionTSF lacks _expand_horizon_cond_to_past_width. "
                "Pass --code-root to ts-sandbox-ordinal-fine (or a tree with the helper)."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    second_name = "blended refine" if kind == "patch_refine" else "fine"
    written: List[Path] = []

    canvas_h = int(getattr(refine.config, "patch_refine_canvas_height", 256) or 256)
    patch_h = int(getattr(refine.config, "patch_refine_patch_height", 16) or 16)
    patch_w = int(getattr(refine.config, "patch_refine_patch_width", 8) or 8)
    n_vars_run = len(run_variate_indices(run))
    n_plot = int(args.variables_to_plot)
    if n_plot <= 0:
        n_plot = n_vars_run
    else:
        n_plot = min(n_plot, n_vars_run)

    for s_i, pool_i in enumerate(picks):
        past_t, future_t = pool[int(pool_i)]
        past = past_t.unsqueeze(0).to(device)
        future = future_t.unsqueeze(0).to(device)
        torch.manual_seed(
            int(args.seed) + int(pool_i) + int(s_i) * int(args.draw_seed_step)
        )
        result = generate_staged_forecast(
            coarse,
            refine,
            past,
            vertical_dual=False,
            sampler=args.sampler,
            num_inference_steps=int(args.num_sampling_steps),
            require_guidance_pred=True,
        )
        overlap = int(getattr(refine.config, "lookback_overlap", 0) or 0)
        gt = future[0, :, overlap:] if overlap else future[0]
        pred = result["prediction_global_norm"][0]
        if pred.shape != gt.shape:
            h = min(int(pred.shape[-1]), int(gt.shape[-1]))
            pred = pred[..., -h:]
            gt = gt[..., -h:]
        guide = result["guidance_prediction_global_norm"][0]
        if guide.shape != pred.shape:
            h = min(int(guide.shape[-1]), int(pred.shape[-1]))
            guide = guide[..., -h:]

        past_np = past[0].detach().cpu().numpy().astype(np.float32)
        gt_np = gt.detach().cpu().numpy().astype(np.float32)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        guide_np = guide.detach().cpu().numpy().astype(np.float32)
        coarse_2d = result["future_2d_coarse"][0].detach().cpu().numpy().astype(np.float32)
        second_2d = result["future_2d_fine"][0].detach().cpu().numpy().astype(np.float32)

        p1 = out_dir / f"sample{s_i:02d}_pool{pool_i}_1d.jpg"
        _plot_1d(
            path=p1,
            past=past_np,
            gt=gt_np,
            pred=pred_np,
            guidance=guide_np,
            title=(
                f"{run_name}/{dataset} pool={pool_i} kind={kind} sampler={args.sampler} "
                f"(GT + refine + guidance)"
            ),
            n_vars=n_plot,
            dpi=args.jpeg_dpi,
        )
        p2 = out_dir / f"sample{s_i:02d}_pool{pool_i}_2d_coarse_fine.jpg"
        _plot_2d_coarse_fine(
            path=p2,
            coarse=coarse_2d,
            second=second_2d,
            second_name=second_name,
            title=f"{run_name}/{dataset} pool={pool_i} coarse | {second_name}",
            n_vars=n_plot,
            dpi=args.jpeg_dpi,
        )
        written.extend([p1, p2])
        extra_names = []

        if kind == "patch_refine":
            if "patch_cdf_unblended" not in result or "patch_locations" not in result:
                raise RuntimeError(
                    f"{run_name}: patch_refine missing patch_cdf_unblended/patch_locations"
                )
            patch_cdf = result["patch_cdf_unblended"]
            locations = result["patch_locations"]
            coarse_up = _upsample_coarse_to_canvas(coarse_2d, canvas_h)
            v_show = min(n_plot, coarse_2d.shape[0])
            for v in range(v_show):
                pv = out_dir / f"sample{s_i:02d}_pool{pool_i}_v{v}_refine_boxes.jpg"
                _plot_refine_boxes(
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
                    dpi=args.jpeg_dpi,
                )
                written.append(pv)
                extra_names.append(pv.name)

        print(
            f"[{run_name}] wrote {p1.name} {p2.name}"
            + ((" " + " ".join(extra_names)) if extra_names else ""),
            flush=True,
        )

    (out_dir / "manifest.txt").write_text(
        "\n".join(
            [
                f"run={run_name}",
                f"kind={kind}",
                f"ckpt={ckpt_root}",
                f"config={config_path}",
                f"sampler={args.sampler}",
                f"steps={args.num_sampling_steps}",
                f"picks={picks}",
                *[str(p) for p in written],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return written


def main() -> None:
    args = parse_args()
    args.output_root = Path(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.code_root is not None:
        resolved = Path(args.code_root).resolve()
        if _CODE_ROOT is not None and resolved != _CODE_ROOT:
            raise RuntimeError(
                f"--code-root {resolved} differs from bootstrap {_CODE_ROOT}; "
                "set TS_SANDBOX_CODE_ROOT or pass --code-root before import side effects"
            )
    from models.diffusion_tsf.diffusion_model import DiffusionTSF

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} output_root={args.output_root}", flush=True)
    print(
        f"code_root={_CODE_ROOT or REPO_ROOT} "
        f"has_expand={hasattr(DiffusionTSF, '_expand_horizon_cond_to_past_width')}",
        flush=True,
    )

    # Shared pool indices once, before loading any models. Variates come from
    # the first run's checkpoint metadata (fail if later runs disagree).
    specs = _parse_runs(args.runs)
    if not specs:
        raise ValueError("--runs is empty")
    first_ckpt = Path(specs[0]["ckpt"])
    if not first_ckpt.is_absolute():
        first_ckpt = REPO_ROOT / first_ckpt
    first_run, _, _ = load_ablation_run(args.dataset, first_ckpt)
    pool_vars = run_variate_indices(first_run)
    for spec in specs[1:]:
        ckpt_i = Path(spec["ckpt"])
        if not ckpt_i.is_absolute():
            ckpt_i = REPO_ROOT / ckpt_i
        run_i, _, _ = load_ablation_run(args.dataset, ckpt_i)
        vars_i = run_variate_indices(run_i)
        if vars_i != pool_vars:
            raise ValueError(
                f"run {spec['name']} variates {vars_i} != first-run {pool_vars}; "
                "pass one dataset subset per viz invocation"
            )
    print(
        f"loading shared pack pool dataset={args.dataset} vars={pool_vars} "
        f"lb={args.lookback} hz={args.horizon} test_stride={args.pack_test_stride} "
        f"splits={args.pack_splits}",
        flush=True,
    )
    pool, _starts, _splits, _, _ = load_tsf_pack_pool(
        args.dataset,
        pool_vars,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=1,
        test_stride=int(args.pack_test_stride),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    picks = _pick_indices(len(pool), args.n_samples, args.seed, args.pool_indices)
    print(f"shared picks (n_pool={len(pool)}): {picks}", flush=True)

    all_paths: List[Path] = []
    for spec in specs:
        ckpt = Path(spec["ckpt"])
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        all_paths.extend(
            viz_run(
                args,
                run_name=spec["name"],
                ckpt_root=ckpt,
                config_path=spec["config"],
                device=device,
                picks=picks,
                pool=pool,
            )
        )
    print(f"\ndone: {len(all_paths)} panels -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
