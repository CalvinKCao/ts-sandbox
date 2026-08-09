#!/usr/bin/env python3
"""Throwaway: past_native ETTh1 g-ablation — coarse/fine GT vs pred 2D maps → wandb.

These Jul-12 runs set skip_eval_visualizations=true, so leaderboard runs have no
forecast panels. This regenerates a few windows (GT coarse/fine + pred coarse/fine
occupancy CDFs + 1D GT/pred) and logs them onto the existing pipeline runs.

Needs coarse/fine best.pt + patch guidance under results/ckpts (or $SCRATCH).
Local WSL usually only has metadata — run on Killarney where the weights live.

  source .venv/bin/activate
  # on Killarney, repo under $SCRATCH/ts-sandbox typically:
  python temp/scripts/viz_past_native_etth1_gt_pred_maps.py --n-windows 3 --wandb

  # local check without inference:
  python temp/scripts/viz_past_native_etth1_gt_pred_maps.py --check-only
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
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("viz_past_native_gt_pred")

# Jul-12 ETTh1 sequential past_native g grid (leaderboard pipeline runs).
RUNS: List[Dict[str, str]] = [
    {
        "g_label": "1.0",
        "job": "4205389",
        "config": "binary_noise_sched_ablation_past_native_g1p0",
        "wandb_id": "5srlbb8u",
    },
    {
        "g_label": "1.0_s43",
        "job": "4205429",
        "config": "binary_noise_sched_ablation_past_native_g1p0_s43",
        "wandb_id": "t13w2f5q",
    },
    {
        "g_label": "1.0_s44",
        "job": "4205433",
        "config": "binary_noise_sched_ablation_past_native_g1p0_s44",
        "wandb_id": "72ep9ehk",
    },
    {
        "g_label": "1.5",
        "job": "4205393",
        "config": "binary_noise_sched_ablation_past_native_g1p5",
        "wandb_id": "z9ejucaj",
    },
    {
        "g_label": "3.0",
        "job": "4205397",
        "config": "binary_noise_sched_ablation_past_native_g3p0",
        "wandb_id": "dpvnhozt",
    },
    {
        "g_label": "4.0",
        "job": "4205401",
        "config": "binary_noise_sched_ablation_past_native_g4p0",
        "wandb_id": "57sxd8df",
    },
    {
        "g_label": "5.0",
        "job": "4205405",
        "config": "binary_noise_sched_ablation_past_native_g5p0",
        "wandb_id": "e1yc2p2g",
    },
    {
        "g_label": "6.0",
        "job": "4205417",
        "config": "binary_noise_sched_ablation_past_native_g6p0",
        "wandb_id": "cuiluu93",
    },
    {
        "g_label": "7.0",
        "job": "4205409",
        "config": "binary_noise_sched_ablation_past_native_g7p0",
        "wandb_id": "lkw04gar",
    },
    {
        "g_label": "8.0",
        "job": "4205421",
        "config": "binary_noise_sched_ablation_past_native_g8p0",
        "wandb_id": "c5tz62wr",
    },
    {
        "g_label": "9.0",
        "job": "4205425",
        "config": "binary_noise_sched_ablation_past_native_g9p0",
        "wandb_id": "v6csajdi",
    },
    {
        "g_label": "10.0",
        "job": "4205413",
        "config": "binary_noise_sched_ablation_past_native_g10p0",
        "wandb_id": "g2lausaf",
    },
]

GUIDANCE_GLOBS = (
    "{subset}_patch_guidance.pt",
    "{subset}_patch_guidance_hp_best.pt",
    "patch_guidance_synthetic.pt",
)


def _default_ckpts_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        cand = Path(scratch) / "ts-sandbox" / "results" / "ckpts"
        if cand.is_dir():
            return cand
    return REPO / "results" / "ckpts"


def _stem(job: str, config: str) -> str:
    return f"07-12-{job}-ETTh1-{config}"


def _resolve_run_dir(ckpts_root: Path, job: str, config: str) -> Path:
    stem = _stem(job, config)
    direct = ckpts_root / stem
    if direct.is_dir():
        return direct
    # tolerate date prefix drift
    cands = sorted(
        (p for p in ckpts_root.iterdir() if p.is_dir() and f"-{job}-ETTh1-{config}" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"no ckpt dir for job={job} config={config} under {ckpts_root}")
    return cands[0]


def _resolve_guidance(run_dir: Path, subset: str) -> Path:
    for tmpl in GUIDANCE_GLOBS:
        cand = run_dir / tmpl.format(subset=subset)
        if cand.is_file():
            return cand
    alts = sorted(run_dir.glob("*_patch_guidance*.pt"))
    if alts:
        return alts[0]
    raise FileNotFoundError(f"no patch guidance under {run_dir}")


def _load_tuned(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.is_file():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        obj = json.load(f)
    return dict(obj.get("tuned_params") or {})


def _check_run(ckpts_root: Path, row: Dict[str, str]) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(ckpts_root, row["job"], row["config"])
    subset = "ETTh1"
    coarse = run_dir / subset / "coarse" / "best.pt"
    fine = run_dir / subset / "fine" / "best.pt"
    guide = None
    try:
        guide = _resolve_guidance(run_dir, subset)
    except FileNotFoundError:
        pass
    return {
        "stem": run_dir.name,
        "run_dir": str(run_dir),
        "coarse": coarse.is_file(),
        "fine": fine.is_file(),
        "guidance": bool(guide and guide.is_file()),
        "guidance_path": str(guide) if guide else None,
        "wandb_id": row["wandb_id"],
        "ok": coarse.is_file() and fine.is_file() and guide is not None and guide.is_file(),
    }


def _apply_length_schedule(state, pipeline_mod, tuned: Dict[str, Any], leaf_cfg: Dict[str, Any]) -> None:
    exp = leaf_cfg.get("experiment") or {}
    mode = str(tuned.get("binary_length_mode") or exp.get("binary_length_mode") or "none")
    g = float(tuned.get("binary_length_g") or exp.get("binary_length_g") or 1.0)
    scale = float(tuned.get("binary_length_scale") or exp.get("binary_length_scale") or 1.0)
    pipeline_mod.BINARY_LENGTH_MODE = mode
    pipeline_mod.BINARY_LENGTH_G = g
    pipeline_mod.BINARY_LENGTH_SCALE = scale
    state.binary_length_mode = mode
    state.binary_length_g = g
    state.binary_length_scale = scale


def _viz_one_run(
    *,
    row: Dict[str, str],
    ckpts_root: Path,
    out_root: Path,
    indices: List[int],
    ds,
    state,
    pipeline_mod,
    device: torch.device,
    sampler: str,
    steps: int,
    n_vars: int,
    upload_wandb: bool,
    project: str,
    entity: str,
) -> List[str]:
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.visualize_utils import (
        _load_staged_diffusion_from_ckpt,
        _plot_dual_concat_synth_panel,
    )

    run_dir = _resolve_run_dir(ckpts_root, row["job"], row["config"])
    subset = "ETTh1"
    coarse_pt = run_dir / subset / "coarse" / "best.pt"
    fine_pt = run_dir / subset / "fine" / "best.pt"
    guide = _resolve_guidance(run_dir, subset)
    if not coarse_pt.is_file() or not fine_pt.is_file():
        raise FileNotFoundError(
            f"missing best.pt under {run_dir}/{subset}/{{coarse,fine}}/ "
            "(weights live on Killarney scratch; sync or run there)"
        )

    leaf = REPO / "configs" / f"{row['config']}.yaml"
    # s43/s44 leaves may only override seed — fall back to g1p0 yaml
    if not leaf.is_file() and row["config"].endswith(("_s43", "_s44")):
        leaf = REPO / "configs" / "binary_noise_sched_ablation_past_native_g1p0.yaml"
    leaf_cfg = load_experiment_config(str(leaf)) if leaf.is_file() else {}
    tuned = _load_tuned(fine_pt.parent / "metadata.json")
    _apply_length_schedule(state, pipeline_mod, tuned, leaf_cfg)

    n_variates = len(state.variate_indices or [])
    coarse_model, _ = _load_staged_diffusion_from_ckpt(
        ckpt_path=str(coarse_pt),
        stage="coarse",
        itrans_ckpt_path=str(guide),
        n_vars=n_variates,
        device=device,
        tuned_params=tuned,
        guidance_type=getattr(state, "guidance_type", None),
    )
    fine_model, _ = _load_staged_diffusion_from_ckpt(
        ckpt_path=str(fine_pt),
        stage="fine",
        itrans_ckpt_path=str(guide),
        n_vars=n_variates,
        device=device,
        tuned_params=tuned,
        guidance_type=getattr(state, "guidance_type", None),
    )

    tag = f"past_native_g{row['g_label']}"
    out_dir = out_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    lb = int(state.lookback_length)
    saved: List[str] = []
    log_payload: Dict[str, Any] = {}

    for row_i, idx in enumerate(indices):
        past, future = ds[idx]
        if not torch.is_tensor(past):
            past = torch.as_tensor(past, dtype=torch.float32)
        if not torch.is_tensor(future):
            future = torch.as_tensor(future, dtype=torch.float32)
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)

        with torch.no_grad():
            n_steps = 1 if sampler == "anchor" else int(steps)
            coarse_out = coarse_model.generate(
                past_b, sampler=sampler, num_inference_steps=n_steps,
            )
            fine_out = fine_model.generate(
                past_b,
                sampler=sampler,
                num_inference_steps=n_steps,
                future_coarse_2d=coarse_out["future_2d_coarse"],
            )
            _pn, future_norm, _ = fine_model._normalize_sequence(past_b, future_b)
            gt_maps = fine_model._encode_staged_maps(future_norm)

        pred = fine_out.get("prediction", fine_out.get("prediction_global_norm"))
        if pred is None:
            raise KeyError("fine generate missing prediction")
        if "future_2d_coarse" not in coarse_out or "future_2d_fine" not in fine_out:
            raise KeyError("generate missing future_2d_coarse/fine")

        pred_np = pred[0].detach().cpu().numpy()
        pred_c = coarse_out["future_2d_coarse"][0].detach().cpu().numpy()
        pred_f = fine_out["future_2d_fine"][0].detach().cpu().numpy()
        gt_c = gt_maps["coarse"][0].detach().cpu().numpy()
        gt_f = gt_maps["fine"][0].detach().cpu().numpy()
        past_np = past.detach().cpu().numpy()
        future_np = future.detach().cpu().numpy()
        if pred_np.shape[-1] <= future_np.shape[-1]:
            future_core = future_np[..., -pred_np.shape[-1] :]
        else:
            future_core = future_np
        common = min(future_core.shape[-1], pred_np.shape[-1])
        future_core = future_core[..., -common:]
        pred_np = pred_np[..., -common:]

        path = out_dir / f"{tag}_sample{row_i:02d}_idx{idx}_maps.jpg"
        saved.append(
            _plot_dual_concat_synth_panel(
                past_np=past_np,
                future_core=future_core,
                pred=pred_np,
                gt_coarse=gt_c,
                gt_fine=gt_f,
                pred_coarse=pred_c,
                pred_fine=pred_f,
                lookback=lb,
                sample_idx=int(idx),
                stage="sequential_coarse_fine",
                sampler=sampler,
                output_path=str(path),
                variables_to_plot=n_vars,
                jpeg_dpi=100,
                title=(
                    f"past_native g={row['g_label']} | ETTh1 | window {idx} | "
                    f"{sampler} | GT vs pred coarse/fine CDF"
                ),
            )
        )
        logger.info("wrote %s", path)
        if upload_wandb:
            import wandb

            key = f"viz/gt_pred_maps/g{row['g_label']}/sample{row_i:02d}_idx{idx}"
            log_payload[key] = wandb.Image(str(path), caption=path.name)

    if upload_wandb:
        import wandb

        run = wandb.init(
            project=project,
            entity=entity,
            id=row["wandb_id"],
            resume="allow",
            job_type="gt_pred_maps_backfill",
        )
        wandb.log(log_payload)
        logger.info(
            "logged %d images → https://wandb.ai/%s/%s/runs/%s",
            len(log_payload),
            entity,
            project,
            row["wandb_id"],
        )
        run.finish()

    # free GPU between g's
    del coarse_model, fine_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts-root", default="", help="default: $SCRATCH/.../ckpts or results/ckpts")
    p.add_argument("--out-dir", default="")
    p.add_argument("--g-labels", default="", help="comma subset of g_label (default: all)")
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--windows", default="", help="explicit test indices, comma-separated")
    p.add_argument("--n-vars", type=int, default=3)
    p.add_argument("--sampler", default="anchor", choices=("anchor", "dpmpp", "ddim", "quad_t"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--device", default=None)
    p.add_argument("--wandb", action="store_true", help="resume each pipeline run and log images")
    p.add_argument("--project", default="ts-sandbox-leaderboard")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "calvincao"))
    p.add_argument("--check-only", action="store_true", help="print ckpt presence and exit")
    args = p.parse_args()

    ckpts_root = Path(args.ckpts_root).expanduser() if args.ckpts_root else _default_ckpts_root()
    ckpts_root = ckpts_root.resolve()
    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (REPO / "results" / "viz" / "past_native_etth1_gt_pred_maps")
    )

    want = {x.strip() for x in args.g_labels.split(",") if x.strip()} if args.g_labels else None
    rows = [r for r in RUNS if want is None or r["g_label"] in want]

    print(f"ckpts_root={ckpts_root}")
    status = [_check_run(ckpts_root, r) for r in rows]
    for r, st in zip(rows, status):
        flag = "OK" if st["ok"] else "MISSING"
        print(
            f"[{flag}] g={r['g_label']:>7} job={r['job']} "
            f"coarse={st['coarse']} fine={st['fine']} guide={st['guidance']} "
            f"→ {st['run_dir']}"
        )
    if args.check_only:
        n_ok = sum(1 for s in status if s["ok"])
        print(f"{n_ok}/{len(status)} runs have weights")
        sys.exit(0 if n_ok == len(status) else 1)

    missing = [r["g_label"] for r, s in zip(rows, status) if not s["ok"]]
    if missing:
        raise SystemExit(
            f"missing weights for g={missing}. "
            "Run on Killarney (or rsync best.pt + *_patch_guidance*.pt), "
            "or pass --ckpts-root to the scratch mirror."
        )

    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from models.diffusion_tsf.pipeline.state import PipelineState
    from models.diffusion_tsf.pipeline.visualize_utils import pick_sample_indices
    from models.diffusion_tsf.train_multivariate_pipeline import (
        generate_dataset_job,
        load_dataset,
    )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    base_yaml = REPO / "configs" / "binary_noise_sched_ablation_past_native_g1p0.yaml"
    cfg0 = load_experiment_config(
        str(base_yaml),
        cli_overrides={
            "dataset": "ETTh1",
            "checkpoint_dir": str(ckpts_root),
            "results_dir": str(out_root),
            "seed": args.seed,
        },
    )
    state = PipelineState.from_config(cfg0)
    state.dataset = "ETTh1"
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
    _train, val_ds, test_ds, norm_stats = load_dataset(
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
    logger.info("windows=%s split=%s n=%d wandb=%s", indices, args.split, len(ds), args.wandb)

    out_root.mkdir(parents=True, exist_ok=True)
    all_saved: List[str] = []
    for row in rows:
        logger.info("=== g=%s job=%s ===", row["g_label"], row["job"])
        all_saved.extend(
            _viz_one_run(
                row=row,
                ckpts_root=ckpts_root,
                out_root=out_root,
                indices=indices,
                ds=ds,
                state=state,
                pipeline_mod=pipeline_mod,
                device=device,
                sampler=args.sampler,
                steps=args.steps,
                n_vars=args.n_vars,
                upload_wandb=bool(args.wandb),
                project=args.project,
                entity=args.entity,
            )
        )
    logger.info("done — %d jpgs under %s", len(all_saved), out_root)


if __name__ == "__main__":
    main()
