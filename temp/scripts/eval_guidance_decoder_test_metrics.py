#!/usr/bin/env python3
"""MSE/MAE of canvas128 patch-guidance decoder alone on each dataset test fold.

Decoder input = window-normalized GT lookback (same space as patch_guidance
finetune). Predictions are scored in **dataset-global-z** (train-slice mean/std
from `load_dataset` — same space as staged_eval `prediction_global_norm`) by
undoing the per-window affine. Window-norm MSE/MAE are also reported.

No coarse/refine diffusion sampling.

Example:
  python temp/scripts/eval_guidance_decoder_test_metrics.py --all
  python temp/scripts/eval_guidance_decoder_test_metrics.py --datasets ETTh1,ETTh2 --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    _patch_guidance_batch,
    load_dataset,
    load_patch_guidance_from_checkpoint,
    resolve_pipeline_data_subset,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    ckpt_dir: str
    config: str


# Live canvas128 packs (same stems as lean-disc / viz_c128 helpers).
CANVAS128_SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "ETTh1",
        "results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
    ),
    DatasetSpec(
        "ETTh2",
        "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml",
    ),
    DatasetSpec(
        "electricity",
        "results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml",
    ),
    DatasetSpec(
        "traffic",
        "results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml",
    ),
    DatasetSpec(
        "exchange_rate",
        "results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml",
    ),
    DatasetSpec(
        "PeMS",
        "results/ckpts/08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml",
    ),
    DatasetSpec(
        "solar_Alabama",
        "results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml",
    ),
    DatasetSpec(
        "ETTm1",
        "results/ckpts/08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml",
    ),
    DatasetSpec(
        "ETTm2",
        "results/ckpts/08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2",
        "configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml",
    ),
)


def _resolve_guidance_ckpt(ckpt_dir: Path, subset_id: str, dataset: str) -> Path:
    wanted = ckpt_dir / f"{subset_id}_patch_guidance.pt"
    if wanted.is_file():
        return wanted
    for legacy in (
        ckpt_dir / f"{dataset}_patch_guidance.pt",
        ckpt_dir / "patch_guidance.pt",
        ckpt_dir / f"{subset_id}_patch_guidance_hp_best.pt",
        ckpt_dir / f"{dataset}_patch_guidance_hp_best.pt",
    ):
        if legacy.is_file():
            print(f"[guidance] {dataset}: using {legacy.name} (wanted {wanted.name})", flush=True)
            return legacy
    raise FileNotFoundError(
        f"{dataset}: missing guidance ckpt under {ckpt_dir} "
        f"(expected {wanted.name})"
    )


def _select_specs(want: Optional[Sequence[str]]) -> List[DatasetSpec]:
    if not want:
        return list(CANVAS128_SPECS)
    want_set = {w.strip() for w in want if w.strip()}
    known = {s.dataset for s in CANVAS128_SPECS}
    unknown = sorted(want_set - known)
    if unknown:
        raise ValueError(f"unknown dataset(s): {unknown}; known={sorted(known)}")
    return [s for s in CANVAS128_SPECS if s.dataset in want_set]


def _window_affine_from_past(past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match `_window_norm_past_future` center/std (mean center, std floor / low-var)."""
    if pipeline_mod.WINDOW_NORM_CENTER == "last":
        center = past[..., -1:]
    elif pipeline_mod.WINDOW_NORM_CENTER == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {pipeline_mod.WINDOW_NORM_CENTER!r}")
    past_std = past.std(dim=-1, keepdim=True)
    thr = float(pipeline_mod.WINDOW_NORM_LOW_VAR_THRESHOLD)
    floor = float(pipeline_mod.WINDOW_NORM_STD_FLOOR)
    unit_v = float(pipeline_mod.WINDOW_NORM_LOW_VAR_UNIT_STD)
    if thr > 0.0:
        std_floor = past_std.clamp_min(floor)
        unit = torch.full_like(past_std, unit_v)
        low_var = past_std < thr
        flat = past_std <= floor
        std = torch.where(flat | low_var, unit, std_floor)
    else:
        std = past_std.clamp_min(floor)
    return center, std


@torch.no_grad()
def eval_one(
    spec: DatasetSpec,
    *,
    device: torch.device,
    batch_size: int,
    smoke_test: bool,
    max_batches: Optional[int],
    test_stride_override: Optional[int],
) -> Dict[str, Any]:
    cfg_path = REPO_ROOT / spec.config
    ckpt_dir = REPO_ROOT / spec.ckpt_dir
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{spec.dataset}: missing config {cfg_path}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"{spec.dataset}: missing ckpt dir {ckpt_dir}")

    cfg = load_experiment_config(str(cfg_path), cli_overrides={"dataset": spec.dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = spec.dataset
    state.checkpoint_dir = str(ckpt_dir.resolve())
    resolve_pipeline_data_subset(state)
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    if bool(state.use_ordinal_window_norm):
        raise RuntimeError(
            f"{spec.dataset}: ordinal window norm unexpected for canvas128 leaf "
            f"{spec.config}"
        )
    if not bool(state.use_window_normalization):
        raise RuntimeError(f"{spec.dataset}: window normalization must be on")

    subset_id = state.subset_id or spec.dataset
    variate_indices = list(state.variate_indices or [])
    if not variate_indices:
        raise RuntimeError(f"{spec.dataset}: empty variate_indices after subset resolve")

    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))
    # Default stride 4 matches MMPD / lean-disc fair packs. Canvas128 staged_eval
    # YAML often uses stride 16 + fraction 0.25 (expensive diffusion subsample) —
    # override with --test-stride if you want that sparse pool.
    eval_stride = 4 if test_stride_override is None else int(test_stride_override)

    guidance_path = _resolve_guidance_ckpt(ckpt_dir, subset_id, spec.dataset)
    stack = load_patch_guidance_from_checkpoint(
        str(guidance_path), len(variate_indices), device,
    )

    _, _, test_ds, _ = load_dataset(
        state.dataset,
        variate_indices,
        lookback=state.lookback_length,
        horizon=state.forecast_length,
        stride=train_stride,
        test_stride=eval_stride,
        lookback_overlap=state.lookback_overlap,
        ordinal_tie_atol=float(getattr(state, "ordinal_tie_atol", 1e-6)),
        use_ordinal_window_norm=False,
    )

    loader_bs = batch_size
    if smoke_test:
        loader_bs = 1
    loader = DataLoader(test_ds, batch_size=loader_bs, shuffle=False, num_workers=0)

    # window-norm space (decoder native) + dataset-global-z (loader / staged_eval space)
    sse_w = sae_w = 0.0
    sse_g = sae_g = 0.0
    n_elem = 0
    n_windows = 0
    n_batches = 0
    data_is_ranked = bool(getattr(test_ds, "yields_ordinal_ranks", False))
    K = int(pipeline_mod.LOOKBACK_OVERLAP)

    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        if past.ndim == 2:
            past = past.unsqueeze(1)
            future = future.unsqueeze(1)
        past_norm, y_true = _patch_guidance_batch(
            past,
            future,
            device,
            apply_ood_shift=False,
            data_is_ranked=data_is_ranked,
        )
        pred_w = stack.forecast(past_norm)
        if pred_w.shape != y_true.shape:
            raise RuntimeError(
                f"{spec.dataset}: pred {tuple(pred_w.shape)} != target {tuple(y_true.shape)}"
            )
        diff_w = pred_w - y_true
        sse_w += float(diff_w.pow(2).sum().item())
        sae_w += float(diff_w.abs().sum().item())

        # Loader tensors are already train-slice dataset-z (staged eval global_norm).
        # Undo per-window affine to report MSE/MAE in that comparable space.
        center, std = _window_affine_from_past(past)
        pred_g = pred_w * std + center
        y_g = future[..., K : K + y_true.shape[-1]]
        if pred_g.shape != y_g.shape:
            raise RuntimeError(
                f"{spec.dataset}: global pred {tuple(pred_g.shape)} != "
                f"target {tuple(y_g.shape)}"
            )
        diff_g = pred_g - y_g
        sse_g += float(diff_g.pow(2).sum().item())
        sae_g += float(diff_g.abs().sum().item())

        n_elem += int(diff_w.numel())
        n_windows += int(past.shape[0])
        n_batches += 1
        if smoke_test and n_batches >= 1:
            break
        if max_batches is not None and n_batches >= max_batches:
            break

    if n_elem <= 0:
        raise RuntimeError(f"{spec.dataset}: empty eval (no elements)")

    mse_w = sse_w / n_elem
    mae_w = sae_w / n_elem
    mse_g = sse_g / n_elem
    mae_g = sae_g / n_elem
    row = {
        "dataset": spec.dataset,
        "subset_id": subset_id,
        "n_variates": len(variate_indices),
        "lookback": int(state.lookback_length),
        "horizon": int(state.forecast_length),
        "lookback_overlap": int(state.lookback_overlap),
        "test_stride": eval_stride,
        "window_norm_center": str(getattr(state, "window_norm_center", "mean")),
        "metric_space_primary": "dataset_global_z",
        "decoder_input": "GT_window_norm_past",
        "target_global": "GT_dataset_z_future_core",
        "guidance_ckpt": str(guidance_path.relative_to(REPO_ROOT)),
        "config": spec.config,
        "ckpt_dir": spec.ckpt_dir,
        "n_windows": n_windows,
        "n_elements": n_elem,
        "mse": mse_g,
        "mae": mae_g,
        "mse_window_norm": mse_w,
        "mae_window_norm": mae_w,
        "mse_dataset_global_z": mse_g,
        "mae_dataset_global_z": mae_g,
        "smoke_test": bool(smoke_test),
    }
    print(
        f"[ok] {spec.dataset}: global_z mse={mse_g:.6f} mae={mae_g:.6f} "
        f"(win-norm mse={mse_w:.6f} mae={mae_w:.6f}) "
        f"n_windows={n_windows} stride={eval_stride} ckpt={guidance_path.name}",
        flush=True,
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=str, default="", help="Comma-separated; default all")
    p.add_argument("--all", action="store_true", help="Eval all canvas128 specs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument(
        "--test-stride",
        type=int,
        default=4,
        help="Test window stride (default 4; canvas128 staged_eval YAML often uses 16)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="temp/lean_disc_c128_results/guidance_decoder_test_metrics",
    )
    args = p.parse_args()

    want = None
    if args.datasets.strip():
        want = [x.strip() for x in args.datasets.split(",") if x.strip()]
    # else: None → all canvas128 specs (--all is optional / documented)

    specs = _select_specs(want)
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("device=cpu", flush=True)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for spec in specs:
        try:
            rows.append(
                eval_one(
                    spec,
                    device=device,
                    batch_size=args.batch_size,
                    smoke_test=args.smoke_test,
                    max_batches=args.max_batches,
                    test_stride_override=int(args.test_stride),
                )
            )
        except Exception as e:
            print(f"[skip] {spec.dataset}: {e}", flush=True)
            skipped.append({"dataset": spec.dataset, "error": str(e)})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_utc": stamp,
        "repo": str(REPO_ROOT),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "metric_space": "dataset_global_z",
        "protocol": {
            "decoder_input": "GT lookback after window-norm (mean center / std floor)",
            "target_primary": (
                "GT future core in dataset-global-z (train-slice mean/std; "
                "same space as staged_eval prediction_global_norm)"
            ),
            "also_reported": "window_norm MSE/MAE (decoder native training space)",
            "no_diffusion": True,
            "test_stride_default": 4,
        },
        "rows": rows,
        "skipped": skipped,
    }
    out_json = out_dir / f"summary_{stamp}.json"
    latest = out_dir / "summary_latest.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    latest.write_text(json.dumps(payload, indent=2) + "\n")

    # Markdown table for quick paste (primary = dataset-global-z)
    md_lines = [
        "# Guidance decoder alone — test MSE/MAE (dataset-global-z)",
        "",
        f"Created: `{stamp}`  ",
        f"Job: `{payload['slurm_job_id'] or 'local'}`  ",
        "",
        "Primary metrics match staged-eval `prediction_global_norm` space ",
        "(train-slice dataset z). Window-norm columns are decoder-native.",
        "",
        "| dataset | MSE (global-z) | MAE (global-z) | MSE (win-norm) | MAE (win-norm) | n_windows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['mse_dataset_global_z']:.6f} | {r['mae_dataset_global_z']:.6f} | "
            f"{r['mse_window_norm']:.6f} | {r['mae_window_norm']:.6f} | {r['n_windows']} |"
        )
    if skipped:
        md_lines.extend(["", "## Skipped", ""])
        for s in skipped:
            md_lines.append(f"- **{s['dataset']}**: {s['error']}")
    md_path = out_dir / "summary_latest.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"[done] wrote {out_json}", flush=True)
    print(f"[done] wrote {latest}", flush=True)
    print(f"[done] wrote {md_path}", flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
