#!/usr/bin/env python3
"""One-off robust texture eval for selected staged binary checkpoints vs MMPD."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.reused_paths import find_reused_binary_staged_root
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_mmpd_gaussian_anchor import (
    AnchorRun,
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    load_tsf_pack_pool,
    load_tsf_test_subset,
    make_eval_indices,
    make_pack_pool_indices,
    parse_pack_splits,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
    stable_dataset_seed,
    summarize_prediction_pack,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import (
    _load_staged_bundle,
    _window_lengths,
)


ROBUST_TEXTURE_KEYS = [
    "texture_increment_wasserstein",
    "texture_curvature_wasserstein",
    "texture_haar_detail_jsd",
    "texture_jump_plateau_distance",
    "texture_derivative_motif_jsd",
]

DEFAULT_SUBSET_DATASETS = (
    "ETTh1",
    "ETTh2",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "solar_Alabama",
)

DEFAULT_ANCHOR_CONFIG = "binary_anchor_stationary_flat_subsets_ema099"
DEFAULT_CKPT_BASE = REPO_ROOT / "results" / "ckpts"
DEFAULT_MMPD_OUTPUT_ROOT = REPO_ROOT / "results" / "datasets" / "06-13-binary-mmpd-subset-compare"


def resolve_staged_ckpt_dir(ckpt_base: Path, dataset: str, anchor_config: str) -> Path:
    """Prefer reused/binary/<stem>, else newest results/ckpts/*-{dataset}-{stem}."""
    reused = find_reused_binary_staged_root(anchor_config, dataset)
    if reused is not None:
        return Path(reused)
    if not ckpt_base.is_dir():
        raise FileNotFoundError(
            f"No reused binary root for {anchor_config}/{dataset} and missing ckpt_base {ckpt_base}"
        )
    matches = sorted(
        [
            p
            for p in ckpt_base.iterdir()
            if p.is_dir() and p.name.endswith(f"-{dataset}-{anchor_config}")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No ckpt dir matching *-{dataset}-{anchor_config} under {ckpt_base} "
            f"(also no reused/binary/{anchor_config})"
        )
    return matches[0]


def _binary_config_path(args: argparse.Namespace, dataset: str) -> str:
    by_ds = getattr(args, "binary_config_by_dataset", None) or {}
    if dataset in by_ds:
        return str(by_ds[dataset])
    cfg = getattr(args, "binary_config", None)
    if cfg:
        return str(cfg)
    anchor = getattr(args, "anchor_config", None)
    if anchor:
        return str(REPO_ROOT / "configs" / f"{anchor}.yaml")
    # Legacy fallback for old dual-scale one-offs.
    return str(REPO_ROOT / "configs" / "binary_dual_scale_staged.yaml")


def load_ordinal_ladder_for_run(
    args: argparse.Namespace,
    run: AnchorRun,
) -> Any:
    """Build the train-split global ordinal ladder used by binary ordinal decode."""
    lookback, horizon = dataset_window_lengths_for_run(args, run)
    config_path = _binary_config_path(args, run.dataset)
    subset_id = run_subset_id(run)
    state = _build_state(run.root, run.dataset, subset_id, config_path)
    resolve_pipeline_data_subset(state)
    if not bool(state.use_ordinal_window_norm):
        raise ValueError(
            f"{run.dataset}: binary config {config_path} does not enable ordinal_window_norm; "
            "cannot quantize MMPD onto ordinal bins"
        )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    _train_ds, _val_ds, _test_ds, norm_stats = load_dataset(
        run.dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{run.dataset}: load_dataset did not return ordinal_ladder")
    return ladder


# Legacy checkpoints kept for old one-off scripts.
DEFAULT_STAGED_CKPTS: Mapping[str, str] = {
    "ETTh1": "/scratch/ccao87/ts-sandbox/results/ckpts/06-02-3849018-ETTh1-binary_dual_scale_staged",
    "dalia": "/scratch/ccao87/ts-sandbox/results/ckpts/06-02-3849021-dalia-binary_dual_scale_staged",
    "traffic": "/scratch/ccao87/ts-sandbox/results/ckpts/06-02-3849023-traffic-binary_dual_scale_staged",
    "exchange_rate": "/scratch/ccao87/ts-sandbox/results/ckpts/06-02-3852949-exchange_rate-binary_dual_scale_staged",
    "PeMS": "/scratch/ccao87/ts-sandbox/results/ckpts/06-02-3852953-PeMS-binary_dual_scale_staged",
}


def staged_anchor_run(dataset: str, checkpoint_dir: Path, test_stride: int) -> Tuple[AnchorRun, Dict[str, Any]]:
    sub = _load_staged_bundle(checkpoint_dir, dataset)
    subset_id = str(sub["subset_id"])
    guidance_path, guidance_type = _resolve_guidance_ckpt(checkpoint_dir, subset_id, "auto")

    meta = dict(sub["fine_metadata"])
    meta["dataset_name"] = dataset
    meta["dataset"] = dataset
    meta["subset_id"] = subset_id
    meta["variate_indices"] = [int(i) for i in sub["variate_indices"]]
    meta["guidance_type"] = guidance_type
    data_subset = dict(meta.get("data_subset") or {})
    data_subset["test_stride"] = int(test_stride)
    meta["data_subset"] = data_subset

    run = AnchorRun(
        variant="binary",
        dataset=dataset,
        root=checkpoint_dir,
        subset_dir=Path(sub["fine_pt"]).parent,
        best_pt=Path(sub["fine_pt"]),
        itrans_pt=guidance_path,
        metadata=meta,
    )
    return run, sub


def make_indices(args: argparse.Namespace, run: AnchorRun) -> List[int]:
    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    pool, series_starts, splits, part_lengths, _stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
    )
    fraction = float(getattr(args, "pack_fraction", None) or args.test_fraction)
    indices = make_pack_pool_indices(
        len(pool),
        fraction=fraction,
        seed=stable_dataset_seed(args.seed, run.dataset),
        max_items=args.test_max_items,
    )
    selected_starts = series_starts[np.asarray(indices, dtype=np.int64)]
    print(
        f"[pack-pool] {run.dataset}: splits={splits} part_lengths={part_lengths} "
        f"pool={len(pool)} kept={len(indices)} fraction={fraction:g} "
        f"abs_start_range=[{int(selected_starts.min())},{int(selected_starts.max())}]",
        flush=True,
    )
    # Stash for callers that materialize packs in the same process.
    args._pack_series_starts_full = series_starts
    args._pack_splits_resolved = splits
    return indices


def dataset_window_lengths_for_run(args: argparse.Namespace, run: AnchorRun) -> Tuple[int, int]:
    return int(args.lookback), int(args.horizon)


def _prediction_tensor(result: Dict[str, torch.Tensor]) -> torch.Tensor:
    return result.get("prediction_global_norm", result["prediction"])


def generate_staged_forecast(
    coarse_model: Any,
    fine_model: Any,
    past: torch.Tensor,
    *,
    vertical_dual: bool,
    fine_seed: Optional[int] = None,
    **generate_kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Generate one final forecast through the checkpoint's actual refinement path."""
    if vertical_dual:
        # The vertical model samples the stacked 16+16 canvas and performs the
        # fine decode / overlap trim internally.  It must not be fed back into
        # a separately-instantiated 16-row fine model.
        return coarse_model.generate(past, **generate_kwargs)
    coarse_out = coarse_model.generate(past, **generate_kwargs)
    if fine_seed is not None:
        torch.manual_seed(int(fine_seed))
    return fine_model.generate(
        past,
        future_coarse_2d=coarse_out["future_2d_coarse"],
        **generate_kwargs,
    )


def evaluate_staged_binary(
    args: argparse.Namespace,
    run: AnchorRun,
    sub: Dict[str, Any],
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from torch.utils.data import Subset

    raw_path = args.output_dir / "raw" / f"binary_staged_{run.dataset}.npz"
    if raw_path.exists() and not args.force_binary_eval:
        with np.load(raw_path) as data:
            return {key: data[key] for key in data.files}

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    pool, series_starts_full, splits, _part_lengths, norm_stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
        ordinal_tie_atol=1e-6,
        use_ordinal_window_norm=None,
    )
    subset = Subset(pool, list(indices))
    # Keep micro-batches small for lb336/hz720 maps.
    batch_size = min(int(args.binary_batch_size), 2)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    config_path = _binary_config_path(args, run.dataset)
    subset_id = run_subset_id(run)
    state = _build_state(run.root, run.dataset, subset_id, config_path)
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    # Rebuild ladder under the same binary config knobs as training.
    _, _, _test_ds, norm_stats = load_dataset(
        run.dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    guidance_type = str(run.metadata.get("guidance_type") or "auto")
    guidance_path, guidance_type = _resolve_guidance_ckpt(run.root, subset_id, guidance_type)
    guidance_model = load_wrapped_guidance(
        str(guidance_path),
        len(run_variate_indices(run)),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    # A vertical-dual run has one H=(Hc+Hf) model, not independent Hc/Hf
    # checkpoints.  _load_staged_bundle deliberately aliases coarse_pt/fine_pt
    # to that one file for metadata compatibility, so dispatch on ``stage``
    # before constructing the model.  Loading it as ``coarse`` or ``fine``
    # silently drops the 32-row decoder parameters and invalidates the sample.
    vertical_dual = (
        str(sub.get("stage") or "") == "vertical_dual"
        or bool(getattr(state, "use_vertical_dual_concat", False))
    )
    if vertical_dual:
        coarse_model = _load_stage_model(
            state,
            "vertical_dual",
            Path(sub["coarse_pt"]),
            guidance_model,
            len(run_variate_indices(run)),
            device,
            strict_non_guidance_shapes=True,
        )
        fine_model = coarse_model
    else:
        coarse_model = _load_stage_model(
            state, "coarse", Path(sub["coarse_pt"]), guidance_model, len(run_variate_indices(run)), device,
            strict_non_guidance_shapes=True,
        )
        fine_model = _load_stage_model(
            state, "fine", Path(sub["fine_pt"]), guidance_model, len(run_variate_indices(run)), device,
            strict_non_guidance_shapes=True,
        )
    # Pool windows are z-score series (not pre-ranked) even under ordinal configs.
    for m in (coarse_model, fine_model):
        m._ordinal_input_is_ranked = False
        m._ordinal_apply_ood_shift = bool(state.use_ordinal_window_norm)

    prob_kwargs = {"sampler": args.probabilistic_sampler, "num_inference_steps": args.num_sampling_steps}
    y_true_all: List[np.ndarray] = []
    det_all: List[np.ndarray] = []
    samples_all: List[np.ndarray] = []
    progress = EvalProgress(f"binary-staged/{run.dataset}", len(loader))
    print(
        f"[binary-staged] {run.dataset}: windows={len(indices)} batches={len(loader)} "
        f"samples={args.sample_num} pack_splits={splits} stride_train={run_train_stride(run)} "
        f"config={config_path}",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            t_batch = time.time()
            past = past.to(device)
            future = future.to(device)
            K = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
            if K > 0:
                future = future[..., K:]
            y_true_all.append(future.cpu().numpy())

            torch.manual_seed(args.seed + batch_idx)
            fine_det = generate_staged_forecast(
                coarse_model,
                fine_model,
                past,
                vertical_dual=vertical_dual,
                sampler="anchor",
            )
            det_all.append(_prediction_tensor(fine_det).cpu().numpy())

            batch_samples = []
            for sample_idx in range(args.sample_num):
                seed = args.seed + batch_idx * 1009 + sample_idx * 17
                torch.manual_seed(seed)
                fine_sample = generate_staged_forecast(
                    coarse_model,
                    fine_model,
                    past,
                    vertical_dual=vertical_dual,
                    fine_seed=seed,
                    **prob_kwargs,
                )
                batch_samples.append(_prediction_tensor(fine_sample).cpu().numpy())
            samples_all.append(np.stack(batch_samples, axis=2))

            progress.maybe_log(
                batch_idx + 1,
                extra=(
                    f"last_batch={fmt_duration(time.time() - t_batch)} "
                    f"elapsed={fmt_duration(time.time() - t0)}"
                ),
            )
    progress.done(extra=f"writing {raw_path}")

    idx_arr = np.asarray(indices, dtype=np.int64)
    pack = {
        "y_true": np.concatenate(y_true_all, axis=0),
        "deterministic": np.concatenate(det_all, axis=0),
        "samples": np.concatenate(samples_all, axis=0),
        "indices": idx_arr,
        "series_starts": series_starts_full[idx_arr],
        "pack_splits": np.asarray(splits),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **pack)
    return pack


def summarize(pack: Dict[str, np.ndarray], args: argparse.Namespace, dataset: str) -> Dict[str, float]:
    return summarize_prediction_pack(
        pack,
        gmm_components=args.gmm_components,
        seed=stable_dataset_seed(args.seed, dataset),
        topk_max=args.topk_max,
        texture_per_sample=True,
    )


def write_outputs(args: argparse.Namespace, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    fields = [
        "dataset",
        "model",
        "n_windows",
        "n_variates",
        "n_samples",
        *ROBUST_TEXTURE_KEYS,
        *[f"prob_{key}" for key in ROBUST_TEXTURE_KEYS],
    ]
    with (args.output_dir / "robust_texture_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in results:
            for model in ("binary_staged", "mmpd"):
                row = {"dataset": dataset, "model": model}
                metrics = results[dataset][model]
                row.update({key: metrics.get(key) for key in fields if key not in row})
                writer.writerow(row)

    print("\nRobust texture summary")
    print(",".join(fields))
    for dataset in results:
        for model in ("binary_staged", "mmpd"):
            metrics = results[dataset][model]
            row = [dataset, model]
            for key in fields[2:]:
                val = metrics.get(key, float("nan"))
                row.append(f"{val:.6f}" if isinstance(val, (float, int)) else "")
            print(",".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_STAGED_CKPTS))
    for dataset, path in DEFAULT_STAGED_CKPTS.items():
        parser.add_argument(f"--{dataset}-ckpt", type=Path, default=Path(path))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "datasets" / "06-03-trend-robust-texture-staged-vs-mmpd")
    parser.add_argument("--mmpd-output-root", type=Path, default=REPO_ROOT / "results" / "datasets" / "06-01-mmpd-binary-aligned")
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=2)
    parser.add_argument("--sample-num", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--probabilistic-sampler", choices=["quad_t", "ddim_quad", "ddim", "ddpm"], default="quad_t")
    parser.add_argument("--gmm-components", type=int, default=1)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--topk-max", type=int, default=3)
    parser.add_argument("--binary-batch-size", type=int, default=8)
    parser.add_argument(
        "--binary-config",
        type=str,
        default=str(REPO_ROOT / "configs" / "binary_dual_scale_staged.yaml"),
        help="Leaf YAML used to build PipelineState for binary staged generate()",
    )
    parser.add_argument("--mmpd-eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-binary-eval", action="store_true")
    parser.add_argument("--force-mmpd-eval", action="store_true")
    parser.add_argument("--no-update-mmpd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [dataset for dataset in args.datasets if dataset not in DEFAULT_STAGED_CKPTS]
    if unknown:
        raise ValueError(f"No default staged checkpoint for: {', '.join(unknown)}")
    if args.sample_num < 1:
        raise ValueError("--sample-num must be >= 1")

    ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    manifest: Dict[str, Any] = {
        "datasets": args.datasets,
        "test_fraction": args.test_fraction,
        "test_stride": args.test_stride,
        "sample_num": args.sample_num,
        "mmpd_output_root": str(args.mmpd_output_root),
        "staged_ckpts": {},
    }

    for dataset in args.datasets:
        ckpt_dir = getattr(args, f"{dataset}_ckpt").resolve()
        run, sub = staged_anchor_run(dataset, ckpt_dir, args.test_stride)
        manifest["staged_ckpts"][dataset] = str(ckpt_dir)
        indices = make_indices(args, run)
        print(
            f"\n[{dataset}] subset={run_subset_id(run)} variates={len(run_variate_indices(run))} "
            f"train_stride={run_train_stride(run)} test_stride={run_test_stride(run)} "
            f"indices={len(indices)}",
            flush=True,
        )

        binary_pack = evaluate_staged_binary(args, run, sub, indices, device)
        binary_metrics = summarize(binary_pack, args, dataset)

        mmpd_pack = run_mmpd_eval(args, run, indices)
        mmpd_metrics = summarize(mmpd_pack, args, dataset)

        results[dataset] = {
            "binary_staged": binary_metrics,
            "mmpd": mmpd_metrics,
        }
        write_outputs(args, results)

    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
