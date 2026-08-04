#!/usr/bin/env python3
"""Learned discriminator texture eval for staged binary vs MMPD outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dual_scale_bin_filter import (
    BIN_MATCH_CHOICES,
    align_mmpd_to_binary_dataset_norm,
    apply_bin_match_to_bundle,
)
from utils.binary_disc_debias import (
    debias_binary_staged_fakes,
    quantize_to_ordinal_ladder,
    resolve_dual_scale_bin_params,
)
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    load_tsf_pack_pool,
    mmpd_data_split,
    mmpd_staged_filename_for_run,
    load_tsf_test_subset,
    parse_pack_splits,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import (
    DEFAULT_ANCHOR_CONFIG,
    DEFAULT_CKPT_BASE,
    DEFAULT_MMPD_OUTPUT_ROOT,
    DEFAULT_SUBSET_DATASETS,
    _binary_config_path,
    dataset_window_lengths_for_run,
    evaluate_staged_binary,
    load_ordinal_ladder_for_run,
    make_indices,
    resolve_staged_ckpt_dir,
    staged_anchor_run,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration


FAKE_SOURCES = ("binary_staged", "mmpd")
LOG2 = math.log(2.0)


@dataclass
class RawBundle:
    run: Any
    sub: Dict[str, Any]
    indices: List[int]
    past: np.ndarray
    y_true_by_source: Dict[str, np.ndarray]
    fakes: Dict[str, np.ndarray]
    series_starts: np.ndarray
    pack_splits: List[str]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def pack_path(raw_eval_dir: Path, fake_source: str, dataset: str) -> Path:
    if fake_source == "binary_staged":
        return raw_eval_dir / "raw" / f"binary_staged_{dataset}.npz"
    if fake_source == "mmpd":
        return raw_eval_dir / "raw" / f"mmpd_{dataset}.npz"
    raise ValueError(f"unknown fake source: {fake_source}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validate_stochastic_pack(path: Path, pack: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in ("y_true", "samples", "indices") if key not in pack]
    if missing:
        raise KeyError(f"{path} missing required arrays: {missing}")
    samples = pack["samples"]
    if samples.ndim != 4 or samples.shape[2] < 1:
        raise ValueError(f"{path} samples must have shape [N, C, S, H] with S>=1, got {samples.shape}")
    if pack["y_true"].shape != samples[:, :, 0, :].shape:
        raise ValueError(
            f"{path} y_true/sample0 shape mismatch: "
            f"{pack['y_true'].shape} vs {samples[:, :, 0, :].shape}"
        )


def validate_variate_alignment(
    dataset: str,
    run: Any,
    sub: Mapping[str, Any],
    past: np.ndarray,
    y_true_by_source: Mapping[str, np.ndarray],
    fakes: Mapping[str, np.ndarray],
) -> None:
    expected = [int(i) for i in run_variate_indices(run)]
    if not expected:
        raise ValueError(f"{dataset}: staged metadata has no variate_indices")

    for name, value in (
        ("bundle", sub.get("variate_indices")),
        ("fine metadata", (sub.get("fine_metadata") or {}).get("variate_indices")),
        ("coarse metadata", (sub.get("coarse_metadata") or {}).get("variate_indices")),
    ):
        if value is None:
            continue
        observed = [int(i) for i in value]
        if observed != expected:
            raise ValueError(
                f"{dataset}: {name} variate_indices {observed} do not match staged run {expected}"
            )

    n_vars = len(expected)
    if past.shape[1] != n_vars:
        raise ValueError(
            f"{dataset}: past has {past.shape[1]} variates but staged subset has "
            f"{n_vars}: {expected}"
        )
    for fake_source, y_true in y_true_by_source.items():
        fake = fakes[fake_source]
        if y_true.shape[1] != n_vars:
            raise ValueError(
                f"{dataset}/{fake_source}: y_true has {y_true.shape[1]} variates "
                f"but staged subset has {n_vars}: {expected}"
            )
        if fake.shape[1] != n_vars:
            raise ValueError(
                f"{dataset}/{fake_source}: fake has {fake.shape[1]} variates "
                f"but staged subset has {n_vars}: {expected}"
            )


def saved_indices(raw_eval_dir: Path, dataset: str) -> Optional[List[int]]:
    for fake_source in FAKE_SOURCES:
        path = pack_path(raw_eval_dir, fake_source, dataset)
        if path.is_file():
            pack = load_npz(path)
            if "indices" in pack:
                return [int(i) for i in pack["indices"].tolist()]
    index_json = raw_eval_dir / "raw" / f"indices_{dataset}_mmpd_eval.json"
    if index_json.is_file():
        return [int(i) for i in load_json(index_json)]
    return None


def _parse_dataset_map(raw: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw:
        return out
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"expected dataset:value in map entry, got {item!r}")
        dataset, value = item.split(":", 1)
        out[dataset.strip()] = value.strip()
    return out


def _anchor_config_for(args: argparse.Namespace, dataset: str) -> str:
    by_ds = getattr(args, "anchor_config_by_dataset", None) or {}
    return str(by_ds.get(dataset, args.anchor_config))


def _binary_config_for(args: argparse.Namespace, dataset: str) -> str:
    by_ds = getattr(args, "binary_config_by_dataset", None) or {}
    if dataset in by_ds:
        return str(by_ds[dataset])
    cfg = getattr(args, "binary_config", None)
    if cfg:
        return str(cfg)
    return str(REPO_ROOT / "configs" / f"{_anchor_config_for(args, dataset)}.yaml")


def _link_or_copy(src: Path, dest: Path) -> None:
    """Idempotent link/copy safe under concurrent shard jobs."""
    src_resolved = src.resolve()
    if dest.is_symlink():
        try:
            if dest.resolve() == src_resolved:
                return
        except OSError:
            pass
    elif dest.is_file():
        try:
            if dest.samefile(src_resolved):
                return
        except OSError:
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        try:
            tmp.symlink_to(src_resolved)
        except OSError:
            import shutil

            shutil.copy2(src_resolved, tmp)
        os.replace(tmp, dest)
        print(f"[import] linked {dest} -> {src}", flush=True)
    except Exception:
        if tmp.exists() or tmp.is_symlink():
            try:
                tmp.unlink()
            except OSError:
                pass
        # Another shard may have won the race with an identical target.
        if dest.is_symlink() or dest.is_file():
            try:
                if dest.resolve() == src_resolved or dest.samefile(src_resolved):
                    return
            except OSError:
                pass
        raise


def import_mmpd_packs(args: argparse.Namespace) -> None:
    """Copy/symlink existing mmpd_*.npz (+ indices JSON) into raw_eval_dir/raw/ when missing."""
    src_root = getattr(args, "import_mmpd_packs_from", None)
    if src_root is None:
        return
    src_root = Path(src_root)
    dest_raw = Path(args.raw_eval_dir) / "raw"
    dest_raw.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    for dataset in args.datasets:
        dest = pack_path(args.raw_eval_dir, "mmpd", dataset)
        if not (dest.is_file() and not args.force_raw_eval):
            candidates = [
                src_root / "raw" / f"mmpd_{dataset}.npz",
                src_root / f"mmpd_{dataset}.npz",
            ]
            src = next((p for p in candidates if p.is_file()), None)
            if src is None:
                missing.append(dataset)
                print(f"[import] no mmpd pack for {dataset} under {src_root}", flush=True)
            else:
                _link_or_copy(src, dest)
        idx_name = f"indices_{dataset}_mmpd_eval.json"
        dest_idx = dest_raw / idx_name
        if dest_idx.is_file() and not args.force_raw_eval:
            continue
        for cand in (src_root / "raw" / idx_name, src_root / idx_name):
            if cand.is_file():
                _link_or_copy(cand, dest_idx)
                break
    if missing and "mmpd" in set(getattr(args, "fake_sources", ()) or ()):
        raise FileNotFoundError(
            "Missing mmpd_*.npz under --import-mmpd-packs-from for: "
            f"{', '.join(missing)}. Refusing silent MMPD regeneration. "
            "Point --import-mmpd-packs-from at a dir with sample packs "
            "(e.g. results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset), "
            "or omit --import-mmpd-packs-from to allow regeneration."
        )


def raw_eval_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    out = copy.copy(args)
    out.output_dir = args.raw_eval_dir
    out.mmpd_output_root = args.mmpd_output_root
    out.mmpd_backbone = getattr(args, "mmpd_backbone", "Decoder")
    out.patch_size = None  # dataset rules (e.g. electricity/traffic patch 24)
    out.force_binary_eval = args.force_raw_eval
    out.force_mmpd_eval = args.force_raw_eval
    out.binary_batch_size = args.raw_binary_batch_size
    out.mmpd_eval_batch_size = args.raw_mmpd_batch_size
    out.sample_num = 1
    out.anchor_config = _anchor_config_for(args, dataset)
    out.binary_config = _binary_config_for(args, dataset)
    out.binary_config_by_dataset = {
        dataset: out.binary_config,
        **dict(getattr(args, "binary_config_by_dataset", None) or {}),
    }
    out.pack_splits = getattr(args, "pack_splits", "test")
    out.pack_fraction = getattr(args, "pack_fraction", None)
    # The ordinal MMPD campaign was trained without instance normalization.
    # Keep that model-internal representation separate from the final binary
    # dataset-z output coordinate, which is harmonized below.
    out.use_ordinal_window_norm = bool(getattr(args, "mmpd_ordinal_norm", False))
    out.mmpd_instance_norm = bool(getattr(args, "mmpd_instance_norm", False))
    if out.use_ordinal_window_norm and out.mmpd_instance_norm:
        raise ValueError("--mmpd-ordinal-norm and --mmpd-instance-norm are mutually exclusive")
    return out


def ensure_raw_packs(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any], List[int], Dict[str, Dict[str, np.ndarray]], np.ndarray, List[str]]:
    anchor_config = _anchor_config_for(args, dataset)
    ckpt_dir = resolve_staged_ckpt_dir(args.ckpt_base, dataset, anchor_config)
    run, sub = staged_anchor_run(dataset, ckpt_dir, args.test_stride)
    indices = saved_indices(args.raw_eval_dir, dataset)
    eval_args = raw_eval_args(args, dataset)
    if indices is None:
        indices = make_indices(eval_args, run)

    if "binary_staged" in args.fake_sources:
        binary_path = pack_path(args.raw_eval_dir, "binary_staged", dataset)
        if args.force_raw_eval or not binary_path.is_file():
            print(f"[raw] materializing binary_staged/{dataset} -> {binary_path}", flush=True)
            evaluate_staged_binary(eval_args, run, sub, indices, device)

    if "mmpd" in args.fake_sources:
        mmpd_path = pack_path(args.raw_eval_dir, "mmpd", dataset)
        if args.force_raw_eval or not mmpd_path.is_file():
            print(f"[raw] materializing mmpd/{dataset} -> {mmpd_path}", flush=True)
            ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
            run_mmpd_eval(eval_args, run, indices)

    packs: Dict[str, Dict[str, np.ndarray]] = {}
    for fake_source in args.fake_sources:
        path = pack_path(args.raw_eval_dir, fake_source, dataset)
        if not path.is_file():
            raise FileNotFoundError(f"raw pack missing after materialization: {path}")
        pack = load_npz(path)
        validate_stochastic_pack(path, pack)
        packs[fake_source] = pack

    series_starts, pack_splits = _resolve_pack_series_meta(args, run, indices, packs)
    # Persist meta onto MMPD packs that predate series_starts.
    for fake_source, pack in packs.items():
        if "series_starts" not in pack or "pack_splits" not in pack:
            path = pack_path(args.raw_eval_dir, fake_source, dataset)
            merged = dict(pack)
            merged["series_starts"] = series_starts
            merged["pack_splits"] = np.asarray(pack_splits)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **merged)
            packs[fake_source] = merged

    return run, sub, indices, packs, series_starts, pack_splits


def _resolve_pack_series_meta(
    args: argparse.Namespace,
    run: Any,
    indices: Sequence[int],
    packs: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[np.ndarray, List[str]]:
    for pack in packs.values():
        if "series_starts" in pack:
            starts = np.asarray(pack["series_starts"], dtype=np.int64)
            if starts.shape[0] != len(indices):
                raise ValueError(
                    f"{run.dataset}: series_starts length {starts.shape[0]} != n_indices {len(indices)}"
                )
            splits = (
                [str(x) for x in np.asarray(pack["pack_splits"]).tolist()]
                if "pack_splits" in pack
                else parse_pack_splits(getattr(args, "pack_splits", None))
            )
            return starts, splits

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    _pool, series_starts_full, splits, _lens, _stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
    )
    idx_arr = np.asarray(indices, dtype=np.int64)
    return series_starts_full[idx_arr], splits


def load_past_windows(
    args: argparse.Namespace,
    run: Any,
    indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    from torch.utils.data import Subset

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    pool, _starts, _splits, _lens, _stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
    )
    subset = Subset(pool, list(indices))
    loader = DataLoader(
        subset,
        batch_size=args.raw_load_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    past_all: List[np.ndarray] = []
    for past, _future in loader:
        past_all.append(past.numpy())
    return np.concatenate(past_all, axis=0)


def binary_mmpd_train_scaler_map(args: argparse.Namespace, run: Any) -> Dict[str, np.ndarray]:
    """Read the two training-set scalers that define the output coordinates.

    Binary uses its full training split's z-score scaler.  MMPD's persisted
    pack uses ``Dataset_MTS``'s StandardScaler on the staged subset CSV.  This
    function intentionally never looks at selected evaluation targets: it
    derives the conversion solely from the scalers used to train/evaluate the
    two models.
    """
    lookback, horizon = dataset_window_lengths_for_run(args, run)
    _pool, _starts, _splits, _lengths, binary_stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=("test",),
    )
    binary_mean = np.asarray(binary_stats["mean"], dtype=np.float64).reshape(-1)
    binary_std = np.asarray(binary_stats["std"], dtype=np.float64).reshape(-1)

    csv_path = Path(args.mmpd_data_dir) / mmpd_staged_filename_for_run(run)
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not csv_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"{run.dataset}: missing MMPD staged CSV/scaler metadata at {csv_path}; "
            "materialize the MMPD pack with the same --mmpd-data-dir first"
        )
    split_text = mmpd_data_split(run, Path(args.mmpd_data_dir))
    train_rows = int(split_text.split(",", 1)[0])
    frame = pd.read_csv(csv_path)
    raw_train = frame.iloc[:train_rows, 1:].to_numpy(dtype=np.float64, copy=True)
    mmpd_mean = raw_train.mean(axis=0)
    mmpd_std = raw_train.std(axis=0)
    n_vars = len(run_variate_indices(run))
    for name, values in (
        ("binary_mean", binary_mean),
        ("binary_std", binary_std),
        ("mmpd_mean", mmpd_mean),
        ("mmpd_std", mmpd_std),
    ):
        if values.shape != (n_vars,) or not np.isfinite(values).all():
            raise ValueError(f"{run.dataset}: invalid {name} shape/values: {values.shape}")
    return {
        "binary_mean": binary_mean,
        "binary_std": binary_std,
        "mmpd_mean": mmpd_mean,
        "mmpd_std": mmpd_std,
    }


def build_raw_bundle(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> RawBundle:
    run, sub, indices, packs, series_starts, pack_splits = ensure_raw_packs(args, dataset, device)
    past = load_past_windows(args, run, indices, device)
    y_true_by_source: Dict[str, np.ndarray] = {}
    fakes: Dict[str, np.ndarray] = {}
    ref_shape: Optional[Tuple[int, ...]] = None
    from utils.forecast_pack_reduce import assert_not_anchor_agg, reduce_pack_forecast

    fake_agg = str(getattr(args, "fake_agg", "sample0") or "sample0")
    assert_not_anchor_agg(fake_agg)
    for fake_source, pack in packs.items():
        y_true = pack["y_true"].astype(np.float32)
        # Default: first stochastic draw (sample0). No mean-over-S; anchor
        # rejected via assert_not_anchor_agg. Existing S>1 packs still work.
        fake = reduce_pack_forecast(pack, agg=fake_agg)
        if ref_shape is None:
            ref_shape = y_true.shape
        elif y_true.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: y_true shape differs from first pack")
        if fake.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: fake shape differs from y_true")
        if not np.array_equal(pack["indices"], np.asarray(indices, dtype=pack["indices"].dtype)):
            raise ValueError(f"{dataset}/{fake_source}: raw pack indices do not match discriminator indices")
        y_true_by_source[fake_source] = y_true
        fakes[fake_source] = fake
    print(f"[{dataset}] disc fake aggregation: {fake_agg}", flush=True)

    if past.shape[0] != ref_shape[0]:
        raise ValueError(f"{dataset}: past/y_true window mismatch {past.shape[0]} vs {ref_shape[0]}")
    validate_variate_alignment(dataset, run, sub, past, y_true_by_source, fakes)

    if (
        bool(getattr(args, "mmpd_to_binary_dataset_norm", False))
        and "binary_staged" in y_true_by_source
        and "mmpd" in y_true_by_source
    ):
        scalers = binary_mmpd_train_scaler_map(args, run)
        aligned_mmpd, align_stats = align_mmpd_to_binary_dataset_norm(
            binary_y_true=y_true_by_source["binary_staged"],
            mmpd_y_true=y_true_by_source["mmpd"],
            mmpd_fakes=fakes["mmpd"],
            **scalers,
        )
        # Labels must use exactly one GT tensor.  This is deliberate rather
        # than an approximate post-hoc equality check: both model forecasts
        # are scored against binary's train-split dataset-z target values.
        y_true_by_source["mmpd"] = y_true_by_source["binary_staged"].copy()
        fakes["mmpd"] = aligned_mmpd
        print(
            f"[{dataset}] MMPD→binary dataset-norm map: "
            f"scale=[{align_stats['scale_min']:.8f},{align_stats['scale_max']:.8f}] "
            f"offset=[{align_stats['offset_min']:.8f},{align_stats['offset_max']:.8f}] "
            f"target_rmse_max={align_stats['target_rmse_max']:.2e} "
            f"target_max_abs={align_stats['target_max_abs']:.2e}",
            flush=True,
        )

    if len(y_true_by_source) > 1:
        sources = list(y_true_by_source)
        ref = y_true_by_source[sources[0]]
        for src in sources[1:]:
            other = y_true_by_source[src]
            mse = float(np.mean((ref - other) ** 2))
            if mse > 1e-6:
                msg = (
                    f"{dataset}: y_true differs between {sources[0]} and {src} "
                    f"(mse={mse:.6f}); packs are not in the same coordinate space"
                )
                if getattr(args, "ordinal_ladder_quantize", False) and "mmpd" in fakes:
                    raise ValueError(
                        msg + "; refusing --ordinal-ladder-quantize onto a mismatched ladder"
                    )
                print(f"[warn] {msg}; each discriminator uses its own pack GT.", flush=True)

    if args.bin_match_filter:
        # Same path binary ordinal_norm uses: train z-score → ordinal ranks
        # (+ OOD constant shift) → [optional stride subsample] → bounded coarse/fine
        # → upsample → ordinal decode. No instance norm.
        ladder = load_ordinal_ladder_for_run(args, run)
        _ms, coarse_h, fine_h = resolve_dual_scale_bin_params(
            dataset,
            sub,
            fallback_max_scale=args.bin_max_scale,
            coarse_height=args.bin_coarse_height or args.bin_image_height,
            fine_height=args.bin_fine_height or args.bin_image_height,
        )
        from models.diffusion_tsf.pipeline.config import load_experiment_config

        cfg = load_experiment_config(str(_binary_config_path(args, run.dataset)))
        repr_stride = int(
            (cfg.get("experiment") or {}).get("representation_time_stride", 1) or 1
        )
        print(
            f"[{dataset}] applying ordinal dual-scale bin-match filter={args.bin_match_filter} "
            f"(coarse={coarse_h}, fine={fine_h}, repr_stride={repr_stride}, "
            f"decoder={args.bin_decoder}, ood_shift=on, no instance-norm)",
            flush=True,
        )
        args._resolved_bin_repr_time_stride = repr_stride
        if args.binary_debias_quantization:
            raise ValueError(
                "--bin-match-filter already canonicalizes all selected sources onto the "
                "binary ordinal lattice; refuse combining with --binary-debias-quantization "
                "(would jitter only binary_staged after the shared round-trip)"
            )
        y_true_by_source, fakes = apply_bin_match_to_bundle(
            mode=args.bin_match_filter,
            past=past.astype(np.float32),
            y_true_by_source=y_true_by_source,
            fakes=fakes,
            ladder=ladder,
            coarse_height=coarse_h,
            fine_height=fine_h,
            decoder=args.bin_decoder,
            device=device,
            repr_time_stride=repr_stride,
        )

    if args.binary_debias_quantization and "binary_staged" in fakes:
        max_scale, coarse_h, fine_h = resolve_dual_scale_bin_params(
            dataset,
            sub,
            fallback_max_scale=args.bin_max_scale,
            coarse_height=args.bin_coarse_height,
            fine_height=args.bin_fine_height,
        )
        debiased, debias_stats = debias_binary_staged_fakes(
            fakes["binary_staged"],
            max_scale=max_scale,
            coarse_height=coarse_h,
            fine_height=fine_h,
            seed=args.seed,
            dataset=dataset,
        )
        fakes["binary_staged"] = debiased
        print(
            f"[{dataset}] binary debias: max_scale={max_scale} coarse_h={coarse_h} "
            f"fine_h={fine_h} half_fine_bin={debias_stats['half_fine_bin']:.6f} "
            f"flatline_frac={debias_stats['flatline_frac']:.3f} "
            f"debias_frac={debias_stats['debias_frac']:.3f}",
            flush=True,
        )

    if getattr(args, "ordinal_ladder_quantize", False) and args.bin_match_filter != "all":
        # Snap ALL horizons (GT, MMPD fakes, binary fakes) onto the global ordinal ladder.
        # Binary preds are post stride-2 linear upsample and are mostly off-ladder otherwise.
        ladder = load_ordinal_ladder_for_run(args, run)
        for src in list(fakes):
            quantized, q_stats = quantize_to_ordinal_ladder(fakes[src], ladder)
            fakes[src] = quantized
            print(
                f"[{dataset}] {src} fake ordinal-quantize: changed_frac={q_stats['changed_frac']:.4f} "
                f"mean_abs_delta={q_stats['mean_abs_delta']:.6f} "
                f"max_abs_delta={q_stats['max_abs_delta']:.6f} "
                f"n_unique_max={int(q_stats['n_unique_max'])}",
                flush=True,
            )
        for src in list(y_true_by_source):
            quantized_gt, gt_stats = quantize_to_ordinal_ladder(y_true_by_source[src], ladder)
            y_true_by_source[src] = quantized_gt
            print(
                f"[{dataset}] {src} GT ordinal-quantize: changed_frac={gt_stats['changed_frac']:.4f} "
                f"mean_abs_delta={gt_stats['mean_abs_delta']:.6f} "
                f"max_abs_delta={gt_stats['max_abs_delta']:.6f}",
                flush=True,
            )

    elif getattr(args, "ordinal_ladder_quantize", False):
        # apply_bin_match_to_bundle(mode=all) is already the exact binary
        # ordinal -> 16x16 bounded dual decode -> ordinal denorm path.  A
        # second generic nearest-global-ladder pass would move non-uniform
        # ladder values off its legal 256 decoded rungs.
        print(
            f"[{dataset}] skipping generic ordinal quantize: bin-match=all already "
            "produced the exact binary 256-bin dataset-z decode lattice",
            flush=True,
        )

    native_stride = int(getattr(args, "native_repr_stride", 1) or 1)
    if native_stride > 1:
        print(
            f"[{dataset}] native-repr downsample stride={native_stride} "
            f"(aligned [::{native_stride}] on GT+fakes)",
            flush=True,
        )
        for src in list(y_true_by_source):
            y_true_by_source[src] = np.ascontiguousarray(y_true_by_source[src][..., ::native_stride])
        for src in list(fakes):
            fakes[src] = np.ascontiguousarray(fakes[src][..., ::native_stride])
        ref_shape = next(iter(y_true_by_source.values())).shape

    expected_variates = [int(i) for i in run_variate_indices(run)]
    print(
        f"[{dataset}] staged subset={run_subset_id(run)} variates={expected_variates} "
        f"pack_splits={pack_splits} n_windows={len(indices)}",
        flush=True,
    )

    return RawBundle(
        run=run,
        sub=sub,
        indices=[int(i) for i in indices],
        past=past.astype(np.float32),
        y_true_by_source=y_true_by_source,
        fakes=fakes,
        series_starts=np.asarray(series_starts, dtype=np.int64),
        pack_splits=list(pack_splits),
    )


def window_time_bounds(
    dataset: str,
    indices: Sequence[int],
    lookback: int,
    horizon: int,
    test_stride: int,
    *,
    series_starts: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if series_starts is not None:
        starts = np.asarray(series_starts, dtype=np.int64)
        if starts.shape[0] != len(indices):
            raise ValueError(
                f"{dataset}: series_starts length {starts.shape[0]} != n_indices {len(indices)}"
            )
        span = int(lookback) + int(horizon)
        ends = starts + span
        return starts, ends
    raw = np.asarray(indices, dtype=np.int64)
    if dataset == "dalia":
        starts = raw
        span = 1
    else:
        starts = raw * max(1, int(test_stride))
        span = int(lookback) + int(horizon)
    ends = starts + span
    return starts, ends


def _purge_nonoverlapping(
    order: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    blocked_start: Optional[int] = None,
) -> np.ndarray:
    """Keep windows in `order` that end before blocked_start and do not overlap earlier kept ones."""
    kept: List[int] = []
    last_end = None
    for idx in order:
        idx = int(idx)
        if blocked_start is not None and int(ends[idx]) > int(blocked_start):
            continue
        if last_end is not None and int(starts[idx]) < int(last_end):
            continue
        kept.append(idx)
        last_end = int(ends[idx])
    return np.asarray(kept, dtype=np.int64)


def split_windows(
    n_windows: int,
    args: argparse.Namespace,
    dataset: str,
    *,
    indices: Optional[Sequence[int]] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    test_stride: Optional[int] = None,
    series_starts: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    if args.max_windows is not None:
        n_windows = min(n_windows, int(args.max_windows))
    if indices is None or lookback is None or horizon is None or test_stride is None:
        raise ValueError("split_windows requires indices, lookback, horizon, and test_stride")
    raw_indices = [int(i) for i in list(indices)[:n_windows]]
    if len(raw_indices) != n_windows:
        raise ValueError(f"{dataset}: got {len(raw_indices)} split indices for {n_windows} windows")
    starts_all = None if series_starts is None else list(series_starts)[:n_windows]

    starts, ends = window_time_bounds(
        dataset,
        raw_indices,
        int(lookback),
        int(horizon),
        int(test_stride),
        series_starts=starts_all,
    )
    order = np.argsort(starts, kind="mergesort")
    n_train_target = max(1, int(round(len(order) * args.train_fraction)))
    n_val_target = max(1, int(round(len(order) * args.val_fraction)))
    if n_train_target + n_val_target >= len(order):
        n_val_target = max(1, len(order) - n_train_target - 1)
    n_test = len(order) - n_train_target - n_val_target
    if n_test < 1:
        raise ValueError(f"not enough windows for train/val/test split: {len(order)}")

    test = order[-n_test:]
    test_start = int(starts[test].min())
    # Hold out anything whose span reaches into the test region. No silent fallback.
    train_val_pool = np.asarray(
        [idx for idx in order[:-n_test] if int(ends[idx]) <= test_start],
        dtype=np.int64,
    )
    if len(train_val_pool) < 2:
        raise ValueError(
            f"{dataset}: hard temporal purge left {len(train_val_pool)} train/val windows "
            f"(need >=2) before test_start={test_start}. "
            f"windows={len(order)} test={n_test} lookback={lookback} horizon={horizon}. "
            f"Raise --pack-fraction / enlarge pack_splits — overlapping fallback is disabled."
        )

    # Allow overlap *within* train and within val (needed at lb336/hz720). The leak we
    # kill is train/val ↔ test absolute-time overlap, not within-split density.
    val_ratio = args.val_fraction / max(args.train_fraction + args.val_fraction, 1e-8)
    n_val = max(1, int(round(len(train_val_pool) * val_ratio)))
    if n_val >= len(train_val_pool):
        n_val = len(train_val_pool) - 1
    # Chronological train then val within the purged pool.
    tv_order = train_val_pool[np.argsort(starts[train_val_pool], kind="mergesort")]
    train = tv_order[:-n_val]
    val = tv_order[-n_val:]

    print(
        f"[split] {dataset}: pack={len(order)} -> train/val/test="
        f"{len(train)}/{len(val)}/{len(test)} "
        f"(raw targets {n_train_target}/{n_val_target}/{n_test}; "
        f"test_start={test_start}; train/val purged vs test only)",
        flush=True,
    )
    return {
        "train": np.sort(train),
        "val": np.sort(val),
        "test": np.sort(test),
    }


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def zscore_time(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-5)


class HorizonSliceDataset(Dataset):
    def __init__(
        self,
        past: np.ndarray,
        real: np.ndarray,
        fake: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
        include_past: bool = True,
        apply_zscore: bool = True,
    ) -> None:
        if real.shape != fake.shape:
            raise ValueError(f"real/fake shape mismatch: {real.shape} vs {fake.shape}")
        if real.shape[0] != past.shape[0]:
            raise ValueError(f"past/real window mismatch: {past.shape[0]} vs {real.shape[0]}")
        if slice_len > real.shape[-1]:
            raise ValueError(f"slice_len={slice_len} exceeds horizon={real.shape[-1]}")

        self.past = past
        self.real = real
        self.fake = fake
        self.slice_len = int(slice_len)
        self.include_past = bool(include_past)
        self.apply_zscore = bool(apply_zscore)
        offsets = list(range(0, real.shape[-1] - slice_len + 1, max(1, int(offset_stride))))
        real_items = [(int(w), int(o), 0) for w in windows for o in offsets]
        fake_items = [(int(w), int(o), 1) for w in windows for o in offsets]

        rng = np.random.default_rng(seed)
        n = min(len(real_items), len(fake_items))
        if max_examples is not None:
            n = min(n, max(1, int(max_examples) // 2))
        real_idx = rng.choice(len(real_items), size=n, replace=False)
        fake_idx = rng.choice(len(fake_items), size=n, replace=False)
        items = [real_items[i] for i in real_idx] + [fake_items[i] for i in fake_idx]
        rng.shuffle(items)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window, offset, label = self.items[idx]
        candidate_src = self.fake if label == 1 else self.real
        candidate = candidate_src[window, :, offset : offset + self.slice_len]
        if self.apply_zscore:
            norm = zscore_time
        else:
            def norm(t: np.ndarray) -> np.ndarray:
                return np.asarray(t, dtype=np.float32)
        if self.include_past:
            past = self.past[window]
            x = np.concatenate([norm(past), norm(candidate)], axis=-1).astype(np.float32)
        else:
            # Local texture only: no lookback continuity cue.
            x = norm(candidate).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(offset, dtype=torch.long),
            torch.tensor(float(label), dtype=torch.float32),
            torch.tensor(int(window), dtype=torch.long),
        )


class InvertedSliceDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        max_offset: int,
        d_model: int,
        n_heads: int,
        depth: int,
        d_ff: int,
        dropout: float,
        *,
        use_offset_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.use_offset_embedding = bool(use_offset_embedding)
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.offset_embedding = (
            nn.Embedding(max_offset + 1, d_model) if self.use_offset_embedding else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]. Like iTransformer, variates are tokens and time is embedded.
        tokens = self.value_embedding(x)
        if self.offset_embedding is not None:
            tokens = tokens + self.offset_embedding(offsets).unsqueeze(1)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def window_level_metrics(
    windows: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    variates: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Mean P(fake) over offsets, then AUROC.

    Multivariate disc examples already pack all variates in one tensor, so the
    default key is ``(window, label)`` (average over offsets only).

    Univariate disc emits one example per variate — pass ``variates`` so the
    key is ``(window, variate, label)`` and we do **not** pool across series.
    """
    keys: Dict[Tuple[Any, ...], List[float]] = {}
    if variates is None:
        for w, y, p in zip(windows.tolist(), labels.tolist(), probs.tolist()):
            key: Tuple[Any, ...] = (int(w), int(y))
            keys.setdefault(key, []).append(float(p))
    else:
        if len(variates) != len(windows):
            raise ValueError(
                f"variates length {len(variates)} != windows length {len(windows)}"
            )
        for w, v, y, p in zip(
            windows.tolist(), variates.tolist(), labels.tolist(), probs.tolist()
        ):
            key = (int(w), int(v), int(y))
            keys.setdefault(key, []).append(float(p))
    if not keys:
        return {
            "disc_acc_window": float("nan"),
            "disc_auroc_window": float("nan"),
            "n_windows_scored": 0.0,
        }
    y_win = []
    p_win = []
    for key, vals in keys.items():
        y_win.append(float(key[-1]))
        p_win.append(float(np.mean(vals)))
    y_arr = np.asarray(y_win, dtype=np.float64)
    p_arr = np.asarray(p_win, dtype=np.float64)
    preds = (p_arr >= 0.5).astype(np.float64)
    return {
        "disc_acc_window": float((preds == y_arr).mean()),
        "disc_auroc_window": binary_auroc(y_arr, p_arr),
        "n_windows_scored": float(len(y_win)),
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    windows_all: List[np.ndarray] = []
    for batch in loader:
        if len(batch) == 4:
            x, offsets, labels, windows = batch
        else:
            x, offsets, labels = batch
            windows = torch.zeros_like(labels, dtype=torch.long)
        x = x.to(device)
        offsets = offsets.to(device)
        labels = labels.to(device)
        logits = model(x, offsets)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(labels.numel())
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        windows_all.append(windows.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    windows_np = np.concatenate(windows_all)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (logits_np >= 0.0).astype(np.float32)
    out = {
        "disc_bce": total_loss / max(1, total_count),
        "disc_acc": float((preds == labels_np).mean()),
        "disc_auroc": binary_auroc(labels_np, probs),
        "n_examples": float(total_count),
        "positive_rate": float(labels_np.mean()),
    }
    out.update(window_level_metrics(windows_np, labels_np, probs))
    return out


def train_classifier(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: RawBundle,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    fake = bundle.fakes[fake_source]
    y_true = bundle.y_true_by_source[fake_source]
    max_offset = y_true.shape[-1] - slice_len
    seed_base = args.seed + stable_hash(f"{dataset}:{fake_source}:{slice_len}")
    include_past = not bool(getattr(args, "candidate_only", False))
    offset_stride = int(getattr(args, "offset_stride", 1) or 1)
    if bool(getattr(args, "nonoverlapping_patches", False)):
        offset_stride = int(slice_len)
    use_offset_embedding = not bool(getattr(args, "no_offset_embedding", False))
    apply_zscore = not bool(getattr(args, "disc_bin_center_shift", False))
    ds_kwargs = dict(
        offset_stride=offset_stride,
        include_past=include_past,
        apply_zscore=apply_zscore,
    )
    ds_train = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["train"],
        slice_len,
        seed=seed_base,
        max_examples=args.max_train_examples,
        **ds_kwargs,
    )
    ds_val = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["val"],
        slice_len,
        seed=seed_base + 1,
        max_examples=args.max_eval_examples,
        **ds_kwargs,
    )
    ds_test = HorizonSliceDataset(
        bundle.past,
        y_true,
        fake,
        splits["test"],
        slice_len,
        seed=seed_base + 2,
        max_examples=args.max_eval_examples,
        **ds_kwargs,
    )
    generator = torch.Generator()
    generator.manual_seed(seed_base)
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    seq_len = int(slice_len if not include_past else bundle.past.shape[-1] + slice_len)
    print(
        f"[disc] {dataset}/{fake_source}/L{slice_len}: "
        f"candidate_only={not include_past} offset_stride={offset_stride} "
        f"offset_emb={use_offset_embedding} seq_len={seq_len}",
        flush=True,
    )
    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=max_offset,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_offset_embedding=use_offset_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    progress = EvalProgress(f"disc/{dataset}/{fake_source}/L{slice_len}", args.epochs)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_idx, batch in enumerate(train_loader):
            x, offsets, labels = batch[0], batch[1], batch[2]
            x = x.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, offsets)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * int(labels.numel())
            train_count += int(labels.numel())
            if getattr(args, "max_batches_per_epoch", None) and batch_idx + 1 >= args.max_batches_per_epoch:
                break

        val_metrics = evaluate_classifier(model, val_loader, device)
        train_bce = train_loss / max(1, train_count)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        progress.maybe_log(
            epoch + 1,
            extra=(
                f"train_bce={train_bce:.4f} val_bce={val_metrics['disc_bce']:.4f} "
                f"val_auc={val_metrics['disc_auroc']:.3f} "
                f"val_auc_win={val_metrics.get('disc_auroc_window', float('nan')):.3f} "
                f"elapsed={fmt_duration(time.time() - t0)}"
            ),
        )
        if stale >= args.patience:
            break

    progress.done(extra=f"best_epoch={best_epoch} best_val_bce={best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device)

    if bool(getattr(args, "save_checkpoints", False)):
        ckpt_path = (
            args.output_dir
            / "checkpoints"
            / f"{dataset}_{fake_source}_L{slice_len}_discriminator.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "dataset": dataset,
                "fake_source": fake_source,
                "slice_len": slice_len,
            },
            ckpt_path,
        )

    out = {
        **test_metrics,
        "best_val_bce": float(best_val),
        "best_epoch": float(best_epoch),
        "epochs_run": float(epoch + 1),
        "n_train": float(len(ds_train)),
        "n_val": float(len(ds_val)),
        "n_test": float(len(ds_test)),
        "n_windows_train": float(len(splits["train"])),
        "n_windows_val": float(len(splits["val"])),
        "n_windows_test": float(len(splits["test"])),
        "slice_len": float(slice_len),
        "horizon": float(y_true.shape[-1]),
        "n_variates": float(y_true.shape[1]),
        "log2_bce_gap": float(abs(test_metrics["disc_bce"] - LOG2)),
        "candidate_only": float(1.0 if not include_past else 0.0),
        "offset_stride": float(offset_stride),
        "no_offset_embedding": float(0.0 if use_offset_embedding else 1.0),
        "native_repr_stride": float(getattr(args, "native_repr_stride", 1) or 1),
    }
    return out


def partial_path(output_dir: Path, dataset: str, fake_source: str) -> Path:
    return output_dir / "partials" / f"{dataset}__{fake_source}.json"


def legacy_partial_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "partials" / f"{dataset}.json"


def write_source_partial(
    output_dir: Path,
    dataset: str,
    fake_source: str,
    by_len: Mapping[str, Mapping[str, float]],
) -> None:
    write_json(partial_path(output_dir, dataset, fake_source), dict(by_len))


def collect_partials(output_dir: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    merged: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    partial_dir = output_dir / "partials"
    if not partial_dir.is_dir():
        return merged
    for path in sorted(partial_dir.glob("*.json")):
        stem = path.stem
        data = load_json(path)
        if "__" in stem:
            dataset, fake_source = stem.split("__", 1)
            merged.setdefault(dataset, {})[fake_source] = data
            continue
        if not isinstance(data, dict):
            continue
        if data and all(key in FAKE_SOURCES for key in data):
            merged[stem] = data
    return merged


def existing_combo(
    output_dir: Path,
    dataset: str,
    fake_source: str,
    slice_len: int,
) -> Optional[Dict[str, float]]:
    path = partial_path(output_dir, dataset, fake_source)
    if path.is_file():
        metrics = load_json(path).get(str(slice_len))
        return metrics if isinstance(metrics, dict) else None
    legacy = legacy_partial_path(output_dir, dataset)
    if legacy.is_file():
        metrics = load_json(legacy).get(fake_source, {}).get(str(slice_len))
        return metrics if isinstance(metrics, dict) else None
    return None


def merge_partial_metrics(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    merged = collect_partials(args.output_dir)
    if not merged:
        return {}
    write_json(args.output_dir / "metrics.json", merged)
    fields = [
        "dataset",
        "fake_source",
        "slice_len",
        "disc_bce",
        "log2_bce_gap",
        "disc_acc",
        "disc_auroc",
        "disc_acc_window",
        "disc_auroc_window",
        "n_windows_scored",
        "best_val_bce",
        "best_epoch",
        "epochs_run",
        "n_train",
        "n_val",
        "n_test",
        "n_windows_train",
        "n_windows_val",
        "n_windows_test",
        "n_variates",
        "horizon",
        "offset_stride",
        "no_offset_embedding",
        "native_repr_stride",
        "candidate_only",
    ]
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, by_source in merged.items():
            for fake_source, by_len in by_source.items():
                for slice_key, metrics in by_len.items():
                    row = {"dataset": dataset, "fake_source": fake_source, "slice_len": int(slice_key)}
                    row.update({key: metrics.get(key) for key in fields if key not in row})
                    writer.writerow(row)

    merged_datasets = sorted(merged.keys()) or list(args.datasets)
    merged_sources = sorted({src for by_source in merged.values() for src in by_source})
    manifest = {
        "datasets": merged_datasets,
        "fake_sources": merged_sources or list(args.fake_sources),
        "slice_lengths": args.slice_lengths,
        "raw_eval_dir": str(args.raw_eval_dir),
        "test_fraction": args.test_fraction,
        "test_stride": args.test_stride,
        "staged_ckpts": {
            d: str(resolve_staged_ckpt_dir(args.ckpt_base, d, _anchor_config_for(args, d)))
            for d in merged_datasets
        },
        "anchor_config": args.anchor_config,
        "anchor_config_by_dataset": dict(getattr(args, "anchor_config_by_dataset", None) or {}),
        "binary_config": getattr(args, "binary_config", None),
        "binary_config_by_dataset": dict(getattr(args, "binary_config_by_dataset", None) or {}),
        "binary_debias_quantization": bool(getattr(args, "binary_debias_quantization", False)),
        "ordinal_ladder_quantize": bool(getattr(args, "ordinal_ladder_quantize", False)),
        "candidate_only": bool(getattr(args, "candidate_only", False)),
        "pack_splits": getattr(args, "pack_splits", "test"),
        "pack_fraction": getattr(args, "pack_fraction", None),
        "nonoverlapping_patches": bool(getattr(args, "nonoverlapping_patches", False)),
        "no_offset_embedding": bool(getattr(args, "no_offset_embedding", False)),
        "native_repr_stride": int(getattr(args, "native_repr_stride", 1) or 1),
        "mmpd_output_root": str(args.mmpd_output_root),
    }
    if getattr(args, "bin_match_filter", None):
        manifest["bin_match_filter"] = args.bin_match_filter
        manifest["bin_image_height"] = args.bin_image_height
        manifest["bin_coarse_height"] = args.bin_coarse_height
        manifest["bin_fine_height"] = args.bin_fine_height
        manifest["bin_match_space"] = "ordinal_bounded_dual_scale"
        manifest["bin_repr_time_stride"] = int(
            getattr(args, "_resolved_bin_repr_time_stride", 0) or 0
        )
        manifest["bin_decoder"] = args.bin_decoder
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        f"[merge] wrote metrics for datasets={merged_datasets} fake_sources={manifest['fake_sources']}",
        flush=True,
    )
    return merged


def write_outputs(args: argparse.Namespace, results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_results in results.items():
        for fake_source, by_len in dataset_results.items():
            if not by_len:
                continue
            path = partial_path(args.output_dir, dataset, fake_source)
            existing = load_json(path) if path.is_file() else {}
            existing.update(by_len)
            write_source_partial(args.output_dir, dataset, fake_source, existing)
    if args.merge_metrics:
        merge_partial_metrics(args)


def run_merge_only(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_partial_metrics(args)
    if not merged:
        raise FileNotFoundError(f"No partial metrics found under {args.output_dir / 'partials'}")


def valid_slice_lengths(horizon: int, requested: Sequence[int]) -> Tuple[List[int], List[int]]:
    valid = [int(length) for length in requested if int(length) <= horizon]
    skipped = [int(length) for length in requested if int(length) > horizon]
    return valid, skipped


def run_eval(args: argparse.Namespace) -> None:
    if not set(args.fake_sources).issubset(set(FAKE_SOURCES)):
        raise ValueError(f"--fake-sources must be within {FAKE_SOURCES}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}")
    import_mmpd_packs(args)

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for dataset in args.datasets:
        print(f"\n[{dataset}] loading/materializing raw packs", flush=True)
        bundle = build_raw_bundle(args, dataset, device)
        n = next(iter(bundle.y_true_by_source.values())).shape[0]
        ref_y = next(iter(bundle.y_true_by_source.values()))
        splits = split_windows(
            n,
            args,
            dataset,
            indices=bundle.indices,
            lookback=bundle.past.shape[-1],
            horizon=ref_y.shape[-1],
            test_stride=run_test_stride(bundle.run),
            series_starts=bundle.series_starts,
        )
        print(
            f"[{dataset}] windows={n} train/val/test="
            f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
            f"variates={ref_y.shape[1]} horizon={ref_y.shape[-1]} "
            f"subset={run_subset_id(bundle.run)} pack_splits={bundle.pack_splits}",
            flush=True,
        )
        results.setdefault(dataset, {})
        horizon = int(ref_y.shape[-1])
        valid_lens, skipped_lens = valid_slice_lengths(horizon, args.slice_lengths)
        if skipped_lens:
            print(
                f"[{dataset}] skipping slice lengths {skipped_lens} (horizon={horizon})",
                flush=True,
            )
        if not valid_lens:
            print(f"[{dataset}] no valid slice lengths for horizon={horizon}; skipping", flush=True)
            continue
        for fake_source in args.fake_sources:
            results[dataset].setdefault(fake_source, {})
            for slice_len in valid_lens:
                trained = False
                if not args.force_train:
                    existing = existing_combo(args.output_dir, dataset, fake_source, int(slice_len))
                    if existing is not None:
                        print(f"[skip] existing metrics dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                        results[dataset][fake_source][str(slice_len)] = existing
                    else:
                        print(f"[train] dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                        metrics = train_classifier(
                            args, dataset, fake_source, int(slice_len), bundle, splits, device
                        )
                        results[dataset][fake_source][str(slice_len)] = metrics
                        trained = True
                else:
                    print(f"[train] dataset={dataset} fake={fake_source} L={slice_len}", flush=True)
                    metrics = train_classifier(
                        args, dataset, fake_source, int(slice_len), bundle, splits, device
                    )
                    results[dataset][fake_source][str(slice_len)] = metrics
                    trained = True

                if args.visualize_confusions:
                    try:
                        from utils.visualize_discriminator_texture_confusions import visualize_combo
                    except ModuleNotFoundError as exc:
                        print(f"[viz] skip {dataset}/{fake_source}/L{slice_len}: {exc}", flush=True)
                        continue

                    try:
                        visualize_combo(
                            args,
                            dataset,
                            fake_source,
                            int(slice_len),
                            bundle,
                            splits,
                            device,
                        )
                    except FileNotFoundError as exc:
                        if trained:
                            raise
                        print(f"[viz] skip {dataset}/{fake_source}/L{slice_len}: {exc}", flush=True)
            write_outputs(args, {dataset: {fake_source: results[dataset][fake_source]}})


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n, c, lookback, horizon = 18, 3, 32, 32
    indices = [i * (lookback + horizon) for i in range(n)]
    past = rng.normal(size=(n, c, lookback)).astype(np.float32)
    y = rng.normal(size=(n, c, horizon)).astype(np.float32)
    fake = (0.7 * y + 0.3 * rng.normal(size=(n, c, horizon))).astype(np.float32)
    bundle = RawBundle(
        run=None,
        sub={},
        indices=indices,
        past=past,
        y_true_by_source={"binary_staged": y},
        fakes={"binary_staged": fake},
        series_starts=np.asarray([i * (lookback + horizon) for i in range(n)], dtype=np.int64),
        pack_splits=["test"],
    )
    args.datasets = ["selftest"]
    args.fake_sources = ["binary_staged"]
    args.slice_lengths = [8]
    args.epochs = min(args.epochs, 2)
    args.patience = 2
    args.max_train_examples = 128
    args.max_eval_examples = 64
    args.batch_size = min(args.batch_size, 32)
    args.max_batches_per_epoch = 2
    device = torch.device("cpu")
    splits = split_windows(
        n,
        args,
        "selftest",
        indices=bundle.indices,
        lookback=lookback,
        horizon=horizon,
        test_stride=1,
        series_starts=bundle.series_starts,
    )
    metrics = train_classifier(args, "selftest", "binary_staged", 8, bundle, splits, device)
    print(json.dumps(metrics, indent=2, sort_keys=True))


DEFAULT_DISC_OUTPUT = (
    REPO_ROOT / "results" / "datasets" / "06-14-disc-texture-flat-subsets-ema099-vs-mmpd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_SUBSET_DATASETS))
    parser.add_argument("--anchor-config", type=str, default=DEFAULT_ANCHOR_CONFIG)
    parser.add_argument(
        "--anchor-config-by-dataset",
        type=str,
        default=None,
        help="Comma map dataset:stem (overrides --anchor-config per dataset)",
    )
    parser.add_argument(
        "--binary-config",
        type=str,
        default=None,
        help="Leaf YAML for PipelineState (default: configs/<anchor-config>.yaml)",
    )
    parser.add_argument(
        "--binary-config-by-dataset",
        type=str,
        default=None,
        help="Comma map dataset:yaml_path for per-dataset binary leaf configs",
    )
    parser.add_argument("--ckpt-base", type=Path, default=DEFAULT_CKPT_BASE)
    parser.add_argument("--fake-sources", nargs="+", default=list(FAKE_SOURCES), choices=list(FAKE_SOURCES))
    parser.add_argument("--slice-lengths", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DISC_OUTPUT,
    )
    parser.add_argument(
        "--raw-eval-dir",
        type=Path,
        default=DEFAULT_DISC_OUTPUT.parent / "06-14-raw-texture-flat-subsets-ema099-vs-mmpd",
    )
    parser.add_argument(
        "--import-mmpd-packs-from",
        type=Path,
        default=None,
        help="Reuse existing mmpd_*.npz packs (e.g. compare report raw/) when present",
    )
    parser.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD_OUTPUT_ROOT)
    parser.add_argument(
        "--mmpd-backbone",
        choices=["Decoder", "MaskAE"],
        default="Decoder",
        help="MMPD backbone for checkpoint lookup (paper lb336/hz720 uses Decoder).",
    )
    parser.add_argument("--mmpd-repo", type=Path, default=DEFAULT_MMPD_REPO)
    parser.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--test-max-items", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--probabilistic-sampler", choices=["quad_t", "ddim_quad", "ddim", "ddpm"], default="quad_t")
    parser.add_argument("--gmm-components", type=int, default=1)
    parser.add_argument("--gmm-iterations", type=int, default=10)
    parser.add_argument("--topk-max", type=int, default=3)
    parser.add_argument("--raw-binary-batch-size", type=int, default=8)
    parser.add_argument("--raw-mmpd-batch-size", type=int, default=16)
    parser.add_argument("--raw-load-batch-size", type=int, default=64)
    parser.add_argument("--force-raw-eval", action="store_true")
    parser.add_argument("--no-update-mmpd", action="store_true")
    parser.add_argument(
        "--mmpd-ordinal-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the ordinal MMPD Decoder in its no-instance-norm ordinal representation.",
    )
    parser.add_argument(
        "--mmpd-instance-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MMPD's legacy per-window instance normalization (incompatible with ordinal MMPD runs).",
    )
    parser.add_argument(
        "--mmpd-to-binary-dataset-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affinely map MMPD output z-scores to binary's dataset-z coordinates from both train-split scalers.",
    )
    parser.add_argument(
        "--bin-match-filter",
        choices=list(BIN_MATCH_CHOICES),
        default=None,
        help="Round-trip horizons through binary ordinal_norm path "
        "(train-set z-score → ordinal ranks + OOD shift → bounded coarse/fine). "
        "No instance/window norm. mmpd=fakes only; both=both fakes; all=GT+fakes.",
    )
    parser.add_argument("--bin-image-height", type=int, default=16)
    parser.add_argument("--bin-coarse-height", type=int, default=16)
    parser.add_argument("--bin-fine-height", type=int, default=16)
    parser.add_argument(
        "--bin-max-scale",
        type=float,
        default=3.5,
        help="Legacy fallback for --binary-debias-quantization only; unused by ordinal bin-match.",
    )
    parser.add_argument(
        "--bin-std-floor",
        type=float,
        default=1e-8,
        help="Unused (kept for CLI compat); ordinal bin-match has no instance norm.",
    )
    parser.add_argument(
        "--binary-debias-quantization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Jitter non-flatline binary_staged fakes by up to ±½ fine bin (discriminator only).",
    )
    parser.add_argument(
        "--ordinal-ladder-quantize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Snap GT + all fakes (MMPD and binary_staged) to the global ordinal ladder. "
        "Needed after stride-2 linear upsample so binary is not left off-ladder.",
    )
    parser.add_argument(
        "--candidate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed only the z-scored horizon patch (no lookback). Isolates local texture from past-continuity cues.",
    )
    parser.add_argument(
        "--disc-bin-center-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace zscore_time with centered bin-index mean shift (utils.disc_bin_center_shift). "
             "Ordinal evaluator defaults this ON; texture/univariate base defaults OFF.",
    )
    parser.add_argument(
        "--disc-bin-center-reduce",
        choices=["per_variate", "joint"],
        default="per_variate",
        help="How to average centered bin indices before the integer shift (default: per_variate).",
    )
    parser.add_argument(
        "--save-classification-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-patch discriminator probabilities for forecast/classification visualizations.",
    )
    parser.add_argument(
        "--pack-splits",
        type=str,
        default="test",
        help="Comma list of TSF splits forming the generation/disc pool (e.g. train,val or test).",
    )
    parser.add_argument(
        "--pack-fraction",
        type=float,
        default=None,
        help="Fraction of pack-pool windows to keep for inference (default: --test-fraction).",
    )
    parser.add_argument(
        "--nonoverlapping-patches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force offset_stride=slice_len (non-overlapping L-blocks).",
    )
    parser.add_argument(
        "--no-offset-embedding",
        action="store_true",
        default=False,
        help="Disable horizon-offset embedding in the discriminator.",
    )
    parser.add_argument(
        "--native-repr-stride",
        type=int,
        default=1,
        help="If >1, downsample GT+fakes with [::stride] before slicing (native stride-2 grid).",
    )
    parser.add_argument(
        "--bin-decoder",
        choices=["mean", "expectation", "pdf_expectation"],
        default="mean",
        help="decode_dual mode for round-trip filter (match staged eval decoder).",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--offset-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--visualize-confusions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each combo, save TP/TN/FP/FN PNGs under output-dir/disc_confusions/ "
             "(default: on; disable with --no-visualize-confusions).",
    )
    parser.add_argument(
        "--fake-agg",
        choices=["prob_mean", "sample0"],
        default="sample0",
        help="How to reduce pack['samples'] into the disc fake trajectory. "
             "sample0 (default) = first stochastic draw; "
             "prob_mean = mean over S (intentional averaging only). "
             "Anchor/deterministic is refused.",
    )
    parser.add_argument("--viz-per-bucket", type=int, default=2)
    parser.add_argument("--viz-variate", type=int, default=0)
    parser.add_argument("--viz-lookback-tail", type=int, default=32)
    parser.add_argument(
        "--viz-plot-dir",
        type=Path,
        default=None,
        help="Override confusion plot root (default: <output-dir>/disc_confusions).",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--merge-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, merge partials into metrics.json (disable for parallel shard jobs).",
    )
    parser.add_argument(
        "--merge-partials-only",
        action="store_true",
        help="Only merge partials/ into metrics.json + CSV + manifest.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.anchor_config_by_dataset = _parse_dataset_map(args.anchor_config_by_dataset)
    args.binary_config_by_dataset = _parse_dataset_map(args.binary_config_by_dataset)
    if bool(getattr(args, "visualize_confusions", False)):
        args.save_checkpoints = True
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-smoke"
    if args.bin_match_filter and args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-binmatch-{args.bin_match_filter}"
    args.datasets = args.datasets[:1]
    args.fake_sources = args.fake_sources[:1]
    args.slice_lengths = args.slice_lengths[:1]
    args.test_max_items = min(args.test_max_items or 8, 8)
    args.max_windows = min(args.max_windows or 8, 8)
    args.max_train_examples = min(args.max_train_examples or 128, 128)
    args.max_eval_examples = min(args.max_eval_examples or 64, 64)
    args.batch_size = min(args.batch_size, 32)
    args.epochs = min(args.epochs, 2)
    args.patience = min(args.patience, 2)
    args.max_batches_per_epoch = min(args.max_batches_per_epoch or 2, 2)


def main() -> None:
    args = parse_args()
    if args.bin_match_filter and args.output_dir == DEFAULT_DISC_OUTPUT:
        args.output_dir = DEFAULT_DISC_OUTPUT.parent / f"{DEFAULT_DISC_OUTPUT.name}-binmatch-{args.bin_match_filter}"
    if args.visualize_confusions:
        args.save_checkpoints = True
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test(args)
    elif args.merge_partials_only:
        run_merge_only(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
