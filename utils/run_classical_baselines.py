#!/usr/bin/env python3
"""Classical statistical forecasting baselines aligned with DL staged_eval metrics.

Evaluates AutoARIMA / AutoETS / AutoTheta / SeasonalNaive (univariate) and
VAR / VECM / DynamicFactor (multivariate) on the same splits, global z-score
normalization, and test windows as the diffusion pipeline, then logs to
ts-sandbox-leaderboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from utils.load_dotenv import load_repo_dotenv

    load_repo_dotenv(REPO_ROOT)
except ImportError:
    pass

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
    load_dataset,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)
from utils.leaderboard_config_nicknames import (  # noqa: E402
    CLASSICAL_BASELINES_NICKNAME,
    CLASSICAL_BASELINES_RAW,
)

logger = logging.getLogger(__name__)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = os.environ.get("WANDB_ENTITY", "calvincao")
JOB_TYPE = "classical_baseline"
EVAL_TEST_STRIDE = 4
VAR_MAX_VARIATES = 20
DFM_MIN_VARIATES = 10
DEFAULT_CONFIG = REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml"

ALL_DATASETS = [
    name for name in DATASET_REGISTRY if name != "dalia"
]

UNIVARIATE_METHODS = ("AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive", "classical_ensemble")
MULTIVARIATE_METHODS = ("VAR", "VECM", "DFM")


def _deterministic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }


def _wandb_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Map to the same keys staged_eval uses for leaderboard panels."""
    out = {
        "eval/staged_anchor_mse": metrics["mse"],
        "eval/staged_anchor_mae": metrics["mae"],
        "eval/staged_prob_mse": metrics["mse"],
        "eval/staged_prob_mae": metrics["mae"],
        "eval/staged_sample_mean_mse": metrics["mse"],
        "eval/staged_sample_mean_mae": metrics["mae"],
    }
    return out


def _leaderboard_group(
    dataset: str,
    config_stem: str,
    job_id: Optional[str] = None,
) -> str:
    jid = job_id or os.environ.get("SLURM_JOB_ID") or "local"
    date_str = datetime.now().strftime("%m-%d")
    return f"{date_str}-{jid}-{dataset}-{config_stem}"


def _load_raw_normalized(
    dataset: str,
    variate_indices: Sequence[int],
    lookback: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    path, date_col = _resolve_registry_path(dataset)
    data = _load_dataset_array(path, date_col)
    if variate_indices:
        data = data[:, list(variate_indices)]
    n = len(data)
    border1s, border2s = _paper_split_borders(dataset, n, lookback)
    train_end = border2s[0]
    train_slice = data[:train_end]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    data = ((data - mean) / std).astype(np.float32)
    meta = {
        "border1s": border1s,
        "border2s": border2s,
        "mean": mean,
        "std": std,
        "n_rows": n,
    }
    return data, meta


def _select_window_indices(n_windows: int, fraction: float, seed: int) -> List[int]:
    if fraction >= 1.0 or n_windows <= 1:
        return list(range(n_windows))
    k = max(1, int(round(n_windows * fraction)))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(n_windows), k))
    return idx


def _history_cap(origin: int, max_fit_rows: int) -> int:
    if max_fit_rows <= 0 or origin <= max_fit_rows:
        return 0
    return origin - max_fit_rows


_STATSFORECAST_FREQ = "D"
_STATSFORECAST_ORIGIN = "2000-01-01"


def _build_long_df(
    data: np.ndarray,
    origin: int,
    history_start: int,
) -> pd.DataFrame:
    n_vars = data.shape[1]
    rows: List[Dict[str, Any]] = []
    for vid in range(n_vars):
        for t in range(history_start, origin):
            rows.append({"unique_id": str(vid), "ds": t, "y": float(data[t, vid])})
    df = pd.DataFrame(rows)
    # statsforecast 1.5.x + pandas 2.x rejects integer freq=1; use daily timestamps.
    df["ds"] = pd.to_datetime(df["ds"], unit="D", origin=_STATSFORECAST_ORIGIN)
    return df


def _uni_model_factory(name: str, season_length: int):
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    factories = {
        "AutoARIMA": lambda: AutoARIMA(season_length=season_length),
        "AutoETS": lambda: AutoETS(season_length=season_length),
        "AutoTheta": lambda: AutoTheta(season_length=season_length),
        "SeasonalNaive": lambda: SeasonalNaive(season_length=season_length),
    }
    if name not in factories:
        raise KeyError(name)
    return factories[name]()


def _fit_univariate_window(
    data: np.ndarray,
    origin: int,
    horizon: int,
    season_length: int,
    n_jobs: int,
    max_fit_rows: int,
    methods: Sequence[str],
) -> Dict[str, np.ndarray]:
    from statsforecast import StatsForecast

    history_start = _history_cap(origin, max_fit_rows)
    train_df = _build_long_df(data, origin, history_start)
    if train_df.empty:
        raise ValueError(f"empty train_df at origin={origin}")

    fit_methods = [m for m in methods if m in UNIVARIATE_METHODS and m != "classical_ensemble"]
    if not fit_methods:
        return {}

    sf = StatsForecast(
        models=[_uni_model_factory(m, season_length) for m in fit_methods],
        freq=_STATSFORECAST_FREQ,
        n_jobs=n_jobs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf.fit(train_df)
        pred_df = sf.predict(h=horizon)
        if "unique_id" not in pred_df.columns:
            pred_df = pred_df.reset_index()

    n_vars = data.shape[1]
    out: Dict[str, np.ndarray] = {}
    for col in fit_methods:
        if col not in pred_df.columns:
            continue
        arr = np.full((n_vars, horizon), np.nan, dtype=np.float64)
        for vid in range(n_vars):
            sub = pred_df[pred_df["unique_id"] == str(vid)].sort_values("ds")
            vals = sub[col].to_numpy(dtype=np.float64)
            if len(vals) >= horizon:
                arr[vid] = vals[:horizon]
        out[col] = arr

    if "classical_ensemble" in methods:
        ensemble_parts = [out[k] for k in ("AutoARIMA", "AutoETS", "AutoTheta") if k in out]
        if ensemble_parts:
            out["classical_ensemble"] = np.nanmean(np.stack(ensemble_parts, axis=0), axis=0)
    return out


def _fit_multivariate_window(
    data: np.ndarray,
    origin: int,
    horizon: int,
    n_vars: int,
    max_fit_rows: int,
    methods: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

    history_start = _history_cap(origin, max_fit_rows)
    train = data[history_start:origin, :]
    notes: Dict[str, Any] = {}
    preds: Dict[str, np.ndarray] = {}

    if "VAR" in methods or "VECM" in methods:
        if n_vars > VAR_MAX_VARIATES:
            notes["var_skipped"] = f"n_variates={n_vars}>{VAR_MAX_VARIATES}"
            notes["vecm_skipped"] = notes["var_skipped"]
        elif n_vars < 2:
            notes["var_skipped"] = "n_variates<2"
            notes["vecm_skipped"] = notes["var_skipped"]
        else:
            train_df = pd.DataFrame(train, columns=[f"v{i}" for i in range(n_vars)])
            maxlags = max(1, min(15, len(train_df) // 10))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    var_model = VAR(train_df)
                    lag_sel = var_model.select_order(maxlags=maxlags)
                    lag_order = int(lag_sel.aic) if lag_sel.aic is not None else 1
                    lag_order = max(1, min(lag_order, maxlags))
                    if "VAR" in methods:
                        var_res = var_model.fit(lag_order)
                        fc = var_res.forecast(train_df.values[-lag_order:], steps=horizon)
                        preds["VAR"] = fc.T.astype(np.float64)
                        notes["var_lag_order"] = lag_order
                    if "VECM" in methods:
                        joh = coint_johansen(train_df, det_order=0, k_ar_diff=lag_order)
                        rank = int(np.sum(joh.lr1 > joh.cvt[:, 1]))
                        notes["vecm_coint_rank"] = rank
                        if rank > 0:
                            k_ar = max(1, lag_order - 1)
                            vecm = VECM(
                                train_df,
                                k_ar_diff=k_ar,
                                coint_rank=rank,
                                deterministic="ci",
                            )
                            vecm_res = vecm.fit()
                            vecm_fc = vecm_res.predict(steps=horizon)
                            preds["VECM"] = np.asarray(vecm_fc, dtype=np.float64).T
                        else:
                            notes["vecm_skipped"] = "not_cointegrated"
            except Exception as exc:
                notes["var_error"] = str(exc)
                if "VAR" in methods:
                    notes["var_skipped"] = "fit_failed"
                if "VECM" in methods:
                    notes["vecm_skipped"] = "var_failed"

    if "DFM" in methods:
        if n_vars >= DFM_MIN_VARIATES:
            train_df = pd.DataFrame(train, columns=[f"v{i}" for i in range(n_vars)])
            k_factors = max(1, min(3, n_vars // 5))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dfm = DynamicFactor(train_df, k_factors=k_factors, factor_order=1)
                    dfm_res = dfm.fit(disp=False)
                    dfm_fc = dfm_res.forecast(steps=horizon)
                    preds["DFM"] = np.asarray(dfm_fc, dtype=np.float64).T
                    notes["dfm_k_factors"] = k_factors
            except Exception as exc:
                notes["dfm_error"] = str(exc)
        else:
            notes["dfm_skipped"] = f"n_variates={n_vars}<{DFM_MIN_VARIATES}"

    return preds, notes


def evaluate_dataset(
    dataset: str,
    *,
    subset_meta: Dict[str, Any],
    season_length: int,
    lookback: int,
    horizon: int,
    overlap: int,
    test_stride: int,
    eval_fraction: float,
    seed: int,
    n_jobs: int,
    max_fit_rows: int,
    max_windows: Optional[int],
    methods: Sequence[str],
) -> Dict[str, Any]:
    variate_indices = list(subset_meta["variate_indices"])
    _, _, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        lookback=lookback,
        horizon=horizon,
        stride=int(subset_meta.get("train_stride", 1)),
        test_stride=test_stride,
        lookback_overlap=overlap,
    )
    data, split_meta = _load_raw_normalized(dataset, variate_indices, lookback)
    border1s = split_meta["border1s"]
    test_base = border1s[2]

    window_indices = _select_window_indices(len(test_ds), eval_fraction, seed)
    if max_windows is not None:
        window_indices = window_indices[:max_windows]

    method_preds: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    targets: List[np.ndarray] = []

    for wi in window_indices:
        past, future = test_ds[wi]
        target = future.numpy()[..., overlap:]
        targets.append(target)
        origin = test_base + wi * test_stride + lookback

        uni_methods = [m for m in methods if m in UNIVARIATE_METHODS]
        if uni_methods:
            uni_preds = _fit_univariate_window(
                data, origin, horizon, season_length, n_jobs, max_fit_rows, uni_methods,
            )
            for m in uni_methods:
                if m in uni_preds:
                    method_preds[m].append(uni_preds[m])

        mv_methods = [m for m in methods if m in MULTIVARIATE_METHODS]
        if mv_methods:
            mv_preds, _ = _fit_multivariate_window(
                data, origin, horizon, len(variate_indices), max_fit_rows, mv_methods,
            )
            for m in mv_methods:
                if m in mv_preds:
                    method_preds[m].append(mv_preds[m])

    y_true = np.stack(targets, axis=0)
    results: Dict[str, Any] = {
        "dataset": dataset,
        "subset_id": subset_meta["subset_id"],
        "n_windows": len(window_indices),
        "n_variates": len(variate_indices),
        "methods": {},
    }
    for method in methods:
        chunks = method_preds.get(method) or []
        if not chunks:
            results["methods"][method] = {"skipped": True}
            continue
        y_pred = np.stack(chunks, axis=0)
        if y_pred.shape != y_true.shape:
            results["methods"][method] = {
                "skipped": True,
                "reason": f"shape mismatch pred={y_pred.shape} true={y_true.shape}",
            }
            continue
        metrics = _deterministic_metrics(y_true, y_pred)
        results["methods"][method] = {"metrics": metrics, "n_windows": len(chunks)}
    return results


def _wandb_api_key_usable() -> bool:
    try:
        from models.diffusion_tsf.pipeline import wandb_utils

        return wandb_utils._api_key_usable()
    except Exception:
        return bool(os.environ.get("WANDB_API_KEY", "").strip())


def leaderboard_marker_path(output_dir: Path, dataset: str, method: str) -> Path:
    return output_dir / "partials" / f".leaderboard_{dataset}_{method}.json"


def log_method_to_wandb(
    dataset: str,
    method: str,
    metrics: Dict[str, float],
    *,
    config: Dict[str, Any],
    config_path: Path,
    output_dir: Path,
    job_id: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[str]:
    if not _wandb_api_key_usable():
        logger.warning("WANDB_API_KEY not set; skip wandb for %s/%s", dataset, method)
        return None

    marker = leaderboard_marker_path(output_dir, dataset, method)
    if not force and marker.is_file():
        logger.info("%s/%s: wandb already logged (marker)", dataset, method)
        return None

    from models.diffusion_tsf.pipeline.wandb_utils import make_phase_run_name

    import wandb

    config_stem = config_path.stem
    group = _leaderboard_group(dataset, config_stem, job_id=job_id)
    phase_slug = method.replace("_", "-")
    run_name = make_phase_run_name(group, phase_slug)
    tags = [dataset, "eval", "classical-baseline", config_stem]
    wandb_config = {
        **config,
        "config_nickname": CLASSICAL_BASELINES_NICKNAME,
        "baseline": CLASSICAL_BASELINES_RAW,
        "dataset": dataset,
        "model": method,
        "run_type": JOB_TYPE,
        "data_subset_config": str(config_path.resolve()),
    }

    if dry_run:
        logger.info("dry-run wandb: %s | group=%s | tags=%s", run_name, group, tags)
        return None

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=run_name,
        group=group,
        job_type=JOB_TYPE,
        tags=tags,
        notes=f"classical baseline {method} on {dataset}",
        config=wandb_config,
        settings=wandb.Settings(console="off"),
        reinit=True,
    )
    try:
        payload = _wandb_metrics(metrics)
        wandb.log(payload, step=0)
        for k, v in payload.items():
            run.summary[k] = v
        url = run.url
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("w", encoding="utf-8") as f:
            json.dump({"group": group, "run_id": run.id, "url": url}, f, indent=2)
        logger.info("[leaderboard] %s/%s: %s", dataset, method, url)
        return url
    finally:
        run.finish()


def _write_partial(output_dir: Path, dataset: str, method: str, payload: Dict[str, Any]) -> None:
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / f"{dataset}_{method}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML with experiment.data_subset (default: flat subsets config)",
    )
    p.add_argument("--datasets", default=",".join(ALL_DATASETS))
    p.add_argument("--datasets-dir", default=str(REPO_ROOT / "datasets"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-test-fraction", type=float, default=1.0)
    p.add_argument("--test-stride", type=int, default=EVAL_TEST_STRIDE)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-fit-rows", type=int, default=8192)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Skip wandb.init")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--methods",
        default=",".join(list(UNIVARIATE_METHODS) + list(MULTIVARIATE_METHODS)),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import models.diffusion_tsf.train_multivariate_pipeline as pipe

    pipe.DATASETS_DIR = os.path.abspath(args.datasets_dir)

    if args.smoke_test:
        args.datasets = "illness"
        args.eval_test_fraction = 1.0
        args.max_windows = 1
        args.n_jobs = 1
        args.max_fit_rows = 512
        # AutoARIMA/ETS on even one window can take minutes; smoke only needs the path.
        args.methods = "SeasonalNaive,VAR"

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASET_REGISTRY or d == "dalia"]
    if unknown:
        raise SystemExit(f"Unknown or excluded datasets: {unknown}")

    policy = _load_data_subset_policy(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    job_tail = (os.environ.get("SLURM_JOB_ID") or "local")[-3:]
    date_str = datetime.now().strftime("%m-%d")
    stem = f"{date_str}-{job_tail}-classical-baselines"
    output_dir = args.output_dir or (REPO_ROOT / "results" / "datasets" / stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"datasets": {}, "stem": stem, "config": str(args.config)}

    for dataset in datasets:
        logger.info("=== %s ===", dataset)
        _, _, season_length = DATASET_REGISTRY[dataset]
        subset_meta = resolve_subset_meta_for_dataset(dataset, policy, args.seed)
        lookback = LOOKBACK_LENGTH
        horizon = FORECAST_LENGTH
        overlap = LOOKBACK_OVERLAP

        try:
            eval_out = evaluate_dataset(
                dataset,
                subset_meta=subset_meta,
                season_length=season_length,
                lookback=lookback,
                horizon=horizon,
                overlap=overlap,
                test_stride=args.test_stride,
                eval_fraction=args.eval_test_fraction,
                seed=args.seed,
                n_jobs=args.n_jobs,
                max_fit_rows=args.max_fit_rows,
                max_windows=args.max_windows,
                methods=methods,
            )
        except FileNotFoundError as exc:
            logger.error("%s: dataset file missing (%s)", dataset, exc)
            summary["datasets"][dataset] = {"error": str(exc)}
            continue
        except Exception as exc:
            logger.exception("%s: evaluation failed", dataset)
            summary["datasets"][dataset] = {"error": str(exc)}
            continue

        summary["datasets"][dataset] = eval_out
        base_config = {
            "horizon": horizon,
            "lookback": lookback,
            "season_length": season_length,
            "split": subset_meta["subset_id"],
            "test_stride": args.test_stride,
            "n_jobs": args.n_jobs,
            "compute": "l40s-cpu",
            "max_fit_rows": args.max_fit_rows,
            "data_subset_config": str(args.config.resolve()),
            "variate_indices": list(subset_meta["variate_indices"]),
        }

        for method, method_out in eval_out["methods"].items():
            if method_out.get("skipped"):
                logger.warning("%s/%s skipped: %s", dataset, method, method_out)
                continue
            metrics = method_out["metrics"]
            payload = {
                "dataset": dataset,
                "method": method,
                "metrics": metrics,
                "wandb": _wandb_metrics(metrics),
                "config": {**base_config, "model": method},
            }
            _write_partial(output_dir, dataset, method, payload)
            logger.info(
                "%s/%s mse=%.6f mae=%.6f (n_windows=%s)",
                dataset, method, metrics["mse"], metrics["mae"], method_out.get("n_windows"),
            )
            if not args.no_wandb:
                log_method_to_wandb(
                    dataset,
                    method,
                    metrics,
                    config=payload["config"],
                    config_path=args.config,
                    output_dir=output_dir,
                    job_id=os.environ.get("SLURM_JOB_ID"),
                    dry_run=args.dry_run,
                )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
