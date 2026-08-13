#!/usr/bin/env python3
"""Paired, native-horizon shape descriptors for binary, MMPD, and iTransformer.

The binary pack defines the common test origins.  Its ground-truth horizons are
matched numerically to the corresponding MMPD and iTransformer pack rows before
any descriptor is calculated.  L=8 uses its eight observed values directly;
this script never interpolates or resamples a forecast horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "electricity", "traffic", "exchange_rate", "PeMS", "solar_Alabama", "ETTm1", "ETTm2")
SEED = 20260812
EPS = 1e-10
ALIGN_TOL_MSE = 1e-8
MODELS = ("binary_quad_t", "mmpd", "itransformer")

BINARY_REL = {
    "ETTh1": "08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6/raw/staged_dpmpp_samples_ETTh1.npz",
    "ETTh2": "08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2/raw/staged_dpmpp_samples_ETTh2.npz",
    "electricity": "08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity/raw/staged_dpmpp_samples_electricity.npz",
    "traffic": "08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic/raw/staged_dpmpp_samples_traffic.npz",
    "exchange_rate": "08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate/raw/staged_dpmpp_samples_exchange_rate.npz",
    "PeMS": "08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems/raw/staged_dpmpp_samples_PeMS.npz",
    "solar_Alabama": "08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama/raw/staged_dpmpp_samples_solar_Alabama.npz",
    "ETTm1": "08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1/raw/staged_dpmpp_samples_ETTm1.npz",
    "ETTm2": "08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2/raw/staged_dpmpp_samples_ETTm2.npz",
}
MMPD_REL = {
    "ETTh1": "07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_ETTh1.npz",
    "ETTh2": "08-04-mmpd-decoder-paper-lb336-hz96-ETTh2/raw/mmpd_ETTh2.npz",
    "electricity": "07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_electricity.npz",
    "traffic": "07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_traffic.npz",
    "exchange_rate": "07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_exchange_rate.npz",
    "PeMS": "08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_PeMS.npz",
    "solar_Alabama": "08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_solar_Alabama.npz",
    "ETTm1": "08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_ETTm1.npz",
    "ETTm2": "08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_ETTm2.npz",
}
ITRANS_TAG = {
    "ETTh1": "L2_D256_lr0.0001", "ETTh2": "L2_D128_lr0.0001", "electricity": "L3_D512_lr0.0005",
    "traffic": "L4_D512_lr0.001", "exchange_rate": "L2_D128_lr0.0001", "PeMS": "L4_D512_lr0.001",
    "solar_Alabama": "L2_D512_lr0.0005", "ETTm1": "L2_D128_lr0.0001", "ETTm2": "L2_D128_lr0.0001",
}
# All packs agree on the selected variates except Solar.  Its historical MMPD
# pack used a different two-variate subset; these are the one raw series shared
# by all three saved packs (binary, MMPD, iTransformer respectively).
COMMON_CHANNELS = {
    "solar_Alabama": {"binary_quad_t": (1,), "mmpd": (0,), "itransformer": (1,)},
}


def as_nvh(values: np.ndarray, source: Path) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{source}: expected 3-D array, got {values.shape}")
    if values.shape[-1] == 96:
        return values
    if values.shape[1] == 96:
        return np.transpose(values, (0, 2, 1))
    raise ValueError(f"{source}: no 96-step horizon in {values.shape}")


def load_probabilistic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        y_true = as_nvh(pack["y_true"], path)
        samples = np.asarray(pack["samples"], dtype=np.float64)
    if samples.ndim != 4 or samples.shape[:2] != y_true.shape[:2] or samples.shape[-1] != y_true.shape[-1]:
        raise ValueError(f"{path}: incompatible true={y_true.shape}, samples={samples.shape}")
    return y_true, samples


def resolve_itransformer(root: Path, dataset: str) -> Path:
    matches = sorted(root.glob(f"{dataset}_336_96_{ITRANS_TAG[dataset]}_iTransformer_*"))
    matches = [p for p in matches if (p / "true.npy").is_file() and (p / "pred.npy").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"{dataset}: expected exactly one iTransformer result, found {matches}")
    return matches[0]


def nearest_alignment(query: np.ndarray, pool: np.ndarray, label: str) -> tuple[np.ndarray, dict[str, float]]:
    """Match each binary ground-truth horizon to one unique reference row."""
    if query.shape[1:] != pool.shape[1:]:
        raise ValueError(f"{label}: incompatible GT shapes {query.shape} vs {pool.shape}")
    chosen = np.empty(len(query), dtype=np.int64)
    errors = np.empty(len(query), dtype=np.float64)
    for i, target in enumerate(query):
        mse = np.mean((pool - target[None, :, :]) ** 2, axis=(1, 2))
        chosen[i] = int(np.argmin(mse))
        errors[i] = mse[chosen[i]]
    if errors.max(initial=0.0) > ALIGN_TOL_MSE:
        raise ValueError(f"{label}: cannot align all binary origins; max GT MSE={errors.max():.3e}")
    if len(np.unique(chosen)) != len(chosen):
        raise ValueError(f"{label}: ambiguous alignment: {len(chosen) - len(np.unique(chosen))} duplicate reference rows")
    return chosen, {"max_gt_mse": float(errors.max(initial=0.0)), "mean_gt_mse": float(errors.mean())}


def select_sample(samples: np.ndarray, rows: np.ndarray, dataset: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + sum(ord(c) for c in f"{dataset}:{model}:one-sample"))
    ids = rng.integers(samples.shape[2], size=len(rows))
    return samples[rows, :, ids, :], ids


def subwindows(values: np.ndarray, length: int) -> np.ndarray:
    if length == values.shape[-1]:
        return values
    if not 2 <= length <= values.shape[-1]:
        raise ValueError(f"invalid native subwindow length {length}")
    windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=-1)
    return windows.transpose(0, 2, 1, 3).reshape(-1, values.shape[1], length)


def normalize(curves: np.ndarray, mode: str) -> np.ndarray:
    centered = curves - curves.mean(axis=1, keepdims=True)
    if mode == "demean":
        return centered
    if mode == "zscore":
        scale = centered.std(axis=1, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > EPS)
    raise ValueError(f"unknown normalization {mode}")


def signs_without_zeros(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values).astype(np.int8)
    for row in signs:
        nonzero = np.flatnonzero(row)
        if len(nonzero) == 0:
            continue
        row[:nonzero[0]] = row[nonzero[0]]
        for left, right in zip(nonzero[:-1], nonzero[1:]):
            row[left + 1:right] = row[left]
        row[nonzero[-1] + 1:] = row[nonzero[-1]]
    return signs


def spacing_stats(turns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(len(turns))
    std = np.zeros(len(turns))
    denominator = max(turns.shape[1] - 1, 1)
    for i, turn in enumerate(turns):
        positions = np.flatnonzero(turn)
        if len(positions) > 1:
            gaps = np.diff(positions) / denominator
            mean[i], std[i] = gaps.mean(), gaps.std()
    return mean, std


def haar_features(curves: np.ndarray) -> dict[str, np.ndarray]:
    approx = curves.copy()
    energies: list[np.ndarray] = []
    while approx.shape[1] >= 2 and approx.shape[1] % 2 == 0:
        even, odd = approx[:, 0::2], approx[:, 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approx = (even + odd) / np.sqrt(2.0)
        energies.append(np.mean(detail**2, axis=1))
    energies.append(np.mean(approx**2, axis=1))
    stacked = np.stack(energies, axis=1)
    stacked /= np.maximum(stacked.sum(axis=1, keepdims=True), EPS)
    return {f"haar_energy_level_{i + 1}": stacked[:, i] for i in range(stacked.shape[1])}


def descriptor_features(curves: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    z = curves
    length = z.shape[1]
    d1 = np.diff(z, axis=1)
    d2 = np.diff(d1, axis=1)
    s1, s2 = signs_without_zeros(d1), signs_without_zeros(d2)
    extrema = s1[:, 1:] != s1[:, :-1]
    inflections = s2[:, 1:] != s2[:, :-1]
    extrema_mean, extrema_std = spacing_stats(extrema)
    inflect_mean, inflect_std = spacing_stats(inflections)
    output: dict[str, np.ndarray] = {
        "deriv_mean": d1.mean(axis=1), "deriv_std": d1.std(axis=1),
        "deriv_abs_mean": np.abs(d1).mean(axis=1), "deriv_p95_abs": np.quantile(np.abs(d1), .95, axis=1),
        "second_deriv_std": d2.std(axis=1), "second_deriv_abs_mean": np.abs(d2).mean(axis=1),
        "deriv_zero_crossing_rate": extrema.mean(axis=1), "local_extrema_count": extrema.sum(axis=1).astype(float),
        "local_extrema_spacing_mean": extrema_mean, "local_extrema_spacing_std": extrema_std,
        "inflection_count": inflections.sum(axis=1).astype(float),
        "inflection_spacing_mean": inflect_mean, "inflection_spacing_std": inflect_std,
    }
    family = {name: "derivative" for name in output if name.startswith("deriv_")}
    family.update({name: "curvature" for name in output if name.startswith("second_")})
    family.update({name: "turning_structure" for name in output if name.startswith(("local_", "inflection_"))})

    spectrum = np.abs(np.fft.rfft(z, axis=1)) ** 2
    spectrum[:, 0] = 0.0
    total = spectrum.sum(axis=1, keepdims=True)
    power = np.divide(spectrum, total, out=np.zeros_like(spectrum), where=total > EPS)
    freqs = np.fft.rfftfreq(length)
    output["spectral_centroid"] = (power * freqs).sum(axis=1)
    non_dc = power[:, 1:]
    output["spectral_entropy"] = -(non_dc * np.log(np.maximum(non_dc, EPS))).sum(axis=1) / np.log(max(non_dc.shape[1], 2))
    for band, entries in enumerate(np.array_split(non_dc, 4, axis=1), 1):
        output[f"spectral_energy_band_{band}"] = entries.sum(axis=1)
    family.update({name: "spectrum" for name in output if name.startswith("spectral_")})

    output.update(haar_features(z))
    family.update({name: "wavelet" for name in output if name.startswith("haar_")})

    x = np.linspace(-1.0, 1.0, length)
    design = np.stack((x, np.ones_like(x)), axis=1)
    beta, *_ = np.linalg.lstsq(design, z.T, rcond=None)
    residual = z - (design @ beta).T
    residual -= residual.mean(axis=1, keepdims=True)
    denom = np.sum(residual**2, axis=1)
    max_lag = min(16, length - 1)
    acf = np.stack([
        np.divide(np.sum(residual[:, :-lag] * residual[:, lag:], axis=1), denom, out=np.zeros_like(denom), where=denom > EPS)
        for lag in range(1, max_lag + 1)
    ], axis=1)
    output["residual_acf_lag1"] = acf[:, 0]
    output["residual_acf_abs_mean"] = np.abs(acf).mean(axis=1)
    below = np.abs(acf) <= np.exp(-1.0)
    output["residual_acf_decay_lag"] = np.where(below.any(axis=1), below.argmax(axis=1) + 1, max_lag + 1).astype(float)
    family.update({name: "regularity" for name in output if name.startswith("residual_")})

    tv = np.abs(d1).sum(axis=1)
    arc = np.sqrt((1.0 / (length - 1)) ** 2 + d1**2).sum(axis=1)
    output["total_variation_per_step"] = tv / (length - 1)
    output["total_variation_per_arc_length"] = tv / np.maximum(arc, EPS)
    family.update({name: "variation" for name in output if name.startswith("total_variation_")})
    return output, family


def feature_rows(dataset: str, model: str, y_true: np.ndarray, pred: np.ndarray, length: int, mode: str) -> list[dict[str, Any]]:
    target, family = descriptor_features(normalize(y_true.reshape(-1, length), mode))
    forecast, _ = descriptor_features(normalize(pred.reshape(-1, length), mode))
    rows: list[dict[str, Any]] = []
    for feature in sorted(target):
        diff = forecast[feature] - target[feature]
        gt_std = float(target[feature].std())
        mae = float(np.abs(diff).mean())
        rows.append({
            "dataset": dataset, "model": model, "length": length, "normalization": mode,
            "family": family[feature], "feature": feature, "n_curves": len(diff),
            "gt_mean": float(target[feature].mean()), "gt_std": gt_std,
            "pred_mean": float(forecast[feature].mean()), "bias": float(diff.mean()),
            "mae": mae, "rmse": float(np.sqrt(np.mean(diff**2))),
            "normalized_mae": None if gt_std <= EPS else mae / gt_std,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty report")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = row["normalized_mae"]
        if value is not None and np.isfinite(value):
            grouped.setdefault((row["dataset"], row["model"], row["family"]), []).append(float(value))
    output = []
    for (dataset, model, family), values in sorted(grouped.items()):
        output.append({"dataset": dataset, "model": model, "family": family, "n_features": len(values),
                       "mean_normalized_mae": float(np.mean(values)), "median_normalized_mae": float(np.median(values))})
    return output


def macro_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["family"]), []).append(float(row["mean_normalized_mae"]))
    return [{"model": model, "family": family, "n_datasets": len(values), "macro_mean_normalized_mae": float(np.mean(values)),
             "macro_median_normalized_mae": float(np.median(values))}
            for (model, family), values in sorted(grouped.items())]


def markdown(macro: list[dict[str, Any]], length: int, mode: str) -> str:
    by_key = {(r["family"], r["model"]): r["macro_mean_normalized_mae"] for r in macro}
    families = sorted({r["family"] for r in macro})
    lines = [f"# Paired native shape descriptors: L={length}, {mode}", "", "Lower is better. Each cell is an equal-dataset macro average of family descriptor MAEs, each divided by that descriptor's cross-curve GT standard deviation. Features with zero GT variance are excluded; no interpolation or per-bin win counts are used.", "", "| family | binary Quad-T | MMPD | iTransformer |", "|---|---:|---:|---:|"]
    for family in families:
        values = [by_key[(family, model)] for model in MODELS]
        best = min(values)
        formatted = [f"**{value:.4f}**" if value == best else f"{value:.4f}" for value in values]
        lines.append(f"| {family} | " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--lengths", nargs="+", type=int, default=[8, 96])
    parser.add_argument("--normalizations", nargs="+", choices=("zscore", "demean"), default=["zscore", "demean"])
    args = parser.parse_args()
    if any(length not in (8, 96) for length in args.lengths):
        raise ValueError("only L=8 and L=96 are supported")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"protocol": {"alignment": "binary GT horizons matched to unique MMPD and iTransformer GT rows by native 96-step MSE <= 1e-8", "subwindows": "all sliding, paired native subwindows; no interpolation or resampling", "normalizations": {"zscore": "per-curve de-mean then divide by its own standard deviation", "demean": "per-curve de-mean only; no variance scaling"}, "sample": "one deterministic seeded saved full-horizon sample for binary and MMPD; deterministic iTransformer output"}, "datasets": {}}
    all_features: dict[tuple[int, str], list[dict[str, Any]]] = {(l, n): [] for l in args.lengths for n in args.normalizations}
    for dataset in args.datasets:
        binary_path = args.binary_root / "results/datasets" / BINARY_REL[dataset]
        mmpd_path = args.reference_root / "results/datasets" / MMPD_REL[dataset]
        itrans_path = resolve_itransformer(args.reference_root / "temp/iTransformer/results", dataset)
        binary_y, binary_samples = load_probabilistic(binary_path)
        mmpd_y, mmpd_samples = load_probabilistic(mmpd_path)
        itrans_y = as_nvh(np.load(itrans_path / "true.npy", mmap_mode="r"), itrans_path / "true.npy")
        itrans_pred = as_nvh(np.load(itrans_path / "pred.npy", mmap_mode="r"), itrans_path / "pred.npy")
        channel_selection = COMMON_CHANNELS.get(dataset)
        if channel_selection is not None:
            binary_channels = channel_selection["binary_quad_t"]
            mmpd_channels = channel_selection["mmpd"]
            itrans_channels = channel_selection["itransformer"]
            binary_y, binary_samples = binary_y[:, binary_channels], binary_samples[:, binary_channels]
            mmpd_y, mmpd_samples = mmpd_y[:, mmpd_channels], mmpd_samples[:, mmpd_channels]
            itrans_y, itrans_pred = itrans_y[:, itrans_channels], itrans_pred[:, itrans_channels]
        mmpd_rows, mmpd_info = nearest_alignment(binary_y, mmpd_y, f"{dataset}/MMPD")
        itrans_rows, itrans_info = nearest_alignment(binary_y, itrans_y, f"{dataset}/iTransformer")
        rows = np.arange(len(binary_y))
        binary_pred, binary_ids = select_sample(binary_samples, rows, dataset, "binary_quad_t")
        mmpd_pred, mmpd_ids = select_sample(mmpd_samples, mmpd_rows, dataset, "mmpd")
        paired = {"binary_quad_t": (binary_y, binary_pred), "mmpd": (mmpd_y[mmpd_rows], mmpd_pred), "itransformer": (itrans_y[itrans_rows], itrans_pred[itrans_rows])}
        manifest["datasets"][dataset] = {"binary_origins": len(binary_y), "n_variates": int(binary_y.shape[1]), "channel_selection": channel_selection, "mmpd_alignment": mmpd_info, "itransformer_alignment": itrans_info, "binary_sample_ids": binary_ids.tolist(), "mmpd_sample_ids": mmpd_ids.tolist()}
        for length in args.lengths:
            for model, (target, pred) in paired.items():
                target_sub, pred_sub = subwindows(target, length), subwindows(pred, length)
                for mode in args.normalizations:
                    all_features[(length, mode)].extend(feature_rows(dataset, model, target_sub, pred_sub, length, mode))
        print(f"done {dataset}: {len(binary_y)} exactly aligned origins", flush=True)
    for (length, mode), rows in all_features.items():
        features_path = args.output_dir / f"feature_errors_l{length}_{mode}.csv"
        write_csv(features_path, rows)
        summary = family_summary(rows)
        macro = macro_family_summary(summary)
        write_csv(args.output_dir / f"family_summary_l{length}_{mode}.csv", summary)
        write_csv(args.output_dir / f"family_macro_l{length}_{mode}.csv", macro)
        (args.output_dir / f"report_l{length}_{mode}.md").write_text(markdown(macro, length, mode), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
