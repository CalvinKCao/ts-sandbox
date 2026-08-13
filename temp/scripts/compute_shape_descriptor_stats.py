#!/usr/bin/env python3
"""Level/scale-invariant shape descriptors for canvas128 binary, MMPD, iTransformer.

For each dataset and model, the binary saved pack defines the number of forecast
windows. MMPD and iTransformer are sampled without replacement to that count.
Binary/MMPD use one seeded saved whole-horizon draw per selected window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "electricity", "traffic", "exchange_rate", "PeMS", "solar_Alabama", "ETTm1", "ETTm2")
SEED = 20260812
N_RESAMPLED = 256
EPS = 1e-8

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
    "ETTh1": "L2_D256_lr0.0001",
    "ETTh2": "L2_D128_lr0.0001",
    "electricity": "L3_D512_lr0.0005",
    "traffic": "L4_D512_lr0.001",
    "exchange_rate": "L2_D128_lr0.0001",
    "PeMS": "L4_D512_lr0.001",
    "solar_Alabama": "L2_D512_lr0.0005",
    "ETTm1": "L2_D128_lr0.0001",
    "ETTm2": "L2_D128_lr0.0001",
}


def as_nvh(y: np.ndarray, *, source: Path) -> np.ndarray:
    """Require (window, variate, horizon), with the 96-step horizon last."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"{source}: expected 3-D values, got {y.shape}")
    if y.shape[-1] == 96:
        return y
    if y.shape[1] == 96:
        return np.transpose(y, (0, 2, 1))
    raise ValueError(f"{source}: cannot locate 96-step horizon in {y.shape}")


def sample_indices(n_total: int, n_take: int, *, dataset: str, model: str) -> np.ndarray:
    if n_take > n_total:
        raise ValueError(f"{dataset}/{model}: need {n_take} windows, source only has {n_total}")
    if n_take == n_total:
        return np.arange(n_total)
    seed = SEED + sum(ord(c) for c in f"{dataset}:{model}")
    return np.sort(np.random.default_rng(seed).choice(n_total, n_take, replace=False))


def load_probabilistic(path: Path, n_take: int, dataset: str, model: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as pack:
        y = as_nvh(pack["y_true"], source=path)
        samples = np.asarray(pack["samples"], dtype=np.float64)
    if samples.ndim != 4 or samples.shape[0] != y.shape[0] or samples.shape[1] != y.shape[1] or samples.shape[-1] != y.shape[-1]:
        raise ValueError(f"{path}: incompatible y_true={y.shape}, samples={samples.shape}")
    indices = sample_indices(y.shape[0], n_take, dataset=dataset, model=model)
    rng = np.random.default_rng(SEED + 10_000 + sum(ord(c) for c in f"{dataset}:{model}"))
    sample_ids = rng.integers(samples.shape[2], size=n_take)
    pred = samples[indices, :, sample_ids, :]
    return y[indices], pred, {"source_windows": int(y.shape[0]), "selected_windows": indices.tolist(), "sample_ids": sample_ids.tolist()}


def resolve_itransformer_result(root: Path, dataset: str) -> Path:
    tag = ITRANS_TAG[dataset]
    matches = sorted(root.glob(f"{dataset}_336_96_{tag}_iTransformer_*"))
    matches = [p for p in matches if (p / "pred.npy").is_file() and (p / "true.npy").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"{dataset}: expected one iTransformer result for {tag}, found {matches}")
    return matches[0]


def load_itransformer(root: Path, n_take: int, dataset: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    result = resolve_itransformer_result(root, dataset)
    y = as_nvh(np.load(result / "true.npy", mmap_mode="r"), source=result / "true.npy")
    pred = as_nvh(np.load(result / "pred.npy", mmap_mode="r"), source=result / "pred.npy")
    if y.shape != pred.shape:
        raise ValueError(f"{result}: true {y.shape} != pred {pred.shape}")
    indices = sample_indices(y.shape[0], n_take, dataset=dataset, model="iTransformer")
    return y[indices], pred[indices], {"source": str(result), "source_windows": int(y.shape[0]), "selected_windows": indices.tolist()}


def normalize_resample(curves: np.ndarray) -> np.ndarray:
    """Uniformly resample horizon to 256 and z-score y independently per curve."""
    if curves.ndim != 2:
        raise ValueError(f"expected (curves, horizon), got {curves.shape}")
    x_old = np.arange(curves.shape[1], dtype=np.float64)
    x_new = np.linspace(0.0, curves.shape[1] - 1.0, N_RESAMPLED)
    out = np.empty((curves.shape[0], N_RESAMPLED), dtype=np.float64)
    for i, curve in enumerate(curves):
        out[i] = np.interp(x_new, x_old, curve)
    out -= out.mean(axis=1, keepdims=True)
    std = out.std(axis=1, keepdims=True)
    return np.divide(out, std, out=np.zeros_like(out), where=std > EPS)


def _sign_without_zeros(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values).astype(np.int8)
    for row in signs:
        nz = np.flatnonzero(row)
        if not len(nz):
            continue
        row[:nz[0]] = row[nz[0]]
        for left, right in zip(nz[:-1], nz[1:]):
            row[left + 1:right] = row[left]
        row[nz[-1] + 1:] = row[nz[-1]]
    return signs


def _spacing_stats(turns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(len(turns), dtype=np.float64)
    stds = np.zeros(len(turns), dtype=np.float64)
    for i, mask in enumerate(turns):
        locations = np.flatnonzero(mask)
        if len(locations) > 1:
            gaps = np.diff(locations) / float(N_RESAMPLED - 1)
            means[i], stds[i] = gaps.mean(), gaps.std()
    return means, stds


def haar_energy_features(z: np.ndarray, levels: int = 6) -> dict[str, np.ndarray]:
    approx = z.copy()
    energies = []
    for _ in range(levels):
        even, odd = approx[:, 0::2], approx[:, 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approx = (even + odd) / np.sqrt(2.0)
        energies.append(np.mean(detail**2, axis=1))
    energies.append(np.mean(approx**2, axis=1))
    energy = np.stack(energies, axis=1)
    energy /= np.maximum(energy.sum(axis=1, keepdims=True), EPS)
    return {f"haar_energy_level_{i + 1}": energy[:, i] for i in range(levels + 1)}


def descriptor_features(curves: np.ndarray) -> dict[str, np.ndarray]:
    z = normalize_resample(curves)
    d1 = np.diff(z, axis=1)
    d2 = np.diff(d1, axis=1)
    s1 = _sign_without_zeros(d1)
    s2 = _sign_without_zeros(d2)
    extrema = s1[:, 1:] != s1[:, :-1]
    inflections = s2[:, 1:] != s2[:, :-1]
    extrema_spacing_mean, extrema_spacing_std = _spacing_stats(extrema)
    inflection_spacing_mean, inflection_spacing_std = _spacing_stats(inflections)
    out: dict[str, np.ndarray] = {
        "deriv_mean": d1.mean(axis=1),
        "deriv_std": d1.std(axis=1),
        "deriv_abs_mean": np.abs(d1).mean(axis=1),
        "deriv_p95_abs": np.quantile(np.abs(d1), 0.95, axis=1),
        "second_deriv_std": d2.std(axis=1),
        "second_deriv_abs_mean": np.abs(d2).mean(axis=1),
        "deriv_zero_crossing_rate": extrema.mean(axis=1),
        "local_extrema_count": extrema.sum(axis=1).astype(np.float64),
        "local_extrema_spacing_mean": extrema_spacing_mean,
        "local_extrema_spacing_std": extrema_spacing_std,
        "inflection_count": inflections.sum(axis=1).astype(np.float64),
        "inflection_spacing_mean": inflection_spacing_mean,
        "inflection_spacing_std": inflection_spacing_std,
        "flat_derivative_fraction": (np.abs(d1) < 1e-3).mean(axis=1),
    }
    # Histograms are normalized distributions, with outer bins catching sharp steps.
    edges = np.array([-np.inf, -1.0, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 0.5, 1.0, np.inf])
    for order, values in (("deriv", d1), ("second_deriv", d2)):
        for b in range(len(edges) - 1):
            out[f"{order}_hist_{b:02d}"] = ((values >= edges[b]) & (values < edges[b + 1])).mean(axis=1)

    spectrum = np.abs(np.fft.rfft(z, axis=1)) ** 2
    spectrum[:, 0] = 0.0
    total = spectrum.sum(axis=1)
    power = np.divide(spectrum, total[:, None], out=np.zeros_like(spectrum), where=total[:, None] > EPS)
    freqs = np.fft.rfftfreq(N_RESAMPLED)
    out["spectral_centroid"] = (power * freqs).sum(axis=1)
    out["spectral_entropy"] = -(power * np.log(np.maximum(power, EPS))).sum(axis=1) / np.log(power.shape[1])
    for i, (lo, hi) in enumerate(((0.0, .0625), (.0625, .125), (.125, .25), (.25, .5))):
        mask = (freqs >= lo) & (freqs < hi if hi < .5 else freqs <= hi)
        out[f"spectral_energy_band_{i + 1}"] = power[:, mask].sum(axis=1)
    out.update(haar_energy_features(z))

    x = np.linspace(-1.0, 1.0, N_RESAMPLED)
    coeff = np.polyfit(x, z.T, deg=1)
    residual = z - (coeff[0][None, :] * x[:, None] + coeff[1][None, :]).T
    residual -= residual.mean(axis=1, keepdims=True)
    denom = np.sum(residual**2, axis=1)
    acf = []
    for lag in range(1, 17):
        num = np.sum(residual[:, :-lag] * residual[:, lag:], axis=1)
        acf.append(np.divide(num, denom, out=np.zeros_like(num), where=denom > EPS))
    acf = np.stack(acf, axis=1)
    out["residual_acf_lag1"] = acf[:, 0]
    out["residual_acf_abs_mean_lags_1_16"] = np.abs(acf).mean(axis=1)
    below = np.abs(acf) <= np.exp(-1.0)
    out["residual_acf_decay_lag"] = np.where(below.any(axis=1), below.argmax(axis=1) + 1, 17).astype(float)
    total_variation = np.abs(d1).sum(axis=1)
    arc_length = np.sqrt((1.0 / (N_RESAMPLED - 1)) ** 2 + d1**2).sum(axis=1)
    out["total_variation_per_step"] = total_variation / (N_RESAMPLED - 1)
    out["total_variation_per_arc_length"] = total_variation / np.maximum(arc_length, EPS)
    return out


def summarize_pair(y_true: np.ndarray, pred: np.ndarray) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    target = descriptor_features(y_true.reshape(-1, y_true.shape[-1]))
    forecast = descriptor_features(pred.reshape(-1, pred.shape[-1]))
    if target.keys() != forecast.keys():
        raise AssertionError("feature key mismatch")
    summary: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for name in sorted(target):
        difference = forecast[name] - target[name]
        item = {
            "gt_mean": float(target[name].mean()),
            "pred_mean": float(forecast[name].mean()),
            "bias": float(difference.mean()),
            "mae": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
        }
        summary[name] = item
        rows.append({"feature": name, **item})
    return summary, rows


def subwindows(values: np.ndarray, length: int) -> np.ndarray:
    """Return every sliding horizon subwindow as (window, variate, subwindow)."""
    if length == values.shape[-1]:
        return values
    if not 2 <= length <= values.shape[-1]:
        raise ValueError(f"window length must be in [2, {values.shape[-1]}], got {length}")
    windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=-1)
    return windows.transpose(0, 2, 1, 3).reshape(-1, values.shape[1], length)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty report")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True, help="MMPD + iTransformer root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--window-length", type=int, default=96, help="96 for full horizons; shorter values use every sliding subwindow")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "seed": SEED,
        "resampled_points": N_RESAMPLED,
        "window_length": args.window_length,
        "normalization": "each individual GT/pred analysis window is linearly resampled to 256 then independently z-scored; constant curves become zero",
        "sampling": "binary pack determines dataset window count; MMPD/iTransformer sample that many windows without replacement; binary/MMPD choose one seeded whole-horizon saved sample per selected window",
        "datasets": {},
    }
    for dataset in args.datasets:
        binary_path = args.binary_root / "results/datasets" / BINARY_REL[dataset]
        mmpd_path = args.reference_root / "results/datasets" / MMPD_REL[dataset]
        if not binary_path.is_file() or not mmpd_path.is_file():
            raise FileNotFoundError(f"{dataset}: missing binary={binary_path.is_file()} mmpd={mmpd_path.is_file()}")
        with np.load(binary_path, allow_pickle=False) as pack:
            binary_windows = int(np.asarray(pack["y_true"]).shape[0])
        dataset_result: dict[str, Any] = {"binary_window_count": binary_windows, "models": {}}
        loaders: list[tuple[str, Callable[[], tuple[np.ndarray, np.ndarray, dict[str, Any]]]]] = [
            ("binary_quad_t", lambda p=binary_path: load_probabilistic(p, binary_windows, dataset, "binary_quad_t")),
            ("mmpd", lambda p=mmpd_path: load_probabilistic(p, binary_windows, dataset, "mmpd")),
            ("itransformer", lambda: load_itransformer(args.reference_root / "temp/iTransformer/results", binary_windows, dataset)),
        ]
        for model, loader in loaders:
            y_true, pred, provenance = loader()
            if y_true.shape != pred.shape:
                raise ValueError(f"{dataset}/{model}: y {y_true.shape} != pred {pred.shape}")
            y_true = subwindows(y_true, args.window_length)
            pred = subwindows(pred, args.window_length)
            feature_summary, feature_rows = summarize_pair(y_true, pred)
            for row in feature_rows:
                all_rows.append({"dataset": dataset, "model": model, "n_windows": y_true.shape[0], "n_variates": y_true.shape[1], **row})
            dataset_result["models"][model] = {
                "shape": list(y_true.shape),
                "provenance": provenance,
                "features": feature_summary,
            }
        manifest["datasets"][dataset] = dataset_result
        print(f"done {dataset}: {binary_windows} windows", flush=True)
    write_csv(output / "feature_errors.csv", all_rows)
    (output / "shape_descriptor_stats.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected = ["deriv_abs_mean", "second_deriv_abs_mean", "deriv_zero_crossing_rate", "spectral_centroid", "spectral_entropy", "haar_energy_level_1", "haar_energy_level_6", "total_variation_per_arc_length", "residual_acf_lag1"]
    md = ["# Shape descriptor errors", "", manifest["normalization"], "", manifest["sampling"], "", "All entries are paired GT-vs-pred feature MAE; lower means closer shape texture.", ""]
    for dataset in args.datasets:
        md += [f"## {dataset}", "", "| model | " + " | ".join(selected) + " |", "|---|" + "---:|" * len(selected)]
        for model in ("binary_quad_t", "mmpd", "itransformer"):
            values = manifest["datasets"][dataset]["models"][model]["features"]
            md.append("| " + model + " | " + " | ".join(f"{values[key]['mae']:.6f}" for key in selected) + " |")
        md.append("")
    (output / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
