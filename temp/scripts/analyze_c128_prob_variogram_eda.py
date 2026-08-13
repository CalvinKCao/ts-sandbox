#!/usr/bin/env python3
"""EDA and forecast-path plots for the solar/PeMS probabilistic variogram gap."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compute_c128_variogram_cloud import REPO, RESULTS, SPECS


REPORT = REPO / "reports" / "c128_prob_variogram_eda.md"
FIG_DIR = REPORT.with_suffix("")
SELECTIONS = RESULTS / "variogram_cloud_prob_sample_gap16.json"
DATASETS = ("solar", "PeMS")
MC_REPLICATES = 2_000


def _load_selected_paths(dataset: str, method: str) -> dict[str, Any]:
    rows = {row["dataset"]: row for row in json.loads(SELECTIONS.read_text())["rows"]}
    key = "binary_quad_t" if method == "binary" else "mmpd_probabilistic"
    spec_key = "binary" if method == "binary" else "mmpd"
    path = Path(SPECS[dataset][spec_key])
    choices = np.asarray(rows[dataset][key]["sample_choices"], dtype=np.int64)
    with np.load(path, allow_pickle=False) as pack:
        y_true = np.asarray(pack["y_true"], dtype=np.float64)
        samples = np.asarray(pack["samples"], dtype=np.float64)
        window_indices = np.asarray(pack["window_indices" if "window_indices" in pack else "indices"])
    prediction = samples[np.arange(len(y_true)), :, choices, :]
    return {
        "path": path,
        "y_true": y_true,
        "prediction": prediction,
        "error": prediction - y_true,
        "window_indices": window_indices,
        "sample_choices": choices,
    }


def _scores(error: np.ndarray, max_gap: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_gap_var = np.stack(
        [np.mean((error[..., gap:] - error[..., :-gap]) ** 2, axis=-1) for gap in range(1, max_gap + 1)],
        axis=-1,
    )
    # [window, variate, gap], [window], [variate]
    return by_gap_var, by_gap_var.mean(axis=(1, 2)), by_gap_var.mean(axis=(0, 2))


def _per_gap(error: np.ndarray, max_gap: int = 16) -> np.ndarray:
    return np.asarray(
        [np.mean((error[..., gap:] - error[..., :-gap]) ** 2) for gap in range(1, max_gap + 1)]
    )


def _sample_path_mc(path: Path, dataset: str, method: str) -> np.ndarray:
    """Monte Carlo the one-random-path/window protocol from all saved paths."""
    with np.load(path, allow_pickle=False) as pack:
        y_true = np.asarray(pack["y_true"], dtype=np.float64)
        samples = np.asarray(pack["samples"], dtype=np.float64)
    error = samples - y_true[:, :, None, :]
    per_window_sample = np.stack(
        [np.mean((error[..., gap:] - error[..., :-gap]) ** 2, axis=(1, 3)) for gap in range(1, 17)],
        axis=-1,
    ).mean(axis=-1)
    rng = np.random.default_rng(np.random.SeedSequence([20260810, sum(map(ord, dataset)), sum(map(ord, method))]))
    choice = rng.integers(per_window_sample.shape[1], size=(MC_REPLICATES, per_window_sample.shape[0]))
    return per_window_sample[np.arange(per_window_sample.shape[0])[None, :], choice].mean(axis=1)


def _plot_summary(dataset: str, binary: dict[str, Any], mmpd: dict[str, Any]) -> dict[str, Any]:
    b_gap_var, b_window, b_var = _scores(binary["error"])
    m_gap_var, m_window, m_var = _scores(mmpd["error"])
    b_gap = _per_gap(binary["error"])
    m_gap = _per_gap(mmpd["error"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    gaps = np.arange(1, 17)
    axes[0].plot(gaps, b_gap, marker="o", label="binary quad_t")
    axes[0].plot(gaps, m_gap, marker="o", label="MMPD sample")
    axes[0].set(xlabel="gap h", ylabel="increment-error MSE", title="Variogram-cloud curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].boxplot([b_window, m_window], tick_labels=["binary", "MMPD"], showfliers=False)
    axes[1].set(ylabel="window variogram cloud", title="Window-level distribution")
    axes[1].grid(axis="y", alpha=0.25)

    x = np.arange(len(b_var))
    width = 0.42
    axes[2].bar(x - width / 2, b_var, width, label="binary")
    axes[2].bar(x + width / 2, m_var, width, label="MMPD")
    axes[2].set(xlabel="variate", ylabel="variogram cloud", title="Per-variate contribution")
    axes[2].set_xticks(x)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False)
    fig.suptitle(f"{dataset}: one probabilistic path per window", fontsize=13)
    output = FIG_DIR / f"{dataset}_probabilistic_variogram_eda.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "binary_gap": b_gap,
        "mmpd_gap": m_gap,
        "binary_window": b_window,
        "mmpd_window": m_window,
        "binary_var": b_var,
        "mmpd_var": m_var,
        "binary_gap_var": b_gap_var,
        "mmpd_gap_var": m_gap_var,
    }


def _plot_pems_paired(binary: dict[str, Any], mmpd: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    # The raw-pack window indices use different pool strides, so align the few
    # target-identical windows by their forecast target rather than row index.
    def target_key(target: np.ndarray) -> bytes:
        return np.round(target.reshape(-1)[:16], 5).tobytes()

    positions = {target_key(target): i for i, target in enumerate(mmpd["y_true"])}
    pairs = []
    for b_i, target in enumerate(binary["y_true"]):
        m_i = positions.get(target_key(target))
        if m_i is not None and np.allclose(binary["y_true"][b_i], mmpd["y_true"][m_i], atol=3e-5, rtol=0):
            pairs.append((b_i, m_i))
    if not pairs:
        raise RuntimeError("PeMS has no aligned target windows")
    b_idx = np.asarray([pair[0] for pair in pairs])
    m_idx = np.asarray([pair[1] for pair in pairs])
    b_scores = summary["binary_window"][b_idx]
    m_scores = summary["mmpd_window"][m_idx]
    pair = pairs[int(np.argmax(b_scores - m_scores))]
    b_i, m_i = pair
    var = int(np.argmax(summary["binary_gap_var"][b_i].mean(axis=-1) - summary["mmpd_gap_var"][m_i].mean(axis=-1)))
    t = np.arange(binary["y_true"].shape[-1])
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.plot(t, binary["y_true"][b_i, var], color="black", lw=1.6, label="GT")
    ax.plot(t, binary["prediction"][b_i, var], lw=1.3, label="binary quad_t")
    ax.plot(t, mmpd["prediction"][m_i, var], lw=1.3, label="MMPD sample")
    ax.set(
        xlabel="forecast step",
        ylabel="dataset-global z",
        title=f"PeMS aligned window {int(binary['window_indices'][b_i])}, variate {var}",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    output = FIG_DIR / "pems_aligned_binary_worse_path.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "paired_windows": len(pairs),
        "window_index": int(binary["window_indices"][b_i]),
        "variate": var,
        "binary_window_score": float(summary["binary_window"][b_i]),
        "mmpd_window_score": float(summary["mmpd_window"][m_i]),
    }


def _plot_solar_examples(binary: dict[str, Any], mmpd: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    b_i = int(np.argmax(summary["binary_window"]))
    m_i = int(np.argmin(np.abs(summary["mmpd_window"] - np.quantile(summary["mmpd_window"], 0.9))))
    b_var = int(np.argmax(summary["binary_gap_var"][b_i].mean(axis=-1)))
    m_var = int(np.argmax(summary["mmpd_gap_var"][m_i].mean(axis=-1)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, constrained_layout=True)
    for ax, name, pack, index, var in (
        (axes[0], "binary quad_t", binary, b_i, b_var),
        (axes[1], "MMPD sample", mmpd, m_i, m_var),
    ):
        t = np.arange(pack["y_true"].shape[-1])
        ax.plot(t, pack["y_true"][index, var], color="black", lw=1.5, label="GT")
        ax.plot(t, pack["prediction"][index, var], lw=1.25, label=name)
        ax.set(
            xlabel="forecast step",
            ylabel="dataset-global z",
            title=f"{name}: source window {int(pack['window_indices'][index])}, v{var}",
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Solar high-variogram examples from each run's own evaluation pool", fontsize=12)
    output = FIG_DIR / "solar_high_variogram_paths.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "binary_window_index": int(binary["window_indices"][b_i]),
        "binary_variate": b_var,
        "binary_window_score": float(summary["binary_window"][b_i]),
        "mmpd_window_index": int(mmpd["window_indices"][m_i]),
        "mmpd_variate": m_var,
        "mmpd_window_score": float(summary["mmpd_window"][m_i]),
    }


def _format_gap(delta: np.ndarray) -> str:
    return ", ".join(f"h={idx + 1}: {value:+.4f}" for idx, value in enumerate(delta))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    analysis: dict[str, Any] = {}
    packs: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for dataset in DATASETS:
        binary = _load_selected_paths(dataset, "binary")
        mmpd = _load_selected_paths(dataset, "mmpd")
        summary = _plot_summary(dataset, binary, mmpd)
        binary_mc = _sample_path_mc(binary["path"], dataset, "binary")
        mmpd_mc = _sample_path_mc(mmpd["path"], dataset, "mmpd")
        packs[dataset] = binary, mmpd, summary
        analysis[dataset] = {
            "binary": {
                "mean": float(summary["binary_window"].mean()),
                "median": float(np.median(summary["binary_window"])),
                "p90": float(np.quantile(summary["binary_window"], 0.9)),
                "sample_mse": float(np.mean(binary["error"] ** 2)),
                "sample_mae": float(np.mean(np.abs(binary["error"]))),
                "random_path_mc_mean": float(binary_mc.mean()),
                "random_path_mc_ci95": np.quantile(binary_mc, [0.025, 0.975]).tolist(),
                "per_gap": summary["binary_gap"].tolist(),
                "per_variate": summary["binary_var"].tolist(),
            },
            "mmpd": {
                "mean": float(summary["mmpd_window"].mean()),
                "median": float(np.median(summary["mmpd_window"])),
                "p90": float(np.quantile(summary["mmpd_window"], 0.9)),
                "sample_mse": float(np.mean(mmpd["error"] ** 2)),
                "sample_mae": float(np.mean(np.abs(mmpd["error"]))),
                "random_path_mc_mean": float(mmpd_mc.mean()),
                "random_path_mc_ci95": np.quantile(mmpd_mc, [0.025, 0.975]).tolist(),
                "per_gap": summary["mmpd_gap"].tolist(),
                "per_variate": summary["mmpd_var"].tolist(),
            },
            "delta_mmpd_minus_binary_mc_mean": float((mmpd_mc - binary_mc).mean()),
            "delta_mmpd_minus_binary_mc_ci95": np.quantile(mmpd_mc - binary_mc, [0.025, 0.975]).tolist(),
        }
    analysis["PeMS"]["example"] = _plot_pems_paired(*packs["PeMS"])
    analysis["solar"]["example"] = _plot_solar_examples(*packs["solar"])
    (FIG_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Probabilistic variogram-cloud EDA: solar and PeMS",
        "",
        "Scope: the same one-random-path-per-window selection used in `variogram_cloud_prob_sample_gap16.json`; lower values are better. Binary and MMPD retain their saved run pools.",
        "",
    ]
    for dataset in DATASETS:
        block = analysis[dataset]
        b, m = block["binary"], block["mmpd"]
        delta = np.asarray(m["per_gap"]) - np.asarray(b["per_gap"])
        winner = "MMPD" if m["mean"] < b["mean"] else "binary"
        lines += [
            f"## {dataset}",
            "",
            f"{winner} has the lower sampled variogram cloud: binary {b['mean']:.6f}, MMPD {m['mean']:.6f} (delta {m['mean'] - b['mean']:+.6f}).",
            "",
            f"- Binary window score: median {b['median']:.6f}; p90 {b['p90']:.6f}.",
            f"- MMPD window score: median {m['median']:.6f}; p90 {m['p90']:.6f}.",
            f"- Pointwise sampled MSE/MAE: binary {b['sample_mse']:.6f}/{b['sample_mae']:.6f}; MMPD {m['sample_mse']:.6f}/{m['sample_mae']:.6f}.",
            f"- Across {MC_REPLICATES} redraws of one saved path per window: MMPD minus binary {block['delta_mmpd_minus_binary_mc_mean']:+.6f}, 95% interval [{block['delta_mmpd_minus_binary_mc_ci95'][0]:+.6f}, {block['delta_mmpd_minus_binary_mc_ci95'][1]:+.6f}].",
            f"- Per-gap MMPD minus binary: {_format_gap(delta)}.",
            f"- [Per-gap, distribution, and variate plot]({REPORT.stem}/{dataset}_probabilistic_variogram_eda.png)",
            "",
        ]
    example = analysis["PeMS"]["example"]
    lines += [
        "## Forecast-path checks",
        "",
        f"- PeMS has {example['paired_windows']} target-identical paired windows. The displayed pair is window {example['window_index']}, variate {example['variate']}: binary {example['binary_window_score']:.6f} vs MMPD {example['mmpd_window_score']:.6f}.",
        f"- [PeMS aligned example]({REPORT.stem}/pems_aligned_binary_worse_path.png)",
        "",
        "- Solar's saved binary and MMPD target arrays do not align exactly under the shared window indices, so the displayed solar paths are representative high-variogram examples from each method's own evaluation pool rather than a pointwise paired overlay.",
        f"- [Solar examples]({REPORT.stem}/solar_high_variogram_paths.png)",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for dataset in DATASETS:
        b, m = analysis[dataset]["binary"], analysis[dataset]["mmpd"]
        print(f"{dataset}: binary={b['mean']:.6f}, mmpd={m['mean']:.6f}, delta={m['mean'] - b['mean']:+.6f}")


if __name__ == "__main__":
    main()
