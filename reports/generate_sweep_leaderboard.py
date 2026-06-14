"""Build sweep grid report + per-dataset leaderboards from YAML-first sweep runs."""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(REPO, "results", "datasets")
MMPD_DIR_LEGACY = os.path.join(RESULTS, "06-12-sweep-subset-mmpd", "partials")
MMPD_DIR_SUBSET = os.path.join(RESULTS, "06-13-binary-mmpd-subset-compare", "partials")
MMPD_SOURCE_LEGACY = "06-12-sweep-subset-mmpd"
MMPD_SOURCE_SUBSET = "06-13-binary-mmpd-subset-compare"
RUN_GLOBS = [
    os.path.join(RESULTS, "06-12-*"),
    os.path.join(RESULTS, "06-13-395119*-binary_anchor_stationary_flat_subsets"),
    os.path.join(RESULTS, "06-14-39515*-binary_anchor_stationary_flat_subsets_ema099"),
]

BASELINE = "sweep_baseline"
REPORT_DIR = os.path.join(REPO, "reports", "sweep_grid_report")
GRID_PATH = os.path.join(REPO, "reports", "sweep_grid_report.md")
LEADERBOARD_PATH = os.path.join(REPO, "reports", "sweep_grid_report_leaderboard.md")
SUBSET_COMPARE_PATH = os.path.join(REPO, "reports", "binary_mmpd_subset_compare.md")

SUBSET_DATASETS = [
    "ETTh1",
    "ETTh2",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "solar_Alabama",
]

MMPD_SUBSET_JOBS = {
    "ETTh1": "3951201",
    "ETTh2": "3951202",
    "exchange_rate": "3951203",
    "weather": "3951204",
    "electricity": "3951205",
    "traffic": "3951206",
    "solar_Alabama": "3951207",
}

DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "solar_Alabama",
]

CONFIG_ALIASES = {
    "ordinal_d3pm_staged": "**Discrete**",
    "ordinal_d3pm_mae_staged_subsets": "**MAE Discrete**",
    "binary_anchor_stationary_flat": "**Binary flat**",
    "binary_anchor_stationary_flat_subsets": "**Flat subsets**",
    "binary_anchor_stationary_flat_subsets_ema099": "**Flat subsets EMA0.99**",
    "hp_max_scale_tuning": "**MS tune**",
}

# Optuna-tuned max_scale from hp_max_scale_tuning (jobs 3943934–3943937, Jun 12 sweep).
HP_MS_TUNE_JOBS = {
    "ETTh1": "3943934",
    "ETTm1": "3943936",
    "exchange_rate": "3943935",
    "weather": "3943937",
}
HP_MS_TUNE_MAX_SCALE = {
    "ETTh1": 5.2,
    "ETTm1": 7.7,
    "exchange_rate": 10.6,
    "weather": 9.3,
}


def short_config(name: str) -> str:
    if name.startswith("binary_dual_scale_staged_arch_"):
        return name.replace("binary_dual_scale_staged_arch_", "")
    return name


def display_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, short_config(name))


def parse_run_dir(path: str) -> Optional[Tuple[str, str, str]]:
    base = os.path.basename(path.rstrip("/"))
    m = re.match(r"\d{2}-\d{2}-(\d+)-([^-]+)-(.+)$", base)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def load_partial(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def drank_str(drank: Optional[int]) -> str:
    if drank is None:
        return "—"
    if drank > 0:
        return f"+{drank}"
    return str(drank)


def load_mmpd(dataset: str) -> Optional[Dict[str, Any]]:
    for mmpd_dir in (MMPD_DIR_SUBSET, MMPD_DIR_LEGACY):
        path = os.path.join(mmpd_dir, f"{dataset}_mmpd.json")
        if os.path.isfile(path):
            data = load_partial(path)
            return {
                "anchor_mse": data.get("mse"),
                "anchor_mae": data.get("mae"),
                "crps": data.get("crps"),
                "source": (
                    MMPD_SOURCE_SUBSET
                    if mmpd_dir == MMPD_DIR_SUBSET
                    else MMPD_SOURCE_LEGACY
                ),
            }
    return None


def mmpd_config_label(dataset: str, mmpd: Dict[str, Any]) -> str:
    if mmpd.get("source") == MMPD_SOURCE_SUBSET:
        return "**MMPD (subset)**"
    return "**MMPD**"


def collect_runs() -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    best_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_dirs: set[str] = set()

    for run_glob in RUN_GLOBS:
        for run_dir in sorted(glob.glob(run_glob)):
            if run_dir in seen_dirs:
                continue
            seen_dirs.add(run_dir)
            parsed = parse_run_dir(run_dir)
            if not parsed:
                continue
            job_id, dataset, raw_config = parsed
            config = display_config(raw_config)

            partials = glob.glob(os.path.join(run_dir, "partials", "*_staged_anchor.json"))
            if not partials:
                continue
            metrics = load_partial(partials[0])

            row = {
                "dataset": dataset,
                "config": config,
                "job_id": job_id,
                "anchor_mse": metrics.get("anchor_mse"),
                "anchor_mae": metrics.get("anchor_mae"),
                "crps": metrics.get("crps"),
                "sample_mean_mse": metrics.get("sample_mean_mse"),
                "status": "**OK**",
            }
            key = (dataset, config)
            prev = best_rows.get(key)
            if prev is not None and int(prev["job_id"]) >= int(job_id):
                continue
            best_rows[key] = row

    grid_rows = list(best_rows.values())
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        by_dataset[row["dataset"]].append(
            {
                "Config": row["config"],
                "anchor_mse": row["anchor_mse"],
                "anchor_mae": row["anchor_mae"],
                "crps": row["crps"],
                "Status": "OK",
            }
        )

    for dataset, rows in by_dataset.items():
        mmpd = load_mmpd(dataset)
        if mmpd and mmpd["anchor_mse"] is not None:
            rows.append(
                {
                    "Config": mmpd_config_label(dataset, mmpd),
                    "anchor_mse": mmpd["anchor_mse"],
                    "anchor_mae": mmpd["anchor_mae"],
                    "crps": mmpd["crps"],
                    "Status": "ref",
                }
            )

    return grid_rows, by_dataset


def append_mmpd_grid_rows(grid_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(grid_rows)
    for dataset in SUBSET_DATASETS:
        mmpd = load_mmpd(dataset)
        if not mmpd or mmpd.get("source") != MMPD_SOURCE_SUBSET:
            continue
        rows.append(
            {
                "dataset": dataset,
                "config": "**MMPD (subset)**",
                "job_id": MMPD_SUBSET_JOBS.get(dataset, "—"),
                "anchor_mse": mmpd["anchor_mse"],
                "anchor_mae": mmpd["anchor_mae"],
                "crps": mmpd["crps"],
                "sample_mean_mse": None,
                "status": "ref",
            }
        )
    return rows


def rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["anchor_mse"] is None, r["anchor_mse"]))


def baseline_rank(rows: List[Dict[str, Any]]) -> Optional[int]:
    ranked = rank_rows(rows)
    for i, r in enumerate(ranked):
        if r["Config"] == BASELINE:
            return i + 1
    return None


def delta_ranks(by_dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Optional[int]]]:
    out: Dict[str, Dict[str, Optional[int]]] = defaultdict(dict)
    for dataset, rows in by_dataset.items():
        ranked = rank_rows(rows)
        base = baseline_rank(rows)
        if base is None:
            continue
        for i, r in enumerate(ranked):
            cfg = r["Config"]
            if cfg in {"**MMPD**", "**MMPD (subset)**"}:
                out[cfg][dataset] = None
            else:
                out[cfg][dataset] = (i + 1) - base
    return out


def subset_row(grid_rows: List[Dict[str, Any]], config: str, dataset: str) -> Optional[Dict[str, Any]]:
    for row in grid_rows:
        if row["dataset"] == dataset and row["config"] == config:
            return row
    return None


def write_subset_compare(path: str, grid_rows: List[Dict[str, Any]]) -> None:
    flat_cfg = "**Flat subsets**"
    ema_cfg = "**Flat subsets EMA0.99**"
    ms_cfg = "**MS tune**"
    subset_ids = {
        "ETTh1": "ETTh1",
        "ETTh2": "ETTh2",
        "exchange_rate": "exchange_rate",
        "weather": "weather_4v_s2",
        "electricity": "electricity_4v_s1",
        "traffic": "traffic_4v_s1",
        "solar_Alabama": "solar_Alabama_2v_s1",
        "ETTm1": "ETTm1_4v_s3",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Binary flat vs MMPD — ETTh1-capped subset comparison\n\n")
        f.write(
            "Apples-to-apples runs on matched variate subsets (`binary_anchor_stationary_flat_subsets`). "
            "Binary eval: `dpmpp`, 20 steps, 20 samples. MMPD: same subset indices from binary ckpt metadata, "
            "20 samples, full test (`06-13-binary-mmpd-subset-compare`). "
            "**MS tune** = `hp_max_scale_tuning` (`configs/tuning_sweep/hp_max_scale_tuning.yaml`, jobs "
            "`3943934`–`3943937`): Optuna searches `max_scale ∈ [2.5, 14.0]` during diffusion finetune HP "
            "(same ETTh1-capped subsets as Jun 12 sweep). Merge job `3951208` failed but all MMPD partials exist.\n\n"
        )

        header = [
            "Dataset",
            "subset_id",
            "Flat subsets anchor_mse",
            "Flat subsets crps",
            "Flat subsets job",
            "Flat subsets EMA0.99 anchor_mse",
            "Flat subsets EMA0.99 crps",
            "EMA job",
            "MS tune anchor_mse",
            "MS tune crps",
            "tuned max_scale",
            "MS job",
            "MMPD (subset) anchor_mse",
            "MMPD (subset) crps",
            "MMPD job",
            "Best CRPS",
        ]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")

        crps_wins = {flat_cfg: 0, ema_cfg: 0, ms_cfg: 0, "**MMPD (subset)**": 0}
        for dataset in SUBSET_DATASETS:
            flat = subset_row(grid_rows, flat_cfg, dataset)
            ema = subset_row(grid_rows, ema_cfg, dataset)
            ms = subset_row(grid_rows, ms_cfg, dataset)
            mmpd = load_mmpd(dataset)
            crps_vals = []
            if flat and flat["crps"] is not None:
                crps_vals.append((flat_cfg, flat["crps"]))
            if ema and ema["crps"] is not None:
                crps_vals.append((ema_cfg, ema["crps"]))
            if ms and ms["crps"] is not None:
                crps_vals.append((ms_cfg, ms["crps"]))
            if mmpd and mmpd["crps"] is not None:
                crps_vals.append(("**MMPD (subset)**", mmpd["crps"]))
            best = min(crps_vals, key=lambda x: x[1])[0] if crps_vals else "—"
            if best in crps_wins:
                crps_wins[best] += 1

            ms_scale = HP_MS_TUNE_MAX_SCALE.get(dataset)
            f.write(
                "| "
                + " | ".join(
                    [
                        dataset,
                        subset_ids.get(dataset, dataset),
                        fmt(flat["anchor_mse"] if flat else None),
                        fmt(flat["crps"] if flat else None),
                        str(flat["job_id"]) if flat else "—",
                        fmt(ema["anchor_mse"] if ema else None),
                        fmt(ema["crps"] if ema else None),
                        str(ema["job_id"]) if ema else "—",
                        fmt(ms["anchor_mse"] if ms else None),
                        fmt(ms["crps"] if ms else None),
                        fmt(ms_scale) if ms_scale is not None else "—",
                        HP_MS_TUNE_JOBS.get(dataset, "—"),
                        fmt(mmpd["anchor_mse"] if mmpd else None),
                        fmt(mmpd["crps"] if mmpd else None),
                        MMPD_SUBSET_JOBS.get(dataset, "—"),
                        best,
                    ]
                )
                + " |\n"
            )

        f.write("\n## CRPS win count (7 datasets)\n\n")
        f.write(f"- **Flat subsets**: {crps_wins[flat_cfg]}\n")
        f.write(f"- **Flat subsets EMA0.99**: {crps_wins[ema_cfg]}\n")
        if crps_wins[ms_cfg]:
            f.write(f"- **MS tune**: {crps_wins[ms_cfg]}\n")
        f.write(f"- **MMPD (subset)**: {crps_wins['**MMPD (subset)**']}\n")

        f.write("\n## MS tune only (`hp_max_scale_tuning`, Jun 12 sweep)\n\n")
        f.write(
            "Config extends `base/binary_staged.yaml` with `max_scale_tuning: true`. "
            "Only these four datasets were submitted in the sweep arm.\n\n"
        )
        ms_header = [
            "Dataset",
            "subset_id",
            "tuned max_scale",
            "anchor_mse",
            "anchor_mae",
            "crps",
            "sample_mean_mse",
            "Job",
        ]
        f.write("| " + " | ".join(ms_header) + " |\n")
        f.write("|" + "|".join(["---"] * len(ms_header)) + "|\n")
        for dataset in ["ETTh1", "ETTm1", "exchange_rate", "weather"]:
            ms = subset_row(grid_rows, ms_cfg, dataset)
            if not ms:
                continue
            f.write(
                "| "
                + " | ".join(
                    [
                        dataset,
                        subset_ids.get(dataset, dataset),
                        fmt(HP_MS_TUNE_MAX_SCALE.get(dataset)),
                        fmt(ms["anchor_mse"]),
                        fmt(ms["anchor_mae"]),
                        fmt(ms["crps"]),
                        fmt(ms.get("sample_mean_mse")),
                        str(ms["job_id"]),
                    ]
                )
                + " |\n"
            )


def write_grid(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "Dataset",
        "Config",
        "Status",
        "anchor_mse",
        "anchor_mae",
        "crps",
        "sample_mean_mse",
        "Job",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# YAML-First Sweep Grid Report\n\n")
        f.write(
            "Fixed-HP binary sweep (`configs/sweep/`, Jun 12 2026) plus ordinal D3PM staged runs: "
            "**Discrete** (CE, `ordinal_d3pm_staged`), **MAE Discrete** (expectation MAE + uniform `1/H` anchor, "
            "`ordinal_d3pm_mae_staged_subsets`), **Binary flat** (full variates, `binary_anchor_stationary_flat`), "
            "**Flat subsets** / **Flat subsets EMA0.99** (ETTh1-capped subsets, jobs `3951193`–`3951199` / "
            "`3951527`–`3951533`), **MS tune** (`hp_max_scale_tuning`, jobs `3943934`–`3943937`), and "
            "**MMPD (subset)** (`06-13-binary-mmpd-subset-compare`, jobs `3951201`–`3951207`). "
            "Probabilistic metrics: `dpmpp` sampler, 20 steps, 20 samples.\n\n"
        )
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in sorted(rows, key=lambda r: (r["dataset"], r["config"])):
            cfg = row["config"]
            cfg_disp = cfg if cfg.startswith("**") else f"`{cfg}`"
            f.write(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        cfg_disp,
                        row["status"],
                        fmt(row["anchor_mse"]),
                        fmt(row["anchor_mae"]),
                        fmt(row["crps"]),
                        fmt(row["sample_mean_mse"]),
                        row["job_id"],
                    ]
                )
                + " |\n"
            )


def write_leaderboard(path: str, by_dataset: Dict[str, List[Dict[str, Any]]], deltas: Dict[str, Dict[str, Optional[int]]]) -> None:
    datasets_present = [d for d in DATASET_ORDER if d in by_dataset]
    configs = sorted(
        {r["Config"] for rows in by_dataset.values() for r in rows if r["Config"] not in {"**MMPD**", "**MMPD (subset)**", "**Discrete**"}},
        key=lambda c: (
            sum(deltas.get(c, {}).get(ds, 0) or 0 for ds in datasets_present) / max(
                sum(1 for ds in datasets_present if ds in deltas.get(c, {}) and deltas[c][ds] is not None),
                1,
            )
        ),
    )
    if any(
        r["Config"] == "**Discrete**"
        for rows in by_dataset.values()
        for r in rows
    ):
        configs.append("**Discrete**")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# YAML-First Sweep Leaderboard\n\n")
        f.write(
            "Probabilistic metrics from `dpmpp` sampler with `20` steps. "
            f"Baseline is `{BASELINE}` (fixed 3e-5 LR, linear noise, epsilon target). "
            "**Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). "
            "**MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). "
            "**Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). "
            "**Flat subsets** / **Flat subsets EMA0.99** use ETTh1-capped variate subsets "
            "(`binary_anchor_stationary_flat_subsets` / `_ema099`). "
            "**MS tune** is Optuna `max_scale` search (`hp_max_scale_tuning`, jobs `3943934`–`3943937`). "
            f"**MMPD (subset)** from `{MMPD_SOURCE_SUBSET}` (same subsets as flat runs, 20 samples, full test). "
            f"Legacy **MMPD** from `{MMPD_SOURCE_LEGACY}` where subset MMPD is unavailable.\n\n"
        )

        # Cross-dataset avg Δrank
        f.write("## Average Δrank vs baseline\n\n")
        f.write(
            f"Δrank = config rank − `{BASELINE}` rank per dataset (negative = better anchor MSE). "
            "Avg Δrank averages over datasets where the config ran.\n\n"
        )
        avg_header = ["Rank", "Config", "avg Δrank"] + [f"{ds} Δrank" for ds in datasets_present] + ["Status"]
        f.write("| " + " | ".join(avg_header) + " |\n")
        f.write("|" + "|".join(["---"] * len(avg_header)) + "|\n")

        avg_rows = []
        for cfg in configs:
            per_ds = [deltas.get(cfg, {}).get(ds) for ds in datasets_present]
            vals = [v for v in per_ds if v is not None]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            avg_rows.append((avg, cfg, per_ds, vals))

        avg_rows.sort(key=lambda x: x[0])

        for i, (avg, cfg, per_ds, _) in enumerate(avg_rows, start=1):
            ds_cells = [drank_str(v) for v in per_ds]
            f.write(
                "| "
                + " | ".join(
                    [
                        str(i),
                        f"`{cfg}`" if not cfg.startswith("**") else cfg,
                        f"{avg:+.2f}",
                        *ds_cells,
                        "**OK**",
                    ]
                )
                + " |\n"
            )

        if any(
            load_mmpd(ds) and load_mmpd(ds).get("source") == MMPD_SOURCE_SUBSET
            for ds in SUBSET_DATASETS
            if ds in datasets_present
        ):
            subset_cells = []
            for ds in datasets_present:
                mmpd = load_mmpd(ds)
                if mmpd and mmpd.get("source") == MMPD_SOURCE_SUBSET:
                    subset_cells.append(drank_str(None))
                else:
                    subset_cells.append("—")
            f.write(
                "| — | **MMPD (subset)** | — | "
                + " | ".join(subset_cells)
                + " | ref |\n"
            )
        elif any(load_mmpd(ds) for ds in datasets_present):
            f.write(
                "| — | **MMPD** | — | "
                + " | ".join(["—"] * len(datasets_present))
                + " | ref |\n"
            )

        f.write("\n")

        for dataset in datasets_present:
            rows = rank_rows(by_dataset[dataset])
            base = baseline_rank(rows)
            n = len(rows)
            f.write(f"### {dataset}\n\n")
            if base is not None:
                f.write(
                    f"Baseline `{BASELINE}` rank: **{base}** / {n} "
                    "(lower anchor MSE is better). "
                    "Δrank = config rank − baseline rank (negative = improvement).\n\n"
                )
            else:
                f.write(f"Baseline `{BASELINE}` missing. Total configs: {n}\n\n")

            header = ["Rank", "Config", "anchor_mse", "anchor_mae", "crps", "Δrank", "Status"]
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---"] * len(header)) + "|\n")

            for i, r in enumerate(rows):
                rank = i + 1
                cfg = r["Config"]
                if cfg in {"**MMPD**", "**MMPD (subset)**"} or base is None:
                    dstr = "—"
                else:
                    dstr = drank_str(rank - base)
                disp = cfg if cfg.startswith("**") else f"`{cfg}`"
                status = r["Status"]
                if status == "OK":
                    status = "**OK**"
                elif status == "ref":
                    status = "ref"
                f.write(
                    "| "
                    + " | ".join(
                        [
                            str(rank),
                            disp,
                            fmt(r["anchor_mse"]),
                            fmt(r["anchor_mae"]),
                            fmt(r["crps"]),
                            dstr,
                            status,
                        ]
                    )
                    + " |\n"
                )
            f.write("\n")


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    grid_rows, by_dataset = collect_runs()
    grid_rows = append_mmpd_grid_rows(grid_rows)
    deltas = delta_ranks(by_dataset)
    write_grid(GRID_PATH, grid_rows)
    write_leaderboard(LEADERBOARD_PATH, by_dataset, deltas)
    write_subset_compare(SUBSET_COMPARE_PATH, grid_rows)
    print(f"Wrote {GRID_PATH}")
    print(f"Wrote {LEADERBOARD_PATH}")
    print(f"Wrote {SUBSET_COMPARE_PATH}")
    print(f"Datasets: {', '.join(sorted(by_dataset))}")
    print(f"Grid rows: {len(grid_rows)}")


if __name__ == "__main__":
    main()
