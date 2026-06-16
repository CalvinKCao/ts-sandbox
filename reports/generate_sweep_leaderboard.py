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
MMPD_DIR_MASKAE = os.path.join(
    RESULTS, "06-15-mmpd-maskae-grad-accum-200-lr-lo-tune", "partials"
)
MMPD_DIR_MASKAE_150 = os.path.join(
    RESULTS, "06-16-mmpd-maskae-grad-accum-150-lr-lo-subset", "partials"
)
MMPD_DIR_MASKAE_FAIR = os.path.join(
    RESULTS, "06-17-mmpd-maskae-fair-13d", "partials"
)
MMPD_SOURCE_LEGACY = "06-12-sweep-subset-mmpd"
MMPD_SOURCE_SUBSET = "06-13-binary-mmpd-subset-compare"
MMPD_SOURCE_MASKAE = "06-15-mmpd-maskae-grad-accum-200-lr-lo-tune"
MMPD_SOURCE_MASKAE_150 = "06-16-mmpd-maskae-grad-accum-150-lr-lo-subset"
MMPD_SOURCE_MASKAE_FAIR = "06-17-mmpd-maskae-fair-13d"
MMPD_MASKAE_LABEL = "**MMPD (MaskedAE)**"
MMPD_REF_CONFIGS = frozenset({"**MMPD**", "**MMPD (subset)**", MMPD_MASKAE_LABEL})
LOGS = os.path.join(REPO, "results", "logs")
RUN_GLOBS = [
    os.path.join(RESULTS, "06-12-*"),
    os.path.join(RESULTS, "06-13-*"),
    os.path.join(RESULTS, "06-14-*"),
    os.path.join(RESULTS, "06-15-*"),
    os.path.join(RESULTS, "06-16-*"),
]
LOG_GLOBS = [
    os.path.join(LOGS, "06-12-*.log"),
    os.path.join(LOGS, "06-13-*.log"),
    os.path.join(LOGS, "06-14-*.log"),
    os.path.join(LOGS, "06-15-*.log"),
    os.path.join(LOGS, "06-16-*.log"),
]
EVAL_DONE_RE = re.compile(
    r"staged eval done: .*?prob_mse=([\d.]+).*?anchor_mse=([\d.]+) "
    r"anchor_mae=([\d.]+) crps=([\d.]+)"
)
MAX_SCALE_RE = re.compile(r"hp/(?:coarse|fine)_diff_ft_max_scale ([\d.]+)")

BASELINE = "sweep_baseline"
REPORT_DIR = os.path.join(REPO, "reports", "sweep_grid_report")
GRID_PATH = os.path.join(REPO, "reports", "sweep_grid_report.md")
LEADERBOARD_PATH = os.path.join(REPO, "reports", "sweep_grid_report_leaderboard.md")
SUBSET_COMPARE_PATH = os.path.join(REPO, "reports", "binary_mmpd_subset_compare.md")
EMA099_VS_MMPD_PATH = os.path.join(REPO, "reports", "flat_subsets_ema099_vs_mmpd.md")

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

MMPD_MASKAE_JOBS = {
    "ETTh1": "3965321",
    "ETTh2": "3965322",
    "exchange_rate": "3965323",
    "weather": "3965324",
    "electricity": "3965325",
    "traffic": "3965326",
    "solar_Alabama": "3965327",
}

MMPD_MASKAE_150_DATASETS = [
    "ETTm1",
    "ETTm2",
    "illness",
    "PeMS",
    "dalia",
    "dynamic",
]

MMPD_MASKAE_JOBS_150 = {
    "ETTm1": "3968154",
    "ETTm2": "3968155",
    "illness": "3968156",
    "PeMS": "3968537",
    "dalia": "3968158",
    "dynamic": "3968538",
}

EMA099_EXTRA_JOBS = {
    "ETTm1": "3967251",
    "ETTm2": "3967252",
    "illness": "3967253",
    "PeMS": "3967254",
    "dalia": "3967255",
    "dynamic": "3967256",
}

EMA099_VS_MMPD_SUBSET_IDS = {
    "ETTm1": "ETTm1_4v_s3",
    "ETTm2": "ETTm2_7v_s4",
    "illness": "illness",
    "PeMS": "PeMS_7v_s1",
    "dalia": "dalia_5v_s2",
    "dynamic": "dynamic_7v_s29",
}

DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "illness",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "PeMS",
    "solar_Alabama",
    "dalia",
    "dynamic",
]

CONFIG_ALIASES = {
    "ordinal_d3pm_staged": "**Discrete**",
    "ordinal_d3pm_mae_staged_subsets": "**MAE Discrete**",
    "binary_anchor_stationary_flat": "**Binary flat**",
    "binary_anchor_stationary_flat_subsets": "**Flat subsets**",
    "binary_anchor_stationary_flat_subsets_ema099": "**Flat subsets EMA0.99**",
    "binary_anchor_stationary_flat_subsets_ema_sweep_090": "**Flat subsets EMA0.90**",
    "binary_anchor_stationary_flat_subsets_ema_sweep_095": "**Flat subsets EMA0.95**",
    "binary_anchor_stationary_flat_subsets_ema_sweep_098": "**Flat subsets EMA0.98**",
    "binary_anchor_stationary_flat_subsets_ema_sweep_0995": "**Flat subsets EMA0.995**",
    "binary_anchor_stationary_flat_subsets_ema_sweep_0999": "**Flat subsets EMA0.999**",
    "binary_anchor_stationary_flat_subsets_grad_accum_125": "**Flat subsets accum1.25x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_150": "**Flat subsets accum1.5x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_200": "**Flat subsets accum2.0x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_400": "**Flat subsets accum4x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo": "**Flat subsets accum1.5x LR-lo**",
    "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi": "**Flat subsets accum1.5x LR-hi**",
    "binary_anchor_stationary_flat_subsets_grad_accum_200_lr_lo": "**Flat subsets accum2.0x LR-lo**",
    "binary_anchor_stationary_flat_subsets_grad_accum_200_lr_hi": "**Flat subsets accum2.0x LR-hi**",
    "binary_anchor_stationary_flat_subsets_grad_accum_guidance_150": "**Flat subsets guidance accum1.5x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_guidance_200": "**Flat subsets guidance accum2x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_guidance_400": "**Flat subsets guidance accum4x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_guidance_800": "**Flat subsets guidance accum8x**",
    "binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi_guidance": "**2d-guidance**",
    "binary_anchor_stationary_flat_subsets_ema099_lb336_hz96": "**Flat subsets EMA0.99 LB336/H96**",
    "binary_anchor_stationary_flat_subsets_ema099_lb96_hz720": "**Flat subsets EMA0.99 LB96/H720**",
    "binary_anchor_ar_grad_accum_400": "**AR accum4x**",
    "binary_anchor_ar_grad_accum_800": "**AR accum8x**",
    "binary_anchor_ar_grad_accum_1600": "**AR accum16x**",
    "binary_anchor_ar_grad_accum_150": "**AR accum1.5x**",
    "binary_anchor_ar_grad_accum_200": "**AR accum2x**",
    "binary_anchor_ar_lb336_hz96_grad_accum_150": "**AR LB336/H96 accum1.5x**",
    "binary_anchor_ar_lb336_hz96_grad_accum_200": "**AR LB336/H96 accum2x**",
    "binary_anchor_ar_lb96_hz720_grad_accum_150": "**AR LB96/H720 accum1.5x**",
    "binary_anchor_ar_lb96_hz720_grad_accum_200": "**AR LB96/H720 accum2x**",
    "hp_max_scale_tuning": "**MS tune**",
}

# Pre-fix Jun 12 runs: training-section YAML never reached Optuna / finetune scheduler.
PREFIX_INVALID_JOBS: Dict[str, frozenset[str]] = {
    "hp_max_scale_tuning": frozenset({
        "3943934", "3943935", "3943936", "3943937",
        "3947879", "3947880", "3947881",
    }),
    "hp_lr_cosine_warmup2": frozenset({"3943882", "3943924", "3943883", "3943925"}),
    "hp_lr_cosine_warmup5": frozenset({"3943884", "3943926", "3943885", "3943927"}),
}

HP_MS_POSTFIX_MIN = 3956629
MS_TUNE_DATASETS = ("ETTh1", "ETTm1", "exchange_rate", "weather")

PREFIX_INVALID_NOTE = (
    "**Pre-fix invalid / incomplete:** Jun 12 `hp_max_scale_tuning` jobs `3943934`–`3943937` "
    "(and resumes `3947879`–`3947881`) never searched `max_scale`. Post-fix cosine+warmup "
    "`3956633`–`3956640` replaces pre-fix `3943882`–`3943927`. Post-fix MS tune: "
    "`3956629`–`3956632` (3h wall); Jun 15 resume `3960877`–`3960879`. "
    "**OK:** exchange_rate `3956631`, ETTm1 `3960878`. "
    "**Incomplete** (tuned `max_scale≈13.43`, eval pending): ETTh1 `3960877`, weather `3960879`."
)

# Legacy pre-fix job ids (for MS tune table footnotes only).
HP_MS_TUNE_JOBS_PREFIX = {
    "ETTh1": "3943934",
    "ETTm1": "3943936",
    "exchange_rate": "3943935",
    "weather": "3943937",
}


def short_config(name: str) -> str:
    if name.startswith("binary_dual_scale_staged_arch_"):
        return name.replace("binary_dual_scale_staged_arch_", "")
    return name


def display_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, short_config(name))


def run_status(raw_config: str, job_id: str) -> str:
    if job_id in PREFIX_INVALID_JOBS.get(raw_config, frozenset()):
        return "pre-fix invalid"
    return "OK"


def ms_tune_has_postfix_attempt(job_id: str, log_path: str) -> bool:
    if int(job_id) >= HP_MS_POSTFIX_MIN:
        return True
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except OSError:
        return False
    return "06-14-39566" in head or "_res39566" in os.path.basename(log_path)


def status_by_config(grid_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    priority = {"OK": 3, "incomplete": 2, "pre-fix invalid": 1}
    for row in grid_rows:
        cfg = row["config"]
        st = row.get("status", "OK")
        if cfg not in out or priority.get(st, 0) > priority.get(out[cfg], 0):
            out[cfg] = st
    return out


def parse_run_dir(path: str) -> Optional[Tuple[str, str, str]]:
    base = os.path.basename(path.rstrip("/"))
    m = re.match(r"\d{2}-\d{2}-(\d+)-([^-]+)-(.+)$", base)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def parse_log_name(path: str) -> Optional[Tuple[str, str, str]]:
    base = os.path.basename(path)
    if not base.endswith(".log"):
        return None
    stem = base[:-4]
    m = re.match(r"\d{2}-\d{2}-(\d+)-([^-]+)-(.+)$", stem)
    if not m:
        return None
    job_id, dataset, rest = m.group(1), m.group(2), m.group(3)
    res = re.match(r"(.+)_res(\d+)$", rest)
    raw_config = res.group(1) if res else rest
    return job_id, dataset, raw_config


def find_run_log(job_id: str, dataset: str, raw_config: str) -> Optional[str]:
    patterns = [
        os.path.join(LOGS, f"*-{job_id}-{dataset}-{raw_config}.log"),
        os.path.join(LOGS, f"*-{dataset}-{raw_config}_res{job_id}.log"),
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def is_smoke_log(path: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except OSError:
        return False
    return "--smoke-test" in head or "Job: smoke-" in head


def load_log_metrics(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    if "PIPELINE COMPLETE" not in text and "staged eval done" not in text:
        return None
    anchor_mse = anchor_mae = crps = sample_mean_mse = None
    tuned_max_scale = None
    for line in text.splitlines():
        m = EVAL_DONE_RE.search(line)
        if m:
            sample_mean_mse = float(m.group(1))
            anchor_mse = float(m.group(2))
            anchor_mae = float(m.group(3))
            crps = float(m.group(4))
        ms = MAX_SCALE_RE.search(line)
        if ms:
            tuned_max_scale = float(ms.group(1))
    if anchor_mse is None:
        return None
    out: Dict[str, Any] = {
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "crps": crps,
        "sample_mean_mse": sample_mean_mse,
    }
    if tuned_max_scale is not None:
        out["tuned_max_scale"] = tuned_max_scale
    return out


def load_log_tuned_max_scale(path: str) -> Optional[float]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    scales = [float(m.group(1)) for m in MAX_SCALE_RE.finditer(text)]
    return max(scales) if scales else None


def reconcile_ms_tune_rows(best_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    """Pick best hp_max_scale row per dataset; prefer post-fix eval over stale pre-fix."""
    ms_cfg = display_config("hp_max_scale_tuning")
    candidates: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in MS_TUNE_DATASETS}

    for log_glob in LOG_GLOBS:
        for log_path in glob.glob(log_glob):
            parsed = parse_log_name(log_path)
            if not parsed:
                continue
            job_id, dataset, raw_config = parsed
            if raw_config != "hp_max_scale_tuning" or dataset not in MS_TUNE_DATASETS:
                continue
            if is_smoke_log(log_path):
                continue

            metrics = load_log_metrics(log_path)
            tuned = (metrics or {}).get("tuned_max_scale") or load_log_tuned_max_scale(log_path)
            has_eval = metrics is not None and metrics.get("anchor_mse") is not None
            postfix = ms_tune_has_postfix_attempt(job_id, log_path)

            if run_status(raw_config, job_id) == "pre-fix invalid":
                status = "pre-fix invalid"
            elif has_eval:
                status = "OK"
            elif postfix and tuned is not None:
                status = "incomplete"
            elif postfix:
                status = "incomplete"
            else:
                status = "incomplete"

            candidates[dataset].append(
                {
                    "dataset": dataset,
                    "config": ms_cfg,
                    "raw_config": raw_config,
                    "job_id": job_id,
                    "anchor_mse": metrics.get("anchor_mse") if metrics else None,
                    "anchor_mae": metrics.get("anchor_mae") if metrics else None,
                    "crps": metrics.get("crps") if metrics else None,
                    "sample_mean_mse": metrics.get("sample_mean_mse") if metrics else None,
                    "tuned_max_scale": tuned,
                    "status": status,
                    "has_eval": has_eval,
                    "postfix": postfix,
                }
            )

    for dataset, cands in candidates.items():
        if not cands:
            continue
        if any(c["postfix"] for c in cands):
            cands = [c for c in cands if c["status"] != "pre-fix invalid"]
        cands.sort(key=lambda c: (c["has_eval"], int(c["job_id"])), reverse=True)
        pick = cands[0]
        tuned_vals = [c["tuned_max_scale"] for c in cands if c.get("tuned_max_scale") is not None]
        if tuned_vals:
            pick["tuned_max_scale"] = max(tuned_vals)
        if pick["status"] == "incomplete":
            pick = {
                **pick,
                "anchor_mse": None,
                "anchor_mae": None,
                "crps": None,
                "sample_mean_mse": None,
            }
        pick.pop("has_eval", None)
        pick.pop("postfix", None)
        best_rows[(dataset, ms_cfg)] = pick


def _merge_row(
    best_rows: Dict[Tuple[str, str], Dict[str, Any]],
    row: Dict[str, Any],
) -> None:
    if row.get("raw_config") != "hp_max_scale_tuning":
        row.pop("tuned_max_scale", None)
    key = (row["dataset"], row["config"])
    prev = best_rows.get(key)
    if prev is not None and int(prev["job_id"]) >= int(row["job_id"]):
        return
    best_rows[key] = row


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


def _load_mmpd_partial(mmpd_dir: str, source: str, dataset: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(mmpd_dir, f"{dataset}_mmpd.json")
    if not os.path.isfile(path):
        return None
    data = load_partial(path)
    anchor_mse = data.get("anchor_mse")
    if anchor_mse is None:
        anchor_mse = data.get("mse")
    anchor_mae = data.get("anchor_mae")
    if anchor_mae is None:
        anchor_mae = data.get("mae")
    return {
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "crps": data.get("crps"),
        "source": source,
    }


def load_mmpd(dataset: str) -> Optional[Dict[str, Any]]:
    for mmpd_dir, source in (
        (MMPD_DIR_SUBSET, MMPD_SOURCE_SUBSET),
        (MMPD_DIR_LEGACY, MMPD_SOURCE_LEGACY),
    ):
        row = _load_mmpd_partial(mmpd_dir, source, dataset)
        if row is not None:
            return row
    return None


def load_mmpd_maskae(dataset: str) -> Optional[Dict[str, Any]]:
    row_fair = _load_mmpd_partial(MMPD_DIR_MASKAE_FAIR, MMPD_SOURCE_MASKAE_FAIR, dataset)
    if row_fair is not None:
        return row_fair
    row_150 = _load_mmpd_partial(MMPD_DIR_MASKAE_150, MMPD_SOURCE_MASKAE_150, dataset)
    row_200 = _load_mmpd_partial(MMPD_DIR_MASKAE, MMPD_SOURCE_MASKAE, dataset)
    if dataset in MMPD_MASKAE_150_DATASETS and row_150 is not None:
        return row_150
    return row_200 or row_150


def mmpd_maskae_job_id(dataset: str) -> str:
    if os.path.isfile(os.path.join(MMPD_DIR_MASKAE_150, f"{dataset}_mmpd.json")):
        return MMPD_MASKAE_JOBS_150.get(dataset, "—")
    return MMPD_MASKAE_JOBS.get(dataset, "—")


def mmpd_config_label(dataset: str, mmpd: Dict[str, Any]) -> str:
    if mmpd.get("source") == MMPD_SOURCE_SUBSET:
        return "**MMPD (subset)**"
    return "**MMPD**"


def enrich_rows_from_logs(best_rows: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    for log_glob in LOG_GLOBS:
        for log_path in glob.glob(log_glob):
            parsed = parse_log_name(log_path)
            if not parsed:
                continue
            job_id, dataset, raw_config = parsed
            if raw_config != "hp_max_scale_tuning":
                continue
            metrics = load_log_metrics(log_path)
            if not metrics:
                continue
            config = display_config(raw_config)
            row = best_rows.get((dataset, config))
            if row is None:
                continue
            if int(row["job_id"]) != int(job_id):
                continue
            if metrics.get("tuned_max_scale") is not None:
                row["tuned_max_scale"] = metrics["tuned_max_scale"]


def collect_runs() -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    best_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_dirs: set[str] = set()
    seen_logs: set[str] = set()

    for run_glob in RUN_GLOBS:
        for run_dir in sorted(glob.glob(run_glob)):
            if run_dir in seen_dirs:
                continue
            seen_dirs.add(run_dir)
            parsed = parse_run_dir(run_dir)
            if not parsed:
                continue
            job_id, dataset, raw_config = parsed
            log_path = find_run_log(job_id, dataset, raw_config)
            if log_path and is_smoke_log(log_path):
                continue
            config = display_config(raw_config)

            partials = glob.glob(os.path.join(run_dir, "partials", "*_staged_anchor.json"))
            if not partials:
                continue
            metrics = load_partial(partials[0])

            row = {
                "dataset": dataset,
                "config": config,
                "raw_config": raw_config,
                "job_id": job_id,
                "anchor_mse": metrics.get("anchor_mse"),
                "anchor_mae": metrics.get("anchor_mae"),
                "crps": metrics.get("crps"),
                "sample_mean_mse": metrics.get("sample_mean_mse"),
                "tuned_max_scale": metrics.get("tuned_max_scale"),
                "status": run_status(raw_config, job_id),
            }
            _merge_row(best_rows, row)

    for log_glob in LOG_GLOBS:
        for log_path in sorted(glob.glob(log_glob)):
            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)
            parsed = parse_log_name(log_path)
            if not parsed:
                continue
            job_id, dataset, raw_config = parsed
            if is_smoke_log(log_path):
                continue
            metrics = load_log_metrics(log_path)
            if not metrics:
                continue
            config = display_config(raw_config)
            key = (dataset, config)
            if key in best_rows and int(best_rows[key]["job_id"]) >= int(job_id):
                continue
            row = {
                "dataset": dataset,
                "config": config,
                "raw_config": raw_config,
                "job_id": job_id,
                "anchor_mse": metrics.get("anchor_mse"),
                "anchor_mae": metrics.get("anchor_mae"),
                "crps": metrics.get("crps"),
                "sample_mean_mse": metrics.get("sample_mean_mse"),
                "tuned_max_scale": metrics.get("tuned_max_scale"),
                "status": run_status(raw_config, job_id),
            }
            _merge_row(best_rows, row)

    enrich_rows_from_logs(best_rows)
    reconcile_ms_tune_rows(best_rows)

    grid_rows = list(best_rows.values())
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        by_dataset[row["dataset"]].append(
            {
                "Config": row["config"],
                "anchor_mse": row["anchor_mse"],
                "anchor_mae": row["anchor_mae"],
                "crps": row["crps"],
                "Status": row["status"],
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
        mmpd_maskae = load_mmpd_maskae(dataset)
        if mmpd_maskae and mmpd_maskae["anchor_mse"] is not None:
            rows.append(
                {
                    "Config": MMPD_MASKAE_LABEL,
                    "anchor_mse": mmpd_maskae["anchor_mse"],
                    "anchor_mae": mmpd_maskae["anchor_mae"],
                    "crps": mmpd_maskae["crps"],
                    "Status": "ref",
                }
            )

    return grid_rows, by_dataset


def append_mmpd_grid_rows(grid_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(grid_rows)
    maskae_datasets = list(dict.fromkeys(SUBSET_DATASETS + MMPD_MASKAE_150_DATASETS))
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
    for dataset in maskae_datasets:
        mmpd_maskae = load_mmpd_maskae(dataset)
        if mmpd_maskae and mmpd_maskae["anchor_mse"] is not None:
            rows.append(
                {
                    "dataset": dataset,
                    "config": MMPD_MASKAE_LABEL,
                    "job_id": mmpd_maskae_job_id(dataset),
                    "anchor_mse": mmpd_maskae["anchor_mse"],
                    "anchor_mae": mmpd_maskae["anchor_mae"],
                    "crps": mmpd_maskae["crps"],
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
            if cfg in MMPD_REF_CONFIGS:
                out[cfg][dataset] = None
            elif r.get("anchor_mse") is None or r.get("Status") == "incomplete":
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
            "**MS tune** = `hp_max_scale_tuning` (`configs/tuning_sweep/hp_max_scale_tuning.yaml`): "
            "Optuna `max_scale ∈ [2.5, 14.0]` with other HPs fixed like `sweep_baseline`. "
            "Merge job `3951208` failed but all MMPD partials exist.\n\n"
            f"{PREFIX_INVALID_NOTE}\n\n"
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

            ms_job = str(ms["job_id"]) if ms else HP_MS_TUNE_JOBS_PREFIX.get(dataset, "—")
            ms_scale = "—"
            if ms and ms.get("tuned_max_scale") is not None:
                ms_scale = fmt(ms["tuned_max_scale"])
            elif ms and ms.get("status") == "pre-fix invalid":
                ms_scale = "policy (not tuned)"
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
                        ms_scale,
                        ms_job,
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

        f.write("\n## MS tune only (`hp_max_scale_tuning`)\n\n")
        f.write(f"{PREFIX_INVALID_NOTE}\n\n")
        f.write(
            "Config extends `fixed_lr_pipeline_base` with `max_scale_tuning: true` and `search_space: lr_only` "
            "(only `max_scale` is Optuna-searched). Four datasets in the sweep arm.\n\n"
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
            if ms.get("tuned_max_scale") is not None:
                scale_disp = fmt(ms["tuned_max_scale"])
            elif ms.get("status") == "pre-fix invalid":
                scale_disp = "policy (not tuned)"
            elif ms.get("status") == "incomplete":
                scale_disp = fmt(ms.get("tuned_max_scale"))
            else:
                scale_disp = "—"
            f.write(
                "| "
                + " | ".join(
                    [
                        dataset,
                        subset_ids.get(dataset, dataset),
                        scale_disp,
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
            "**Flat subsets** (`3951193`–`3951199`), **Flat subsets EMA0.99** (`3951527`–`3951533`; Jun 16 `3967251`–`3967256`), "
            "EMA reuse sweep (`ema_sweep_{090,095,098,0995,0999}`, jobs `3953317`–`3953351`), "
            "grad-accum reuse sweep (`grad_accum_{125,150,200}`, jobs `3953944`–`3953964`; "
            "LR-band split `grad_accum_{150,200}_lr_{lo,hi}`, jobs `3954784`–`3954810`), "
            "**Flat subsets guidance accum** {1.5×, 2×, 4×, 8×} (`grad_accum_guidance_{150,200,400,800}`, jobs `3961419`–`3961447`), "
            "**2d-guidance** (iTrans 2D ghost + guidance channel, `grad_accum_150_lr_hi_guidance`, jobs `3965290`–`3965296`), "
            "**Flat subsets accum4x** no guidance (`grad_accum_400`, jobs `3963967`–`3963973`), "
            "EMA0.99 lookback variants (`ema099_lb336_hz96`, `ema099_lb96_hz720`, jobs `3955091`–`3955098`), "
            "**AR accum4x/8x** (`binary_anchor_ar_grad_accum_{400,800}`, LB96/H96 base), "
            "**AR LB336/H96 accum1.5x** (`3961448`–`3961454`), "
            "**AR LB96/H720 accum1.5x** (`3961455`, `3961457`, `3961460`), "
            "**MS tune** (`hp_max_scale_tuning`), and "
            "**MMPD (subset)** (`06-13-binary-mmpd-subset-compare`, jobs `3951201`–`3951207`). "
            f"**MMPD (MaskedAE)** ETTh1-capped seven (`{MMPD_SOURCE_MASKAE}`) plus Jun 16 "
            f"remaining six (`{MMPD_SOURCE_MASKAE_150}`). "
            "Probabilistic metrics: `dpmpp` sampler, 20 steps, 20 samples.\n\n"
            f"{PREFIX_INVALID_NOTE}\n\n"
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
                        row["status"] if row["status"] == "pre-fix invalid" else f"**{row['status']}**",
                        fmt(row["anchor_mse"]),
                        fmt(row["anchor_mae"]),
                        fmt(row["crps"]),
                        fmt(row["sample_mean_mse"]),
                        row["job_id"],
                    ]
                )
                + " |\n"
            )


def write_leaderboard(
    path: str,
    by_dataset: Dict[str, List[Dict[str, Any]]],
    deltas: Dict[str, Dict[str, Optional[int]]],
    grid_rows: List[Dict[str, Any]],
) -> None:
    datasets_present = [d for d in DATASET_ORDER if d in by_dataset]
    configs = sorted(
        {
            r["Config"]
            for rows in by_dataset.values()
            for r in rows
            if r["Config"] not in MMPD_REF_CONFIGS | {"**Discrete**"}
        },
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

    st_map = status_by_config(grid_rows)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# YAML-First Sweep Leaderboard\n\n")
        f.write(
            "Probabilistic metrics from `dpmpp` sampler with `20` steps. "
            f"Baseline is `{BASELINE}` (fixed 3e-5 LR, linear noise, epsilon target). "
            "**Discrete** is ordinal D3PM CE (`ordinal_d3pm_staged`). "
            "**MAE Discrete** is expectation-MAE + uniform `1/H` anchor (`ordinal_d3pm_mae_staged_subsets`). "
            "**Binary flat** is flat `0.5` XOR anchor on full variates (`binary_anchor_stationary_flat`). "
            "**Flat subsets** (`binary_anchor_stationary_flat_subsets`, jobs `3951193`–`3951199`). "
            "**Flat subsets EMA0.99** (`3951527`–`3951533`; Jun 16 remaining datasets `3967251`–`3967256`). "
            "EMA reuse sweep: `diffusion_ema_decay` ∈ {0.90, 0.95, 0.98, 0.995, 0.999} (jobs `3953317`–`3953351`). "
            "Grad-accum reuse sweep: effective batch {1.25×, 1.5×, 2.0×} (jobs `3953944`–`3953964`); "
            "LR-band split on 1.5×/2.0× (`3954784`–`3954810`). "
            "**Flat subsets guidance accum** {1.5×, 2×, 4×, 8×} (jobs `3961419`–`3961447`). "
            "**2d-guidance** (iTrans 2D ghost + guidance channel, `grad_accum_150_lr_hi_guidance`, jobs `3965290`–`3965296`). "
            "**Flat subsets accum4x** no guidance (jobs `3963967`–`3963973`). "
            "EMA0.99 lookback variants: LB336/H96 and LB96/H720 (`3955091`–`3955098`). "
            "**AR accum4x/8x** (`binary_anchor_ar_grad_accum_{400,800}`, LB96/H96). "
            "**AR LB336/H96 accum1.5x** (`3961448`–`3961454`); "
            "**AR LB96/H720 accum1.5x** (partial: `3961455`, `3961457`, `3961460`). "
            "**MS tune** (`hp_max_scale_tuning`; post-fix `3956631` exchange_rate, `3960878` ETTm1; "
            "incomplete `3960877` ETTh1 / `3960879` weather; cosine+warmup post-fix `3956633`–`3956640`). "
            f"**MMPD (subset)** from `{MMPD_SOURCE_SUBSET}` (Decoder backbone, same subsets as flat runs, "
            "20 samples, full test). "
            f"**MMPD (MaskedAE)** from `{MMPD_SOURCE_MASKAE}` (ETTh1-capped 7 datasets, "
            "grad_accum_200_lr_lo, jobs `3965321`–`3965327`) and "
            f"`{MMPD_SOURCE_MASKAE_150}` (remaining 6 datasets, grad_accum_150_lr_lo MaskAE, "
            "jobs `3968154`–`3968538`). "
            f"Legacy **MMPD** from `{MMPD_SOURCE_LEGACY}` where subset MMPD is unavailable.\n\n"
            f"{PREFIX_INVALID_NOTE}\n\n"
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
            cfg_st = st_map.get(cfg, "OK")
            st_cell = "**pre-fix invalid**" if cfg_st == "pre-fix invalid" else (
                "**incomplete**" if cfg_st == "incomplete" else "**OK**"
            )
            f.write(
                "| "
                + " | ".join(
                    [
                        str(i),
                        f"`{cfg}`" if not cfg.startswith("**") else cfg,
                        f"{avg:+.2f}",
                        *ds_cells,
                        st_cell,
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
        if any(load_mmpd_maskae(ds) for ds in datasets_present):
            maskae_cells = []
            for ds in datasets_present:
                if load_mmpd_maskae(ds):
                    maskae_cells.append(drank_str(None))
                else:
                    maskae_cells.append("—")
            f.write(
                f"| — | {MMPD_MASKAE_LABEL} | — | "
                + " | ".join(maskae_cells)
                + " | ref |\n"
            )
        if any(
            (m := load_mmpd(ds)) and m.get("source") == MMPD_SOURCE_LEGACY
            for ds in datasets_present
        ):
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
                if cfg in MMPD_REF_CONFIGS or base is None:
                    dstr = "—"
                else:
                    dstr = drank_str(rank - base)
                disp = cfg if cfg.startswith("**") else f"`{cfg}`"
                status = r["Status"]
                if status == "OK":
                    status = "**OK**"
                elif status == "pre-fix invalid":
                    status = "**pre-fix invalid**"
                elif status == "incomplete":
                    status = "**incomplete**"
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


def write_ema099_vs_mmpd_compare(
    path: str,
    grid_rows: List[Dict[str, Any]],
) -> None:
    ema_cfg = "**Flat subsets EMA0.99**"
    datasets = list(EMA099_VS_MMPD_SUBSET_IDS.keys())
    mse_wins = {ema_cfg: 0, MMPD_MASKAE_LABEL: 0}
    crps_wins = {ema_cfg: 0, MMPD_MASKAE_LABEL: 0}

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Flat subsets EMA0.99 vs MMPD MaskAE\n\n")
        f.write(
            "Head-to-head on the six **remaining** flat-subset datasets (Jun 16, 2026). "
            "Both use the same YAML `data_subset` policy (`binary_anchor_stationary_flat_subsets`).\n\n"
            "- **EMA0.99**: `binary_anchor_stationary_flat_subsets_ema099`, `dpmpp` 20 steps / 20 samples "
            "(jobs `3967251`–`3967256`).\n"
            f"- **MMPD MaskAE**: `{MMPD_SOURCE_MASKAE_150}`, UP2ME MaskAE backbone, "
            "grad_accum_150_lr_lo, Optuna tune (7 trials), 20 diffusion samples, full test "
            "(jobs `3968154`–`3968538`). MMPD indices come from subset-config (no binary ckpt).\n\n"
            "Lower is better for MSE and CRPS.\n\n"
        )
        header = [
            "Dataset",
            "subset_id",
            "EMA0.99 mse",
            "EMA0.99 crps",
            "EMA job",
            "MMPD mse",
            "MMPD crps",
            "MMPD job",
            "Better MSE",
            "Better CRPS",
        ]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")

        for dataset in datasets:
            ema = subset_row(grid_rows, ema_cfg, dataset)
            mmpd = subset_row(grid_rows, MMPD_MASKAE_LABEL, dataset)
            ema_mse = ema.get("anchor_mse") if ema else None
            ema_crps = ema.get("crps") if ema else None
            mmpd_mse = mmpd.get("anchor_mse") if mmpd else None
            mmpd_crps = mmpd.get("crps") if mmpd else None

            better_mse = better_crps = "—"
            if ema_mse is not None and mmpd_mse is not None:
                if ema_mse < mmpd_mse:
                    better_mse = "EMA0.99"
                    mse_wins[ema_cfg] += 1
                elif mmpd_mse < ema_mse:
                    better_mse = "MMPD"
                    mse_wins[MMPD_MASKAE_LABEL] += 1
                else:
                    better_mse = "tie"
            if ema_crps is not None and mmpd_crps is not None:
                if ema_crps < mmpd_crps:
                    better_crps = "EMA0.99"
                    crps_wins[ema_cfg] += 1
                elif mmpd_crps < ema_crps:
                    better_crps = "MMPD"
                    crps_wins[MMPD_MASKAE_LABEL] += 1
                else:
                    better_crps = "tie"

            ema_job = (
                ema.get("job_id")
                if ema
                else EMA099_EXTRA_JOBS.get(dataset, "—")
            )
            mmpd_job = (
                mmpd.get("job_id")
                if mmpd
                else MMPD_MASKAE_JOBS_150.get(dataset, "—")
            )
            f.write(
                "| "
                + " | ".join(
                    [
                        dataset,
                        EMA099_VS_MMPD_SUBSET_IDS[dataset],
                        fmt(ema_mse),
                        fmt(ema_crps),
                        str(ema_job),
                        fmt(mmpd_mse),
                        fmt(mmpd_crps),
                        str(mmpd_job),
                        better_mse,
                        better_crps,
                    ]
                )
                + " |\n"
            )

        n = len(datasets)
        f.write("\n## Summary\n\n")
        f.write(f"- **MSE**: EMA0.99 {mse_wins[ema_cfg]}/{n}, MMPD {mse_wins[MMPD_MASKAE_LABEL]}/{n}\n")
        f.write(f"- **CRPS**: EMA0.99 {crps_wins[ema_cfg]}/{n}, MMPD {crps_wins[MMPD_MASKAE_LABEL]}/{n}\n")


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    grid_rows, by_dataset = collect_runs()
    grid_rows = append_mmpd_grid_rows(grid_rows)
    deltas = delta_ranks(by_dataset)
    write_grid(GRID_PATH, grid_rows)
    write_leaderboard(LEADERBOARD_PATH, by_dataset, deltas, grid_rows)
    write_subset_compare(SUBSET_COMPARE_PATH, grid_rows)
    write_ema099_vs_mmpd_compare(EMA099_VS_MMPD_PATH, grid_rows)
    print(f"Wrote {GRID_PATH}")
    print(f"Wrote {LEADERBOARD_PATH}")
    print(f"Wrote {SUBSET_COMPARE_PATH}")
    print(f"Wrote {EMA099_VS_MMPD_PATH}")
    print(f"Datasets: {', '.join(sorted(by_dataset))}")
    print(f"Grid rows: {len(grid_rows)}")


if __name__ == "__main__":
    main()
