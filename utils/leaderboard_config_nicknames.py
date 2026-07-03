"""Leaderboard display names for wandb runs (config nicknames)."""

from __future__ import annotations

import importlib.util
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

RUN_STEM_RE = re.compile(r"^\d{2}-\d{2}-(\d+)-([^-]+)-(.+)$")

CLASSICAL_BASELINES_RAW = "classical_baselines"
CLASSICAL_BASELINES_NICKNAME = "Classical baselines"

MMPD_SUBSET_RAW = "mmpd_subset"
MMPD_SUBSET_NICKNAME = "MMPD (subset)"
MMPD_SUBSET_CAMPAIGN_DATE = "06-13"

MMPD_MASKAE_FAIR_13D_RAW = "mmpd_maskae_fair_13d"
MMPD_MASKAE_FAIR_13D_NICKNAME = "MMPD MaskedAE"
MMPD_MASKAE_FAIR_13D_CAMPAIGN_DATE = "06-16"
MMPD_MASKAE_FAIR_13D_DIR = os.path.join(
    REPO, "results", "datasets", "06-16-mmpd-maskae-fair-13d"
)

MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW = "mmpd_decoder_flat_subsets_grad_accum_200_lr_lo"
MMPD_DECODER_GRAD_ACCUM_200_LR_LO_NICKNAME = "MMPD Decoder (subset tuned)"
MMPD_DECODER_GRAD_ACCUM_200_LR_LO_CAMPAIGN_DATE = "07-02"
MMPD_DECODER_GRAD_ACCUM_200_LR_LO_DIR = os.path.join(
    REPO, "results", "datasets", "07-02-mmpd-decoder-grad-accum-200-lr-lo-subset"
)
MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS: Dict[str, str] = {
    "ETTh1": "4037459",
    "ETTh2": "4037460",
    "ETTm1": "4037461",
    "ETTm2": "4037462",
    "illness": "4037463",
    "exchange_rate": "4037464",
    "weather": "4038279",
    "electricity": "4038280",
    "traffic": "4038281",
    "PeMS": "4038282",
    "solar_Alabama": "4038283",
    "dalia": "4037470",
    "dynamic": "4038284",
}

MMPD_MASKAE_FAIR_13D_JOBS: Dict[str, str] = {
    "ETTh1": "3969137",
    "ETTh2": "3969138",
    "ETTm1": "3969139",
    "ETTm2": "3969140",
    "illness": "3969141",
    "exchange_rate": "3969142",
    "weather": "3969143",
    "electricity": "3969144",
    "traffic": "3969145",
    "PeMS": "3969146",
    "solar_Alabama": "3969147",
    "dalia": "3969148",
    "dynamic": "3969149",
}

MMPD_SUBSET_JOBS: Dict[str, str] = {
    "ETTh1": "3951201",
    "ETTh2": "3951202",
    "exchange_rate": "3951203",
    "weather": "3951204",
    "electricity": "3951205",
    "traffic": "3951206",
    "solar_Alabama": "3951207",
}

MMPD_DIR_SUBSET = os.path.join(
    REPO, "results", "datasets", "06-13-binary-mmpd-subset-compare", "partials"
)


def strip_leaderboard_markdown(label: str) -> str:
    s = label.strip()
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        return s[2:-2]
    return s


@lru_cache(maxsize=1)
def _gsl_module():
    for rel in (
        "archive/reports/generate_sweep_leaderboard.py",
        "reports/generate_sweep_leaderboard.py",
    ):
        path = os.path.join(REPO, rel)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("gsl", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("generate_sweep_leaderboard.py not found")


def display_config(raw_config: str) -> str:
    gsl = _gsl_module()
    return gsl.display_config(raw_config)


def leaderboard_nickname(
    *,
    raw_config: Optional[str] = None,
    config_label: Optional[str] = None,
    yaml_path: Optional[str] = None,
) -> str:
    if config_label:
        return strip_leaderboard_markdown(config_label)
    if raw_config == MMPD_SUBSET_RAW:
        return MMPD_SUBSET_NICKNAME
    if raw_config == MMPD_MASKAE_FAIR_13D_RAW:
        return MMPD_MASKAE_FAIR_13D_NICKNAME
    if raw_config == MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW:
        return MMPD_DECODER_GRAD_ACCUM_200_LR_LO_NICKNAME
    if raw_config == CLASSICAL_BASELINES_RAW:
        return CLASSICAL_BASELINES_NICKNAME
    if raw_config:
        return strip_leaderboard_markdown(display_config(raw_config))
    if yaml_path:
        stem = os.path.splitext(os.path.basename(str(yaml_path)))[0]
        return strip_leaderboard_markdown(display_config(stem))
    return ""


def parse_run_stem(stem: str) -> Optional[Tuple[str, str, str]]:
    m = RUN_STEM_RE.match(stem or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def mmpd_subset_run_stem(dataset: str, job_id: Optional[str] = None) -> str:
    jid = job_id or MMPD_SUBSET_JOBS[dataset]
    return f"{MMPD_SUBSET_CAMPAIGN_DATE}-{jid}-{dataset}-{MMPD_SUBSET_RAW}"


def mmpd_leaderboard_run_stem(
    dataset: str,
    raw_config: str,
    *,
    job_id: str,
    campaign_date: Optional[str] = None,
) -> str:
    date = campaign_date or datetime.now().strftime("%m-%d")
    return f"{date}-{job_id}-{dataset}-{raw_config}"


def mmpd_fair_13d_run_stem(dataset: str, job_id: Optional[str] = None) -> str:
    jid = job_id or MMPD_MASKAE_FAIR_13D_JOBS[dataset]
    return mmpd_leaderboard_run_stem(
        dataset,
        MMPD_MASKAE_FAIR_13D_RAW,
        job_id=jid,
        campaign_date=MMPD_MASKAE_FAIR_13D_CAMPAIGN_DATE,
    )


def mmpd_decoder_grad_accum_200_lr_lo_run_stem(dataset: str, job_id: Optional[str] = None) -> str:
    jid = job_id or MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS[dataset]
    return mmpd_leaderboard_run_stem(
        dataset,
        MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW,
        job_id=jid,
        campaign_date=MMPD_DECODER_GRAD_ACCUM_200_LR_LO_CAMPAIGN_DATE,
    )


def nickname_for_wandb_run(run: Any) -> str:
    cfg = dict(run.config)
    existing = cfg.get("config_nickname")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    curation = cfg.get("curation") or {}
    label = curation.get("config_label")
    if label:
        return leaderboard_nickname(config_label=str(label))

    yaml_path = cfg.get("_yaml_path")
    if yaml_path:
        return leaderboard_nickname(yaml_path=str(yaml_path))

    parsed = parse_run_stem(run.group or "")
    if parsed:
        _, _, raw_config = parsed
        return leaderboard_nickname(raw_config=raw_config)

    if getattr(run, "job_type", None) == "classical_baseline":
        return CLASSICAL_BASELINES_NICKNAME
    if cfg.get("run_type") == "classical_baseline":
        return CLASSICAL_BASELINES_NICKNAME

    return ""


def load_mmpd_fair_13d_metrics(dataset: str) -> Optional[Dict[str, Any]]:
    partial = os.path.join(MMPD_MASKAE_FAIR_13D_DIR, "partials", f"{dataset}_mmpd.json")
    if not os.path.isfile(partial):
        return None
    import json

    with open(partial, encoding="utf-8") as f:
        data = json.load(f)
    anchor_mse = data.get("anchor_mse")
    if anchor_mse is None:
        anchor_mse = data.get("mse")
    anchor_mae = data.get("anchor_mae")
    if anchor_mae is None:
        anchor_mae = data.get("mae")
    crps = data.get("crps")
    if anchor_mse is None or anchor_mae is None or crps is None:
        return None

    tuning_path = os.path.join(MMPD_MASKAE_FAIR_13D_DIR, "tuning", f"{dataset}_best.json")
    tuned_hparams = None
    if os.path.isfile(tuning_path):
        with open(tuning_path, encoding="utf-8") as f:
            tuned_hparams = json.load(f).get("hparams")

    return {
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "crps": crps,
        "source": "06-16-mmpd-maskae-fair-13d",
        "partial_path": partial,
        "tuning_path": tuning_path if os.path.isfile(tuning_path) else None,
        "tuned_hparams": tuned_hparams,
        "raw": data,
    }


def load_mmpd_decoder_grad_accum_200_lr_lo_metrics(dataset: str) -> Optional[Dict[str, Any]]:
    partial = os.path.join(
        MMPD_DECODER_GRAD_ACCUM_200_LR_LO_DIR, "partials", f"{dataset}_mmpd.json"
    )
    if not os.path.isfile(partial):
        return None
    import json

    with open(partial, encoding="utf-8") as f:
        data = json.load(f)
    anchor_mse = data.get("anchor_mse")
    if anchor_mse is None:
        anchor_mse = data.get("mse")
    anchor_mae = data.get("anchor_mae")
    if anchor_mae is None:
        anchor_mae = data.get("mae")
    crps = data.get("crps")
    if anchor_mse is None or anchor_mae is None or crps is None:
        return None

    tuning_path = os.path.join(
        MMPD_DECODER_GRAD_ACCUM_200_LR_LO_DIR, "tuning", f"{dataset}_best.json"
    )
    tuned_hparams = None
    if os.path.isfile(tuning_path):
        with open(tuning_path, encoding="utf-8") as f:
            tuned_hparams = json.load(f).get("hparams")

    return {
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "crps": crps,
        "source": "07-02-mmpd-decoder-grad-accum-200-lr-lo-subset",
        "partial_path": partial,
        "tuning_path": tuning_path if os.path.isfile(tuning_path) else None,
        "tuned_hparams": tuned_hparams,
        "raw": data,
    }


def load_mmpd_subset_metrics(dataset: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(MMPD_DIR_SUBSET, f"{dataset}_mmpd.json")
    if not os.path.isfile(path):
        return None
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
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
        "source": "06-13-binary-mmpd-subset-compare",
        "partial_path": path,
        "raw": data,
    }


def mmpd_stub_wandb_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Match staged_eval.py keys so MMPD stubs appear in the same wandb panels."""
    raw = metrics.get("raw") or {}
    out: Dict[str, float] = {
        "eval/staged_anchor_mse": float(metrics["anchor_mse"]),
        "eval/staged_anchor_mae": float(metrics["anchor_mae"]),
        "eval/staged_crps": float(metrics["crps"]),
    }
    if raw.get("mse") is not None:
        out["eval/staged_prob_mse"] = float(raw["mse"])
        out["eval/staged_sample_mean_mse"] = float(raw["mse"])
    if raw.get("mae") is not None:
        out["eval/staged_prob_mae"] = float(raw["mae"])
        out["eval/staged_sample_mean_mae"] = float(raw["mae"])
    if raw.get("top1_mse") is not None:
        out["eval/staged_top1_mse"] = float(raw["top1_mse"])
    if raw.get("top3_mse") is not None:
        out["eval/staged_top3_mse"] = float(raw["top3_mse"])
    return out
