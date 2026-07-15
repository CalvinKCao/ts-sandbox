"""Canonical reused checkpoint layout under $SCRATCH/ts-sandbox/reused/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def reused_root() -> str:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "ts-sandbox", "reused")
    return os.path.join(os.getcwd(), "reused")


def reused_pretrain_ckpt(config_suffix: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "pretrain",
        config_suffix,
        f"pretrained_{stage}",
        "pretrained_diffusion.pt",
    )


def reused_guidance_ckpt(config_suffix: str, subset_id: str) -> str:
    return os.path.join(
        reused_root(),
        "guidance",
        config_suffix,
        f"{subset_id}_patch_guidance.pt",
    )


def reused_tuned_params_meta(config_suffix: str, subset_id: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "tuned_params",
        config_suffix,
        subset_id,
        stage,
        "metadata.json",
    )


def reused_stage_best_ckpt(config_suffix: str, subset_id: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "tuned_params",
        config_suffix,
        subset_id,
        stage,
        "best.pt",
    )


def reused_binary_staged_root(config_stem: str) -> str:
    return os.path.join(reused_root(), "binary", config_stem)


def reused_mmpd_campaign_root(config_suffix: str) -> str:
    return os.path.join(reused_root(), "mmpd", config_suffix)


def find_reused_pretrain_ckpt(config_suffix: str, stage: str) -> Optional[str]:
    path = reused_pretrain_ckpt(config_suffix, stage)
    return path if os.path.isfile(path) else None


def find_reused_guidance_ckpt(config_suffix: str, subset_id: str) -> Optional[str]:
    path = reused_guidance_ckpt(config_suffix, subset_id)
    return path if os.path.isfile(path) else None


def find_reused_tuned_params_meta(
    config_suffix: str,
    subset_id: str,
    stage: str,
) -> Optional[str]:
    path = reused_tuned_params_meta(config_suffix, subset_id, stage)
    return path if os.path.isfile(path) else None


def find_reused_binary_staged_root(config_stem: str, dataset: str) -> Optional[str]:
    """Return reused binary run root if it has staged coarse/fine or vertical_dual best.pt."""
    root = Path(reused_binary_staged_root(config_stem))
    if not root.is_dir():
        return None
    for sub_dir in root.iterdir():
        if not sub_dir.is_dir():
            continue
        vd_meta = sub_dir / "vertical_dual" / "metadata.json"
        if (sub_dir / "vertical_dual" / "best.pt").is_file() and vd_meta.is_file():
            try:
                import json

                meta = json.loads(vd_meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("dataset_name") == dataset:
                return str(root)
        fine_meta = sub_dir / "fine" / "metadata.json"
        if not (sub_dir / "coarse" / "best.pt").is_file():
            continue
        if not (sub_dir / "fine" / "best.pt").is_file():
            continue
        if not fine_meta.is_file():
            continue
        try:
            import json

            meta = json.loads(fine_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("dataset_name") == dataset:
            return str(root)
    return None


def find_reused_mmpd_campaign_root(
    config_suffix: str,
    *,
    data_names: Optional[Sequence[str]] = None,
    backbone: str = "Decoder",
) -> Optional[str]:
    """Return reused MMPD campaign dir when it has at least one Decoder-MMPD ckpt."""
    root = Path(reused_mmpd_campaign_root(config_suffix))
    base = root / "mmpd_out" / "checkpoints" / f"{backbone}-MMPD"
    if not base.is_dir():
        return None
    prefixes = [f"data{name}_" for name in (data_names or [])]
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if prefixes and not any(d.name.startswith(pref) for pref in prefixes):
            continue
        if (d / "model_checkpoint.pth").is_file():
            return str(root)
    return None
