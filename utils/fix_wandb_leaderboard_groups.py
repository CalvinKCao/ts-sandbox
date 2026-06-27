"""Normalize wandb groups in ts-sandbox-leaderboard to Slurm run stems.

Each dataset×config job should group all phase runs (pretrain, HP, eval) under one
stem like ``06-15-3965290-ETTh1-binary_anchor_stationary_flat_subsets_...``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.load_dotenv import load_repo_dotenv

load_repo_dotenv(REPO)

PROJECT = "ts-sandbox-leaderboard"
ENTITY = "calvincao"

PHASE_SUFFIXES = (
    "-staged-diffusion-pretrain",
    "-diffusion-coarse-finetune-hp",
    "-diffusion-fine-finetune-hp",
    "-diffusion-finer-finetune-hp",
    "-itrans-finetune-hp",
    "-staged-eval",
)

STEM_FROM_NAME_RE = re.compile(r"^(\d{2}-\d{2}-\d+-.+)$")
RELOG_URL_RE = re.compile(r"relogged from (https://wandb\.ai/\S+)")


def _run_path(run) -> str:
    path = getattr(run, "path", None)
    if isinstance(path, list):
        return "/".join(str(p) for p in path)
    return str(path).lstrip("/")


def stem_from_name(name: str) -> Optional[str]:
    for suffix in sorted(PHASE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if STEM_FROM_NAME_RE.match(candidate):
                return candidate
    if STEM_FROM_NAME_RE.match(name):
        return name
    return None


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_stem_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """Return (dst_path -> stem, src_url -> stem)."""
    dst_to_stem: Dict[str, str] = {}
    url_to_stem: Dict[str, str] = {}

    copy_map_paths = [
        os.path.join(REPO, "reports", "sweep_grid_report", "curated_wandb_copy_map.json"),
        os.path.join(REPO, "archive", "reports", "sweep_grid_report", "curated_wandb_copy_map.json"),
    ]
    for path in copy_map_paths:
        data = load_json(path)
        if not data:
            continue
        for rec in (data.get("copies") or {}).values():
            stem = rec.get("run_stem")
            if not stem:
                continue
            if rec.get("dst_path"):
                dst_to_stem[rec["dst_path"]] = stem
            if rec.get("src_url"):
                url_to_stem[rec["src_url"].rstrip(").,;")] = stem

    manifest_paths = [
        os.path.join(REPO, "reports", "sweep_grid_report", "curated_wandb_manifest.json"),
        os.path.join(REPO, "archive", "reports", "sweep_grid_report", "curated_wandb_manifest.json"),
    ]
    for path in manifest_paths:
        data = load_json(path)
        if not data:
            continue
        for hit in data.get("wandb_runs_found") or []:
            stem = hit.get("run_stem")
            url = hit.get("url")
            if stem and url:
                url_to_stem[url.rstrip(").,;")] = stem
            path_key = hit.get("path")
            if stem and path_key:
                url_to_stem.setdefault(path_key, stem)

    return dst_to_stem, url_to_stem


def resolve_stem(run, dst_to_stem: Dict[str, str], url_to_stem: Dict[str, str]) -> Optional[str]:
    path = _run_path(run)
    if path in dst_to_stem:
        return dst_to_stem[path]

    stem = stem_from_name(run.name or "")
    if stem:
        return stem

    notes = run.notes or ""
    m = RELOG_URL_RE.search(notes)
    if m:
        url = m.group(1).rstrip(").,;")
        if url in url_to_stem:
            return url_to_stem[url]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    dst_to_stem, url_to_stem = build_stem_maps()

    updated = skipped = missing = 0
    for run in api.runs(f"{args.entity}/{args.project}"):
        stem = resolve_stem(run, dst_to_stem, url_to_stem)
        if not stem:
            missing += 1
            continue
        if (run.group or "") == stem:
            skipped += 1
            continue
        if args.dry_run:
            print(f"would set group: {run.name!r}  {run.group!r} -> {stem!r}")
            updated += 1
            continue
        run.group = stem
        run.update()
        updated += 1
        print(f"updated: {run.name} -> group {stem}")

    print(
        f"\n{'would update' if args.dry_run else 'updated'}: {updated}, "
        f"already ok: {skipped}, unresolved: {missing}"
    )


if __name__ == "__main__":
    main()
