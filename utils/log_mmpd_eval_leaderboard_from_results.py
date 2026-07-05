#!/usr/bin/env python3
"""Backfill mmpd_eval rows into ts-sandbox-leaderboard from local partials."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.leaderboard_config_nicknames import (  # noqa: E402
    MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS,
    MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW,
    MMPD_MASKAE_FAIR_13D_JOBS,
    MMPD_MASKAE_FAIR_13D_RAW,
    MMPD_SUBSET_JOBS,
    MMPD_SUBSET_RAW,
)
from utils.load_dotenv import load_repo_dotenv  # noqa: E402
from utils.log_mmpd_eval_leaderboard import log_mmpd_eval_to_leaderboard  # noqa: E402

load_repo_dotenv(REPO)

DATA_ROOT = REPO / "results" / "datasets"
LOG_ROOT = REPO / "results" / "logs"

DIR_RAW_CONFIG: Dict[str, str] = {
    "06-13-binary-mmpd-subset-compare": MMPD_SUBSET_RAW,
    "06-16-mmpd-maskae-fair-13d": MMPD_MASKAE_FAIR_13D_RAW,
    "07-02-mmpd-decoder-grad-accum-200-lr-lo-subset": MMPD_DECODER_GRAD_ACCUM_200_LR_LO_RAW,
    "07-05-mmpd-decoder-paper-lb336-hz720-subset": "mmpd_decoder_flat_subsets_paper_lb336_hz720",
}

DIR_CONFIG_YAML: Dict[str, str] = {
    "07-02-mmpd-decoder-grad-accum-200-lr-lo-subset": "configs/mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.yaml",
    "07-05-mmpd-decoder-paper-lb336-hz720-subset": "configs/mmpd_decoder_flat_subsets_paper_lb336_hz720.yaml",
}

KNOWN_JOB_MAPS: Dict[str, Dict[str, str]] = {
    "06-13-binary-mmpd-subset-compare": MMPD_SUBSET_JOBS,
    "06-16-mmpd-maskae-fair-13d": MMPD_MASKAE_FAIR_13D_JOBS,
    "07-02-mmpd-decoder-grad-accum-200-lr-lo-subset": MMPD_DECODER_GRAD_ACCUM_200_LR_LO_JOBS,
}

_SLURM_LOG_RE = re.compile(r"^mmpd-(.+)-(\d+)\.out$")


def _normalize_repo_path(path_str: str) -> Path:
    p = Path(path_str.replace("/scratch/ccao87/ts-sandbox", str(REPO)))
    return p


def _metrics_from_partial(data: Dict[str, Any]) -> Dict[str, Any]:
    anchor_mse = data.get("anchor_mse", data.get("mse"))
    anchor_mae = data.get("anchor_mae", data.get("mae"))
    crps = data.get("crps")
    return {
        "anchor_mse": anchor_mse,
        "anchor_mae": anchor_mae,
        "crps": crps,
        "raw": data,
    }


def _manifest_config_path(output_dir: Path) -> Optional[Path]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("config", "args"):
        block = manifest.get(key)
        if isinstance(block, dict) and block.get("mmpd_run_config"):
            p = _normalize_repo_path(str(block["mmpd_run_config"]))
            if p.is_file():
                return p
    return None


def _resolve_raw_config(output_dir: Path, config_path: Optional[Path]) -> str:
    if config_path is not None:
        return config_path.stem
    return DIR_RAW_CONFIG.get(output_dir.name, output_dir.name)


def _resolve_config_path(output_dir: Path) -> Optional[Path]:
    from_manifest = _manifest_config_path(output_dir)
    if from_manifest is not None:
        return from_manifest
    rel = DIR_CONFIG_YAML.get(output_dir.name)
    if rel:
        p = REPO / rel
        if p.is_file():
            return p
    return None


def _job_ids_from_logs(output_dir: Path) -> Dict[str, str]:
    log_dir = LOG_ROOT / output_dir.name
    if not log_dir.is_dir():
        return {}
    out: Dict[str, str] = {}
    for path in log_dir.glob("mmpd-*-*.out"):
        m = _SLURM_LOG_RE.match(path.name)
        if not m:
            continue
        ds, jid = m.group(1), m.group(2)
        if ds not in out or int(jid) > int(out[ds]):
            out[ds] = jid
    return out


def _resolve_job_id(output_dir: Path, dataset: str) -> Optional[str]:
    from_logs = _job_ids_from_logs(output_dir).get(dataset)
    if from_logs:
        return from_logs
    known = KNOWN_JOB_MAPS.get(output_dir.name, {})
    return known.get(dataset)


def _discover_output_dirs(selected: Optional[Iterable[Path]] = None) -> list[Path]:
    if selected:
        return [p.resolve() for p in selected]
    dirs: list[Path] = []
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir() or "mmpd" not in path.name:
            continue
        if path.name.startswith("_smoke"):
            continue
        partials = path / "partials"
        if partials.is_dir() and any(partials.glob("*_mmpd.json")):
            dirs.append(path)
    return dirs


def backfill_output_dir(
    output_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Dict[str, str]:
    output_dir = output_dir.resolve()
    config_path = _resolve_config_path(output_dir)
    raw_config = _resolve_raw_config(output_dir, config_path)
    partials_dir = output_dir / "partials"
    results: Dict[str, str] = {}

    for partial_path in sorted(partials_dir.glob("*_mmpd.json")):
        dataset = partial_path.stem.removesuffix("_mmpd")
        job_id = _resolve_job_id(output_dir, dataset)
        if not job_id:
            print(f"[skip] {output_dir.name}/{dataset}: no job_id (logs or known map)")
            results[dataset] = "no_job_id"
            continue

        data = json.loads(partial_path.read_text(encoding="utf-8"))
        metrics = _metrics_from_partial(data)
        if metrics["anchor_mse"] is None or metrics["anchor_mae"] is None or metrics["crps"] is None:
            print(f"[skip] {output_dir.name}/{dataset}: incomplete metrics")
            results[dataset] = "incomplete"
            continue

        url = log_mmpd_eval_to_leaderboard(
            dataset=dataset,
            metrics=metrics,
            output_dir=output_dir,
            mmpd_run_config=config_path,
            raw_config=raw_config,
            job_id=job_id,
            force=force,
            dry_run=dry_run,
            extra_tags=["backfill"],
        )
        results[dataset] = "dry_run" if dry_run and url is None else ("logged" if url else "skipped")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        action="append",
        type=Path,
        dest="output_dirs",
        help="MMPD campaign dir under results/datasets (repeatable). Default: all non-smoke mmpd dirs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-log even if marker/wandb group exists.")
    args = parser.parse_args()

    dirs = _discover_output_dirs(args.output_dirs)
    if not dirs:
        raise SystemExit("no MMPD output dirs found")

    summary: Dict[str, Dict[str, str]] = {}
    for output_dir in dirs:
        print(f"\n== {output_dir.name} ==")
        summary[output_dir.name] = backfill_output_dir(
            output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )

    print("\ndone:", summary)


if __name__ == "__main__":
    main()
