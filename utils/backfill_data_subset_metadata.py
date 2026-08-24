#!/usr/bin/env python3
"""Rewrite artifacts that stored a single-dataset record under ``data_subset``.

New schema:
    metadata["data_subset_by_dataset"][dataset] = compact record

Usage:
    python utils/backfill_data_subset_metadata.py --path results/ckpts/JOB/subset/metadata.json
    python utils/backfill_data_subset_metadata.py --path results/datasets/JOB --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.data_subset import (
    COMPACT_SUBSET_KEYS,
    LEGACY_SUBSET_METADATA_KEY,
    SUBSET_METADATA_KEY,
    compact_subset_record,
    put_subset_record,
)

JSON_NAMES = ("metadata.json",)
JSON_SUFFIXES = (".json",)


def _dataset_from_record(payload: Dict[str, Any], path: Path) -> str:
    for key in ("dataset", "dataset_name"):
        val = payload.get(key)
        if val:
            return str(val)
    parent = path.parent.name
    if parent and parent not in {"coarse", "patch_refine", "fine"}:
        return parent
    raise ValueError(f"{path}: cannot infer dataset for legacy data_subset backfill")


def _maybe_migrate(payload: Dict[str, Any], path: Path) -> bool:
    if not isinstance(payload, dict):
        return False
    if SUBSET_METADATA_KEY in payload and isinstance(payload[SUBSET_METADATA_KEY], dict):
        payload.pop(LEGACY_SUBSET_METADATA_KEY, None)
        return LEGACY_SUBSET_METADATA_KEY in payload
    legacy = payload.get(LEGACY_SUBSET_METADATA_KEY)
    if not isinstance(legacy, dict):
        return False
    dataset = _dataset_from_record(payload, path)
    # Full resolved records from old writers already have compact keys.
    try:
        rec = compact_subset_record(legacy)
    except KeyError as e:
        raise KeyError(
            f"{path}: legacy data_subset cannot be compacted ({e}). "
            "Re-resolve from experiment.data_subset_by_dataset instead of backfilling."
        ) from e
    put_subset_record(payload, dataset, rec)
    return True


def _iter_json_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in JSON_NAMES or (
            path.suffix in JSON_SUFFIXES and "staged_results" in path.name
        ):
            yield path


def migrate_path(root: Path, *, dry_run: bool) -> int:
    n = 0
    for path in _iter_json_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _maybe_migrate(payload, path):
            continue
        n += 1
        print(f"{'[dry-run] ' if dry_run else ''}migrated {path}")
        if not dry_run:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return n


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.path.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"missing path: {root}")
    n = migrate_path(root, dry_run=args.dry_run)
    print(f"updated {n} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
