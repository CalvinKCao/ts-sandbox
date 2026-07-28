#!/usr/bin/env python3
"""Fail fast unless an MMPD campaign used ordinal norm without instance norm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mmpd_root", type=Path)
    args = parser.parse_args()
    manifest_path = args.mmpd_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing campaign manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    saved = manifest.get("args")
    if not isinstance(saved, dict):
        raise ValueError(f"{manifest_path}: missing saved args")
    if saved.get("use_ordinal_window_norm") is not True:
        raise ValueError(
            f"{manifest_path}: expected use_ordinal_window_norm=true, got "
            f"{saved.get('use_ordinal_window_norm')!r}"
        )
    if saved.get("mmpd_instance_norm") is not False:
        raise ValueError(
            f"{manifest_path}: expected mmpd_instance_norm=false, got "
            f"{saved.get('mmpd_instance_norm')!r}"
        )
    print(
        f"[mmpd-campaign] verified ordinal norm / no instance norm: {args.mmpd_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
