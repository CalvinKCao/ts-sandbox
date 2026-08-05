#!/usr/bin/env python3
"""Fail fast unless an MMPD campaign used ordinal norm without instance norm."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mmpd_root", type=Path)
    args = parser.parse_args()
    campaign_root = args.mmpd_root.parent if args.mmpd_root.name == "mmpd_out" else args.mmpd_root
    manifest_path = campaign_root / "run_manifest.json"
    reused_manifest = False
    if not manifest_path.is_file():
        manifest_path = campaign_root / "manifest.json"
        reused_manifest = True
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing campaign manifest: expected {campaign_root / 'run_manifest.json'} "
            f"or {campaign_root / 'manifest.json'}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if reused_manifest:
        expected_suffix = "mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm"
        if manifest.get("config_suffix") != expected_suffix:
            raise ValueError(
                f"{manifest_path}: expected config_suffix={expected_suffix!r}, got "
                f"{manifest.get('config_suffix')!r}"
            )
        config_path = Path(__file__).resolve().parents[2] / "configs" / f"{expected_suffix}.yaml"
        config_text = config_path.read_text(encoding="utf-8")
        if len(re.findall(r"^\s*use_ordinal_window_norm:\s*true\s*$", config_text, flags=re.MULTILINE)) < 2:
            raise ValueError(f"{config_path}: reused campaign config does not enable ordinal normalization")
        if re.search(r"^\s*mmpd_instance_norm:\s*true\s*$", config_text, flags=re.MULTILINE):
            raise ValueError(f"{config_path}: reused campaign config enables instance normalization")
        ckpt_root = campaign_root / "mmpd_out" / "checkpoints" / "Decoder-MMPD"
        if not any(ckpt_root.glob("*/model_checkpoint.pth")):
            raise FileNotFoundError(f"{campaign_root}: no Decoder MMPD model_checkpoint.pth files")
        print(
            f"[mmpd-campaign] verified reused ordinal/no-instance campaign: {campaign_root}",
            flush=True,
        )
        return
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
        f"[mmpd-campaign] verified ordinal norm / no instance norm: {campaign_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
