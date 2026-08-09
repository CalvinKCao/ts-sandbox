#!/usr/bin/env python3
"""Fail fast unless an MMPD campaign used paper instance-normalized evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_CONFIG = "mmpd_decoder_flat_subsets_paper_lb336_hz720.yaml"
EXPECTED_DATASETS = ("ETTh1", "traffic_4v_s1", "exchange_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mmpd_root", type=Path)
    args = parser.parse_args()

    campaign_root = args.mmpd_root.parent if args.mmpd_root.name == "mmpd_out" else args.mmpd_root
    manifest_path = campaign_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing campaign manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    saved = manifest.get("args")
    if not isinstance(saved, dict):
        raise ValueError(f"{manifest_path}: missing saved args")

    run_config = Path(str(saved.get("mmpd_run_config", ""))).name
    if run_config != EXPECTED_CONFIG:
        raise ValueError(
            f"{manifest_path}: expected mmpd_run_config={EXPECTED_CONFIG!r}, got {run_config!r}"
        )
    if saved.get("mmpd_instance_norm") is not True:
        raise ValueError(
            f"{manifest_path}: expected mmpd_instance_norm=true, got "
            f"{saved.get('mmpd_instance_norm')!r}"
        )
    if saved.get("use_ordinal_window_norm") is True:
        raise ValueError(f"{manifest_path}: non-ordinal MMPD campaign unexpectedly enables ordinal norm")

    ckpt_root = campaign_root / "mmpd_out" / "checkpoints" / "Decoder-MMPD"
    missing = [
        dataset
        for dataset in EXPECTED_DATASETS
        if not list(ckpt_root.glob(f"data{dataset}_il336_ol720_backboneDecoder_*/model_checkpoint.pth"))
    ]
    if missing:
        raise FileNotFoundError(f"{ckpt_root}: missing Decoder checkpoints for {', '.join(missing)}")
    print(
        f"[mmpd-campaign] verified paper instance-norm campaign and 3 Decoder checkpoints: "
        f"{campaign_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
