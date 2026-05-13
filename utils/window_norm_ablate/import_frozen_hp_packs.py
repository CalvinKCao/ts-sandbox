#!/usr/bin/env python3
"""Build frozen HP JSON packs from completed Slurm result dirs (see submit script).

Each run dir should look like: results/05-12-3539360-default-ETTh1/
with ckpts/pretrained_dim{N}/itrans_hp.json, diff_hp.json, and
ckpts/{Dataset}_itrans_ft_hp.json, ckpts/{Dataset}/metadata.json (tuned_params).

Usage (on a machine that has those trees):

  python utils/window_norm_ablate/import_frozen_hp_packs.py \\
    results/05-12-3539360-default-ETTh1 \\
    results/05-12-3539361-default-ETTh2 \\
    ...  # six dirs total

Writes utils/window_norm_ablate/frozen_packs/{dataset}.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _slug_to_dataset(slug: str) -> str:
    return slug.replace("-", "_")


def _infer_dataset(run_dir: Path) -> str:
    name = run_dir.name
    mo = re.search(
        r"(exchange-rate|ETTh1|ETTh2|ETTm1|ETTm2|weather|electricity|illness|traffic)",
        name,
    )
    if not mo:
        raise ValueError(f"cannot infer dataset id from directory name: {name}")
    return _slug_to_dataset(mo.group(1))


def _import_one(run_dir: Path, out_root: Path) -> Path:
    run_dir = run_dir.resolve()
    ck = run_dir / "ckpts"
    if not ck.is_dir():
        raise FileNotFoundError(f"missing ckpts dir: {ck}")

    dim_dirs = sorted(ck.glob("pretrained_dim*"))
    if not dim_dirs:
        raise FileNotFoundError(f"no pretrained_dim* under {ck}")
    dim_dir = dim_dirs[0]

    dataset = _infer_dataset(run_dir)
    itrans_s = json.loads((dim_dir / "itrans_hp.json").read_text(encoding="utf-8"))
    diff_s = json.loads((dim_dir / "diff_hp.json").read_text(encoding="utf-8"))
    itrans_ft_path = ck / f"{dataset}_itrans_ft_hp.json"
    if not itrans_ft_path.is_file():
        raise FileNotFoundError(itrans_ft_path)
    itrans_ft = json.loads(itrans_ft_path.read_text(encoding="utf-8"))

    meta_path = ck / dataset / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tuned = meta.get("tuned_params") or {}
    if "learning_rate" not in tuned:
        raise ValueError(f"{meta_path} missing tuned_params.learning_rate")

    if meta.get("variate_indices"):
        n_variates = len(meta["variate_indices"])
    else:
        m = re.search(r"pretrained_dim(\d+)", dim_dir.name)
        if not m:
            raise ValueError(f"cannot infer n_variates from {dim_dir.name}")
        n_variates = int(m.group(1))

    pack = {
        "dataset": dataset,
        "n_variates": n_variates,
        "itrans_synth": itrans_s,
        "diffusion_synth": diff_s,
        "itrans_finetune": {
            "learning_rate": float(itrans_ft["learning_rate"]),
            "batch_size": int(itrans_ft.get("batch_size", 32)),
            "dropout": float(itrans_ft.get("dropout", 0.1)),
        },
        "diffusion_finetune": {
            "learning_rate": float(tuned["learning_rate"]),
            "batch_size": int(tuned.get("batch_size", 8)),
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"{dataset}.json"
    out.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Six results/* job dirs containing ckpts/",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "frozen_packs",
    )
    args = ap.parse_args()
    for d in args.run_dirs:
        try:
            _import_one(d, args.out_dir)
        except Exception as e:
            print(f"ERROR {d}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
