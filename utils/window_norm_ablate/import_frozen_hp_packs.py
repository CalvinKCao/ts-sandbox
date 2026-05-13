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

import torch


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


def _subset_ckpt_dir(ck: Path, dataset: str) -> Path:
    """Resolve ckpts/<subset>/ where subset folder name may differ only by case."""
    want = dataset.lower()
    for child in ck.iterdir():
        if child.is_dir() and child.name.lower() == want:
            return child
    return ck / dataset


def _load_diffusion_finetune_tuned_params(ck: Path, dataset: str) -> dict:
    """tuned_params with learning_rate (+ batch_size) from metadata.json or trial checkpoints."""
    subset = _subset_ckpt_dir(ck, dataset)
    meta_path = subset / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tuned = meta.get("tuned_params") or {}
        if "learning_rate" in tuned:
            return tuned
    best_tp: dict = {}
    best_v = float("inf")
    for p in sorted(subset.glob("_diff_ft_trial_*_best.pt")):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        tp = d.get("tuned_params") or {}
        if "learning_rate" not in tp:
            continue
        v = float(d.get("val_loss", float("inf")))
        if v < best_v:
            best_v = v
            best_tp = tp
    if best_tp:
        return best_tp
    raise FileNotFoundError(
        f"no tuned diffusion finetune params: tried {meta_path} and {subset}/_diff_ft_trial_*_best.pt"
    )


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
        for q in ck.glob("*_itrans_ft_hp.json"):
            if q.name.lower().startswith(f"{dataset.lower()}_"):
                itrans_ft_path = q
                break
    if not itrans_ft_path.is_file():
        raise FileNotFoundError(f"missing {dataset}_itrans_ft_hp.json under {ck}")
    itrans_ft = json.loads(itrans_ft_path.read_text(encoding="utf-8"))

    subset_dir = _subset_ckpt_dir(ck, dataset)
    meta_path = subset_dir / "metadata.json"
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tuned = _load_diffusion_finetune_tuned_params(ck, dataset)

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
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures but keep processing remaining run dirs",
    )
    args = ap.parse_args()
    failed = 0
    for d in args.run_dirs:
        try:
            _import_one(d, args.out_dir)
        except Exception as e:
            print(f"ERROR {d}: {e}", file=sys.stderr)
            failed += 1
            if not args.continue_on_error:
                sys.exit(1)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
