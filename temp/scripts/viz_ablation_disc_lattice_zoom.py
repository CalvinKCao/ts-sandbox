#!/usr/bin/env python3
"""Zoomed L8/L16 disc-input lattice panels from existing snapped/raw packs.

Use this when ablation packs already exist (adapter raw dir), or to smoke the
viz path from the h96 ordinal campaign disc-raw + MMPD packs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.eval_univariate_disc_two_ablations_vs_gt import (  # noqa: E402
    DEFAULT_MMPD,
    _mmpd_pack,
    _snap_bundle,
    _thin_indices,
    _write_zoomed_disc_input_viz,
)
from utils.disc_shared import binary_mmpd_train_scaler_map, write_json  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    DEFAULT_MMPD_DATA,
    run_train_stride,
    run_test_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import (  # noqa: E402
    assert_not_anchor_agg,
    reduce_pack_forecast,
    subset_pack_by_pool_indices,
)
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument(
        "--binary-pack",
        type=Path,
        default=REPO_ROOT
        / "results/datasets/07-31-0925-h96-ordinal-disc-raw/binary_ordinal_patch_refine_ETTh1.npz",
    )
    p.add_argument("--tag", default="campaign_ref")
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument(
        "--ckpt-meta-root",
        type=Path,
        default=REPO_ROOT
        / "results/ckpts/07-29-4462980-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
        help="Only metadata is required (variate indices / strides).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/datasets/disc-ablation-window-norm-vs-ordinal-fine",
    )
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--test-max-items", type=int, default=8)
    p.add_argument("--disc-index-stride", type=int, default=1)
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--viz-windows", type=int, default=2)
    p.add_argument("--viz-variate", type=int, default=0)
    p.add_argument("--viz-zoom-steps", type=int, default=8)
    p.add_argument("--viz-y-rung-pad", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    assert_not_anchor_agg(args.fake_agg)
    args.binary_pack = args.binary_pack.expanduser().resolve()
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.expanduser().resolve()
    args.ckpt_meta_root = args.ckpt_meta_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _load_run(dataset: str, ckpt_root: Path) -> AnchorRun:
    for subset in sorted(ckpt_root.iterdir()):
        meta = subset / "patch_refine" / "metadata.json"
        if not meta.is_file():
            meta = subset / "fine" / "metadata.json"
        if not meta.is_file():
            continue
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        if metadata.get("dataset_name") != dataset:
            continue
        metadata = dict(metadata)
        metadata["dataset_name"] = dataset
        metadata["dataset"] = dataset
        return AnchorRun(
            variant="binary_meta",
            dataset=dataset,
            root=ckpt_root,
            subset_dir=subset,
            best_pt=None,
            itrans_pt=None,
            metadata=metadata,
        )
    raise FileNotFoundError(f"no metadata for {dataset} under {ckpt_root}")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    if not args.binary_pack.is_file():
        raise FileNotFoundError(args.binary_pack)
    with np.load(args.binary_pack) as data:
        binary_pack = {k: data[k] for k in data.files}
    run = _load_run(args.dataset, args.ckpt_meta_root)
    mmpd_full = _mmpd_pack(args.mmpd_output_root, args.dataset)
    # Intersect binary pack indices with MMPD, then thin for a quick viz.
    shared = np.intersect1d(
        np.asarray(binary_pack["indices"], dtype=np.int64),
        np.asarray(mmpd_full["indices"], dtype=np.int64),
    )
    indices = _thin_indices(
        shared,
        seed=int(args.seed),
        test_fraction=float(args.test_fraction),
        disc_index_stride=int(args.disc_index_stride),
        test_max_items=int(args.test_max_items),
    )
    binary_pack = dict(subset_pack_by_pool_indices(binary_pack, indices))
    mmpd_pack = dict(subset_pack_by_pool_indices(mmpd_full, indices))

    _, _, _, norm_stats = load_dataset(
        args.dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=args.lookback,
        horizon=args.horizon,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats["ordinal_ladder"]
    past = binary_pack["past"].astype(np.float32)
    gt_raw = binary_pack["y_true"].astype(np.float32)
    binary_raw = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    scaler_args = SimpleNamespace(
        lookback=args.lookback,
        horizon=args.horizon,
        mmpd_data_dir=args.mmpd_data_dir,
    )
    scalers = binary_mmpd_train_scaler_map(scaler_args, run)
    mmpd_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=gt_raw,
        mmpd_y_true=mmpd_pack["y_true"].astype(np.float32),
        mmpd_fakes=reduce_pack_forecast(mmpd_pack, agg=args.fake_agg),
        **scalers,
    )
    gt, binary, mmpd, legal, lattice = _snap_bundle(
        past=past,
        gt_raw=gt_raw,
        binary_raw=binary_raw,
        mmpd_raw=mmpd_z,
        ladder=ladder,
        device=device,
    )
    lattice["mmpd_alignment"] = align
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"lattice_fixture_{args.tag}_{args.dataset}.json", lattice)
    paths = _write_zoomed_disc_input_viz(
        args,
        tag=args.tag,
        dataset=args.dataset,
        gt=gt,
        binary=binary,
        mmpd=mmpd,
        legal_levels=legal,
        indices=indices,
    )
    print(f"wrote {len(paths)} panels; lattice rows={legal.shape[-1]}", flush=True)
    for path in paths[:6]:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
