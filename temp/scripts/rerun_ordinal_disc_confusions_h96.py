#!/usr/bin/env python3
"""One-off: rerun h96 ordinal disc with one probabilistic sample fakes + confusion PNGs.

Reuses ``07-31-0925-h96-ordinal-disc-raw`` binary packs and MMPD root packs, defaults
``fake_agg=sample0`` (first draw; never mean-over-S or anchor), retrains univariate
discs, writes TP/TN/FP/FN panels under ``results/datasets/<run>/disc_confusions/``.

Does **not** retrain forecast models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.eval_univariate_patch_refine_ordinal_vs_mmpd import (  # noqa: E402
    _binary_lattice_atol,
    _mmpd_pack,
    _pack_test_stride,
)
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier  # noqa: E402
from utils.eval_discriminator_texture_staged_vs_mmpd import (  # noqa: E402
    binary_mmpd_train_scaler_map,
    parse_args as parse_base_args,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    load_tsf_pack_pool,
    parse_pack_splits,
    run_train_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import (  # noqa: E402
    assert_not_anchor_agg,
    reduce_pack_forecast,
    subset_pack_by_pool_indices,
)
from utils.patch_refine_ordinal_ladder import (  # noqa: E402
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.visualize_discriminator_univariate_confusions import (  # noqa: E402
    visualize_univariate_combo,
)
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run  # noqa: E402

DEFAULT_DATASETS = ("electricity", "ETTh1", "dynamic", "traffic")
DEFAULT_BINARY = {
    "electricity": "results/ckpts/07-29-4462979-electricity-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "ETTh1": "results/ckpts/07-29-4462980-ETTh1-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "dynamic": "results/ckpts/07-29-4462981-dynamic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
    "traffic": "results/ckpts/07-29-4462982-traffic-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback",
}
DEFAULT_DISC_RAW = REPO_ROOT / "results/datasets/07-31-0925-h96-ordinal-disc-raw"
DEFAULT_MMPD = REPO_ROOT / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
DEFAULT_OUT = REPO_ROOT / "results/datasets/07-31-h96-ordinal-disc-confusions-prob"


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    custom.add_argument("--disc-raw-dir", type=Path, default=DEFAULT_DISC_RAW)
    custom.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    custom.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    custom.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    custom.add_argument("--slice-lengths", nargs="+", type=int, default=[8])
    custom.add_argument("--per-bucket", type=int, default=2)
    custom.add_argument("--lookback-tail", type=int, default=32)
    custom.add_argument("--pack-test-stride", type=int, default=4)
    custom.add_argument("--smoke-test", action="store_true")
    custom.add_argument("--cpu", action="store_true")
    custom.add_argument("--gpu", type=int, default=0)
    extra, remaining = custom.parse_known_args()

    saved = sys.argv
    sys.argv = [
        saved[0],
        "--fake-sources", "binary_staged", "mmpd",
        "--lookback", "336", "--horizon", "96",
        "--test-stride", "4", "--test-fraction", "0.25",
        "--candidate-only", "--save-classification-scores",
        "--save-checkpoints", "--force-train",
        "--mmpd-instance-norm", "--no-mmpd-ordinal-norm",
        "--pack-splits", "test",
        "--output-dir", str(extra.output_dir),
        "--mmpd-output-root", str(extra.mmpd_output_root),
        "--raw-eval-dir", str(extra.disc_raw_dir),
        *remaining,
    ]
    try:
        args = parse_base_args()
    finally:
        sys.argv = saved

    args.datasets = [d for raw in extra.datasets for d in str(raw).split(",") if d]
    args.disc_raw_dir = extra.disc_raw_dir.expanduser().resolve()
    args.mmpd_output_root = extra.mmpd_output_root.expanduser().resolve()
    args.output_dir = extra.output_dir.expanduser().resolve()
    args.raw_eval_dir = args.disc_raw_dir
    args.fake_agg = str(extra.fake_agg)
    assert_not_anchor_agg(args.fake_agg)
    args.slice_lengths = [int(x) for x in extra.slice_lengths]
    args.viz_per_bucket = int(extra.per_bucket)
    args.viz_lookback_tail = int(extra.lookback_tail)
    args.pack_test_stride = max(1, int(extra.pack_test_stride))
    args.smoke_test = bool(extra.smoke_test)
    args.cpu = bool(extra.cpu)
    args.gpu = int(extra.gpu)
    args.save_checkpoints = True
    args.force_train = True
    args.candidate_only = True
    if args.smoke_test:
        args.datasets = args.datasets[:1]
        args.slice_lengths = args.slice_lengths[:1]
        args.epochs = min(int(args.epochs), 2)
        args.patience = min(int(args.patience), 2)
        args.max_train_examples = min(int(args.max_train_examples or 64), 64)
        args.max_eval_examples = min(int(args.max_eval_examples or 32), 32)
        args.batch_size = min(int(args.batch_size), 32)
    return args


def _prepare_bundle(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> tuple[SimpleNamespace, Mapping[str, np.ndarray], List[int]]:
    binary_path = args.disc_raw_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    if not binary_path.is_file():
        raise FileNotFoundError(f"missing disc-raw binary pack: {binary_path}")
    binary_pack = _load_npz(binary_path)
    indices = [int(i) for i in binary_pack["indices"].tolist()]
    if args.smoke_test and len(indices) > 8:
        indices = indices[:8]
        binary_pack = {
            k: (v[:8] if isinstance(v, np.ndarray) and v.shape[:1] == (binary_pack["y_true"].shape[0],) else v)
            for k, v in binary_pack.items()
        }

    mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
    mmpd_pack = subset_pack_by_pool_indices(mmpd_full, np.asarray(indices, dtype=np.int64))

    ckpt_rel = DEFAULT_BINARY[dataset]
    ckpt_root = (REPO_ROOT / ckpt_rel).resolve()
    run, _stages = load_patch_refine_run(dataset, ckpt_root, test_stride=None)

    binary_gt = binary_pack["y_true"].astype(np.float32)
    binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
    mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    print(
        f"[{dataset}] fake_agg={args.fake_agg} "
        f"binary_samples={binary_pack['samples'].shape} mmpd_samples={mmpd_pack['samples'].shape}",
        flush=True,
    )

    scalers = binary_mmpd_train_scaler_map(args, run)
    mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_pred,
        **scalers,
    )
    # Ladder from ckpt-linked ordinal loader via past windows on the shared pool.
    from utils.visualize_staged_eval_2d_preds import _build_state
    from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
    from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
    from utils.eval_mmpd_gaussian_anchor import run_subset_id, run_test_stride
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    state = _build_state(
        ckpt_root, dataset, run_subset_id(run),
        str(REPO_ROOT / "configs/binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback.yaml"),
    )
    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=336,
        horizon=96,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal ladder missing")
    state.extra["global_ordinal_ladder"] = ladder
    pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    past_pool, _, _, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=336,
        horizon=96,
        train_stride=run_train_stride(run),
        test_stride=_pack_test_stride(args),
        pack_splits=parse_pack_splits("test"),
        use_ordinal_window_norm=False,
    )
    past = np.stack([past_pool[i][0].detach().cpu().numpy() for i in indices]).astype(np.float32)
    legal_levels = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
    gt, _ = snap_to_patch_refine_levels(binary_gt, legal_levels)
    mmpd_snapped, _ = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
    binary_atol = _binary_lattice_atol(legal_levels)
    assert_on_patch_refine_levels(binary_pred, legal_levels, atol=binary_atol)
    assert_on_patch_refine_levels(mmpd_snapped, legal_levels)

    write_json(
        args.output_dir / f"lattice_assertion_{dataset}.json",
        {"align": align, "fake_agg": args.fake_agg, "n_windows": len(indices)},
    )
    bundle = SimpleNamespace(
        fakes={"binary_staged": binary_pred, "mmpd": mmpd_snapped},
        y_true_by_source={"binary_staged": gt, "mmpd": gt.copy()},
        past=past,
        indices=np.asarray(indices, dtype=np.int64),
        series_starts=np.asarray(binary_pack["series_starts"], dtype=np.int64),
        run=run,
        pack_splits=[str(x) for x in np.asarray(binary_pack.get("pack_splits", ["test"])).tolist()],
    )
    return bundle, {"gt": gt, "binary": binary_pred, "mmpd": mmpd_snapped}, indices


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device} fake_agg={args.fake_agg} out={args.output_dir}", flush=True)

    for dataset in args.datasets:
        print(f"\n===== {dataset} =====", flush=True)
        bundle, _snapped, _indices = _prepare_bundle(args, dataset, device)
        splits = split_windows(
            len(bundle.y_true_by_source["binary_staged"]),
            args,
            dataset,
            indices=bundle.indices,
            lookback=336,
            horizon=96,
            test_stride=_pack_test_stride(args),
            series_starts=bundle.series_starts,
        )
        conf_dir = args.output_dir / "disc_confusions"
        for source in ("binary_staged", "mmpd"):
            for length in args.slice_lengths:
                print(f"[train] {dataset}/{source}/L{length}", flush=True)
                metrics = train_classifier(
                    args, dataset, source, int(length), bundle, splits, device,
                )
                write_json(
                    args.output_dir / "partials" / f"{dataset}__{source}.json",
                    {str(int(length)): metrics},
                )
                visualize_univariate_combo(
                    output_dir=args.output_dir,
                    dataset=dataset,
                    fake_source=source,
                    slice_len=int(length),
                    past=bundle.past,
                    y_true=bundle.y_true_by_source[source],
                    fake=bundle.fakes[source],
                    test_windows=splits["test"],
                    device=device,
                    seed=int(args.seed),
                    batch_size=int(args.batch_size),
                    per_bucket=int(args.viz_per_bucket),
                    lookback_tail=int(args.viz_lookback_tail),
                    plot_dir=conf_dir,
                    max_eval_examples=args.max_eval_examples,
                    candidate_only=True,
                    offset_stride=int(args.offset_stride),
                )
        print(f"[{dataset}] confusions under {conf_dir}", flush=True)

    manifest = {
        "datasets": list(args.datasets),
        "fake_agg": args.fake_agg,
        "disc_raw_dir": str(args.disc_raw_dir),
        "mmpd_output_root": str(args.mmpd_output_root),
        "output_dir": str(args.output_dir),
        "note": "probabilistic path: mean-over-samples (never deterministic/anchor)",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
