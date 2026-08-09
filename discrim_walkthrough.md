# Discriminator walkthrough — unique_abs L8/L16 ablation (live path)
#
# Inline comments for this path also live in the source files below (see File map).
# Refreshed against working tree on `exp/ordinal-fine-residual` (2026-08-05).
# Note: this markdown was never committed; earlier excerpts matched an Aug-4
# working-tree snapshot. Major drift since then: training-lattice snap
# (`window_norm_grid` / hybrid_flat), submit defaults `val,test` + 80/20,
# forecast cache, LayerNorm-only disc, lean arches (mlp/cnn1d/flatness).

Read this top-to-bottom in **runtime order**. Code blocks are real excerpts from the repo (line ranges in the fence header). Comments inside the blocks are walkthrough annotations (what happens to tensors / windows / labels); many of the same explanations are mirrored as inline comments in those modules.

## File map (old + new disc path)

| Role | Path | What to read |
|------|------|--------------|
| **OLD arch** | `utils/disc_shared.py` → `InvertedSliceDiscriminator` (~L1020) | iTransformer-style; collapses under unique_abs C=1 |
| **NEW arches + factory** | same file → `FlatnessSliceDiscriminator` / `MLPSliceDiscriminator` / `CNN1DSliceDiscriminator` / `build_slice_discriminator` / `DISC_ARCH_CHOICES` | `--disc-arch {transformer,mlp,cnn1d,flatness}` |
| Split / AUROC / pack protocol | `utils/disc_shared.py` → `split_windows`, `apply_disc_pack_protocol`, `binary_auroc`, `window_level_metrics` | chrono 80/20 + hard purge |
| **LEGACY multivariate dataset** | `utils/disc_shared.py` → `HorizonSliceDataset` | dense windows×offsets; **not** the live ablation |
| Data loading + unique_abs | `utils/eval_discriminator_binary_vs_mmpd_univariate.py` → `_unique_absolute_slice_items`, `UnivariateRealVsFakeDataset` | one (abs_t,variate) → real/fake pair |
| Train + eval disc | same file → `train_classifier`, `evaluate_classifier` | BCE train, collapse detector, per-variate AUROC |
| Orchestrator + snap | `temp/scripts/eval_ablation_disc_l8_l16.py` → `_legal_levels_for_run`, `_snap_bundle`, `run_one` | leaf-aware lattice + nearest-rung snap |
| Window-norm lattice (canvas128) | `utils/patch_refine_value_grid.py` → `legal_window_norm_patch_refine_levels_dataset_z` | H midpoints from past mean/std |
| Ordinal lattice + nearest snap | `utils/patch_refine_ordinal_ladder.py` → `legal_patch_refine_levels_dataset_z`, `snap_to_patch_refine_levels` | ordinal leaves + shared snap helper |
| Bin-center preprocess | `utils/disc_bin_center_shift.py` → `bin_center_shift` | kills absolute level; keeps step texture |
| Hybrid flat (LULL-like) | `utils/hybrid_flat_dataset_norm.py` | flat variates skip window-norm |
| Lean multi-arch runner | `temp/scripts/smoke_lean_disc_arches.py` | pack → snap → loop arches × sources × L |
| Slurm (legacy transformer) | `temp/scripts/submit_ablation_disc_l8_l16.sh` | original unique_abs ablation |
| Slurm (lean sweep) | `temp/scripts/submit_lean_disc_c128_killarney.sh` | canvas128 × datasets × arches |

**Live path (unique_abs ablation disc):** login submit → compute-node Python → pack pool (paper **val+test** by default) → binary + MMPD fakes → **training-lattice** snap → chrono purged split (**80/20** + val-from-train) → unique absolute L-slices → disc (`--disc-arch`, default transformer) → test AUROC.

**Entrypoints:**

- `temp/scripts/submit_ablation_disc_l8_l16.sh` (self-sbatch wrapper)
- `temp/scripts/eval_ablation_disc_l8_l16.py` (symlink: `temp/eval_ablation_disc_l8_l16.py`)
- Shared train/eval: `utils/eval_discriminator_binary_vs_mmpd_univariate.py` (`train_classifier`, `UnivariateRealVsFakeDataset`, `_unique_absolute_slice_items`)
- Shared split / AUROC helpers: `utils/disc_shared.py` (`split_windows`, `apply_disc_pack_protocol`, `binary_auroc`, `window_level_metrics`, `InvertedSliceDiscriminator` + lean arches)
- Ladder / snap: `utils/patch_refine_ordinal_ladder.py` (ordinal leaves), `utils/patch_refine_value_grid.py` (window-norm / canvas128), `utils/hybrid_flat_dataset_norm.py` (flat-variate hybrid), `utils/disc_bin_center_shift.py`

Companion style reference: `full_walkthrough.md` (binary staged train DAG). This doc is the disc side only.

**Defaults that matter for recent ~0.5 AUROC numbers:**

| Flag | Ablation default | Effect |
|------|------------------|--------|
| `--unique-absolute-slices` | **on** | One (window,offset) per absolute `[T,T+L)` × variate |
| `--candidate-only` | **on** | Disc sees L-slice only (no lookback) |
| `--disc-bin-center-shift` | **on** | Per-slice integer bin mean-centering (**zscore hard-off** when BC on) |
| `--slice-lengths` | `8 16` | Short local texture |
| `--pack-splits` | `val,test` (submit) | Combined paper **val+test** pool (stride=`pack_test_stride` on both) |
| `--train-fraction` / `--val-fraction` | `0.8` / `0` (submit) | Last 20% → disc test; early-stop val = last 10% of purged train (≈72/8/20) |
| `--fake-agg` | `sample0` | First stochastic draw |
| snap mode | from ckpt flags | `window_norm_grid` / `window_norm_grid_hybrid_flat` / `ordinal_absolute` |

### Pack protocol: `val,test` + 80/20

`--pack-splits val,test` concatenates paper val and test windows (both at `--pack-test-stride`, usually 4 — not dense train_stride=1 on val). Chronological `split_windows` then takes the last 20% as disc test and purges any earlier window that temporally overlaps that tail. With `--val-fraction 0`, early-stopping val is carved as the **last 10% of the purged train pool** so the final 20% test stays untouched. `apply_disc_pack_protocol` auto-applies 0.8/0 when pack is val+test and fractions are still the legacy 0.7/0.15 defaults. Precomputed test-only MMPD packs are **not** index-compatible with the expanded pool — the ablation rematerializes MMPD into `output_dir/raw/` (donor ckpts via `--mmpd-output-root`).

Legacy multivariate `HorizonSliceDataset` in `disc_shared` (dense windows×offsets, optional zscore) is **not** what this ablation trains. Old ~0.9 AUROC numbers almost certainly came from that denser / leakier protocol (or an earlier fair-disc variant without unique_abs). Mentioned below only where needed.

---

## 1. Login / compute: submit_ablation_disc_l8_l16.sh

Outside Slurm, the script re-sbatches itself (L40S, 8h / 50G full, shorter for `--smoke-test` / `--viz-only`). On the compute node it builds a fresh `$SLURM_TMPDIR` venv and calls the Python evaluator. Prefer `$SCRATCH/ts-sandbox-ordinal-fine` over stale `$SCRATCH/ts-sandbox`.

```bash
# temp/scripts/submit_ablation_disc_l8_l16.sh:1-57
#!/bin/bash
# Ablation L8/L16 candidate-only disc. Ladder height comes from each run's
# patch_refine_canvas_height (256 legacy / 128 canvas128 leaf).
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox-ordinal-fine):
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --viz-only --smoke-test
#   CKPT=results/ckpts/<stamp>-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6 \
#     ./temp/scripts/submit_ablation_disc_l8_l16.sh
#
# Forecast packs auto-cache under results/datasets/disc_forecast_cache/ ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # Login node: pick wall/mem from flags, then sbatch this same script.
    IS_SMOKE=0
    IS_VIZ=0
    for arg in "$@"; do
        [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
        [ "$arg" = "--viz-only" ] && IS_VIZ=1
    done
    ...
    sbatch \
        --job-name="$NAME" \
        --account=aip-boyuwang \
        --gres=gpu:l40s:1 \
        ...
        "$SCRIPT_DIR/submit_ablation_disc_l8_l16.sh" "$@"
    exit 0
fi
```

Python handoff (MMPD root + protocol flags baked in; `"$@"` / `CKPT` / `DISC_CONFIG` override). **Live default pack is val+test / 80/20** (not legacy test-only 70/15/15):

```bash
# temp/scripts/submit_ablation_disc_l8_l16.sh:130-168
STAMP="$(date +%m-%d-%H%M)"
OUT_TAG="${OUT_TAG:-valtest80}"
OUT_DIR="results/datasets/${STAMP}-ablation-disc-l8-l16-${OUT_TAG}"
mkdir -p "$OUT_DIR" results/slurm

CFG="${DISC_CONFIG:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
RUN_NAME="${DISC_RUN_NAME:-window_norm_c128}"
if [ -n "${CKPT:-}" ]; then
    RUN_SPEC="${RUN_NAME}:${CKPT}:${CFG}"
    RUN_ARGS=(--runs "$RUN_SPEC")
else
    RUN_ARGS=()
fi

# Default protocol: paper val+test pack, chrono 80/20 (+ val-from-train early-stop).
# Legacy test-only: --pack-splits test --train-fraction 0.7 --val-fraction 0.15
python temp/scripts/eval_ablation_disc_l8_l16.py \
    --dataset ETTh1 \
    --output-dir "$OUT_DIR" \
    --lookback 336 \
    --horizon 96 \
    --pack-test-stride 4 \
    --pack-splits val,test \
    --train-fraction 0.8 \
    --val-fraction 0 \
    --fake-agg sample0 \
    --slice-lengths 8 16 \
    --candidate-only \
    --disc-bin-center-shift \
    --num-sampling-steps 20 \
    --mmpd-output-root results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \
    "${RUN_ARGS[@]}" \
    "$@"
```

Note: unique_abs is **not** passed on the CLI because the Python flag defaults to on (`--unique-absolute-slices` / `--no-unique-absolute-slices`).

---

## 2. CLI / config load

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:110-236
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="name:ckpt_root:config triples (coarse+patch_refine or coarse+fine layouts)",
    )
    ...
    # Pack pool: paper-test windows at stride 4 by default (MMPD-aligned).
    # Use --pack-splits val,test for paper val+test combined pool (80/20 disc carve).
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--pack-splits", default="test")  # submit overrides → val,test
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    ...
    p.add_argument("--candidate-only", action="store_true", default=True)
    p.add_argument("--disc-bin-center-shift", action="store_true", default=True)
    # BC on → zscore hard-off (mutually exclusive; fail-fast in apply_smoke).
    p.add_argument("--disc-apply-zscore", action="store_true", default=False)
    ...
    p.add_argument(
        "--unique-absolute-slices",
        action="store_true",
        default=True,
        help="One random (window,offset) per absolute L-block ...",
    )
    p.add_argument(
        "--no-unique-absolute-slices",
        action="store_false",
        dest="unique_absolute_slices",
        help="Dense Cartesian product of windows × in-horizon offsets (old slow path).",
    )
    # Disc fractions *inside* the pack. Submit sets 0.8/0 for val+test.
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.15)
    ...
```

Each `--runs` entry is `name:ckpt_root:config`. `load_ablation_run` accepts either **coarse+patch_refine** or **coarse+fine** checkpoint layouts.

Canvas height is **not** guessed: read `patch_refine_canvas_height` from the run YAML/state (256 legacy / 128 coarser leaf), or pass `--canvas-height`.

Shared forecast packs live under `results/datasets/disc_forecast_cache/` (keyed by ckpt+pack+protocol). `--reuse-forecast-cache` / `--require-forecast-cache` fail fast if missing; `--no-forecast-cache` disables the shared layer.

---

## 3. Pack / index loading (“MMPD-aligned”)

`run_one` builds the pool from `--pack-splits`. **test-only** still drives indices from the precomputed MMPD pack. **val,test** (live submit default) builds a ConcatDataset pool, rematerializes MMPD, then regenerates binary on the same rows.

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1537-1617 (trimmed)
def run_one(...):
    """Pack pool → binary + MMPD fakes on same rows → snap → purged split → disc."""
    dataset = str(args.dataset)
    pack_splits = parse_pack_splits(args.pack_splits)

    if list(pack_splits) == ["test"]:
        # Load MMPD npz: y_true, samples, indices (and friends).
        mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
        indices = [int(x) for x in np.asarray(mmpd_full["indices"], dtype=np.int64).tolist()]
        indices, mmpd_pack = _thin_windows(
            indices, mmpd_full, max_windows=args.max_windows, seed=args.seed,
        )
        print(f"[{run_name}] windows={len(indices)} (MMPD-aligned pack_splits=test)", flush=True)
    else:
        # Expanded pack: rematerialize MMPD; refuse test-only index reuse.
        pool, _starts, _splits, part_lengths, _ = load_tsf_pack_pool(
            ..., pack_splits=pack_splits, use_ordinal_window_norm=False,
        )
        indices = list(range(len(pool)))
        ...
        mmpd_pack = _rematerialize_mmpd_pack(args, run=run, indices=indices, pack_splits=pack_splits)

    # Generate (or cache-load) binary_staged pack on those exact pool indices.
    binary_pack, run, ladder, kind, canvas_height = materialize_binary_pack(
        args, dataset=dataset, run_name=run_name, ckpt_root=ckpt_root,
        config_path=config_path, indices=indices, device=device,
    )
```

Binary materialization rebuilds the TSF concat pool with `load_tsf_pack_pool(..., use_ordinal_window_norm=False)` — always **dataset-z** windows, never mixed train-rank / test-z. Absolute `series_starts` come from paper borders + stride:

```python
# utils/eval_mmpd_gaussian_anchor.py:1724-1803 (trimmed)
def _absolute_series_starts_for_splits(...):
    # border1s from _paper_split_borders: train/val/test CSV row starts
    # (val/test lookbacks reach back into previous split — paper protocol).
    ...
    starts.extend(border + i * stride for i in range(int(n_part)))
    return np.asarray(starts, dtype=np.int64)

def load_tsf_pack_pool(...):
    # ConcatDataset in pack_splits order; series_starts aligned 1:1 with pool rows.
    # Pack pools always load z-score series (use_ordinal_window_norm=False).
    ...
    return pool, series_starts, list(pack_splits), part_lengths, stats
```

With `--pack-splits val,test`, disc windows span paper **val and test**. Disc train/val/test below are a *second* chronological carve of that pack — **not** the paper 70/10/20 (or ETT month) borders.

Cached pack keys written under `output_dir/raw/binary_{run}_{dataset}_{pack_tag}.npz` (and optionally shared cache):

- `past` `(N,V,Lb)`, `y_true` `(N,V,H)`, `samples` `(N,V,1,H)` (single sample0 draw stored as S=1)
- `indices` (pool row ids), `series_starts` (absolute CSV row of each past start)
- `canvas_height`, `kind` (`patch_refine` | `fine`), `pack_splits`

---

## 4. Generate or load fakes (binary staged + MMPD)

### Binary staged

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1019-1045 (trimmed)
with torch.no_grad():
    for batch_idx, (past, future) in enumerate(loader):
        past = past.to(device)
        future = future.to(device)
        overlap = int(getattr(refine.config, "lookback_overlap", 0) or 0)
        target = future[..., overlap:] if overlap else future
        torch.manual_seed(int(args.seed) + batch_idx * 1009)
        result = generate_staged_forecast(
            coarse, refine, past,
            vertical_dual=False,
            sampler=args.probabilistic_sampler,
            num_inference_steps=int(args.num_sampling_steps),
        )
        # Dataset-z forecast (same coordinate family as pack y_true).
        pred = result["prediction_global_norm"]
        ...
        samples_all.append(pred.detach().cpu().numpy().astype(np.float32)[:, :, None, :])
```

Window-norm ckpts still load their own training flags for sampling. Disc snap then uses the **matching training lattice** (window-norm H-row grid for canvas128; absolute ordinal ladder only for ordinal leaves) — see `_legal_levels_for_run` below.

### MMPD

Already on disk for test-only packs: `mmpd_output_root/raw/mmpd_{dataset}.npz`. For val+test, rematerialized into `output_dir/raw/`. Reduced with the same `--fake-agg` (`sample0` → `samples[:,:,0,:]`).

### Snap / align onto the training lattice

**This is the biggest correction vs the Aug-4 walkthrough.** Older text claimed disc always snapped onto the absolute ordinal ladder “for fairness.” That was a real bug for canvas128 / window-norm leaves (wrong alphabet). Live path picks the lattice from training flags:

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:798-836
def _legal_levels_for_run(...):
    """Pick the lattice that matches the binary training leaf (not a foreign ordinal one)."""
    if bool(getattr(state, "use_ordinal_window_norm", False)):
        levels = legal_patch_refine_levels_dataset_z(
            past, ladder=ladder, canvas_height=h, device=device,
        )
        return levels, "ordinal_absolute", {"canvas_height": float(h)}

    # Window-norm canvas128 (and non-ordinal window-norm) leaves.
    max_scale = _max_scale_from_ckpt_metadata(ckpt_root, dataset)
    flat_mask = _flat_mask_from_ckpt(ckpt_root, dataset)  # hybrid flat-dsnorm
    grid_cfg = _window_norm_grid_config(
        state, canvas_height=h, max_scale=max_scale,
        skip_window_norm_variate_mask=flat_mask,
    )
    levels = legal_window_norm_patch_refine_levels_dataset_z(past, grid_cfg)
    snap_mode = (
        "window_norm_grid_hybrid_flat" if flat_mask and any(flat_mask)
        else "window_norm_grid"
    )
    return levels, snap_mode, meta
```

Window-norm lattice midpoints (dataset-z, lookback-conditioned):

```python
# utils/patch_refine_value_grid.py:71-106 (trimmed)
def legal_window_norm_patch_refine_levels_dataset_z(past, config) -> np.ndarray:
    """Finite H-row window-norm lattice midpoints in dataset-z. Shape (N, V, H).

    Row i → (-max_scale + (i + 0.5) * step) * std + center
    with center/std from the lookback only (same as training encode).
    """
    past_t = torch.from_numpy(np.asarray(past, dtype=np.float32))
    center, std = window_normalization_stats(past_t, config)
    step = normalized_grid_step(config)
    rows = torch.arange(height, dtype=torch.float32)
    norm_levels = -max_scale + (rows + 0.5) * step  # (H,)
    levels = norm_levels.view(1, 1, -1) * std + center  # (N,V,H)
    return levels.detach().cpu().numpy().astype(np.float32)
```

Bundle snap:

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1070-1168 (trimmed)
def _snap_bundle(...):
    binary_gt = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)
    mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    if not np.array_equal(binary_pack["indices"], mmpd_pack["indices"]):
        raise RuntimeError("binary/MMPD indices differ after thinning")

    # Affine map MMPD z → binary train-scaler z (from saved scalers, not eval GT).
    scalers = binary_mmpd_train_scaler_map(args, run)
    mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt, mmpd_y_true=mmpd_gt, mmpd_fakes=mmpd_pred, **scalers,
    )

    legal_levels, snap_mode, snap_meta = _legal_levels_for_run(...)
    # Nearest-rung snap for GT, aligned MMPD, and binary forecast.
    gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
    mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
    binary, binary_snap = snap_to_patch_refine_levels(binary_raw, legal_levels)
    ...
    return {"gt": gt, "binary": binary, "mmpd": mmpd, "past": past,
            "legal_levels": ..., "snap_mode": snap_mode, ...}
```

After this point every source speaks the **same discrete alphabet** (window-specific canvas rows decoded to dataset-z). Continuous forecast quirks that lived between rungs are gone.

**AUROC flag (protocol, not a one-line bug):** snapping + shared alphabet is exactly what you want for a fair “are these on the same lattice?” disc — and it also **destroys** sub-rung signal. If models mostly differ from GT by sub-bin noise, AUROC collapses toward chance by design. Post-fix wn128grid campaigns still sit ~0.50 — wrong-ladder snap was real, but it was not the sole cause of chance AUROC under the full hardness stack.

---

## 5. Disc train/val/test split (80/20 + purge) vs paper borders

Bundle handed to the univariate trainer:

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1680-1706
bundle = SimpleNamespace(
    fakes={"binary_staged": snapped["binary"], "mmpd": snapped["mmpd"]},
    # Same snapped GT for both fake sources (real-vs-fake, not binary-vs-mmpd).
    y_true_by_source={
        "binary_staged": snapped["gt"],
        "mmpd": snapped["gt"].copy(),
    },
    past=snapped["past"],
    legal_levels=snapped["legal_levels"],
    indices=snapped["indices"],
    series_starts=snapped["series_starts"],
    ...
)
splits = split_windows(
    len(snapped["gt"]),
    args,
    dataset,
    indices=bundle.indices,
    lookback=args.lookback,
    horizon=args.horizon,
    test_stride=int(args.pack_test_stride),
    series_starts=bundle.series_starts,
)
```

### What `split_windows` / `apply_disc_pack_protocol` do

```python
# utils/disc_shared.py:796-930 (trimmed)
def apply_disc_pack_protocol(args):
    # val,test + legacy 0.7/0.15 defaults → auto-switch to 0.8/0.0
    ...

def split_windows(...):
    starts, ends = window_time_bounds(..., series_starts=starts_all)
    order = np.argsort(starts, kind="mergesort")  # chronological by past start

    if val_frac <= 0:
        # 80/20-style: test = last (1 - train_frac); val carved from train pool.
        n_test = max(1, int(round(len(order) * (1.0 - train_frac))))
        ...
    else:
        # Target sizes from --train-fraction / --val-fraction (default 0.7 / 0.15).
        n_train_target = max(1, int(round(len(order) * train_frac)))
        n_val_target = max(1, int(round(len(order) * val_frac)))
        n_test = len(order) - n_train_target - n_val_target

    test = order[-n_test:]                      # latest windows → disc test
    test_start = int(starts[test].min())
    # HARD PURGE: drop any earlier window whose span reaches into test region.
    train_val_pool = [idx for idx in order[:-n_test] if ends[idx] <= test_start]
    ...
    # Within train/val pool: chronological train then val.
    # Overlap *within* train and *within* val is allowed on purpose.
    train = tv_order[:-n_val]
    val = tv_order[-n_val:]
```

### Contrast with paper borders

| | Paper borders (`_paper_split_borders`) | Disc `split_windows` (this path) |
|--|--|--|
| Region | Full CSV: train / val / test | Live pack = **paper val+test** |
| Fractions | ETT months, or ~70/10/20 | 80 / 0 / 20 of the **pack** (+ val-from-train) |
| Lookback leak across paper splits | Val/test lookbacks reach into previous split | N/A inside val+test pack (already paper regions) |
| Train↔test absolute overlap | Possible at paper boundaries by design | **Forbidden** (purge vs disc test) |
| Train↔val absolute overlap | Separate paper regions | **Allowed** inside purged pool |

So: “80/20” here is **not** “train the disc on the paper training set.” It is “chronologically split the MMPD-aligned val+test pack, purge anything that touches the held-out tail.”

**AUROC flag:** relative to a disc that trained on dense overlapping paper-train windows and tested on paper-test without purge, this is **intentionally harder / less leaky**. Chance AUROC can be the honest answer under purge + unique_abs; it is not automatically evidence the generative models are perfect.

---

## 6. Unique absolute slice sampling (vs old dense offsets)

Training calls **univariate** `train_classifier` (not `disc_shared.train_classifier`):

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1713-1727
for source in ("binary_staged", "mmpd"):
    for length in args.slice_lengths:
        if int(length) <= args.horizon:
            raw = train_classifier(
                args, dataset, source, int(length), bundle, splits, device,
            )
            ...
            per_len[str(int(length))] = raw
```

### Dense (old path, `--no-unique-absolute-slices`)

Cartesian product: every `(window ∈ split) × (offset ∈ 0..H-L) × (variate)` for real **and** fake. With `pack-test-stride=4` and `H=96`, neighboring windows share almost the entire horizon → the same absolute L-block is trained dozens of times under different `(window, offset)` ids. That is both slow and a classic **leakage / inflation** machine for AUROC (especially if train/test purge is weak).

### Unique abs (default)

```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:64-112
def _unique_absolute_slice_items(...):
    """One random (window, offset) per absolute L-block × variate (real+fake pair)."""
    offsets = list(range(0, horizon - slice_len + 1, max(1, offset_stride)))
    # Group all (window, offset) that cover the same absolute future start T.
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for w in windows:
        fut0 = int(starts[w]) + int(lookback)   # absolute t of horizon step 0
        for o in offsets:
            abs_t = fut0 + int(o)               # absolute t of L-slice start
            for v in range(n_var):
                groups.setdefault((abs_t, v), []).append((w, int(o)))
    rng = np.random.default_rng(int(seed))
    items = []
    for (abs_t, v), parents in groups.items():
        # Pick one parent window that covers this absolute block.
        w, o = parents[int(rng.integers(0, len(parents)))]
        items.append((w, o, v, 0))  # real
        items.append((w, o, v, 1))  # fake — same (w,o,v)
    rng.shuffle(items)
    return items
```

Same spirit as `UniquePatchSegmentDataset` for patch-refine training: collapse overlapping 96-horizons to unique absolute L-blocks.

**AUROC flag (protocol):** unique_abs removes the dense-overlap free lunch. Expect lower AUROC than the old dense path even with identical snaps. Re-running one leaf with `--no-unique-absolute-slices` (holding snap / candidate_only / bin-center fixed) is the cleanest A/B for “inflation vs true detectability.”

**AUROC flag (residual leakage, wrong direction for 0.5):** train↔val windows may still share absolute `[T,T+L)` because purge is only vs **test**. That would tend to **inflate** val metrics if anything, not explain universal ~0.5 test AUROC.

---

## 7. Dataset construction, labels, candidate_only

```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:114-275 (trimmed)
class UnivariateRealVsFakeDataset(Dataset):
    """Balanced univariate patches: label 1=fake, 0=GT. Pools all variates."""

    def __getitem__(self, idx):
        window, offset, variate, label = self.items[idx]
        src = self.fake if label == 1 else self.real
        # Shape (1, L) — one variate, candidate slice only when candidate_only.
        candidate = src[window, variate : variate + 1, offset : offset + self.slice_len]
        if self.include_past:
            past = self.past[window, variate : variate + 1]
            x = concat(norm(past), norm(candidate))  # rare on this ablation
        else:
            x = self._norm_segment(candidate, window, variate)  # (1, L)
        return x, offset, label, window, variate
```

**Labels:** `0 = GT (real)`, `1 = fake` (`binary_staged` or `mmpd`). Documented at module top; matching multivariate texture disc. **Not inverted** in the live path — an inverted-label bug would still usually produce *high* AUROC after the model learns the flipped mapping, not stable ~0.5 across runs.

**candidate_only:** `include_past = not args.candidate_only` → default **False**. Disc never sees lookback continuity; only the L-slice texture.

### Bin-center shift (replaces zscore on this path)

```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:233-250
def _norm_segment(self, segment, window, variate):
    if self.apply_bin_center_shift:
        levels = self.legal_levels[window, variate : variate + 1, :]  # (1, canvas_H)
        # Mean over *this L-slice only*; integer shift in centered bin coords;
        # remap back onto the same ladder. No std scaling.
        shifted, _ = bin_center_shift(
            seg[None, :, :], levels[None, :, :], reduce=self.bin_center_reduce,
        )
        return shifted[0]
    if self.apply_zscore:
        return zscore_time(seg)
    return seg
```

Applied identically to real and fake. Strips absolute level / bias in bin space; keeps local step pattern. `train_classifier` hard-offs zscore whenever BC is on (mutual exclusion fails fast).

**AUROC flag (protocol):** if fakes are mostly “right shape, wrong level,” bin-center + candidate_only L=8 can erase the usable cue. That is a deliberate texture-focused protocol, not a silent zscore bug. Ambiguous intent: if the research question is “can a disc detect *any* difference including bias,” turn bin-center off; if it is “local ordinal texture only,” ~0.5 may be the right answer when models match local texture.

**AUROC flag (emptying regions):** unique_abs does not drop variates or empty splits by itself; it only collapses parents. `candidate_only` does not filter by occupancy. No obvious “filter deleted all hard negatives” path in the ablation code. If `n_train` / `n_test` in partials look sane, this is not an emptying bug.

---

## 8. Model architecture — `InvertedSliceDiscriminator`

**This is the live discriminator module.** Same class as the multivariate disc (imported via `eval_discriminator_texture_staged_vs_mmpd` / `disc_shared`). Defined here:

```python
# utils/disc_shared.py:1020-1071
class InvertedSliceDiscriminator(nn.Module):
    """iTransformer-style disc: linear over time → token dim, then TransformerEncoder.

    Univariate ablation feeds C=1 → effectively one token + optional offset emb
    (fine for L=8/16). Output logit is P(fake).
    """

    def __init__(self, seq_len, max_offset, d_model, n_heads, depth, d_ff, dropout,
                 *, use_offset_embedding=True):
        super().__init__()
        # Linear over time length → token dim (variates are tokens).
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.offset_embedding = (
            nn.Embedding(max_offset + 1, d_model) if use_offset_embedding else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)  # LayerNorm only — no InstanceNorm
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, offsets):
        # x: [B, C, T] with C=1 for univariate disc → single token after embed.
        tokens = self.value_embedding(x)
        if self.offset_embedding is not None:
            tokens = tokens + self.offset_embedding(offsets).unsqueeze(1)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled).squeeze(-1)  # logit P(fake)
```

Ablation defaults: `d_model=128`, `n_heads=4`, `depth=2`, `d_ff=256`, `dropout=0.1`, offset embedding **on**, `seq_len=L` under candidate_only.

With `C=1` the transformer is effectively a deep MLP on one token plus offset — fine for L=8/16. LULL audit: tiny MLP on the same L8+BC pairs can still get ~0.93 AUROC while this disc collapses to constant ~0.5 (BCE≈ln2) — failure mode looks like **capacity/protocol mismatch / train collapse**, not wrong snap/labels.

---

## 9. Train loop and val metrics

```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:384-547 (trimmed)
def train_classifier(...):
    ...
    # Separate datasets per split; unique_abs re-run with seeds seed_base / +1 / +2
    # so train/val/test each pick their own parent when absolute blocks collide
    # *within* that split.
    ds_train = UnivariateRealVsFakeDataset(..., splits["train"], seed=seed_base, ...)
    ds_val   = UnivariateRealVsFakeDataset(..., splits["val"],   seed=seed_base + 1, ...)
    ds_test  = UnivariateRealVsFakeDataset(..., splits["test"],  seed=seed_base + 2, ...)

    model = InvertedSliceDiscriminator(seq_len=..., max_offset=H - L, ...).to(device)
    optimizer = AdamW(...)

    for epoch in range(args.epochs):          # default 20, patience 5 on val BCE
        for batch in train_loader:
            x, offsets, labels = batch[0], batch[1], batch[2]
            logits = model(x, offsets)
            loss = BCEWithLogits(logits, labels)  # label 1 = fake
            loss.backward(); clip; step
        val_metrics = evaluate_classifier(model, val_loader, device)
        # Checkpoint on best val BCE (not AUROC).
        ...
        # Logged: val_auc=disc_auroc, val_auc_win=disc_auroc_window
    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device)
```

### Metric definitions

```python
# utils/disc_shared.py:1074-1113 + univariate evaluate_classifier (L310+)
# Example-level AUROC: Mann–Whitney on P(fake) vs labels (ties averaged).
disc_auroc = binary_auroc(labels, sigmoid(logits))

# Window-level (univariate): mean P(fake) over offsets for each
# (window, variate, label), then AUROC on those averages.
# Under unique_abs there is usually ~1 offset per (window,variate) →
# disc_auroc_window ≈ disc_auroc.
#
# Also: per_variate_metrics → auroc_by_variate.json (LULL ~0.50 alone).
```

Early stopping on **val BCE**, report **test** `disc_auroc` / `disc_acc` in `auroc_table.json`. Ablation also writes classifier scores under `scores/` for disagreement viz.

---

## 10. Test AUROC reporting

```python
# temp/scripts/eval_ablation_disc_l8_l16.py:1730-1751 + main summary
# Always write disagreement panels (MMPD wrong / binary right and vice versa).
disagree_root = args.output_dir / "viz" / "disc_disagreement" / run_name
...
write_json(disagree_root / "summary.json", disagree_manifests)

# Flat table: run × source × L → disc_auroc / disc_acc
# Also: auroc_by_variate.json, partials/, lattice snap stats, viz/
```

Also written: `partials/{run}__{dataset}__{source}.json` (full metric dict per L), lattice snap stats, optional viz panels under `viz/` (zoom lattice + staged_eval red-box + disagreement).

Chance = **0.5** AUROC / ~0.5 acc / BCE → `log(2)`. Recent unique_abs campaigns sitting on ~0.50 across WN / c128 / ordinal_fine / guided_p8 and elec / traffic / exchange means: **either the protocol removed the old leakage/signal, or real≈fake after snap+bin-center on L=8/16** — see analysis below. Wrong-ladder snap is fixed; residual ~0.5 under correct `window_norm_grid` remains.

---

## End-to-end data shapes (one ETTh1 window, L=8)

```
MMPD / val+test pool row i  →  pool index I
load_tsf_pack_pool(val,test, stride=4)[I]:
    past  (V, 336) dataset-z
    future (V, 96)  dataset-z

generate_staged_forecast → pred (V, 96) dataset-z
reduce sample0 → binary fake (V, 96)
align MMPD fake → binary z (V, 96)

_legal_levels_for_run → legal_levels (V, canvas_H)
  canvas128 / window-norm → window_norm_grid (or hybrid_flat)
  ordinal leaf → ordinal_absolute
snap GT / binary / MMPD → values on those rungs

split_windows → local row ids {train, val, test} with purge (80/20)

unique_abs for L=8:
    abs_t = series_starts[row] + 336 + offset
    one (row, offset, v) per abs_t → two examples (label 0 and 1)

bin_center_shift on (1, 8) → disc input x ∈ R^{1×8}
InvertedSliceDiscriminator → logit → P(fake)
```

---

## Analysis: why unique_abs AUROC ≈ 0.5?

Honest split between **intentional protocol** and **possible bugs**. No clear one-line production bug jumped out that would force chance AUROC while leaving the rest of the pipeline consistent; the live path looks coherent. Post-handoff audits: snap alphabet bug was real and fixed; LULL disc-vs-viz is **not** a data/label bug (see `temp/lull_disc_vs_viz_audit.md`).

### Likely protocol (not bugs) — preferred explanation

1. **unique_abs removed dense-overlap inflation.** Old dense `windows × offsets` reused the same absolute L-blocks many times and (under weaker purge) could leak absolute-time identity across splits. Collapsing to unique abs blocks is meant to kill that. Expect a large drop vs historical ~0.9.

2. **candidate_only + L∈{8,16} + bin_center_shift** strips lookback context and absolute level. Disc judges short mean-centered ladder textures only. Good local texture matching → chance. (Tiny MLP can still separate some LULL pairs — disc capacity/collapse is a separate question.)

3. **Shared training-lattice snap** puts GT / binary / MMPD on one discrete alphabet. Sub-rung errors disappear. Canvas128 now correctly uses `window_norm_grid` (not ordinal absolute).

4. **Pack = paper val+test + purged 80/20** is a harder, cleaner eval than training a disc on paper-train dense windows. Chance here does **not** by itself mean “models are indistinguishable under every protocol.”

5. **MMPD also ~0.5** (if that is what you saw) strongly favors “protocol / information-free after transforms” over “binary is uniquely perfect.” MMPD should still be detectable under a leaky or level-sensitive disc if large systematic errors remain.

### Checked — do **not** look like the 0.5 cause

| Hypothesis | Verdict |
|------------|---------|
| Labels inverted (0/1 swapped) | **No** — consistently `1=fake`, `0=GT`; inversion usually still yields high AUROC after learning |
| Real/fake tensors accidentally identical | **Unlikely as a silent bug** — binary and MMPD come from different packs; snap asserts lattice membership but does not copy GT into fakes |
| Wrong snap alphabet (ordinal on canvas128) | **Was a bug; fixed** — live path uses `window_norm_grid`. Post-fix campaigns still ~0.50 |
| `candidate_only` empties discriminable regions | **No filter** that drops examples by content |
| Seed / subset empties test | Smoke `max_windows` can starve; full runs with logged `n_train`/`n_test` should be checked in partials — if counts look normal, not this |
| Window AUROC definition wrong | Example AUROC and window AUROC both use `binary_auroc`; under unique_abs they nearly coincide — would not systematically pin both at 0.50 from a definition bug alone |
| LULL hybrid / wrong `__getitem__` | **Ruled out** — disc train tensors match viz; LULL alone ~0.50; scores collapsed to ~0.5 |

### Soft / ambiguous issues (ask before “fixing”)

1. **Train↔val absolute overlap still allowed.** Would inflate val if anything. Intentional for long horizons; confirm you still want that under unique_abs.

2. **Is bin-center meant to stay on for all leaves?** It couples tightly with the lattice story. If the question is detectability including bias, try `--no-disc-bin-center-shift` (falls back to per-slice zscore in the univariate trainer).

3. **Disc capacity vs LULL texture.** MLP sees residual nonlinear signal under L8+BC that `InvertedSliceDiscriminator` fails to fit. Optional next: debug train collapse / capacity, or report H96/pre-BC detectability separately.

4. **Early-stop on val BCE** while reporting AUROC — normal, but a model that never leaves chance BCE will report chance AUROC; not a scoring bug.

5. **`sample0` only** — one stochastic draw. Unlikely to force *all* datasets/models to 0.50 unless signal is gone after snap/centering.

### Suggested A/Bs (do not submit from this doc)

Hold snap + pack + purge fixed, vary one knob:

1. `--no-unique-absolute-slices` → if AUROC jumps toward old ~0.9, the collapse was **leakage/inflation**, not “models became perfect.”
2. `--no-disc-bin-center-shift` → tests whether level/bias was the only cue.
3. `--no-candidate-only` → gives lookback continuity (much easier / different task).
4. Compare `binary_staged` vs `mmpd` on the **same** unique_abs run — both at 0.5 vs only binary at 0.5 changes the story.

---

## Refactor smells (disc only)

- Ablation script still owns a lot of snap/viz/disagreement logic; `disc_shared` multivariate `train_classifier` is a footgun (no unique_abs / bin-center) if someone imports the wrong trainer.
- Submit hardcodes `--dataset ETTh1` then relies on `"$@"` to override — easy to miss when sweeping elec/traffic/exchange.
- `binary_mmpd_train_scaler_map` loads pack_splits `("test",)` only to fetch binary train scaler stats from `load_dataset` — works if stats are always train-split, but the pack_splits arg is misleading to readers.
- `utils/patch_refine_value_grid.py` currently duplicates `snap_to_unbounded_patch_refine_grid` / `grid_coordinates` / `assert_on_patch_refine_grid` twice in the file (dead duplicate block at bottom).

---

## File map

| Path | Role |
|------|------|
| `temp/scripts/submit_ablation_disc_l8_l16.sh` | Slurm wrapper (val+test / 80/20 defaults) |
| `temp/scripts/eval_ablation_disc_l8_l16.py` | Ablation orchestrator (`_legal_levels_for_run`, `_snap_bundle`, `run_one`) |
| `utils/eval_discriminator_binary_vs_mmpd_univariate.py` | unique_abs dataset + univariate train/eval |
| `utils/disc_shared.py` | `split_windows`, AUROC, **`InvertedSliceDiscriminator`** |
| `utils/patch_refine_ordinal_ladder.py` | ordinal absolute legal levels + snap |
| `utils/patch_refine_value_grid.py` | **window-norm / canvas128 lattice** in dataset-z |
| `utils/hybrid_flat_dataset_norm.py` | flat-variate detect + skip-window-norm mask |
| `utils/disc_bin_center_shift.py` | per-L bin mean-center |
| `utils/eval_mmpd_gaussian_anchor.py` | `load_tsf_pack_pool`, series starts |
| `utils/dual_scale_bin_filter.py` | MMPD→binary scaler align |
| `utils/forecast_pack_reduce.py` | sample0 / prob_mean |
