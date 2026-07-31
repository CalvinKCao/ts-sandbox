#!/usr/bin/env bash
# Optional helper: pull packs from Killarney then run the lattice-snap viz (CPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REMOTE="${KILLARNEY_HOST:-killarney}"
SCRATCH_REPO="${KILLARNEY_TS_SANDBOX:-/scratch/ccao87/ts-sandbox}"

DISC_RAW="results/datasets/07-31-0925-h96-ordinal-disc-raw"
MMPD_RAW="results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw"
OUT="results/pulled/h96-disc-input-lattice-snap"

mkdir -p "$DISC_RAW" "$MMPD_RAW" "$OUT"

if [[ ! -f "$DISC_RAW/binary_ordinal_patch_refine_ETTh1.npz" ]]; then
  echo "[rsync] disc-raw packs"
  rsync -avz "$REMOTE:$SCRATCH_REPO/$DISC_RAW/" "$DISC_RAW/"
fi
if [[ ! -f "$MMPD_RAW/mmpd_ETTh1.npz" ]]; then
  echo "[rsync] MMPD raw packs"
  rsync -avz "$REMOTE:$SCRATCH_REPO/$MMPD_RAW/" "$MMPD_RAW/"
fi

# Metadata only (no .pt weights) for ladder/scaler run objects.
for spec in \
  "4462979:electricity:electricity_4v_s1" \
  "4462980:ETTh1:ETTh1" \
  "4462981:dynamic:dynamic_2v_s480" \
  "4462982:traffic:traffic_4v_s1"
do
  IFS=: read -r id ds sub <<<"$spec"
  root="results/ckpts/07-29-${id}-${ds}-binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback"
  meta="$root/$sub/patch_refine/metadata.json"
  if [[ ! -f "$meta" ]]; then
    mkdir -p "$root/$sub/coarse" "$root/$sub/patch_refine"
    rsync -avz "$REMOTE:$SCRATCH_REPO/$root/wandb_manifest.json" "$root/" || true
    rsync -avz "$REMOTE:$SCRATCH_REPO/$root/$sub/coarse/metadata.json" "$root/$sub/coarse/"
    rsync -avz "$REMOTE:$SCRATCH_REPO/$root/$sub/patch_refine/metadata.json" "$root/$sub/patch_refine/"
  fi
done

python temp/scripts/viz_disc_input_gt_mmpd_lattice_snap.py \
  --cpu \
  --disc-raw-dir "$DISC_RAW" \
  --mmpd-output-root "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd" \
  --output-dir "$OUT" \
  "$@"
