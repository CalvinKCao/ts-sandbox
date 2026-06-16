#!/bin/bash
# End-to-end MMPD MaskAE flat-subset smoke: init -> tune -> train -> eval -> merge.
# One dataset (default illness: tiny train set; use --dataset ETTm1 for 4-variate kNN cap).
#
# USAGE (repo root, .venv active):
#   ./smoke_mmpd_maskae_subset.sh
#   ./smoke_mmpd_maskae_subset.sh --cpu
#   ./smoke_mmpd_maskae_subset.sh --dataset illness

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="illness"
USE_CPU=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) USE_CPU=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
fi
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

OUT="$SCRIPT_DIR/results/datasets/_smoke-mmpd-maskae-subset"
rm -rf "$OUT"

COMMON=(
    python -u utils/eval_mmpd_gaussian_anchor.py
    --smoke-test
    --subset-config configs/binary_anchor_stationary_flat_subsets.yaml
    --mmpd-only
    --mmpd-backbone MaskAE
    --output-dir "$OUT"
    --mmpd-repo "$SCRIPT_DIR/temp/MMPD"
    --mmpd-data-dir "$SCRIPT_DIR/temp/mmpd_datasets"
    --lookback 96
    --horizon 96
    --seed 2026
    --datasets "$DATASET"
)
if [[ "$USE_CPU" -eq 1 ]]; then
    COMMON+=(--cpu)
fi

echo "=== [1/3] init ==="
"${COMMON[@]}" --phase init

echo "=== [2/3] mmpd (tune + train + eval) ==="
"${COMMON[@]}" --phase mmpd

echo "=== [3/3] merge ==="
"${COMMON[@]}" --phase merge

echo "OK: smoke passed"
echo "  metrics: $OUT/metrics.json"
echo "  partial: $OUT/partials/${DATASET}_mmpd.json"
test -f "$OUT/partials/${DATASET}_mmpd.json"
test -f "$OUT/metrics.json"
test -f "$OUT/tuning/${DATASET}_best.json"
