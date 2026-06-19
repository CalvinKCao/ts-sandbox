#!/bin/bash
# Flat-subset LB96/HZ96 with lookback_overlap=0 (contiguous past→future, MMPD-style).
# Baseline grad_accum_150_lr_lo uses overlap=8.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_no_overlap.sh
#   ./submit_binary_flat_subsets_no_overlap.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_no_overlap.yaml"
DATASETS="ETTh1,ETTm1,electricity,exchange_rate,weather"
WALL_TIME="8:00:00"
EXCLUDE_NODES="kn001"

if [[ "${1:-}" == "--smoke" || "${1:-}" == "--smoke-test" ]]; then
    ./submit_grid.sh --smoke \
        --configs "$CONFIG" \
        --datasets ETTh1 \
        --time "0:45:00" \
        --exclude-nodes "$EXCLUDE_NODES"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIG" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME" \
    --exclude-nodes "$EXCLUDE_NODES"
