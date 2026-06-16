#!/bin/bash
# Full-pipeline flat-subset run: accum1.5x LR-hi HP + iTrans 2D guidance ghost.
# Donor HP/checkpoints: binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_grad_accum_150_lr_hi_guidance.sh
#   ./submit_binary_flat_subsets_grad_accum_150_lr_hi_guidance.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi_guidance.yaml"
DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
WALL_TIME="8:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs "$CONFIG" \
        --datasets ETTh1 \
        --time "$WALL_TIME"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIG" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME"
