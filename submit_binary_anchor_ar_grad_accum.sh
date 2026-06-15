#!/bin/bash
# AR grad-accum sweep: reuse ema099 pretrain; 4 Optuna LR trials/stage on iTrans + diffusion.
# Effective batch = probed_max × {4, 8, 16}.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_anchor_ar_grad_accum.sh
#   ./submit_binary_anchor_ar_grad_accum.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_ar_grad_accum_400.yaml,configs/binary_anchor_ar_grad_accum_800.yaml,configs/binary_anchor_ar_grad_accum_1600.yaml"
WALL_TIME="16:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_ar_grad_accum_400.yaml \
        --datasets ETTh1 \
        --time "$WALL_TIME"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME"
