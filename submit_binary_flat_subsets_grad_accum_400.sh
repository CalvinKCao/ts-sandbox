#!/bin/bash
# Flat-subset LB96/H96, 4x grad accum, NO guidance channel (use_guidance_channel=false).
# Contrast with grad_accum_guidance_400 (iTrans 2D ghost + cross-attn).
# Reuses ema099 pretrain + iTrans; 4 Optuna LR trials/stage.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_grad_accum_400.sh --smoke
#   ./submit_binary_flat_subsets_grad_accum_400.sh
#   ./submit_binary_flat_subsets_grad_accum_400.sh ETTh1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_400.yaml"
WALL_TIME="4:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs "$CONFIG" \
        --datasets ETTh1 \
        --time "$WALL_TIME"
    exit 0
fi

DATASETS="${1:-ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama}"

./submit_grid.sh \
    --configs "$CONFIG" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME"
