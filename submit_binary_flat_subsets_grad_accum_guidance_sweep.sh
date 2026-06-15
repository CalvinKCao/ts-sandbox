#!/bin/bash
# Flat-subset 96/96 grad-accum sweep with iTrans 2D guidance + cross-attn.
# Effective batch = probed_max × {1.5, 2.0, 4.0, 8.0}; 5 Optuna LR trials/stage.
# Reuses flat-subset pretrain + iTrans from completed baseline runs.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_grad_accum_guidance_sweep.sh
#   ./submit_binary_flat_subsets_grad_accum_guidance_sweep.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_150.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_200.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_400.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_800.yaml"
WALL_TIME="4:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_stationary_flat_subsets_grad_accum_guidance_150.yaml \
        --datasets ETTh1 \
        --time "$WALL_TIME"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME"
