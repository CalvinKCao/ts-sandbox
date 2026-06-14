#!/bin/bash
# EMA decay sweep on flat-subset binary runs: reuse synthetic pretrain + iTrans,
# re-run diffusion coarse/fine/finer finetune with fixed diffusion_ema_decay.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_ema_sweep.sh
#   ./submit_binary_flat_subsets_ema_sweep.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIG_GLOB="configs/binary_anchor_stationary_flat_subsets_ema_sweep_0*.yaml"
WANDB_PROJECT="ts-sandbox-flat-subsets-ema-sweep"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_stationary_flat_subsets_ema_sweep_098.yaml \
        --datasets ETTh1 \
        --wandb-project "${WANDB_PROJECT}-smoke"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIG_GLOB" \
    --datasets "$DATASETS" \
    --wandb-project "$WANDB_PROJECT"
