#!/bin/bash
# Cosine LR + warmup sweep (sweep_baseline-fixed diffusion HPs).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_hp_lr_cosine_warmup.sh
#   ./submit_hp_lr_cosine_warmup.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS="configs/sweep/hp_lr_cosine_warmup2.yaml,configs/sweep/hp_lr_cosine_warmup5.yaml"
DATASETS="ETTh1,ETTm1,exchange_rate,weather"
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-anchor-92d3}"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/sweep/hp_lr_cosine_warmup2.yaml \
        --datasets ETTh1 \
        --wandb-project "${WANDB_PROJECT}-cosine-warmup-smoke"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --wandb-project "$WANDB_PROJECT"
