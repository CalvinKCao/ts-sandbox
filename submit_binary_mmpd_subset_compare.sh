#!/bin/bash
# Apples-to-apples binary flat vs MMPD on ETTh1-capped variate subsets (all 7 sweep datasets).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_mmpd_subset_compare.sh binary          # step 1: train binary (all 7)
#   ./submit_binary_mmpd_subset_compare.sh mmpd          # step 2: MMPD eval (after binary ckpts exist)
#   ./submit_binary_mmpd_subset_compare.sh smoke-binary    # quick binary smoke (ETTh1)
#   ./submit_binary_mmpd_subset_compare.sh smoke-mmpd      # quick MMPD smoke (ETTh1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets.yaml"
ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets"
DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
WANDB_PROJECT="ts-sandbox-binary-mmpd-subset-compare"

usage() {
    echo "Usage: $0 {binary|mmpd|smoke-binary|smoke-mmpd}" >&2
    exit 1
}

[[ $# -eq 1 ]] || usage

case "$1" in
    binary)
        ./submit_grid.sh \
            --configs "$CONFIG" \
            --datasets "$DATASETS" \
            --wandb-project "$WANDB_PROJECT"
        echo ""
        echo "When binary jobs finish, run:"
        echo "  ./submit_binary_mmpd_subset_compare.sh mmpd"
        ;;
    mmpd)
        ./submit_mmpd_sweep_subset.sh \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets "$DATASETS" \
            --output-dir "results/datasets/$(date +%m-%d)-binary-mmpd-subset-compare"
        ;;
    smoke-binary)
        ./submit_grid.sh --smoke \
            --configs "$CONFIG" \
            --datasets ETTh1 \
            --wandb-project "${WANDB_PROJECT}-smoke"
        ;;
    smoke-mmpd)
        ./submit_mmpd_sweep_subset.sh --smoke \
            --anchor-config "$ANCHOR_CONFIG" \
            --datasets ETTh1
        ;;
    *)
        usage
        ;;
esac
