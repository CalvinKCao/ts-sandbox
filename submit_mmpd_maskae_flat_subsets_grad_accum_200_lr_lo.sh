#!/bin/bash
# MMPD MaskAE on flat subsets aligned with grad-accum-200-lr-lo binary ckpts.
# Runs 5 Optuna trials/dataset (lr, point_weight, dropout; + TC depth/kNN for MaskAE)
# then full 20-epoch train with the best validation loss config.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh --smoke-test
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_200_lr_lo.sh --resume \
#       --output-dir results/datasets/06-15-mmpd-maskae-grad-accum-200-lr-lo-subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets_grad_accum_200_lr_lo"
DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-maskae-grad-accum-200-lr-lo-subset"
TUNE_ARGS=(--mmpd-tune-trials 5 --mmpd-tune-epochs 10 --mmpd-tune-patience 3 --time 8:00:00)

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke)
            EXTRA+=(--smoke-test)
            TUNE_ARGS=(--mmpd-tune-trials 2 --mmpd-tune-epochs 1 --mmpd-tune-patience 1 --time 1:30:00)
            shift
            ;;
        --resume) EXTRA+=(--resume); shift ;;
        --force) EXTRA+=(--force); shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no-tune) TUNE_ARGS=(--time 3:00:00); shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

exec ./submit_mmpd_sweep_subset.sh \
    --anchor-config "$ANCHOR_CONFIG" \
    --mmpd-backbone MaskAE \
    --datasets "$DATASETS" \
    --lookback 96 \
    --horizon 96 \
    --output-dir "$OUTPUT_DIR" \
    "${TUNE_ARGS[@]}" \
    "${EXTRA[@]}"
