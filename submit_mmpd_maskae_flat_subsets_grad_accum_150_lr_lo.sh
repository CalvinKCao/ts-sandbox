#!/bin/bash
# MMPD MaskAE on flat subsets from YAML data_subset policy (no binary ckpts required).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh --smoke-test
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh --datasets ETTm1,illness
#   ./submit_mmpd_maskae_flat_subsets_grad_accum_150_lr_lo.sh --resume \
#       --output-dir results/datasets/06-16-mmpd-maskae-grad-accum-150-lr-lo-subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUBSET_CONFIG="configs/binary_anchor_stationary_flat_subsets.yaml"
DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
OUTPUT_DIR=""
DEPENDENCY=""
TUNE_ARGS=(--mmpd-tune-trials 7 --mmpd-tune-epochs 10 --mmpd-tune-patience 3 --time 8:00:00)

EXTRA=()
RESUME=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke)
            EXTRA+=(--smoke-test)
            TUNE_ARGS=(--mmpd-tune-trials 2 --mmpd-tune-epochs 1 --mmpd-tune-patience 1 --time 1:30:00)
            shift
            ;;
        --resume) RESUME=1; EXTRA+=(--resume); shift ;;
        --force) EXTRA+=(--force); shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --subset-config) SUBSET_CONFIG="$2"; shift 2 ;;
        --no-tune) TUNE_ARGS=(--time 3:00:00); shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-maskae-grad-accum-150-lr-lo-subset"
fi
if [[ "$RESUME" -eq 1 && ! -d "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="results/datasets/$(basename "$OUTPUT_DIR")"
fi
if [[ "$RESUME" -eq 1 && ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: --resume requires an existing --output-dir (got: $OUTPUT_DIR)" >&2
    exit 1
fi

exec ./submit_mmpd_sweep_subset.sh \
    --mmpd-backbone MaskAE \
    --datasets "$DATASETS" \
    --subset-config "$SUBSET_CONFIG" \
    --lookback 96 \
    --horizon 96 \
    --output-dir "$OUTPUT_DIR" \
    ${DEPENDENCY:+--dependency "$DEPENDENCY"} \
    "${TUNE_ARGS[@]}" \
    "${EXTRA[@]}"
