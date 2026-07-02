#!/bin/bash
# MMPD Decoder on flat subsets from YAML data_subset policy (no binary ckpts required).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.sh --smoke-test
#   ./submit_mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.sh
#   ./submit_mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.sh --dataset weather
#   ./submit_mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.sh --resume \
#       --output-dir results/datasets/06-30-mmpd-decoder-grad-accum-200-lr-lo-subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.yaml"
OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-decoder-grad-accum-200-lr-lo-subset"
DEPENDENCY=""
DATASET=""
TUNE_ARGS=(--mmpd-tune-trials 7 --mmpd-tune-epochs 10 --mmpd-tune-patience 3 --time 0:45:00)

# shellcheck source=utils/mmpd_submit_helpers.sh
source "${SCRIPT_DIR}/utils/mmpd_submit_helpers.sh"

read_datasets_csv() {
    read_mmpd_yaml_datasets "${SCRIPT_DIR}/${CONFIG}"
}

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke)
            EXTRA+=(--smoke-test)
            TUNE_ARGS=(--mmpd-tune-trials 2 --mmpd-tune-epochs 1 --mmpd-tune-patience 1 --time 0:45:00)
            shift
            ;;
        --resume) EXTRA+=(--resume); shift ;;
        --force) EXTRA+=(--force); shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --datasets) DATASET="$2"; shift 2 ;;
        --no-tune) TUNE_ARGS=(--time 0:45:00); shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ -n "$DATASET" ]]; then
    DATASETS_CSV="$DATASET"
else
    DATASETS_CSV="$(read_datasets_csv)"
fi

exec ./submit_mmpd_sweep_subset.sh \
    --mmpd-run-config "$CONFIG" \
    --datasets "$DATASETS_CSV" \
    --output-dir "$OUTPUT_DIR" \
    ${DEPENDENCY:+--dependency "$DEPENDENCY"} \
    "${TUNE_ARGS[@]}" \
    "${EXTRA[@]}"
