#!/bin/bash
# Fair MMPD MaskAE rerun: all 13 datasets, pipeline-aligned splits/test windows,
# anchor_mse + CRPS (20 samples), Optuna 20 trials with EMA decay tuning.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_maskae_fair_13d.sh --smoke-test
#   ./submit_mmpd_maskae_fair_13d.sh
#   ./submit_mmpd_maskae_fair_13d.sh --resume \
#       --output-dir results/datasets/06-17-mmpd-maskae-fair-13d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUBSET_CONFIG="configs/binary_anchor_stationary_flat_subsets.yaml"
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia,dynamic"
OUTPUT_DIR=""
DEPENDENCY=""
TUNE_ARGS=(--mmpd-tune-trials 20 --mmpd-tune-epochs 10 --mmpd-tune-patience 3 --time 8:00:00)

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
        --subset-config) SUBSET_CONFIG="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --no-tune) TUNE_ARGS=(--time 3:00:00); shift ;;
        --skip-mmpd-train) EXTRA+=(--skip-mmpd-train); shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-maskae-fair-13d"
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
