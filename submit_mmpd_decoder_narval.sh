#!/bin/bash
# MMPD Decoder flat subsets on Narval (A100). Thin wrapper around submit_mmpd_sweep_subset.sh.
#
# USAGE (Narval login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_decoder_narval.sh --smoke-test
#   ./submit_mmpd_decoder_narval.sh
#   ./submit_mmpd_decoder_narval.sh --dataset weather
#   ./submit_mmpd_decoder_narval.sh --time 1:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.yaml"
OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-decoder-grad-accum-200-lr-lo-subset"
DEPENDENCY=""
DATASET=""
WALL_TIME="1:30:00"
TUNE_ARGS=(--mmpd-tune-trials 7 --mmpd-tune-epochs 10 --mmpd-tune-patience 3)

read_datasets_csv() {
    python3 - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${SCRIPT_DIR}/${CONFIG}").read_text())
print(",".join(cfg["mmpd"]["datasets"]))
PY
}

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke)
            EXTRA+=(--smoke-test)
            TUNE_ARGS=(--mmpd-tune-trials 2 --mmpd-tune-epochs 1 --mmpd-tune-patience 1)
            WALL_TIME="0:45:00"
            shift
            ;;
        --resume) EXTRA+=(--resume); shift ;;
        --force) EXTRA+=(--force); shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --dataset|--datasets) DATASET="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --no-tune) TUNE_ARGS=(); shift ;;
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
    --gpu a100 \
    --time "$WALL_TIME" \
    ${DEPENDENCY:+--dependency "$DEPENDENCY"} \
    "${TUNE_ARGS[@]}" \
    "${EXTRA[@]}"
