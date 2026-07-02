#!/bin/bash
# Local end-to-end MMPD Decoder (subset YAML, Optuna tune + train + eval).
#
#   ./run_mmpd_decoder_local.sh
#   ./run_mmpd_decoder_local.sh --smoke-test
#   ./run_mmpd_decoder_local.sh --dataset weather
#   ./run_mmpd_decoder_local.sh --resume --output-dir results/datasets/07-01-mmpd-decoder-local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/mmpd_decoder_flat_subsets_grad_accum_200_lr_lo.yaml"
OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-decoder-local"
RESUME=0
DATASET=""

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) EXTRA+=(--smoke-test); shift ;;
        --resume) RESUME=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dataset|--datasets) DATASET="$2"; shift 2 ;;
        --force-mmpd-tune) EXTRA+=(--force-mmpd-tune); shift ;;
        --force-mmpd-train) EXTRA+=(--force-mmpd-train); shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -d temp/MMPD/.git ]]; then
    echo "[setup] Cloning MMPD into temp/MMPD"
    mkdir -p temp
    git clone https://github.com/Thinklab-SJTU/MMPD.git temp/MMPD
fi

source .venv/bin/activate
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_ARGS=()
if [[ -n "$DATASET" ]]; then
    DATASET_ARGS=(--datasets "$DATASET")
fi

if [[ "$RESUME" -eq 0 ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "[run] config=$CONFIG output=$OUTPUT_DIR datasets=${DATASET:-all from yaml}"

exec python -u utils/eval_mmpd_gaussian_anchor.py \
    --phase all \
    --mmpd-run-config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --no-update-mmpd \
    --metrics-profile anchor-compat \
    "${DATASET_ARGS[@]}" \
    "${EXTRA[@]}"
