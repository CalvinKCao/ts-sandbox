#!/bin/bash
# MMPD Decoder paper-fixed hparams, lb336/hz96, flat subsets — parallel Slurm sweep.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd_decoder_flat_subsets_paper_lb336_hz96.sh
#   ./submit_mmpd_decoder_flat_subsets_paper_lb336_hz96.sh --smoke-test
#   ./submit_mmpd_decoder_flat_subsets_paper_lb336_hz96.sh --dataset ETTh1
#   ./submit_mmpd_decoder_flat_subsets_paper_lb336_hz96.sh --resume \
#       --output-dir results/datasets/07-04-mmpd-decoder-paper-lb336-hz96-subset
#
# Leaderboard: config sets mmpd.leaderboard: true → one mmpd_eval row per dataset
# in ts-sandbox-leaderboard (needs WANDB_API_KEY in repo .env or env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/mmpd_decoder_flat_subsets_paper_lb336_hz96.yaml"
OUTPUT_DIR="results/datasets/$(date +%m-%d)-mmpd-decoder-paper-lb336-hz96-subset"
DEPENDENCY=""
DATASET=""
WALL_TIME="6:00:00"
EXTRA=()

# shellcheck source=utils/mmpd_submit_helpers.sh
source "${SCRIPT_DIR}/utils/mmpd_submit_helpers.sh"

read_datasets_csv() {
    read_mmpd_yaml_datasets "${SCRIPT_DIR}/${CONFIG}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke)
            EXTRA+=(--smoke-test)
            WALL_TIME="0:45:00"
            shift
            ;;
        --resume) EXTRA+=(--resume); shift ;;
        --force) EXTRA+=(--force); shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --datasets) DATASET="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
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
    --gpu l40s \
    --time "$WALL_TIME" \
    --mmpd-tune-trials 0 \
    ${DEPENDENCY:+--dependency "$DEPENDENCY"} \
    "${EXTRA[@]}"
