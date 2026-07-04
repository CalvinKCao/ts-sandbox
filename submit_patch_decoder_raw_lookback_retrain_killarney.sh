#!/bin/bash
# Retrain ETTh1 + electricity with raw lookback visual cond channel (skip pretrain/patch-guidance HP).
#
# Donors (under results/ckpts on Killarney):
#   pretrain + patch guidance: binary_anchor_ar_patch_decoder_ctx
#   coarse/fine HP — ETTh1: binary_anchor_ar_patch_decoder_ctx_full_hp
#                       electricity: binary_anchor_ar_patch_decoder_ctx
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_patch_decoder_raw_lookback_retrain_killarney.sh --smoke-test
#   ./submit_patch_decoder_raw_lookback_retrain_killarney.sh
#   ./submit_patch_decoder_raw_lookback_retrain_killarney.sh --time 24:00:00
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_raw_lookback_retrain.yaml"
DATASETS="ETTh1,electricity"
WALL_TIME="24:00:00"
SMOKE=0
RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(--configs "$CONFIG" --gpu l40s --time "$WALL_TIME")
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --gpu l40s --time "0:45:00")
else
    ARGS+=(--datasets "$DATASETS")
fi

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

exec ./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
