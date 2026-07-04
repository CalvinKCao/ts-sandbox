#!/bin/bash
# Healthy window-norm retrain + reduced diffusion HP search (LR + effective batch + EMA preset).
#
# Reuses pretrain + patch-guidance from binary_anchor_ar_patch_decoder_ctx.
# Coarse/fine HP: search_space=reduced_hp (4 trials each).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_patch_decoder_healthy_norm_reduced_hp_killarney.sh --smoke-test
#   ./submit_patch_decoder_healthy_norm_reduced_hp_killarney.sh
#   ./submit_patch_decoder_healthy_norm_reduced_hp_killarney.sh --datasets dynamic,electricity
#   ./submit_patch_decoder_healthy_norm_reduced_hp_killarney.sh --resume
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_healthy_norm_reduced_hp.yaml"
DATASETS="dynamic,electricity"
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

ARGS=(--configs "$CONFIG" --gpu l40s --time "$WALL_TIME" --wandb-project ts-sandbox-leaderboard)
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets dynamic --gpu l40s --time "0:45:00" --wandb-project ts-sandbox-leaderboard)
else
    ARGS+=(--datasets "$DATASETS")
fi

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

exec ./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
