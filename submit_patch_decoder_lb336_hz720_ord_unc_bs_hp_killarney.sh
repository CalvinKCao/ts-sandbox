#!/bin/bash
# Uncompressed lb336/hz720 ordinal: LR + univariate effective-batch HP (3 grids).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_patch_decoder_lb336_hz720_ord_unc_bs_hp_killarney.sh --tier small --smoke-test --datasets ETTh1
#   ./submit_patch_decoder_lb336_hz720_ord_unc_bs_hp_killarney.sh --tier small --datasets ETTh1,exchange_rate,traffic
#   ./submit_patch_decoder_lb336_hz720_ord_unc_bs_hp_killarney.sh --tier mid --datasets ETTh1,exchange_rate,traffic
#   ./submit_patch_decoder_lb336_hz720_ord_unc_bs_hp_killarney.sh --tier xlarge --datasets ETTh1,exchange_rate,traffic
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIER=""
DATASETS="ETTh1,exchange_rate,traffic"
WALL_TIME="12:00:00"
SMOKE=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$TIER" in
    small)
        CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_small.yaml"
        ;;
    mid)
        CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid.yaml"
        ;;
    xlarge|xl|large)
        CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_xlarge.yaml"
        ;;
    *)
        echo "ERROR: --tier required: small | mid | xlarge" >&2
        exit 1
        ;;
esac

if [[ "$SMOKE" -eq 1 ]]; then
    if [[ "$TIER" != "small" ]]; then
        echo "ERROR: --smoke-test only supported with --tier small (uses bs_small_smoke.yaml)" >&2
        exit 1
    fi
    CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_small_smoke.yaml"
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --gpu l40s --time "0:45:00" --wandb-project ts-sandbox-leaderboard)
else
    ARGS=(--configs "$CONFIG" --datasets "$DATASETS" --gpu l40s --time "$WALL_TIME" --wandb-project ts-sandbox-leaderboard)
fi
[[ "$RESUME" -eq 1 ]] && ARGS+=(--resume)

echo "Submitting tier=$TIER config=$CONFIG"
exec ./test_submit.sh "${ARGS[@]}"
