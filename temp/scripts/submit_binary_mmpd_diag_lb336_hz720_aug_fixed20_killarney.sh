#!/bin/bash
# Binary vs MMPD high-|error-diff| window plots for the aug_fixed20 vertical_dual
# campaign (jobs 4241374–4241377) vs the 07-10 paper MMPD Decoder campaign.
#
# Binary: configs/..._per_ds_best_g_aug_fixed20 (vertical_dual, ordinal, lb336/hz720)
# MMPD:   mmpd_decoder_flat_subsets_paper_lb336_hz720
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   git pull
#   ./temp/scripts/submit_binary_mmpd_diag_lb336_hz720_aug_fixed20_killarney.sh --smoke-test
#   ./temp/scripts/submit_binary_mmpd_diag_lb336_hz720_aug_fixed20_killarney.sh
#   ./temp/scripts/submit_binary_mmpd_diag_lb336_hz720_aug_fixed20_killarney.sh --test-fraction 0.125
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

STEM="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_per_ds_best_g_aug_fixed20"
CFG="configs/${STEM}.yaml"

SMOKE=0
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

if [[ "$SMOKE" -eq 1 ]]; then
    DATASETS="ETTh1"
else
    DATASETS="ETTh1,traffic,electricity,exchange_rate"
fi

ARGS=(
    --binary-config "$CFG"
    --binary-ckpt-stem "$STEM"
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720.yaml
    --mmpd-config-suffix mmpd_decoder_flat_subsets_paper_lb336_hz720
    --mmpd-dir results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset
    --datasets "$DATASETS"
    --output-dir reports/binary_vs_mmpd_lb336_hz720_aug_fixed20
    --force-eval
    --force-mmpd-eval
    --time 4:00:00
)
[[ "$SMOKE" -eq 1 ]] && ARGS+=(--smoke-test)
[[ ${#PASS[@]} -gt 0 ]] && ARGS+=("${PASS[@]}")

exec "$REPO_ROOT/submit_binary_mmpd_staged_diag_killarney.sh" "${ARGS[@]}"
