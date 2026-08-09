#!/bin/bash
# =============================================================================
# bce_dist follow-up ablations: geometry Optuna + flat BCE + guidance-cond 3x336
# Datasets: ETTh1, traffic, exchange_rate
#
# USAGE (Killarney login, repo = $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/scripts/submit_bce_dist_ablations_killarney.sh geo
#   ./temp/scripts/submit_bce_dist_ablations_killarney.sh fixed
#   ./temp/scripts/submit_bce_dist_ablations_killarney.sh all
#   ./temp/scripts/submit_bce_dist_ablations_killarney.sh geo --smoke-test
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "$REPO"

MODE="${1:-all}"
shift || true

DATASETS="${DATASETS:-ETTh1,traffic,exchange_rate}"

GEO_CONFIGS="$(
  paste -sd, - <<'EOF'
binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_geo_lb96_hz96
binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_geo_lb336_hz96
binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_geo_lb336_hz336
EOF
)"

FIXED_CONFIGS="$(
  paste -sd, - <<'EOF'
binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_flat_pixel_bce
binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_guidance_cond_3x336
EOF
)"

submit_group() {
  local label="$1"
  local configs="$2"
  local wall="$3"
  shift 3
  echo "Submitting ${label}: configs=${configs} datasets=${DATASETS} time=${wall}"
  ./submit_binary.sh \
    --configs "$configs" \
    --datasets "$DATASETS" \
    --time "$wall" \
    "$@"
}

case "$MODE" in
  geo)
    # Fresh pretrain + patch guidance + 16-trial Hyperband + 20ep refit
    submit_group geo "$GEO_CONFIGS" "1-12:00:00" "$@"
    ;;
  fixed)
    # Reuse decoder (and g1p0 pretrain for flat); fixed HPs; 20ep finetune
    # guidance_cond also does fresh synth pretrain in-run
    submit_group fixed "$FIXED_CONFIGS" "0-12:00:00" "$@"
    ;;
  all)
    submit_group geo "$GEO_CONFIGS" "1-12:00:00" "$@"
    submit_group fixed "$FIXED_CONFIGS" "0-12:00:00" "$@"
    ;;
  *)
    echo "Usage: $0 {geo|fixed|all} [--smoke-test ...]" >&2
    exit 1
    ;;
esac
