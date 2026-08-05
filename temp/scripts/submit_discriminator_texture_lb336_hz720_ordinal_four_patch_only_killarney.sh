#!/bin/bash
# Patch-only discriminator: no lookback continuity — z-scored horizon slices only.
# Same ordinal-four ckpts/packs as the main disc campaign.
#
# Preprocessing: GT + MMPD snapped to binary global ordinal ladder; no binary jitter.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_patch_only_killarney.sh --smoke-test
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_patch_only_killarney.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

export DISC_NO_BINARY_DEBIAS=1
export DISC_ORDINAL_LADDER_QUANTIZE=1
export DISC_CANDIDATE_ONLY=1

BINARY_STEMS="ETTh1:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native,traffic:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5,electricity:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0,exchange_rate:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0"
BINARY_CONFIGS="ETTh1:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml,traffic:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5.yaml,electricity:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0.yaml,exchange_rate:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0.yaml"

exec "$REPO_ROOT/slurm_discriminator_texture_staged_vs_mmpd.sh" \
    --datasets ETTh1,traffic,electricity,exchange_rate \
    --anchor-config binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native \
    --anchor-config-by-dataset "$BINARY_STEMS" \
    --binary-config configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml \
    --binary-config-by-dataset "$BINARY_CONFIGS" \
    --mmpd-run 07-10-mmpd-decoder-paper-lb336-hz720-subset \
    --mmpd-backbone Decoder \
    --lookback 336 \
    --horizon 720 \
    --test-stride 1 \
    --test-fraction 0.125 \
    --disc-run disc-lb336-hz720-ordinal-four-patch-only-gtbin \
    --raw-run disc-lb336-hz720-ordinal-four-raw \
    --report reports/disc_texture_lb336_hz720_ordinal_four_patch_only_gtbin.md \
    --import-mmpd-packs-from reports/binary_vs_mmpd_lb336_hz720_ordinal_four_f0p125 \
    --time 8:00:00 \
    "$@"
