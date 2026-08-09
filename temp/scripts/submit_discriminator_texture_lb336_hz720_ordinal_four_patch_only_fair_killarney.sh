#!/bin/bash
# Fair patch-only discriminator campaign (leakage fixes).
#
# - Pool: >=25% of original TSF train+val windows (not the old 12.5% test subsample)
# - Snap GT + MMPD + binary fakes onto the ordinal ladder
# - Hard non-overlapping temporal split (no relaxed fallback)
# - Non-overlapping L patches + no offset embedding
# - Window-level headline metrics
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_patch_only_fair_killarney.sh --smoke-test
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_patch_only_fair_killarney.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

export DISC_NO_BINARY_DEBIAS=1
export DISC_ORDINAL_LADDER_QUANTIZE=1
export DISC_CANDIDATE_ONLY=1
export DISC_NONOVERLAPPING_PATCHES=1
export DISC_NO_OFFSET_EMBEDDING=1

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
    --pack-splits train,val \
    --pack-fraction 0.25 \
    --force-raw-eval \
    --force-train \
    --disc-run disc-lb336-hz720-ordinal-four-patch-only-fair \
    --raw-run disc-lb336-hz720-ordinal-four-raw-trainval25 \
    --report reports/disc_texture_lb336_hz720_ordinal_four_patch_only_fair.md \
    --time 1:00:00 \
    "$@"
