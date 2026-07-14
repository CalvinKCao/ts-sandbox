#!/bin/bash
# Fair univariate real-vs-fake disc with 16x16 dual-scale bin-match (not ordinal ladder).
#
# Snap GT + binary + MMPD through encode_dual/decode_dual (H=16 coarse/fine) so
# MMPD lives on the same lattice as decoded binary preds.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_discriminator_binary_vs_mmpd_univariate_fair_killarney.sh --smoke-test
#   ./temp/submit_discriminator_binary_vs_mmpd_univariate_fair_killarney.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BINARY_STEMS="ETTh1:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native,traffic:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5,electricity:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0,exchange_rate:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0"
BINARY_CONFIGS="ETTh1:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml,traffic:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5.yaml,electricity:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0.yaml,exchange_rate:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0.yaml"

exec "$REPO_ROOT/slurm_discriminator_binary_vs_mmpd_univariate.sh" \
    --datasets ETTh1,traffic,electricity,exchange_rate \
    --fake-sources binary_staged,mmpd \
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
    --force-train \
    --bin-match-filter all \
    --disc-run disc-lb336-hz720-ordinal-four-patch-only-fair-univariate-bin16 \
    --raw-run disc-lb336-hz720-ordinal-four-raw-trainval25 \
    --time 2:00:00 \
    "$@"
