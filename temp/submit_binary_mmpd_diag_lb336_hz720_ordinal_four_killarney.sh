#!/bin/bash
# Binary vs MMPD diag for the lb336/hz720 past-native ordinal binary grid four
# (jobs 4208596–4208599) against the 07-10 paper MMPD Decoder campaign.
#
# Binary: ordinal past_native per-dataset stems (ETTh1, traffic, electricity, exchange_rate)
# MMPD:  non-ordinal mmpd_decoder_flat_subsets_paper_lb336_hz720
#        (campaign 07-10-mmpd-decoder-paper-lb336-hz720-subset)
#
# Prereq: ./temp/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --apply
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh --smoke-test
#   ./temp/submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh
#   ./temp/submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh --test-fraction 1.0
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BINARY_CONFIGS="ETTh1:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml,traffic:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5.yaml,electricity:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0.yaml,exchange_rate:configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0.yaml"
BINARY_STEMS="ETTh1:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native,traffic:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5,electricity:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0,exchange_rate:binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0"

exec "$REPO_ROOT/submit_binary_mmpd_staged_diag_killarney.sh" \
    --binary-config configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml \
    --binary-config-by-dataset "$BINARY_CONFIGS" \
    --binary-ckpt-stem binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native \
    --binary-ckpt-stem-by-dataset "$BINARY_STEMS" \
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720.yaml \
    --mmpd-config-suffix mmpd_decoder_flat_subsets_paper_lb336_hz720 \
    --mmpd-dir results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset \
    --datasets ETTh1,traffic,electricity,exchange_rate \
    --output-dir reports/binary_vs_mmpd_lb336_hz720_ordinal_four \
    --force-eval \
    --force-mmpd-eval \
    --time 4:00:00 \
    "$@"
