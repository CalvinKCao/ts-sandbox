#!/bin/bash
# Re-run binary vs MMPD diag for the lb336/hz720 ordinal-norm grid four after fixing
# the MMPD OOD-shift leak (models/diffusion_tsf/ordinal_window_norm.py +
# utils/mmpd_patches/exp/exp_forecast.py + utils/eval_mmpd_gaussian_anchor.py).
#
# Bug: shift_window_to_ordinal_envelope() computed the OOD-shift magnitude from
# past+future combined, so MMPD's eval-time prediction decode got de-shifted by an
# amount that leaked the true future's extremity. Fix adds causal_only=True at the
# MMPD eval call sites so the shift is computed from the lookback alone (matching
# how the binary model's own generate() already works — it never sees the future).
# No retraining needed; only eval-time decode/conditioning changes. Writes to a
# fresh output dir so before/after plots + dataset_summary.csv are easy to diff
# against reports/binary_vs_mmpd_lb336_hz720_ordinal_four_f0p125.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_mmpd_causal_shift_reeval_lb336_hz720_ordinal_four_killarney.sh --smoke-test
#   ./temp/submit_mmpd_causal_shift_reeval_lb336_hz720_ordinal_four_killarney.sh
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
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml \
    --mmpd-config-suffix mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm \
    --mmpd-dir results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset \
    --datasets ETTh1,traffic,electricity,exchange_rate \
    --output-dir reports/binary_vs_mmpd_lb336_hz720_ordinal_four_causal_shift_fix \
    --force-eval \
    --force-mmpd-eval \
    --time 4:00:00 \
    "$@"
