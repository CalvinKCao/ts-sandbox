#!/bin/bash
# Discriminator texture: past-native ordinal binary (jobs 4208596–4208599) vs
# paper MMPD Decoder lb336/hz720 on ETTh1, traffic, electricity, exchange_rate.
#
# Preprocessing (disc only; raw packs reused):
#   - binary: no Gaussian lattice jitter
#   - GT + MMPD: snap each timestep to nearest global ordinal ladder rung (binary bins)
# Confusion plots show lookback + GT + model pred for TP/TN/FP/FN.
#
# Reuses compare-report MMPD packs when present (must contain samples):
#   reports/binary_vs_mmpd_lb336_hz720_ordinal_four_f0p125/raw/mmpd_*.npz
# Binary probabilistic packs are regenerated under the disc raw-run dir.
#
# Prereq (once): ./temp/scripts/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --apply
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_killarney.sh --smoke-test
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_killarney.sh
#   ./temp/scripts/submit_discriminator_texture_lb336_hz720_ordinal_four_killarney.sh --dataset ETTh1 --fake-source mmpd
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

# Opt into ordinal-campaign disc preprocessing (shared launcher defaults leave these off).
export DISC_NO_BINARY_DEBIAS=1
export DISC_ORDINAL_LADDER_QUANTIZE=1

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
    --disc-run disc-lb336-hz720-ordinal-four-gtbin \
    --raw-run disc-lb336-hz720-ordinal-four-raw \
    --report reports/disc_texture_lb336_hz720_ordinal_four_gtbin.md \
    --import-mmpd-packs-from reports/binary_vs_mmpd_lb336_hz720_ordinal_four_f0p125 \
    --time 8:00:00 \
    "$@"
