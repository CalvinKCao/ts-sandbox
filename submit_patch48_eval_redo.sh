#!/bin/bash
# =============================================================================
# Re-run patch48 eval only (jobs 3844450–3844457 train ckpts).
#
# Uses eval_mmpd_gaussian_anchor.py — same metrics as submit_cfg_ablation.sh:
#   50% test windows, 1× anchor det, 100× stochastic, CRPS/top-k, texture.
#
# Prereq: train finished (best.pt + *_itransformer_finetuned.pt under each ckpt).
#
# USAGE ($SCRATCH/ts-sandbox):
#   ./submit_patch48_eval_redo.sh
#   SAMPLERS=dpmpp ./submit_patch48_eval_redo.sh          # dpmpp only (8 jobs)
#   EVAL_WALL=8:00:00 ./submit_patch48_eval_redo.sh       # if 6h still tight on ETTm
#   DATASETS=ETTm1,ETTm2 ./submit_patch48_eval_redo.sh    # resubmit subset only
#
# If a prior wave is still in squeue: scancel -u $USER -n smp-
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CKPT_SUFFIX="${CKPT_SUFFIX:-binary_dual_scale_patch48}"
export CKPT_STEM_PREFIX="${CKPT_STEM_PREFIX:-06-02-384445}"
export DATASETS="${DATASETS:-ETTm1,ETTm2,dalia,electricity,exchange_rate,solar_Alabama,traffic,weather}"
export SAMPLERS="${SAMPLERS:-dpmpp,ddim}"
export RUN_STEM="${RUN_STEM:-$(date +%m-%d)-patch48-redo}"

echo "Patch48 eval redo → submit_sampler_ablation.sh"
echo "  ckpt filter: *${CKPT_STEM_PREFIX}*-*-${CKPT_SUFFIX}"
echo "  datasets: $DATASETS"
echo "  samplers: $SAMPLERS"
echo "  run_stem: $RUN_STEM"
echo ""

exec env CKPT_SUFFIX="$CKPT_SUFFIX" CKPT_STEM_PREFIX="$CKPT_STEM_PREFIX" \
    "$SCRIPT_DIR/submit_sampler_ablation.sh" \
    --ckpt-suffix "$CKPT_SUFFIX" \
    --ckpt-stem-prefix "$CKPT_STEM_PREFIX" \
    --run-stem "$RUN_STEM" \
    --datasets "$DATASETS" \
    --samplers "$SAMPLERS" \
    "$@"
