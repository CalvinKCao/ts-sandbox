#!/bin/bash
# CFG-off (w=1) eval for ETTh1/ETTh2/PeMS on the 05-31 full-cond variembed grid (job 3828089 family).
#
# USAGE ($SCRATCH/ts-sandbox):
#   ./submit_cfg_off_etth_pems_redo.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 05-31 binary_dual_scale train grid (CFG off weights; no inference CFG at w=1)
export CFG_CKPT_MAP="ETTh1=05-31-3828089-ETTh1-binary_dual_scale,ETTh2=05-31-3828090-ETTh2-binary_dual_scale,PeMS=05-31-3828098-PeMS-binary_dual_scale"
export EVAL_WALL="5:00:00"
export RUN_STEM="${RUN_STEM:-06-01-cfg-off-etth-pems-redo}"

exec "$SCRIPT_DIR/submit_cfg_ablation.sh" \
    --datasets ETTh1,ETTh2,PeMS \
    --cfg-scales 1 \
    --ckpt-map "$CFG_CKPT_MAP" \
    --run-stem "$RUN_STEM" \
    "$@"
