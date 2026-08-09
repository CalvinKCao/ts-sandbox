#!/bin/bash
# Rebuild eval plots from the Jul 20 Narval fresh fixed-HP checkpoints.
set -euo pipefail

./submit_binary.sh --resume \
  --configs configs/bce_dist_guidance_cond_3x336_narval_fresh_fixed_hp_eval_viz_overlap_temp.yaml \
  --ckpt-config bce_dist_guidance_cond_3x336_narval_fresh_fixed_hp_temp \
  --datasets ETTh1,traffic,exchange_rate \
  --gpu a100 --time 1:30:00 "$@"
