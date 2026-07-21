#!/bin/bash
# Rebuild and log eval plots from the existing Jul 18 hz720 guidance-cond checkpoints.
set -euo pipefail
./submit_binary.sh --resume \
  --configs configs/bce_dist_guidance_cond_3x336_eval_viz_regen_temp.yaml \
  --ckpt-config binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20_bce_dist_guidance_cond_3x336 \
  --datasets ETTh1,traffic,exchange_rate --time 8:00:00 "$@"
