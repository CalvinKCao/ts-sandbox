#!/bin/bash
# Fresh Narval training with prior fixed per-dataset HPs; no checkpoint reuse.
set -euo pipefail
./submit_binary.sh \
  --configs configs/bce_dist_guidance_cond_3x336_narval_fresh_fixed_hp_temp.yaml \
  --datasets ETTh1,traffic,exchange_rate --gpu a100 --time 8:00:00 "$@"
