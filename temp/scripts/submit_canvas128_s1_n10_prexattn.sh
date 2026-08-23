#!/bin/bash
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail
export PATH="/cm/shared/apps/slurm/current/bin:${PATH}"
if [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ]; then
  export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
fi
OF=/scratch/ccao87/ts-sandbox-ordinal-fine-0bf5752a
cd "$OF"
echo "HEAD=$(git rev-parse --short HEAD) cwd=$PWD"

submit() {
  local cfg="$1" ds="$2"
  echo "=== $ds $cfg ==="
  ./submit_binary.sh --configs "$cfg" --datasets "$ds" --time 12:00:00 --mem 80G
}

submit configs/binary_window_norm_patch_refine_canvas128_p64x6_etth1_eval_s1_n10_f66.yaml ETTh1
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_eval_s1_n10_f66.yaml ETTh2
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_eval_s1_n10_f66.yaml electricity
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic_eval_s1_n10_f66.yaml traffic
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_eval_s1_n10_f66.yaml exchange_rate
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_pems_eval_s1_n10_f66.yaml PeMS
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_eval_s1_n10_f66.yaml solar_Alabama
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1_eval_s1_n10_f50.yaml ETTm1
submit configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2_eval_s1_n10_f27.yaml ETTm2
echo DONE
