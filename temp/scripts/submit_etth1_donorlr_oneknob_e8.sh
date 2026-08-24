#!/bin/bash
# Login-node helper (not a submit_*.sh wrapper). PATH before nounset:
# sourcing /etc/profile under set -u aborts the script.
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
set -eo pipefail
eval "$(grep -E '^export WANDB_API_KEY=' ~/.bashrc || true)"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo WANDB_missing
  exit 1
fi
echo WANDB_ok
cd /scratch/ccao87/ts-sandbox-allv-randwin-lr10
test -d datasets -o -L datasets
for s in win05_e8 patch05_e8 anchor4_e8; do
  f="configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_etth1_donorlr_${s}.yaml"
  test -f "$f"
  echo "ok $f"
done
echo "=== submit 3 one-knob ablations ETTh1 8ep ==="
./submit_binary.sh --configs \
  binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_etth1_donorlr_win05_e8,\
binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_etth1_donorlr_patch05_e8,\
binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_etth1_donorlr_anchor4_e8 \
  --datasets ETTh1 --time 1:00:00 --mem 60G
echo "=== squeue ==="
squeue -u ccao87 -o "%.10i %.16P %.2t %.11M %.10l %.8m %R %j"
