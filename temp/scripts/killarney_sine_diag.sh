#!/bin/bash
set -euo pipefail
source /etc/profile >/dev/null 2>&1 || true
echo "HOST=$(hostname)"
echo "SCRATCH=${SCRATCH:-unset}"
echo "DATE=$(date)"
echo ""
echo "===== squeue ====="
squeue -u ccao87 -o "%.18i %.9P %.30j %.2t %.10M %.10l %R" 2>/dev/null | head -40
echo ""
echo "===== sine-linear 5356069 / 5356070 ====="
for j in 5356069 5356070; do
  echo "--- $j ---"
  sacct -j "$j" --format=JobID,JobName%40,State,Elapsed,Timelimit,ExitCode,Start,End -n -P 2>/dev/null | head -8
done
echo ""
echo "===== illness donor ====="
ls -lh "$SCRATCH/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_lb104_hz60_pretrain/pretrained_coarse/pretrained_diffusion.pt" \
       "$SCRATCH/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_lb104_hz60_pretrain/pretrained_patch_refine/pretrained_diffusion.pt" 2>&1 | head
echo ""
echo "===== staged_eval ====="
f="$SCRATCH/ts-sandbox/models/diffusion_tsf/pipeline/phases/staged_eval.py"
if [ -f "$f" ]; then
  grep -n "def _staged_anchor_global_norm" "$f" || echo SYMBOL_MISSING
  ls -l "$f"
else
  echo MISSING
fi
echo ""
echo "===== sine linear config ====="
ls -l "$SCRATCH/ts-sandbox/configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10_pretrain_sine_linear.yaml" 2>&1
echo ""
echo "===== 5356069 logs tail ====="
# find slurm logs
find "$SCRATCH/ts-sandbox" -maxdepth 3 -name '*5356069*' 2>/dev/null | head
find "$SCRATCH/ts-sandbox/results" -name '*5356069*' 2>/dev/null | head
