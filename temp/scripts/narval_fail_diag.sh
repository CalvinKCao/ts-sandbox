#!/bin/bash
set -euo pipefail
source /etc/profile >/dev/null 2>&1 || true
echo "HOST=$(hostname)"
echo "SCRATCH=${SCRATCH:-unset}"
echo "DATE=$(date)"
echo ""
echo "===== PROTECTED JOBS ====="
for j in 2874036 2874118 2874119; do
  echo "--- $j ---"
  squeue -j "$j" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.6D %R" 2>/dev/null || true
  sacct -j "$j" --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,Start,End -n 2>/dev/null | head -5
done
echo ""
echo "===== squeue ccao87 ====="
squeue -u ccao87 -o "%.18i %.9P %.30j %.2t %.10M %.10l %R" 2>/dev/null | head -80
echo ""
echo "===== FAILED JOBS sacct ====="
for j in 2874844 2874851 2874921 2874933 2874941 2874963 2874964 2874965 2874986 2874989 2874996 2874041 2874058 2874061 2874062 2874063 2874076 2874078 2874026 2874027 2874028 2874029 2874030 2874031 2874032 2874033 2874034 2874035 2874037 2890531 2890532 2890533; do
  sacct -j "$j" --format=JobID,JobName%40,State,Elapsed,Timelimit,ExitCode -n -P 2>/dev/null | head -2
done
echo ""
echo "===== staged_eval on cluster ====="
for root in "$SCRATCH/ts-sandbox-fullT" "$SCRATCH/ts-sandbox"; do
  f="$root/models/diffusion_tsf/pipeline/phases/staged_eval.py"
  echo "-- $f --"
  if [ -f "$f" ]; then
    grep -n "def _staged_anchor_global_norm" "$f" || echo "SYMBOL_MISSING"
    ls -l "$f"
  else
    echo MISSING
  fi
done
echo ""
echo "===== flock in apply script ====="
for root in "$SCRATCH/ts-sandbox-fullT" "$SCRATCH/ts-sandbox"; do
  f="$root/temp/scripts/apply_baseline_canvas128_patches.py"
  echo "-- $f --"
  if [ -f "$f" ]; then
    grep -n "fcntl\|LOCK_EX\|_apply_patches_locked" "$f" | head -20
  else
    echo MISSING
  fi
done
echo ""
echo "===== p32 donor ====="
for dest in \
  "$SCRATCH/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_pretrain" \
  "$SCRATCH/ts-sandbox-fullT/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_pretrain"; do
  echo "-- $dest --"
  ls -lh "$dest/pretrained_coarse/pretrained_diffusion.pt" "$dest/pretrained_patch_refine/pretrained_diffusion.pt" 2>&1 | head -10
done
echo ""
echo "===== illness hz60 donor ====="
for dest in \
  "$SCRATCH/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_lb104_hz60_pretrain" \
  "$SCRATCH/ts-sandbox-fullT/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_lb104_hz60_pretrain"; do
  echo "-- $dest --"
  ls -lh "$dest/pretrained_coarse/pretrained_diffusion.pt" "$dest/pretrained_patch_refine/pretrained_diffusion.pt" 2>&1 | head -10
done
echo ""
echo "===== MMPD fail dirs ====="
ls -d "$SCRATCH/ts-sandbox-fullT"/results/datasets/09-11-mmpd-fullT-* 2>/dev/null || true
echo ""
echo "===== binary job names 2874026-37 ====="
for j in $(seq 2874026 2874037); do
  sacct -j "$j" --format=JobID,JobName%50,State,Elapsed,Timelimit,ExitCode,WorkDir%80 -n -P 2>/dev/null | head -1
done
echo ""
echo "===== 2890531-33 details ====="
for j in 2890531 2890532 2890533; do
  echo "--- $j ---"
  sacct -j "$j" --format=JobID,JobName%50,State,Elapsed,Timelimit,ExitCode,WorkDir%80 -n -P 2>/dev/null | head -3
  scontrol show job "$j" 2>/dev/null | egrep -i 'JobName|WorkDir|Command|StdOut|StdErr|Comment|Reason|JobState|TimeLimit' | head -20 || true
done
