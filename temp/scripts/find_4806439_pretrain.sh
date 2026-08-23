#!/bin/bash
set -e
echo "hostname=$(hostname)"
echo "SCRATCH=${SCRATCH:-/scratch/ccao87}"
export SCRATCH="${SCRATCH:-/scratch/ccao87}"
cd "$SCRATCH/ts-sandbox"
echo "=== local branch/sha ==="
git branch --show-current
git rev-parse HEAD
echo "=== 4806439 dirs ==="
ls -ld "$SCRATCH"/ts-sandbox*/results/ckpts/*4806439* 2>/dev/null || true
ls -ld "$SCRATCH"/ts-sandbox/results/ckpts/*4806439* 2>/dev/null || true
CKPT=/scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/ckpts/08-15-4806439-weather-binary_window_norm_patch_refine_canvas128_p64x6_weather_default_xattn_cache_anchor_every8_eval_s16
echo "=== xattn-cache 4806439 listing ==="
if [ -d "$CKPT" ]; then
  ls -la "$CKPT"
  echo "--- pretrained ---"
  ls -la "$CKPT/pretrained_coarse" "$CKPT/pretrained_patch_refine" 2>/dev/null || echo "no pretrained_* dirs"
  find "$CKPT" -maxdepth 3 \( -name pretrained_diffusion.pt -o -name .signature \) -ls
else
  echo "MISSING $CKPT"
fi
echo "=== reused/pretrain ==="
ls -la "$SCRATCH/ts-sandbox/reused/pretrain" 2>/dev/null | head -40 || echo "no reused/pretrain"
echo "=== shared cache xattn ==="
ls /scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/ckpts/_shared_staged_pretrain 2>/dev/null | head || echo "no shared in xattn-cache"
echo "=== shared cache ts-sandbox ==="
ls "$SCRATCH/ts-sandbox/results/ckpts/_shared_staged_pretrain" 2>/dev/null | head || echo "no shared in ts-sandbox"
echo "=== 4849244 ==="
squeue -j 4849244 -o "%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R" || true
echo "=== squeue me ==="
squeue -u ccao87 -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
echo "=== sacct 4806439 ==="
sacct -j 4806439 --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,Start,End -P | head
echo "=== logs 4806439 ==="
ls -la /scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/logs/*4806439* 2>/dev/null || true
ls -la "$SCRATCH/ts-sandbox/results/logs/"*4806439* 2>/dev/null || true
