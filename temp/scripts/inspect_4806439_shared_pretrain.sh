#!/bin/bash
export PYTHONUNBUFFERED=1
export PATH="/opt/software/slurm/bin:/opt/slurm/bin:/usr/bin:/bin:${PATH}"
# Force line-buffered stdout
exec 1> >(stdbuf -oL cat)
exec 2> >(stdbuf -oL cat >&2)
echo "start $(date) host=$(hostname)"
echo "squeue=$(command -v squeue || echo missing)"
SHARED=/scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/ckpts/_shared_staged_pretrain/binary_dual_scale_staged-v4-h16-ms9.3-121b5990f4
echo "=== SHARED ls ==="
ls -la "$SHARED" || echo "missing shared"
echo "=== SHARED coarse ==="
ls -la "$SHARED/coarse" 2>/dev/null || echo "no coarse"
echo "=== SHARED patch_refine ==="
ls -la "$SHARED/patch_refine" 2>/dev/null || echo "no patch_refine"
echo "=== SHARED files ==="
ls -la "$SHARED/coarse/pretrained_diffusion.pt" "$SHARED/patch_refine/pretrained_diffusion.pt" "$SHARED/coarse/.signature" "$SHARED/patch_refine/.signature" "$SHARED/coarse/shared_pretrain_metadata.json" "$SHARED/patch_refine/shared_pretrain_metadata.json" 2>&1
echo "=== metadata coarse ==="
cat "$SHARED/coarse/shared_pretrain_metadata.json" 2>/dev/null || echo "no coarse meta"
echo "=== metadata patch_refine ==="
cat "$SHARED/patch_refine/shared_pretrain_metadata.json" 2>/dev/null || echo "no pr meta"
CKPT=/scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/ckpts/08-15-4806439-weather-binary_window_norm_patch_refine_canvas128_p64x6_weather_default_xattn_cache_anchor_every8_eval_s16
echo "=== 4806439 pretrained paths ==="
ls -ld "$CKPT/pretrained_coarse" "$CKPT/pretrained_patch_refine" 2>&1
ls -la "$CKPT/pretrained_coarse/pretrained_diffusion.pt" "$CKPT/pretrained_patch_refine/pretrained_diffusion.pt" 2>&1
echo "=== 4806439 log ==="
LOG=/scratch/ccao87/ts-sandbox-xattn-cache-20260815/results/logs/08-15-4806439-weather-binary_window_norm_patch_refine_canvas128_p64x6_weather_default_xattn_cache_anchor_every8_eval_s16.log
ls -la "$LOG" 2>&1
if [ -f "$LOG" ]; then
  grep -n -E "shared cached|pretrained_|reuse_pretrain|staged_diffusion_pretrain|_shared_staged_pretrain|signature|cross_attention_only|Elapsed|TIMEOUT|eval/" "$LOG" | head -80
  echo "=== log head 60 ==="
  head -60 "$LOG"
  echo "=== log tail 40 ==="
  tail -40 "$LOG"
fi
echo "=== squeue ==="
squeue -u ccao87 2>&1 | head -30
echo "=== 4849244 ==="
squeue -j 4849244 2>&1 | head
echo "=== sacct 4806439 ==="
sacct -j 4806439 --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode -P 2>&1 | head
echo "done $(date)"
