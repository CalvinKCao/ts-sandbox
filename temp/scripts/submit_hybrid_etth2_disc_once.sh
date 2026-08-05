#!/bin/bash
# Login-node one-shot: hybrid ETTh2 disc (no redbox; MMPD from forecast cache).
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:${PATH:-}
if [ -z "${SLURM_CONF:-}" ]; then
  if [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
  elif [ -f /cm/shared/apps/slurm/var/etc/slurm/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
  fi
fi
export SCRATCH=/scratch/ccao87
cd /scratch/ccao87/ts-sandbox-ordinal-fine
echo "cwd=$(pwd) branch=$(git branch --show-current)"
CKPT=results/ckpts/08-05-4609805-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm
CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2_hybrid_flat_dsnorm.yaml
test -f "$CKPT/ETTh2/patch_refine/best.pt"
export OUT_TAG=ETTh2-c128-hybrid-flat-dsnorm-valtest80-byvar
export DISC_CONFIG="$CFG"
export DISC_RUN_NAME=hybrid_flat_dsnorm
export CKPT
sbatch \
  --job-name=ablation-disc-l8l16 \
  --account=aip-boyuwang \
  --time=8:00:00 \
  --nodes=1 \
  --gres=gpu:l40s:1 \
  --cpus-per-task=8 \
  --mem=50G \
  --exclude=kn010 \
  --export=NONE,OUT_TAG,DISC_CONFIG,DISC_RUN_NAME,CKPT,PATH,SLURM_CONF,SCRATCH,HOME,USER,LANG \
  --output=results/slurm/%x-%j.out \
  --error=results/slurm/%x-%j.err \
  --mail-type=END,FAIL \
  --mail-user=ccao87@uwo.ca \
  ./temp/scripts/submit_ablation_disc_l8_l16.sh \
  --dataset ETTh2 \
  --reuse-forecast-cache \
  --no-redbox-viz
squeue -u ccao87 -o "%.18i %.40j %.2t %.10M %R" | head -10
