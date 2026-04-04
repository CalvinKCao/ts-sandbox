#!/bin/bash
#SBATCH --job-name=ci-lat-boot
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# Shared CI-latent stages 0-2 only (VAE + synthetic iTrans + synthetic diffusion).
# Usually submitted by submit_ci_latent_multidataset_jobs.sh with finetune jobs depending on this.
#
# Manual:
#   sbatch slurm_ci_latent_bootstrap.sh
#   sbatch slurm_ci_latent_bootstrap.sh -- --smoke-test

set -e
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  (bootstrap 0-2)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_ci_latent_common.inc.sh
source "$_SCRIPT_DIR/slurm_ci_latent_common.inc.sh"

echo ""
echo "Shared dir: $SHARED"
echo ""

if [ ! -f "$SHARED/itransformer_pretrained_synth.pt" ] || [ ! -f "$SHARED/diffusion_pretrained_H${IMAGE_H}.pt" ]; then
    echo "=== Stages 0, 1, 2 ==="
    run_py --dataset ETTh2 --stage 0 --stage4-trials 0 --no-wandb $EXTRA_ARGS
    run_py --dataset ETTh2 --stage 1 --stage4-trials 0 --no-wandb $EXTRA_ARGS
    run_py --dataset ETTh2 --stage 2 --stage4-trials 0 --no-wandb $EXTRA_ARGS
else
    echo "=== Skipping 0-2 (shared checkpoints already present) ==="
fi

echo ""
echo "Bootstrap finished: $(date)"
echo "Shared: $SHARED"
echo "=========================================="
