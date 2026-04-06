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

# Slurm copies this script to spool as .../slurm_script — dirname is not the repo.
# Submit from the repo root (e.g. cd $SCRATCH/ts-sandbox) so SLURM_SUBMIT_DIR has the .inc.sh.
_INC=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm_ci_latent_common.inc.sh" ]; then
    _INC="${SLURM_SUBMIT_DIR}/slurm_ci_latent_common.inc.sh"
else
    _SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "${_SD}/slurm_ci_latent_common.inc.sh" ]; then
        _INC="${_SD}/slurm_ci_latent_common.inc.sh"
    fi
fi
if [ -z "$_INC" ]; then
    echo "ERROR: slurm_ci_latent_common.inc.sh not found. Run sbatch from repo root (where the .inc.sh lives), not from a path-only sbatch without cd."
    exit 1
fi
# shellcheck source=slurm_ci_latent_common.inc.sh
source "$_INC"

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
