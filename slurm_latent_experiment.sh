#!/bin/bash
#SBATCH --job-name=latent-dim1
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_h100_b4
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1              # no GPU type in directive — partition picks default
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca     # CHANGE to your email
#SBATCH --signal=B:USR1@120            # Send USR1 120s before wall-time kill

# =============================================================================
# Slurm: full 1-variate latent diffusion experiment (VAE → iTransformer → LDM → ETTh1)
# =============================================================================
#
# Prerequisites on Killarney:
#   - Repo under $SCRATCH/ts-sandbox (not /home)
#   - datasets/ETT-small/ETTh1.csv inside that repo
#   - One-time: ./setup/alliance_setup_killarney.sh
#
# Usage (from scratch repo root):
#   sbatch slurm_latent_experiment.sh
#   sbatch slurm_latent_experiment.sh -- --image-height 96
#   sbatch slurm_latent_experiment.sh -- --skip-vae-train
#   sbatch slurm_latent_experiment.sh -- --smoke-test
#
# Edit --account / --mail-user in the header to match your allocation (same as slurm_pipeline.sh).
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo "Started: $(date)"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    export PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found in SCRATCH or HOME"
    exit 1
fi

if [ -z "${PROJECT:-}" ]; then
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d "$HOME/projects"/def-* "$HOME/projects"/aip-* 2>/dev/null | head -1)
        [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi
if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: Set PROJECT to your allocation, e.g. export PROJECT=\$(readlink -f ~/projects/aip-...)"
    exit 1
fi

export STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf"
mkdir -p "$STORAGE_ROOT/checkpoints" "$STORAGE_ROOT/synthetic_cache" "$STORAGE_ROOT/results"

VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: venv not found at $VENV_PATH"
    echo "Run once on a login node: cd $PROJECT_ROOT && ./setup/alliance_setup_killarney.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

LATENT_CACHE="$STORAGE_ROOT/synthetic_cache/latent_dim1"
mkdir -p "$LATENT_CACHE"

cd "$PROJECT_ROOT"

if [ ! -f datasets/ETT-small/ETTh1.csv ]; then
    echo "ERROR: Missing datasets/ETT-small/ETTh1.csv under $PROJECT_ROOT"
    exit 1
fi

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[SLURM CLEANUP] $(date)"
    kill -- -$$ 2>/dev/null || true
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

ARGS=()
for a in "$@"; do
    [ "$a" = "--" ] && continue
    ARGS+=("$a")
done

echo ""
echo "Running: python -m models.diffusion_tsf.train_latent_experiment --stage all --cache-dir $LATENT_CACHE ${ARGS[*]:-}"
echo ""

python -m models.diffusion_tsf.train_latent_experiment \
    --stage all \
    --cache-dir "$LATENT_CACHE" \
    "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Checkpoints: $PROJECT_ROOT/models/diffusion_tsf/checkpoints_latent/"
echo "Results JSON: $PROJECT_ROOT/models/diffusion_tsf/results/"
echo "=========================================="
