#!/bin/bash
#SBATCH --job-name=latent-dim1
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_h100_b4
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# =============================================================================
# Slurm: full 1-variate latent diffusion experiment (VAE → iTransformer → LDM → ETTh1)
# =============================================================================
#
# Prerequisites on Killarney:
#   - Repo under $SCRATCH/ts-sandbox (not /home)
#   - datasets/ETT-small/ETTh1.csv inside that repo
#   - One-time: ./setup/alliance_setup_killarney.sh  → venv at $PROJECT/$USER/diffusion-tsf/venv
#
# Usage (from scratch repo root):
#   sbatch slurm_latent_experiment.sh
#   sbatch slurm_latent_experiment.sh -- --image-height 96
#   sbatch slurm_latent_experiment.sh -- --skip-vae-train
#   sbatch slurm_latent_experiment.sh -- --smoke-test
#
# If the job exits immediately with almost nothing in .out, check .err for:
#   - "unbound variable" → was caused by `set -u` + missing $SCRATCH (fixed below).
#   - "venv not found" → run alliance_setup on a login node.
#
# If the job stays PD (Resources) for hours:
#   - H100 queue full: try #SBATCH --partition=gpubase_h100_b5 and longer --time
#   - Or shorter smoke: --time=0:30:00 and -- --smoke-test
#   - Inspect: sinfo -p gpubase_h100_b4 -o "%P %a %G"   and   squeue -j JOBID --reason
#
# L40S (often shorter queue for quick tests only — uncomment and remove H100 lines):
#   #SBATCH --partition=<partition_that_has_l40s>   # run: sinfo -o "%P %G"
#   #SBATCH --gres=gpu:l40s:1
#   (and remove --partition gpubase_h100_b4 and --gpus-per-node h100:1)
# =============================================================================

# Do not use `set -u`: $SCRATCH is not always exported on compute nodes → silent fast-fail.
set -eo pipefail

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: ${SLURM_GPUS_ON_NODE:-?}"
echo "Started: $(date)"
echo "=========================================="

# SCRATCH: use env if set, else Alliance-style default (avoids unbound-variable exit)
SCRATCH_DIR="${SCRATCH:-/scratch/${USER}}"
export SCRATCH="$SCRATCH_DIR"
echo "Using SCRATCH=$SCRATCH"

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
echo "PROJECT_ROOT=$PROJECT_ROOT"

if [ -z "${PROJECT:-}" ]; then
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d "$HOME/projects"/def-* "$HOME/projects"/aip-* 2>/dev/null | head -1 || true)
        [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi
if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: Set PROJECT to your allocation, e.g. export PROJECT=\$(readlink -f ~/projects/aip-...)"
    exit 1
fi
echo "PROJECT=$PROJECT"

export STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf"
mkdir -p "$STORAGE_ROOT/checkpoints" "$STORAGE_ROOT/synthetic_cache" "$STORAGE_ROOT/results"

VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: venv not found at $VENV_PATH"
    echo "On a login node run: cd $PROJECT_ROOT && ./setup/alliance_setup_killarney.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
echo "Python: $(command -v python)  $(python -V 2>&1)"

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
    [ "$code" -ne 0 ] && echo "[SLURM CLEANUP] $(date) exit=$code"
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
echo "Running: python -m models.diffusion_tsf.train_latent_experiment --stage all --cache-dir $LATENT_CACHE ${ARGS[*]}"
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
