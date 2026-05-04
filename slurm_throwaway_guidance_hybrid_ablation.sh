#!/bin/bash
# =============================================================================
# Throwaway: guidance channel × hybrid cross-variate context ablation
#
# Same electricity 4-variate subset as slurm_throwaway_diffsteps_sweep.sh
# (consumers 93, 292, 81, 84). Shared iTransformer pretrain, then four runs:
#   both | guidance_only | hybrid_only | neither
# Each run: 10-epoch diffusion pretrain → 20-epoch finetune → eval.
# Default geometry: T=1000, H=128 (pass extra args to the Python module to change).
#
# Usage (from $SCRATCH/ts-sandbox on the login node):
#   sbatch slurm_throwaway_guidance_hybrid_ablation.sh
#
# Logs:        ./results/logs/<MM-DD>-<last3-jobid>-guidance-hybrid-ablation.log
# Checkpoints: $SCRATCH/diffusion-tsf-guidance-hybrid-ablation/
# =============================================================================
#SBATCH --job-name=tsf-guidance-hybrid-abl
#SBATCH --account=aip-boyuwang
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ccao87@uwo.ca

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p results/logs results/ckpts
STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-guidance-hybrid-ablation"
LOG="results/logs/${STEM}.log"

exec >>"$LOG" 2>&1

echo "======================================================="
echo "  Job : $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Start: $(date)"
echo "  Log : $SLURM_SUBMIT_DIR/$LOG"
echo "======================================================="

PROJECT_ROOT="$SLURM_SUBMIT_DIR"

if [ -n "${SCRATCH:-}" ]; then
    STORE="$SCRATCH/diffusion-tsf-guidance-hybrid-ablation"
else
    STORE="$PROJECT_ROOT/results/ckpts/${STEM}"
fi
mkdir -p "$STORE"
echo "[setup] Checkpoint store: $STORE"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

VENV="$SLURM_TMPDIR/env"
echo "[setup] Building venv on $VENV ..."
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch numpy pandas scipy scikit-learn tqdm wandb optuna matplotlib einops -q
pip install reformer-pytorch==1.4.4 -q 2>/dev/null || true
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements.txt" -q 2>/dev/null || true
fi
echo "[setup] Venv ready: $VENV/bin/python"

export WANDB_DIR="$STORE/wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY not set — running without wandb."
    WANDB_FLAG=""
else
    echo "[wandb] API key present."
    WANDB_FLAG="--wandb"
fi

export PYTHONUNBUFFERED=1

echo "[run] guidance × hybrid ablation (T=1000 H=128 by default) ..."
python -u -m models.diffusion_tsf.throwaway_guidance_hybrid_ablation \
    --store "$STORE" \
    --num-diffusion-steps 1000 \
    --image-height 128 \
    --amp \
    --gradient-checkpointing \
    ${WANDB_FLAG:+$WANDB_FLAG} \
    --wandb-project diffusion-tsf

echo "Done: $(date)"
