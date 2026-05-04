#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --job-name=tsf-diffsteps-sweep
#SBATCH --output=/tmp/tsf-diffsteps-sweep-%j.out
#SBATCH --error=/tmp/tsf-diffsteps-sweep-%j.err
# =============================================================================
# Throwaway: diffusion-steps × image-height grid sweep (Killarney, L40S)
#
# Trains iTransformer once, then runs all 8 combos of
#   T ∈ {200,500,1000,1500}  ×  H ∈ {64,128}
# Each combo: 10-epoch diffusion pretrain → 20-epoch ETTh2 finetune → eval.
# Fixed: lr=2.05e-5, batch=16, n_variates=3.
#
# Usage (login node, repo root):
#   ./slurm_throwaway_diffsteps_sweep.sh
#   ACCOUNT=aip-boyuwang ./slurm_throwaway_diffsteps_sweep.sh
#   STORE=/scratch/ccao87/diffusion-tsf-diffsteps-sweep ./slurm_throwaway_diffsteps_sweep.sh
#
# Expected wall time: ~6–8 h on an L40S (iTransformer + 8 × short runs).
# =============================================================================

set -euo pipefail

ACCOUNT="${ACCOUNT:-aip-boyuwang}"
EMAIL="${EMAIL:-ccao87@uwo.ca}"
CPUS="${CPUS:-8}"
MEM="${MEM:-50G}"
WALL="${WALL:-0-10:00:00}"

# Resolve repo root
if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: clone repo to \$SCRATCH/ts-sandbox (or \$HOME/ts-sandbox fallback)."
    exit 1
fi

# Resolve storage root
if [ -z "${STORE:-}" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        echo "ERROR: \$SCRATCH not set. Set STORE manually and re-run."
        exit 1
    fi
    STORE="$SCRATCH/diffusion-tsf-diffsteps-sweep"
fi

LOG_DIR="$STORE/logs"
JOB_DIR="$STORE/job_scripts"
mkdir -p "$LOG_DIR" "$JOB_DIR"

JOB_SCRIPT="$JOB_DIR/diffsteps_sweep.job.sh"

# Write the batch script to disk (single file, no heredoc quoting issues)
cat > "$JOB_SCRIPT" << 'JOB'
#!/bin/bash
set -euo pipefail

echo "======================================================="
echo "  Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Start: $(date)"
echo "======================================================="

# ---- Resolve repo root inside the batch environment ----
if [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found."
    exit 1
fi

if [ -z "${STORE:-}" ]; then
    echo "ERROR: STORE not passed into batch environment."
    exit 1
fi

# ---- Modules ----
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# ---- Build venv on fast local NVMe (avoids slow Lustre imports) ----
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

# ---- wandb ----
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
cd "$PROJECT_ROOT"

echo "[run] Starting diffusion-steps × image-height sweep ..."
python -u -m models.diffusion_tsf.throwaway_diffsteps_sweep \
    --store "$STORE" \
    --amp \
    --gradient-checkpointing \
    ${WANDB_FLAG:+$WANDB_FLAG} \
    --wandb-project diffusion-tsf

echo "Done: $(date)"
JOB

chmod +x "$JOB_SCRIPT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Detected direct sbatch mode (job $SLURM_JOB_ID); running payload in-place."
    export STORE
    exec "$JOB_SCRIPT"
fi

echo "Submitting diffusion-steps sweep ..."
JOB_ID=$(sbatch --parsable \
    --job-name=tsf-diffsteps-sweep \
    --account="$ACCOUNT" \
    --nodes=1 \
    --gres=gpu:l40s:1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$WALL" \
    --export=ALL,STORE="$STORE" \
    --chdir="$PROJECT_ROOT" \
    --output="$LOG_DIR/diffsteps-sweep-%j.out" \
    --error="$LOG_DIR/diffsteps-sweep-%j.err" \
    --mail-type=FAIL,END \
    --mail-user="$EMAIL" \
    "$JOB_SCRIPT")

echo "Submitted: $JOB_ID"
echo "Logs:"
echo "  stdout: $LOG_DIR/diffsteps-sweep-$JOB_ID.out"
echo "  stderr: $LOG_DIR/diffsteps-sweep-$JOB_ID.err"
echo "Tail live output:"
echo "  tail -f $LOG_DIR/diffsteps-sweep-$JOB_ID.out"
echo "Monitor:"
echo "  squeue -j $JOB_ID -o '%.18i %.20j %.10T %.20R'"
echo "Verify GPU:"
echo "  scontrol show job=$JOB_ID | grep -i gres"
