#!/bin/bash
# =============================================================================
# Quick profiling submit wrapper (Killarney / Alliance Canada)
#
# Purpose:
#   Submit a short L40S job that runs the 1-epoch, fine-grained timing profile:
#     python -m models.diffusion_tsf.train_multivariate_pipeline --profile-one-epoch
#
# Usage (login node, repo root):
#   ./slurm_profile_one_epoch.sh
#   ACCOUNT=aip-boyuwang ./slurm_profile_one_epoch.sh
#   STORE=$SCRATCH/diffusion-tsf-profile ./slurm_profile_one_epoch.sh
#
# Notes:
# - Run with bash on the login node (this script submits sbatch).
# - Uses L40S by default for fast queue turn-around.
# =============================================================================

set -euo pipefail

ACCOUNT="${ACCOUNT:-aip-boyuwang}"
EMAIL="${EMAIL:-ccao87@uwo.ca}"
CPUS="${CPUS:-8}"
MEM="${MEM:-50G}"
WALL="${WALL:-0-01:00:00}"

if [ -z "${STORE:-}" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        echo "ERROR: \$SCRATCH is not set. Set STORE manually and re-run."
        exit 1
    fi
    STORE="$SCRATCH/diffusion-tsf-profile-one-epoch"
fi

if [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: clone repo to \$SCRATCH/ts-sandbox (or \$HOME/ts-sandbox fallback)."
    exit 1
fi

LOG_DIR="$STORE/logs"
JOB_DIR="$STORE/job_scripts"
mkdir -p "$LOG_DIR" "$JOB_DIR"

JOB_SCRIPT="$JOB_DIR/profile_one_epoch.job.sh"
cat > "$JOB_SCRIPT" <<'JOB'
#!/bin/bash
set -euo pipefail

echo "======================================================="
echo "  Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Start: $(date)"
echo "======================================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found in SCRATCH or HOME."
    exit 1
fi

if [ -z "${STORE:-}" ]; then
    echo "ERROR: STORE not exported into batch environment."
    exit 1
fi

echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch numpy pandas scipy scikit-learn tqdm wandb optuna matplotlib einops -q
pip install reformer-pytorch==1.4.4 -q
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements.txt" -q || true
fi

export WANDB_DIR="$STORE/wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$PROJECT_ROOT/wandb_api_key.txt" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$PROJECT_ROOT/wandb_api_key.txt")"
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_MODE:-}" ]; then
    export WANDB_MODE=offline
    echo "[wandb] WANDB_API_KEY not found; using offline mode."
fi

export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

echo "[run] profile-one-epoch quick test"
python -u -m models.diffusion_tsf.train_multivariate_pipeline \
    --profile-one-epoch \
    --profile-max-batches 2 \
    --profile-synthetic-samples 256 \
    --profile-max-subsets 1 \
    --amp \
    --wandb \
    --wandb-project diffusion-tsf

echo "Done: $(date)"
JOB
chmod +x "$JOB_SCRIPT"

echo "Submitting quick profile job..."
JOB_ID=$(sbatch --parsable \
    --job-name=tsf-profile-1epoch \
    --account="$ACCOUNT" \
    --nodes=1 \
    --gres=gpu:l40s:1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$WALL" \
    --export=ALL,STORE="$STORE" \
    --chdir="$PROJECT_ROOT" \
    --output="$LOG_DIR/profile-one-epoch-%j.out" \
    --error="$LOG_DIR/profile-one-epoch-%j.err" \
    --mail-type=FAIL,END \
    --mail-user="$EMAIL" \
    "$JOB_SCRIPT")

echo "Submitted: $JOB_ID"
echo "Logs:"
echo "  $LOG_DIR/profile-one-epoch-$JOB_ID.out"
echo "  $LOG_DIR/profile-one-epoch-$JOB_ID.err"
echo "Monitor:"
echo "  squeue -j $JOB_ID -o '%.18i %.20j %.10T %.20R'"
echo "Check GRES:"
echo "  scontrol show job=$JOB_ID | egrep -i 'ReqTRES|TRES=|MinMem'"
