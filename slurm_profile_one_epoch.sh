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
#
# Artifacts: ./results/{logs,ckpts,datasets}/ under the repo (submit dir). One
# combined log per job (stdout+stderr) under results/logs/.
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

if [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: clone repo to \$SCRATCH/ts-sandbox (or \$HOME/ts-sandbox fallback)."
    exit 1
fi

mkdir -p "$PROJECT_ROOT/results/logs" "$PROJECT_ROOT/results/ckpts" "$PROJECT_ROOT/results/datasets"
JOB_DIR="$PROJECT_ROOT/results/logs/profile-job-scripts"
mkdir -p "$JOB_DIR"

JOB_SCRIPT="$JOB_DIR/profile_one_epoch.job.sh"
cat > "$JOB_SCRIPT" <<'JOB'
#!/bin/bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/ckpts results/datasets
ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID: -4}-profile-1epoch"
ALLIANCE_JOB_LOG="$SLURM_SUBMIT_DIR/results/logs/${ALLIANCE_RUN_STEM}.log"
touch "$ALLIANCE_JOB_LOG"
exec >>"$ALLIANCE_JOB_LOG" 2>&1

echo "======================================================="
echo "  Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Start: $(date)"
echo "  Log: $ALLIANCE_JOB_LOG"
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

if [ -x "$STORE/venv/bin/python" ] && [ -f "$STORE/venv/bin/activate" ]; then
    echo "[setup] Using existing venv: $STORE/venv"
    source "$STORE/venv/bin/activate"
else
    echo "[setup] Creating venv: $STORE/venv"
    virtualenv --no-download "$STORE/venv"
    source "$STORE/venv/bin/activate"
fi
pip install --no-index --upgrade pip -q
pip install --no-index torch numpy pandas scipy scikit-learn tqdm wandb optuna matplotlib einops -q
pip install reformer-pytorch==1.4.4 -q
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements.txt" -q || true
fi

CKPT_DIR="$SLURM_SUBMIT_DIR/results/ckpts/$ALLIANCE_RUN_STEM"
RES_DIR="$SLURM_SUBMIT_DIR/results/datasets/$ALLIANCE_RUN_STEM"
mkdir -p "$CKPT_DIR" "$RES_DIR"

export WANDB_DIR="$SLURM_SUBMIT_DIR/results/logs/${ALLIANCE_RUN_STEM}_wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] ERROR: WANDB_API_KEY is not set."
    echo "[wandb] Export WANDB_API_KEY from https://wandb.ai/authorize and re-submit."
    exit 2
fi
echo "[wandb] Using WANDB_API_KEY from environment."

export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

wandb_upload_job_logs() {
    local checkpoint_dir="$1"
    shift
    local run_id_file="${checkpoint_dir}/wandb_run_id.txt"
    if [ ! -f "$run_id_file" ]; then
        echo "[wandb] WARN: no run id file at $run_id_file; skipping log upload."
        return 0
    fi
    local run_id
    run_id="$(tr -d '[:space:]' < "$run_id_file")"
    [ -z "$run_id" ] && echo "[wandb] WARN: empty run id in $run_id_file; skipping." && return 0

    local files=()
    local f
    for f in "$@"; do
        [ -f "$f" ] && files+=("$f")
    done
    [ "${#files[@]}" -eq 0 ] && echo "[wandb] WARN: no log files found to upload." && return 0

    python - "$run_id" "${files[@]}" <<'PY' || true
import os
import sys
import wandb

run_id = sys.argv[1]
files = sys.argv[2:]
project = os.environ.get("WANDB_PROJECT", "diffusion-tsf")
job_id = os.environ.get("SLURM_JOB_ID", "unknown")
job_name = os.environ.get("SLURM_JOB_NAME", "unknown")

run = wandb.init(project=project, id=run_id, resume="must", reinit=True)
artifact = wandb.Artifact(f"slurm-job-logs-{job_id}", type="logs")
artifact.metadata.update({"slurm_job_id": job_id, "slurm_job_name": job_name})
for path in files:
    if os.path.isfile(path):
        artifact.add_file(path)
run.log_artifact(artifact)
run.finish()
print(f"[wandb] Uploaded {len(files)} log file(s) for job {job_id}.")
PY
}

echo "[run] profile-one-epoch quick test"
python -u -m models.diffusion_tsf.train_multivariate_pipeline \
    --profile-one-epoch \
    --profile-max-batches 2 \
    --profile-synthetic-samples 256 \
    --profile-max-subsets 1 \
    --amp \
    --wandb \
    --wandb-project diffusion-tsf \
    --checkpoint-dir "$CKPT_DIR" \
    --results-dir "$RES_DIR"

wandb_upload_job_logs "$CKPT_DIR" "$ALLIANCE_JOB_LOG"

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
    --chdir="$PROJECT_ROOT" \
    --output=/dev/null \
    --error=/dev/null \
    --mail-type=FAIL,END \
    --mail-user="$EMAIL" \
    "$JOB_SCRIPT")

echo "Submitted: $JOB_ID"
echo "Log (combined stdout+stderr) under:"
echo "  $PROJECT_ROOT/results/logs/*-${JOB_ID: -4}-profile-1epoch.log"
echo "List newest: ls -tr $PROJECT_ROOT/results/logs/*profile-1epoch.log | tail -1"
echo "Monitor:"
echo "  squeue -j $JOB_ID -o '%.18i %.20j %.10T %.20R'"
echo "Check GRES:"
echo "  scontrol show job=$JOB_ID | egrep -i 'ReqTRES|TRES=|MinMem'"
