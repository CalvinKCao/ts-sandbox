#!/bin/bash
#SBATCH --job-name=diffusion-smoke
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_h100_b1    # 3h limit, shortest queue
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1         # 1 GPU is enough for smoke test
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@60

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[CLEANUP] $(date) — killing child processes..."
    kill -- -$$ 2>/dev/null || true
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    [ "$code" -ne 0 ] && echo "[CLEANUP] Done."
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

# Resolve paths
if [ -d "$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="$SCRATCH/ts-sandbox"
else
    export PROJECT_ROOT="$HOME/ts-sandbox"
fi

# Detect PROJECT
if [ -z "$PROJECT" ] && [ -d "$HOME/projects" ]; then
    FIRST_PROJECT=$(ls -d $HOME/projects/def-* $HOME/projects/aip-* 2>/dev/null | head -1)
    [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
fi
[ -z "$PROJECT" ] && { echo "ERROR: PROJECT not found"; exit 1; }

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
mkdir -p "$STORAGE_ROOT/checkpoints" "$STORAGE_ROOT/results" "$STORAGE_ROOT/wandb"

if [ ! -d "$STORAGE_ROOT/datasets" ]; then
    cp -r "$PROJECT_ROOT/datasets" "$STORAGE_ROOT/datasets"
fi

VENV_PATH="$STORAGE_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    module purge
    module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer_pytorch -q
    [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt" -q
else
    module purge
    module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
    source "$VENV_PATH/bin/activate"
fi

export WANDB_MODE=offline
export WANDB_DIR="$STORAGE_ROOT/wandb"

echo "=========================================="
echo "SMOKE TEST  |  Job: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
./pipeline.sh --smoke-test --gpus 1 \
    --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
    --results-dir "$STORAGE_ROOT/results"

echo "=========================================="
echo "Smoke test done: $(date)"
echo "Results: $STORAGE_ROOT/results"
echo "=========================================="
