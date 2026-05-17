#!/bin/bash
#SBATCH --job-name=tsf_experimental
#SBATCH --account=aip-boyuwang
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --array=0-5
#SBATCH --output=results_experimental/logs/%x-%A_%a.log
#SBATCH --error=results_experimental/logs/%x-%A_%a.log

set -euo pipefail

# Ensure we're in the right directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p results_experimental/logs

DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Dataset: $DATASET"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Alliance CA best practice is to rebuild venv on $SLURM_TMPDIR for speed
if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Building fast venv on \$SLURM_TMPDIR..."
    python -m venv "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn wandb optuna tqdm matplotlib einops
    # Some specialized packages might need pypi if not in wheels
    pip install reformer-pytorch --index-url https://pypi.org/simple
fi

# Run the experimental pipeline for THIS dataset
echo "Running experimental pipeline for $DATASET..."
python models/diffusion_tsf/train_experimental_pipeline.py --datasets "$DATASET" --epochs-itrans 5 --epochs-diff 10 --job-id "$SLURM_ARRAY_JOB_ID"

# Only the first task of the array (or a separate job) should run aggregation
# But since they all finish at different times, we can just say "Run the aggregation script manually after all jobs finish"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Task 0 will wait briefly then suggest aggregation."
    echo "To aggregate all results later, run: python models/diffusion_tsf/aggregate_experimental_results.py"
fi

echo "=========================================="
echo "Job completed: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="
