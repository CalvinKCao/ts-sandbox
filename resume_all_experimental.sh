#!/bin/bash
# =============================================================================
# Helper script to resume all interrupted experimental runs
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try both common locations for runs folders
if [ -d "$SCRIPT_DIR/results_experimental/runs" ]; then
    RUNS_DIR="$SCRIPT_DIR/results_experimental/runs"
elif [ -d "$SCRIPT_DIR/results/runs" ]; then
    RUNS_DIR="$SCRIPT_DIR/results/runs"
else
    echo "No runs directory found in results_experimental/runs or results/runs"
    exit 1
fi

echo "Scanning $RUNS_DIR for folders to resume..."

for RUN_PATH in "$RUNS_DIR"/*; do
    if [ ! -d "$RUN_PATH" ]; then
        continue
    fi
    
    RUN_FOLDER=$(basename "$RUN_PATH")
    
    # Parse scenario and dataset from folder name. 
    # Example format: 05-18-3650640-exp_A_ETTh1
    REMAINDER="${RUN_FOLDER#*-*-*-}"
    
    if [[ "$REMAINDER" == exp_A_B_* ]]; then
        SCENARIO="A+B"
        DATASET="${REMAINDER#exp_A_B_}"
    elif [[ "$REMAINDER" == exp_A_* ]]; then
        SCENARIO="A"
        DATASET="${REMAINDER#exp_A_}"
    elif [[ "$REMAINDER" == exp_B_* ]]; then
        SCENARIO="B"
        DATASET="${REMAINDER#exp_B_}"
    else
        echo "Could not parse scenario from $RUN_FOLDER, skipping."
        continue
    fi
    
    TARGET_DIM=7
    if [ "$DATASET" = "weather" ]; then TARGET_DIM=21; fi
    if [ "$DATASET" = "exchange_rate" ]; then TARGET_DIM=8; fi
    if [ "$DATASET" = "ETTm1" ] || [ "$DATASET" = "ETTh1" ] || [ "$DATASET" = "ETTh2" ] || [ "$DATASET" = "ETTm2" ]; then TARGET_DIM=7; fi

    DS_TAG="${DATASET//_/-}"
    JOB_NAME="res_${SCENARIO//+/_}_${DS_TAG}"
    
    echo "Submitting resume job for $RUN_FOLDER (Scenario: $SCENARIO, Dataset: $DATASET, Dim: $TARGET_DIM)"
    
    # We use a heredoc to sbatch to avoid the quoting/escaping hell of --wrap
    sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=aip-boyuwang
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --chdir=$SCRIPT_DIR
#SBATCH --output=${RUN_PATH}/logs/resume-%j.out
#SBATCH --error=${RUN_PATH}/logs/resume-%j.err

set -euo pipefail

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Alliance CA best practice is to rebuild venv on \$SLURM_TMPDIR for speed
if [ -n "\${SLURM_TMPDIR:-}" ]; then
    echo "Building fast venv on \$SLURM_TMPDIR..."
    python3 -m venv "\$SLURM_TMPDIR/env"
    source "\$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn wandb optuna tqdm matplotlib einops
    pip install reformer-pytorch --index-url https://pypi.org/simple
else
    echo "Warning: SLURM_TMPDIR not set, using local venv"
    source .venv/bin/activate
fi

# Append to original log
exec >>"${RUN_PATH}/logs/${RUN_FOLDER}.log" 2>&1

echo "=========================================="
echo "RESUMING AT: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "=========================================="

echo "Running Phase 1 (Pretrain)..."
python3 models/diffusion_tsf/train_multivariate_pipeline.py \\
    --mode pretrain \\
    --n-variates $TARGET_DIM \\
    --dataset $DATASET \\
    --model-type unet \\
    --checkpoint-dir ${RUN_PATH}/ckpts \\
    --results-dir ${RUN_PATH}/datasets \\
    --synth-cache-dir $SCRIPT_DIR/synth_data \\
    --experiment $SCENARIO \\
    --subset-id exp_$SCENARIO \\
    --resume

echo "Running Phase 2 (Finetune)..."
python3 models/diffusion_tsf/train_multivariate_pipeline.py \\
    --mode finetune \\
    --n-variates $TARGET_DIM \\
    --dataset $DATASET \\
    --model-type unet \\
    --checkpoint-dir ${RUN_PATH}/ckpts \\
    --results-dir ${RUN_PATH}/datasets \\
    --synth-cache-dir $SCRIPT_DIR/synth_data \\
    --experiment $SCENARIO \\
    --subset-id exp_$SCENARIO \\
    --resume

echo "Pipeline complete."
EOS
done
echo "All resume jobs submitted!"
