#!/bin/bash
# =============================================================================
# Helper script to resume all interrupted experimental runs
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/results_experimental/runs"

if [ ! -d "$RUNS_DIR" ]; then
    echo "No runs directory found at $RUNS_DIR"
    exit 1
fi

for RUN_PATH in "$RUNS_DIR"/*; do
    if [ ! -d "$RUN_PATH" ]; then
        continue
    fi
    
    RUN_FOLDER=$(basename "$RUN_PATH")
    
    # Parse scenario and dataset from folder name. 
    # Example format: 05-18-3650640-exp_A_ETTh1
    # We can extract everything after "exp_"
    
    # Remove the date and job id prefix (e.g. 05-18-3650640-)
    REMAINDER="${RUN_FOLDER#*-*-*-}"
    
    # Now we have exp_A_ETTh1 or exp_A_B_ETTh1
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
    
    DS_TAG="${DATASET//_/-}"
    JOB_NAME="resume_${SCENARIO//+/_}_${DS_TAG}"
    
    echo "Resuming $RUN_FOLDER (Scenario: $SCENARIO, Dataset: $DATASET)"
    
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="12:00:00" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --chdir="$SCRIPT_DIR" \
        --output="${RUN_PATH}/logs/resume-%j.out" \
        --error="${RUN_PATH}/logs/resume-%j.err" \
        --wrap="
module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

if [ -n \"\${SLURM_TMPDIR:-}\" ]; then
    echo \"Building fast venv on \\\$SLURM_TMPDIR...\"
    python -m venv \"\$SLURM_TMPDIR/env\"
    source \"\$SLURM_TMPDIR/env/bin/activate\"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn wandb optuna tqdm matplotlib einops
    pip install reformer-pytorch --index-url https://pypi.org/simple
else
    source .venv/bin/activate
fi

TARGET_DIM=7
if [ \"$DATASET\" = \"weather\" ]; then TARGET_DIM=21; fi
if [ \"$DATASET\" = \"exchange_rate\" ]; then TARGET_DIM=8; fi
if [ \"$DATASET\" = \"ETTm1\" ] || [ \"$DATASET\" = \"ETTh1\" ] || [ \"$DATASET\" = \"ETTh2\" ] || [ \"$DATASET\" = \"ETTm2\" ]; then TARGET_DIM=7; fi

SYNTH_CACHE_ROOT=\"$SCRIPT_DIR/synth_data\"

COMMON_ARGS=(
    \"--dataset\" \"$DATASET\"
    \"--model-type\" \"unet\"
    \"--checkpoint-dir\" \"${RUN_PATH}/ckpts\"
    \"--results-dir\" \"${RUN_PATH}/datasets\"
    \"--synth-cache-dir\" \"\$SYNTH_CACHE_ROOT\"
    \"--experiment\" \"$SCENARIO\"
    \"--subset-id\" \"exp_$SCENARIO\"
    \"--resume\"
)

# Append to the original log file
exec >>\"${RUN_PATH}/logs/${RUN_FOLDER}.log\" 2>&1

echo \"==========================================\"
echo \"RESUMING Job ID: \$SLURM_JOB_ID\"
echo \"==========================================\"

echo \"Running Phase 1 (Pretrain)...\"
python models/diffusion_tsf/train_multivariate_pipeline.py \\
    --mode pretrain \\
    --n-variates \"\$TARGET_DIM\" \\
    \"\${COMMON_ARGS[@]}\"

echo \"Running Phase 2 (Finetune)...\"
python models/diffusion_tsf/train_multivariate_pipeline.py \\
    --mode finetune \\
    --n-variates \"\$TARGET_DIM\" \\
    \"\${COMMON_ARGS[@]}\"

echo \"Pipeline complete.\"
"
done
echo "All resume jobs submitted!"
