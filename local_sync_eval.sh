#!/usr/bin/env bash
# =============================================================================
# Local to Remote Sync & Eval Script
# =============================================================================
# This script runs on your LOCAL machine. It:
# 1. Syncs your local checkpoints to the remote server
# 2. Runs the evaluation script on the remote server via SSH
#
# Usage:
#   ./local_sync_eval.sh                  # Fast eval (stride=8)
#   ./local_sync_eval.sh --stride 1       # Full eval (all samples)
#   ./local_sync_eval.sh --ddim-steps 25  # Faster DDIM
# =============================================================================

# --- CONFIGURATION ---
# PLEASE UPDATE THESE FOR YOUR ENVIRONMENT
REMOTE_USER="root"
REMOTE_HOST="162.243.218.59"
REMOTE_PATH="/root/ts-sandbox"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="models/diffusion_tsf/checkpoints"

# Capture all args to pass to remote eval script
EVAL_ARGS="$@"

# Check if local checkpoints exist
if [[ ! -d "${LOCAL_ROOT}/${CHECKPOINT_DIR}" ]]; then
    echo "❌ Error: Local checkpoint directory not found at ${LOCAL_ROOT}/${CHECKPOINT_DIR}"
    exit 1
fi

# Function to run rsync with retry logic
sync_with_retry() {
    local description="$1"
    shift  # Remove description from args
    local max_retries=3
    local retry_count=0

    echo "🔄 ${description}..."

    while [[ $retry_count -lt $max_retries ]]; do
        if rsync -avz --progress "$@"; then
            echo "✅ ${description} completed successfully!"
            return 0
        else
            retry_count=$((retry_count + 1))
            echo "⚠️  ${description} failed (attempt $retry_count/$max_retries)"

            if [[ $retry_count -lt $max_retries ]]; then
                echo "⏳ Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    echo "❌ ${description} failed after $max_retries attempts"
    return 1
}

echo "🚀 Starting sync to ${REMOTE_USER}@${REMOTE_HOST}..."

# 1. Sync evaluation scripts first (small files)
if ! sync_with_retry "Syncing evaluation scripts" \
  "${LOCAL_ROOT}/evaluate_all.py" \
  "${LOCAL_ROOT}/run_eval.sh" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"; then
    echo "❌ Failed to sync evaluation scripts. Aborting."
    exit 1
fi

# 1.5. Sync datasets (only final CSV files, not split parts)
echo "📊 Syncing datasets..."
if ! sync_with_retry "Syncing datasets" \
  --include="*/" \
  --include="*.csv" \
  --exclude="*_part*.csv" \
  --exclude="*.Zone.Identifier" \
  "${LOCAL_ROOT}/datasets/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/datasets/"; then
    echo "⚠️  Warning: Failed to sync datasets. Continuing anyway..."
fi

# 2. Sync checkpoints (only the 'big' files)
# -avz: archive, verbose, compress
# --progress: show transfer progress
# includes/excludes: focus on best models and params only
if ! sync_with_retry "Syncing checkpoints" \
  --include="*/" \
  --include="best_model.pt" \
  --include="best_params.json" \
  --include="checkpoint.pth" \
  --exclude="trial_*.pt" \
  --exclude="*.db" \
  "${LOCAL_ROOT}/${CHECKPOINT_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${CHECKPOINT_DIR}/"; then
    echo "❌ Failed to sync checkpoints. Aborting."
    exit 1
fi

echo "🎉 All syncs completed successfully!"
echo "🔥 Triggering evaluation on remote..."
if [[ -n "$EVAL_ARGS" ]]; then
    echo "📝 Eval args: $EVAL_ARGS"
fi

# 3. Run the evaluation script on the remote via SSH
# Use nohup to make it run independently of SSH session
echo "📡 Starting remote evaluation (this may take a while)..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_PATH} && chmod +x run_eval.sh && nohup ./run_eval.sh ${EVAL_ARGS} > evaluation.log 2>&1 & echo 'Evaluation started in background. PID: '\$!"

echo ""
echo "✅ Remote evaluation triggered!"
echo "📋 Check progress with: ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${REMOTE_PATH}/evaluation.log'"
echo "📊 When done, view results: ssh ${REMOTE_USER}@${REMOTE_HOST} 'cat ${REMOTE_PATH}/evaluation.log'"


