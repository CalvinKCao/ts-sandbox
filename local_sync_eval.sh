#!/usr/bin/env bash
# =============================================================================
# Local to Remote Sync & Eval Script
# =============================================================================
# This script runs on your LOCAL machine. It:
# 1. Syncs your local checkpoints to the remote server
# 2. Runs the evaluation script on the remote server via SSH
# =============================================================================

# --- CONFIGURATION ---
# PLEASE UPDATE THESE FOR YOUR ENVIRONMENT
REMOTE_USER="root"
REMOTE_HOST="162.243.218.59"
REMOTE_PATH="/root/ts-sandbox"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="models/diffusion_tsf/checkpoints"

# Check if local checkpoints exist
if [[ ! -d "${LOCAL_ROOT}/${CHECKPOINT_DIR}" ]]; then
    echo "❌ Error: Local checkpoint directory not found at ${LOCAL_ROOT}/${CHECKPOINT_DIR}"
    exit 1
fi

echo "🚀 Starting sync to ${REMOTE_USER}@${REMOTE_HOST}..."

# 1. Sync evaluation scripts first (small files)
echo "📄 Syncing evaluation scripts..."
rsync -avz --progress \
  "${LOCAL_ROOT}/evaluate_all.py" \
  "${LOCAL_ROOT}/run_eval.sh" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

# 2. Sync checkpoints (only the 'big' files)
# -avz: archive, verbose, compress
# --progress: show transfer progress
# includes/excludes: focus on best models and params only
echo "📦 Syncing checkpoints..."
rsync -avz --progress \
  --include="*/" \
  --include="best_model.pt" \
  --include="best_params.json" \
  --include="checkpoint.pth" \
  --exclude="trial_*.pt" \
  --exclude="*.db" \
  "${LOCAL_ROOT}/${CHECKPOINT_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${CHECKPOINT_DIR}/"

echo "✅ Sync complete!"
echo "🔥 Triggering evaluation on remote..."

# 3. Run the evaluation script on the remote via SSH
ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_PATH} && chmod +x run_eval.sh && ./run_eval.sh"

echo ""
echo "🎉 Remote evaluation triggered! Check the output above."


