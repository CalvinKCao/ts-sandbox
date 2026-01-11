#!/usr/bin/env bash
# =============================================================================
# Local to Remote Visualization Script
# =============================================================================
# This script runs on your LOCAL machine. It:
# 1. Syncs necessary files to remote server
# 2. Runs visualization on remote (GPU inference)
# 3. Pulls generated images back to local machine
#
# Usage:
#   ./local_sync_viz.sh                    # 5 samples per dataset (fine-tuned)
#   ./local_sync_viz.sh --samples 10       # 10 samples per dataset
#   ./local_sync_viz.sh --dataset ETTh2    # Only visualize ETTh2
#   ./local_sync_viz.sh --pretrain-only    # Use universal pretrained model only
#
# For comparison, run BOTH:
#   ./local_sync_viz.sh --samples 5        # Fine-tuned models
#   ./local_sync_viz.sh --samples 5 --pretrain-only  # Pretrained only
# =============================================================================

# --- CONFIGURATION ---
REMOTE_USER="root"
REMOTE_HOST="162.243.218.59"
REMOTE_PATH="/root/ts-sandbox"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_OUTPUT_DIR="models/diffusion_tsf/final_visualizations"

# Capture all args to pass to remote viz script
VIZ_ARGS="$@"

echo "🎨 DiffusionTSF Visualization Pipeline"
echo "======================================="
echo ""

# 1. Sync visualization scripts
echo "📄 Syncing visualization scripts to remote..."
rsync -avz --progress \
  "${LOCAL_ROOT}/run_visualizations.py" \
  "${LOCAL_ROOT}/run_viz.sh" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to sync scripts. Aborting."
    exit 1
fi

# 1.5. If --pretrain-only is in args, sync the universal pretrained model
if [[ "$VIZ_ARGS" == *"--pretrain-only"* ]]; then
    echo ""
    echo "📦 Syncing universal pretrained model..."
    PRETRAIN_DIR="models/diffusion_tsf/checkpoints/universal_synthetic_pretrain"
    rsync -avz --progress \
      "${LOCAL_ROOT}/${PRETRAIN_DIR}/" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${PRETRAIN_DIR}/"
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to sync pretrained model. Aborting."
        exit 1
    fi
fi

# 2. Run visualization on remote (foreground - we want to see output)
echo ""
echo "🔥 Running visualization on remote GPU..."
if [[ -n "$VIZ_ARGS" ]]; then
    echo "📝 Args: $VIZ_ARGS"
fi
echo ""

ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_PATH} && chmod +x run_viz.sh && ./run_viz.sh ${VIZ_ARGS}"

if [[ $? -ne 0 ]]; then
    echo "❌ Visualization failed on remote."
    exit 1
fi

# 3. Create local output directories
LOCAL_VIZ_DIR="${LOCAL_ROOT}/${VIZ_OUTPUT_DIR}"
LOCAL_VIZ_PRETRAIN_DIR="${LOCAL_ROOT}/models/diffusion_tsf/final_visualizations_pretrain"
mkdir -p "${LOCAL_VIZ_DIR}"
mkdir -p "${LOCAL_VIZ_PRETRAIN_DIR}"

# 4. Pull visualizations back to local (both fine-tuned and pretrain folders)
echo ""
echo "📥 Pulling visualizations to local machine..."

# Pull fine-tuned visualizations
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${VIZ_OUTPUT_DIR}/" \
  "${LOCAL_VIZ_DIR}/" 2>/dev/null || true

# Pull pretrain visualizations (if they exist)
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/models/diffusion_tsf/final_visualizations_pretrain/" \
  "${LOCAL_VIZ_PRETRAIN_DIR}/" 2>/dev/null || true

echo ""
echo "✅ Done! Visualizations saved to:"
echo "   Fine-tuned: ${LOCAL_VIZ_DIR}"
echo "   Pretrained: ${LOCAL_VIZ_PRETRAIN_DIR}"
echo ""
echo "📂 Fine-tuned contents:"
ls -la "${LOCAL_VIZ_DIR}" 2>/dev/null || echo "   (no fine-tuned visualizations yet)"
echo ""
echo "📂 Pretrained contents:"
ls -la "${LOCAL_VIZ_PRETRAIN_DIR}" 2>/dev/null || echo "   (no pretrain visualizations yet)"

