#!/bin/bash

# Usage: ./sync_checkpoints.sh [remote_ip] [--best-only] [--guidance-only] [--all]
# This script copies the checkpoints from a remote machine to the local project.
# Use --best-only to sync only the most recent best_model.pt file.
# Use --guidance-only to sync only iTransformer/guidance model checkpoints.
# Use --all (default) to sync both diffusion and guidance checkpoints.

BEST_ONLY=false
GUIDANCE_ONLY=false
SYNC_GUIDANCE=true
SYNC_DIFFUSION=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --best-only)
            BEST_ONLY=true
            shift
            ;;
        --guidance-only)
            GUIDANCE_ONLY=true
            SYNC_DIFFUSION=false
            shift
            ;;
        --no-guidance)
            SYNC_GUIDANCE=false
            shift
            ;;
        --all)
            SYNC_GUIDANCE=true
            SYNC_DIFFUSION=true
            shift
            ;;
        *)
            REMOTE_IP="$1"
            shift
            ;;
    esac
done

# Check if IP address is provided
if [ -z "$REMOTE_IP" ]; then
    echo "Usage: $0 [remote_ip] [options]"
    echo ""
    echo "Options:"
    echo "  --best-only      Sync only the most recent best_model.pt file"
    echo "  --guidance-only  Sync only iTransformer/guidance checkpoints"
    echo "  --no-guidance    Skip guidance checkpoints"
    echo "  --all            Sync both diffusion and guidance (default)"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.1.100                    # Sync everything"
    echo "  $0 192.168.1.100 --best-only        # Sync only best diffusion model"
    echo "  $0 192.168.1.100 --guidance-only    # Sync only iTransformer checkpoints"
    exit 1
fi

REMOTE_USER="root"

# --- CONFIGURATION ---
# Diffusion TSF checkpoints
REMOTE_DIFFUSION_PATH="~/ts-sandbox/models/diffusion_tsf/checkpoints/"
LOCAL_DIFFUSION_PATH="models/diffusion_tsf/checkpoints/"

# iTransformer / Guidance model checkpoints
REMOTE_GUIDANCE_PATH="~/ts-sandbox/checkpoints/"
LOCAL_GUIDANCE_PATH="checkpoints/"
# ---------------------

# Create local directories if they don't exist
mkdir -p "$LOCAL_DIFFUSION_PATH"
mkdir -p "$LOCAL_GUIDANCE_PATH"

SYNC_SUCCESS=true

# ============================================================================
# Sync Diffusion TSF Checkpoints
# ============================================================================
if [ "$SYNC_DIFFUSION" = true ]; then
    if [ "$BEST_ONLY" = true ]; then
        echo "========================================================"
        echo "📦 Syncing Diffusion TSF (best model only)"
        echo "   Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIFFUSION_PATH}"
        echo "   Local:  ${LOCAL_DIFFUSION_PATH}"
        echo "========================================================"

        # Find all best_model.pt and model_best.pt files on remote and show them
        echo "Searching for best_model.pt and model_best.pt files..."
        echo ""
        echo "All checkpoint files found (sorted by modification time):"
        ssh "${REMOTE_USER}@${REMOTE_IP}" "find ${REMOTE_DIFFUSION_PATH} \( -name 'best_model.pt' -o -name 'model_best.pt' \) -type f -printf '%T+ %p\n' 2>/dev/null | sort"
        echo ""

        # Get the most recent one (full absolute path)
        REMOTE_BEST_FILE=$(ssh "${REMOTE_USER}@${REMOTE_IP}" "find ${REMOTE_DIFFUSION_PATH} \( -name 'best_model.pt' -o -name 'model_best.pt' \) -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-")

        if [ -z "$REMOTE_BEST_FILE" ]; then
            echo "⚠️  No best_model.pt or model_best.pt files found on remote machine."
        else
            echo "Most recent: $REMOTE_BEST_FILE"

            # Local destination - use the same filename as remote
            REMOTE_FILENAME=$(basename "$REMOTE_BEST_FILE")
            LOCAL_BEST_FILE="${LOCAL_DIFFUSION_PATH}${REMOTE_FILENAME}"

            echo "Local destination: $LOCAL_BEST_FILE"

            # Use rsync to sync only this specific file directly
            rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_BEST_FILE}" "$LOCAL_BEST_FILE" || SYNC_SUCCESS=false
        fi
    else
        echo "========================================================"
        echo "📦 Syncing Diffusion TSF Checkpoints (all)"
        echo "   Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIFFUSION_PATH}"
        echo "   Local:  ${LOCAL_DIFFUSION_PATH}"
        echo "========================================================"

        # Use rsync for efficient transfer
        # -a: archive mode (preserves permissions, times, etc.)
        # -v: verbose
        # -z: compress during transfer
        # -P: show progress and allow resuming
        rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIFFUSION_PATH}" "$LOCAL_DIFFUSION_PATH" || SYNC_SUCCESS=false
    fi
    echo ""
fi

# ============================================================================
# Sync Guidance Model Checkpoints (iTransformer, etc.)
# ============================================================================
if [ "$SYNC_GUIDANCE" = true ]; then
    echo "========================================================"
    echo "🤖 Syncing Guidance Model Checkpoints (iTransformer, etc.)"
    echo "   Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_GUIDANCE_PATH}"
    echo "   Local:  ${LOCAL_GUIDANCE_PATH}"
    echo "========================================================"

    # Check if remote guidance directory exists
    REMOTE_GUIDANCE_EXISTS=$(ssh "${REMOTE_USER}@${REMOTE_IP}" "[ -d ${REMOTE_GUIDANCE_PATH} ] && echo 'yes' || echo 'no'")

    if [ "$REMOTE_GUIDANCE_EXISTS" = "yes" ]; then
        # List what's available
        echo "Available guidance checkpoints on remote:"
        ssh "${REMOTE_USER}@${REMOTE_IP}" "find ${REMOTE_GUIDANCE_PATH} -name 'checkpoint.pth' -type f 2>/dev/null | head -20"
        echo ""

        # Sync all guidance checkpoints
        rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_GUIDANCE_PATH}" "$LOCAL_GUIDANCE_PATH" || SYNC_SUCCESS=false
    else
        echo "⚠️  No guidance checkpoints directory found on remote at: ${REMOTE_GUIDANCE_PATH}"
        echo "   (This is normal if you haven't trained an iTransformer yet)"
    fi
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
if [ "$SYNC_SUCCESS" = true ]; then
    echo "========================================================"
    echo "✅ Sync completed successfully!"
    echo "========================================================"
    
    if [ "$SYNC_DIFFUSION" = true ]; then
        echo "Diffusion checkpoints: $LOCAL_DIFFUSION_PATH"
    fi
    if [ "$SYNC_GUIDANCE" = true ]; then
        echo "Guidance checkpoints:  $LOCAL_GUIDANCE_PATH"
    fi
else
    echo "========================================================"
    echo "❌ Error: Some syncs failed. Please check the IP address and SSH access."
    echo "========================================================"
    exit 1
fi

