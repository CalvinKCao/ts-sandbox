#!/bin/bash

# Usage: ./sync_checkpoints.sh [remote_ip] [--best-only]
# This script copies the checkpoints from a remote machine to the local project.
# Use --best-only to sync only the most recent best_model.pt file.

BEST_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --best-only)
            BEST_ONLY=true
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
    echo "Usage: $0 [remote_ip] [--best-only]"
    echo "Example: $0 192.168.1.100"
    echo "Example: $0 192.168.1.100 --best-only"
    exit 1
fi

REMOTE_USER="root"

# --- CONFIGURATION ---
# Path to the checkpoints on the remote machine
# Adjust this if your project is located elsewhere on the remote
REMOTE_PATH="~/ts-sandbox/models/diffusion_tsf/checkpoints/"

# Path to the checkpoints on your local machine
LOCAL_PATH="models/diffusion_tsf/checkpoints/"
# ---------------------

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

if [ "$BEST_ONLY" = true ]; then
    echo "--------------------------------------------------------"
    echo "Syncing only the most recent best_model.pt"
    echo "Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
    echo "Local:  ${LOCAL_PATH}"
    echo "--------------------------------------------------------"

    # Find all best_model.pt files on remote and show them
    echo "Searching for best_model.pt files in: ${REMOTE_PATH}"
    echo ""
    echo "All best_model.pt files found (sorted by modification time):"
    ssh "${REMOTE_USER}@${REMOTE_IP}" "find ${REMOTE_PATH} -name 'best_model.pt' -type f -printf '%T+ %p\n' 2>/dev/null | sort"
    echo ""

    # Get the most recent one (full absolute path)
    REMOTE_BEST_FILE=$(ssh "${REMOTE_USER}@${REMOTE_IP}" "find ${REMOTE_PATH} -name 'best_model.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-")

    if [ -z "$REMOTE_BEST_FILE" ]; then
        echo "No best_model.pt files found on remote machine."
        exit 1
    fi

    echo "Most recent: $REMOTE_BEST_FILE"

    # Local destination
    LOCAL_BEST_FILE="${LOCAL_PATH}best_model.pt"

    echo "Local destination: $LOCAL_BEST_FILE"

    # Use rsync to sync only this specific file directly
    rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_BEST_FILE}" "$LOCAL_BEST_FILE"
else
    echo "--------------------------------------------------------"
    echo "Syncing all checkpoints"
    echo "Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
    echo "Local:  ${LOCAL_PATH}"
    echo "--------------------------------------------------------"

    # Use rsync for efficient transfer
    # -a: archive mode (preserves permissions, times, etc.)
    # -v: verbose
    # -z: compress during transfer
    # -P: show progress and allow resuming
    # --exclude: avoids syncing things you might not want
    rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}" "$LOCAL_PATH"
fi

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "Sync completed successfully!"
else
    echo "--------------------------------------------------------"
    echo "Error: Sync failed. Please check the IP address and SSH access."
fi

