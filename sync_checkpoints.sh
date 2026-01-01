#!/bin/bash

# Usage: ./sync_checkpoints.sh [remote_ip]
# This script copies the checkpoints from a remote machine to the local project.

# Check if IP address is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [remote_ip]"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

REMOTE_IP=$1
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

echo "--------------------------------------------------------"
echo "Syncing checkpoints"
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

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "Sync completed successfully!"
else
    echo "--------------------------------------------------------"
    echo "Error: Sync failed. Please check the IP address and SSH access."
fi

