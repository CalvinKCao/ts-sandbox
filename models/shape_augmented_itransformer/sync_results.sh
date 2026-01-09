#!/bin/bash

# Usage: ./sync_results.sh [remote_ip] [--results-only]
# This script copies the shape-augmented iTransformer results from a remote machine to local.
#
# Examples:
#   ./sync_results.sh 192.168.1.100              # Sync all results
#   ./sync_results.sh 192.168.1.100 --results-only  # Sync only JSON/PNG results (no checkpoints)

RESULTS_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-only)
            RESULTS_ONLY=true
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
    echo "Usage: $0 <remote_ip> [options]"
    echo ""
    echo "Options:"
    echo "  --results-only    Sync only JSON results and PNG plots (skip checkpoints)"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.1.100                    # Sync everything"
    echo "  $0 192.168.1.100 --results-only     # Sync only results/plots"
    exit 1
fi

REMOTE_USER="root"

# --- CONFIGURATION ---
REMOTE_BASE="~/ts-sandbox/models/shape_augmented_itransformer"
LOCAL_BASE="$(dirname "$0")"  # Script directory
REMOTE_RESULTS="${REMOTE_BASE}/results/"
LOCAL_RESULTS="${LOCAL_BASE}/results/"
# ---------------------

# Create local directories if they don't exist
mkdir -p "$LOCAL_RESULTS"

echo "========================================================"
echo "📦 Syncing Shape-Augmented iTransformer Results"
echo "   Remote: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_RESULTS}"
echo "   Local:  ${LOCAL_RESULTS}"
echo "========================================================"

SYNC_SUCCESS=true

# Check if remote results directory exists
REMOTE_EXISTS=$(ssh "${REMOTE_USER}@${REMOTE_IP}" "[ -d ${REMOTE_RESULTS} ] && echo 'yes' || echo 'no'" 2>/dev/null)

if [ "$REMOTE_EXISTS" != "yes" ]; then
    echo "⚠️  No results directory found on remote at: ${REMOTE_RESULTS}"
    echo "   Run the test script first: python test_etth1_comparison.py"
    exit 1
fi

# List available results
echo ""
echo "Available results on remote:"
ssh "${REMOTE_USER}@${REMOTE_IP}" "ls -la ${REMOTE_RESULTS} 2>/dev/null | tail -20"
echo ""

if [ "$RESULTS_ONLY" = true ]; then
    # Sync only JSON and PNG files
    echo "Syncing JSON results and PNG plots only..."
    rsync -avzP \
        --include='*.json' \
        --include='*.png' \
        --exclude='*' \
        "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_RESULTS}" "$LOCAL_RESULTS" || SYNC_SUCCESS=false
else
    # Sync everything
    echo "Syncing all results..."
    rsync -avzP "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_RESULTS}" "$LOCAL_RESULTS" || SYNC_SUCCESS=false
fi

# ============================================================================
# Summary
# ============================================================================
if [ "$SYNC_SUCCESS" = true ]; then
    echo ""
    echo "========================================================"
    echo "✅ Sync completed successfully!"
    echo "========================================================"
    echo "Results saved to: $LOCAL_RESULTS"
    echo ""
    echo "Files synced:"
    ls -la "$LOCAL_RESULTS" 2>/dev/null | tail -10
else
    echo ""
    echo "========================================================"
    echo "❌ Error: Sync failed. Please check the IP address and SSH access."
    echo "========================================================"
    exit 1
fi

