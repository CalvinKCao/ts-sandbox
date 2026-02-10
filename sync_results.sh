#!/bin/bash
# sync_results.sh - Sync training results from remote GPU to local machine
#
# This script downloads the latest results and checkpoints from a remote server.
#
# Usage (run from LOCAL machine):
#   ./sync_results.sh                          # Use default remote
#   ./sync_results.sh user@remote-server       # Specify remote
#   ./sync_results.sh user@remote-server /path/to/remote/project  # Custom path
#
# Examples:
#   ./sync_results.sh cao@gpu-server
#   ./sync_results.sh cao@192.168.1.100 /home/cao/ts-sandbox
#
# What gets synced:
#   - results_7var/       (evaluation results, summary.csv)
#   - checkpoints_7var/   (training manifests, metadata - NOT model weights by default)
#   - training logs

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
DEFAULT_REMOTE_PATH="/home/cao/ts-sandbox"
LOCAL_RESULTS_DIR="./synced_results"

# Parse args
REMOTE_HOST="${1:-}"
REMOTE_PATH="${2:-$DEFAULT_REMOTE_PATH}"

if [ -z "$REMOTE_HOST" ]; then
    echo -e "${RED}Usage: $0 user@remote-server [remote-path]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 cao@gpu-server"
    echo "  $0 cao@192.168.1.100 /home/cao/ts-sandbox"
    exit 1
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Syncing Results from Remote${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Remote: ${GREEN}$REMOTE_HOST:$REMOTE_PATH${NC}"
echo -e "Local:  ${GREEN}$LOCAL_RESULTS_DIR${NC}"
echo ""

# Create local directory
mkdir -p "$LOCAL_RESULTS_DIR"

# ============================================================================
# Sync Results (lightweight - just JSON/CSV)
# ============================================================================
echo -e "${YELLOW}[1/3] Syncing evaluation results...${NC}"

rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_PATH/models/diffusion_tsf/results_7var/" \
    "$LOCAL_RESULTS_DIR/results_7var/" \
    2>/dev/null || echo -e "${YELLOW}  (No results yet)${NC}"

# ============================================================================
# Sync Manifests & Metadata (NOT weights)
# ============================================================================
echo -e "${YELLOW}[2/3] Syncing training metadata...${NC}"

# Only sync JSON files (manifests, metadata), not .pt checkpoint files
rsync -avz --progress \
    --include="*/" \
    --include="*.json" \
    --exclude="*.pt" \
    "$REMOTE_HOST:$REMOTE_PATH/models/diffusion_tsf/checkpoints_7var/" \
    "$LOCAL_RESULTS_DIR/checkpoints_7var/" \
    2>/dev/null || echo -e "${YELLOW}  (No checkpoints yet)${NC}"

# ============================================================================
# Sync Training Log
# ============================================================================
echo -e "${YELLOW}[3/3] Syncing training log...${NC}"

rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_PATH/models/diffusion_tsf/train_7var.log" \
    "$LOCAL_RESULTS_DIR/" \
    2>/dev/null || echo -e "${YELLOW}  (No log yet)${NC}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}  Sync Complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Show what we got
if [ -f "$LOCAL_RESULTS_DIR/results_7var/summary.csv" ]; then
    echo -e "${GREEN}Summary CSV:${NC}"
    echo "  $LOCAL_RESULTS_DIR/results_7var/summary.csv"
    echo ""
    
    # Show last few lines if available
    if command -v column &> /dev/null; then
        echo -e "${GREEN}Latest Results:${NC}"
        head -1 "$LOCAL_RESULTS_DIR/results_7var/summary.csv" | tr ',' '\t'
        tail -5 "$LOCAL_RESULTS_DIR/results_7var/summary.csv" | tr ',' '\t' | column -t -s $'\t'
    else
        echo -e "${GREEN}Latest Results (last 5):${NC}"
        tail -5 "$LOCAL_RESULTS_DIR/results_7var/summary.csv"
    fi
    echo ""
fi

# Count models
if [ -f "$LOCAL_RESULTS_DIR/checkpoints_7var/training_manifest.json" ]; then
    COMPLETE=$(grep -o '"status": "complete"' "$LOCAL_RESULTS_DIR/checkpoints_7var/training_manifest.json" 2>/dev/null | wc -l)
    PENDING=$(grep -o '"status": "pending"' "$LOCAL_RESULTS_DIR/checkpoints_7var/training_manifest.json" 2>/dev/null | wc -l)
    IN_PROGRESS=$(grep -o '"status": "in_progress"' "$LOCAL_RESULTS_DIR/checkpoints_7var/training_manifest.json" 2>/dev/null | wc -l)
    
    echo -e "${GREEN}Training Progress:${NC}"
    echo "  Complete:    $COMPLETE"
    echo "  In Progress: $IN_PROGRESS"
    echo "  Pending:     $PENDING"
fi

echo ""
echo -e "Results saved to: ${GREEN}$LOCAL_RESULTS_DIR${NC}"



