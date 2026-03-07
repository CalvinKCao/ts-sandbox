#!/bin/bash
# run_7var_pipeline.sh - 7-Variate Training Pipeline
#
# Trains diffusion models on all datasets using 7-variate subsets.
# Each model is evaluated immediately after training - results saved incrementally.
# Safe to Ctrl+C anytime; resume with --resume.
#
# Usage:
#   ./run_7var_pipeline.sh                    # Train all (with inline eval)
#   ./run_7var_pipeline.sh --resume           # Resume interrupted training
#   ./run_7var_pipeline.sh --smoke-test       # Quick validation run
#   ./run_7var_pipeline.sh --status           # Show training status
#   ./run_7var_pipeline.sh --list             # List all subsets
#
# To sync results to local machine:
#   ./sync_results.sh user@this-server

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_DIR="models/diffusion_tsf/checkpoints_7var"
RESULTS_DIR="models/diffusion_tsf/results_7var"
TRAFFIC_DIR="datasets/traffic"

# Default settings
MODE="train"          # train, status, list
RESUME=false
SMOKE_TEST=false

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --status)
            MODE="status"
            shift
            ;;
        --list)
            MODE="list"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume        Resume interrupted training"
            echo "  --smoke-test    Quick validation run"
            echo "  --status        Show training status"
            echo "  --list          List all subsets"
            echo "  --help          Show this help message"
            echo ""
            echo "Note: Evaluation runs automatically after each model trains."
            echo "Use sync_results.sh to pull results to local machine."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  7-Variate Training Pipeline${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}[✓] Activated venv${NC}"
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}[✗] Python not found${NC}"
    exit 1
fi

# ============================================================================
# Traffic Data Recombination
# ============================================================================

recombine_traffic() {
    echo -e "${YELLOW}[INFO] Checking traffic data...${NC}"
    
    TRAFFIC_CSV="$TRAFFIC_DIR/traffic.csv"
    PART1="$TRAFFIC_DIR/traffic_part1.csv"
    PART2="$TRAFFIC_DIR/traffic_part2.csv"
    
    if [ -f "$TRAFFIC_CSV" ]; then
        echo -e "${GREEN}[✓] traffic.csv already exists${NC}"
        return 0
    fi
    
    if [ ! -f "$PART1" ] || [ ! -f "$PART2" ]; then
        echo -e "${YELLOW}[WARN] Traffic part files not found, skipping${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}[INFO] Recombining traffic data...${NC}"
    
    # Get header from part1
    head -n 1 "$PART1" > "$TRAFFIC_CSV"
    
    # Append data (skip headers)
    tail -n +2 "$PART1" >> "$TRAFFIC_CSV"
    tail -n +2 "$PART2" >> "$TRAFFIC_CSV"
    
    LINES=$(wc -l < "$TRAFFIC_CSV")
    echo -e "${GREEN}[✓] Created traffic.csv with $LINES rows${NC}"
}

# ============================================================================
# Status Check
# ============================================================================

show_status() {
    echo -e "${BLUE}Training Status:${NC}"
    python -m models.diffusion_tsf.train_7var_pipeline --status
}

# ============================================================================
# List Subsets
# ============================================================================

list_subsets() {
    echo -e "${BLUE}Available Subsets:${NC}"
    python -m models.diffusion_tsf.train_7var_pipeline --list-subsets
}

# ============================================================================
# Training
# ============================================================================

run_training() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Training (with inline evaluation)${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    # Build command
    CMD="python -m models.diffusion_tsf.train_7var_pipeline"
    
    if [ "$RESUME" = true ]; then
        CMD="$CMD --resume"
        echo -e "${YELLOW}[INFO] Resuming from last checkpoint${NC}"
    fi
    
    if [ "$SMOKE_TEST" = true ]; then
        CMD="$CMD --smoke-test"
        echo -e "${YELLOW}[INFO] Running smoke test (minimal data)${NC}"
    fi
    
    echo -e "${YELLOW}[INFO] Starting training...${NC}"
    echo -e "${YELLOW}[INFO] Each model is evaluated immediately after training${NC}"
    echo -e "${YELLOW}[INFO] Press Ctrl+C to safely interrupt${NC}"
    echo ""
    
    # Run training (allow interruption)
    set +e
    $CMD
    TRAIN_EXIT=$?
    set -e
    
    if [ $TRAIN_EXIT -eq 0 ]; then
        echo -e "${GREEN}[✓] All training & evaluation complete${NC}"
    elif [ $TRAIN_EXIT -eq 130 ]; then
        echo -e "${YELLOW}[INFO] Interrupted by user${NC}"
        echo -e "${YELLOW}[INFO] Run with --resume to continue${NC}"
    else
        echo -e "${RED}[✗] Failed with exit code $TRAIN_EXIT${NC}"
    fi
    
    return $TRAIN_EXIT
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Always recombine traffic first
    recombine_traffic
    
    case $MODE in
        "status")
            show_status
            ;;
        "list")
            list_subsets
            ;;
        "train")
            run_training
            
            # Show results summary if any exist
            if [ -f "$RESULTS_DIR/summary.csv" ]; then
                echo ""
                echo -e "${GREEN}Results saved to: $RESULTS_DIR${NC}"
                echo -e "${GREEN}Sync to local with: ./sync_results.sh user@this-server${NC}"
            fi
            ;;
    esac
}

# Run main
main

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Pipeline Complete${NC}"
echo -e "${BLUE}======================================${NC}"

