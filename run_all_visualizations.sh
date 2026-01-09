#!/bin/bash

# Script to run visualization on all January 5 direct training models
# Logs output to run_all_visualizations.log

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_FILE="run_all_visualizations.log"
echo "Starting visualization run at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Activate virtual environment
echo "Activating virtual environment..." | tee -a "$LOG_FILE"
source venv/bin/activate

cd models/diffusion_tsf

# List of all Jan 5 direct training directories with model_best.pt
directories=(
    "direct_train_20260105_143327"
    "direct_train_20260105_171309"
    "direct_train_20260105_173518"
    "direct_train_20260105_173834"
    "direct_train_20260105_174154"
    "direct_train_20260105_174334"
    "direct_train_20260105_175110"
    "direct_train_20260105_194838"
    "direct_train_20260105_195026"
    "direct_train_20260105_210011"
    "direct_train_20260105_210148"
    "direct_train_20260105_210639"
    "direct_train_20260105_213057"
    "direct_train_20260105_213206"
    "direct_train_20260105_213727"
)

total=${#directories[@]}
completed=0

echo "Found $total models to process" | tee -a "../$LOG_FILE"
echo "" | tee -a "../$LOG_FILE"

for dir in "${directories[@]}"; do
    completed=$((completed + 1))
    echo -e "${BLUE}[$(date '+%H:%M:%S')] Processing $completed/$total: $dir${NC}" | tee -a "../$LOG_FILE"
    echo "Command: python3 visualize.py --model-path checkpoints/$dir/model_best.pt --output-dir visualizations_$dir" | tee -a "../$LOG_FILE"

    start_time=$(date +%s)

    # Run the command and capture both stdout and stderr
    if python3 visualize.py --model-path checkpoints/$dir/model_best.pt --output-dir visualizations_$dir --num-samples 2 >> "../$LOG_FILE" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✓ Completed $dir in ${duration}s${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}✗ Failed $dir${NC}" | tee -a "../$LOG_FILE"
    fi

    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

echo "" | tee -a "../$LOG_FILE"
echo -e "${GREEN}All visualizations completed!${NC}" | tee -a "../$LOG_FILE"
echo "Total processed: $completed/$total" | tee -a "../$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "../$LOG_FILE"
echo "Finished at $(date)" | tee -a "../$LOG_FILE"
