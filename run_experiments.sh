#!/bin/bash
# run_experiments.sh
# This script runs two experimental scenarios across multiple time series datasets.
# Scenario 1: attention only near the bottleneck (level 2) + 0.2 guidance penalty
# Scenario 2: no attention in up/down paths (empty string) + 0.2 guidance penalty
#             This forces the bottleneck to be the *only* place with cross-variate attention.

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
source .venv/bin/activate

DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "exchange_rate")

for ds in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "Running Scenario 1: attn-near-bottleneck + 0.2 penalty on $ds"
    echo "=========================================================="
    python3 models/diffusion_tsf/train_multivariate_pipeline.py \
        --dataset "$ds" \
        --attention-levels "2" \
        --guidance-penalty-weight 0.2 \
        --subset-id "attn-bottleneck-pen-0.2"

    echo "=========================================================="
    echo "Running Scenario 2: 100% univariate (no cross-attn) + 0.2 penalty on $ds"
    echo "=========================================================="
    python3 models/diffusion_tsf/train_multivariate_pipeline.py \
        --dataset "$ds" \
        --disable-cross-attention \
        --guidance-penalty-weight 0.2 \
        --subset-id "100pct-univariate-pen-0.2"
done

echo "All experimental scenarios completed successfully!"
