#!/bin/bash
source /home/cao/ts-sandbox/venv/bin/activate
cd /home/cao/ts-sandbox/models/diffusion_tsf
mkdir -p visualizations/batch_check_20260103

checkpoints=(
    "best_model"
    "trial_0_best"
    "trial_3_best"
    "trial_6_best"
    "trial_9_best"
    "trial_12_best"
)

for trial in "${checkpoints[@]}"; do
    echo "Running visualization for $trial..."
    python3 visualize.py --model-path checkpoints/diffusion_tsf_electricity_20260103_182614/${trial}.pt --output-dir visualizations/batch_check_20260103/${trial} --num-samples 5
done

