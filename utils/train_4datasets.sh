#!/bin/bash
# Throwaway script to train full 4-phase pipeline on electricity, traffic, PeMS, and solar_Alabama
# using DiT backbone via Slurm sbatch.

mkdir -p results/bootstrap

DATASETS=("electricity" "traffic" "PeMS" "solar_Alabama")

for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" = "electricity" ] || [ "$DATASET" = "traffic" ]; then
        WALLTIME="48:00:00"
    else
        WALLTIME="08:00:00"
    fi
    
    echo "==================================================================="
    echo "Submitting full 4-phase pipeline for $DATASET ($WALLTIME)"
    echo "==================================================================="
    
    cat <<EOF | sbatch
#!/bin/bash
#SBATCH --job-name=dit-full-${DATASET}
#SBATCH --account=aip-boyuwang
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=results/bootstrap/%x-%j.out
#SBATCH --error=results/bootstrap/%x-%j.err

source .venv/bin/activate

echo "Starting training for $DATASET using DiT backbone"
python3 models/diffusion_tsf/train_multivariate_pipeline.py \\
    --mode full \\
    --dataset "${DATASET}" \\
    --model-type dit
EOF

done

echo "All jobs submitted."
