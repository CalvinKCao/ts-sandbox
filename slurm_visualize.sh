#!/bin/bash
#SBATCH --job-name=diffusion-viz
#SBATCH --account=def-boyuwang
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Quick visualization job — should take < 10 minutes

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

# Auto-detect PROJECT
if [ -z "$PROJECT" ]; then
    FIRST_PROJECT=$(ls -d $HOME/projects/def-* 2>/dev/null | head -1)
    if [ -n "$FIRST_PROJECT" ]; then
        export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
source "$STORAGE_ROOT/venv/bin/activate"

cd "$HOME/ts-sandbox"

# Checkpoints: try PROJECT storage first, fall back to local repo
if [ -f "$STORAGE_ROOT/checkpoints/training_manifest.json" ]; then
    CKPT_DIR="$STORAGE_ROOT/checkpoints"
elif [ -f "$HOME/ts-sandbox/models/diffusion_tsf/checkpoints_7var/training_manifest.json" ]; then
    CKPT_DIR="$HOME/ts-sandbox/models/diffusion_tsf/checkpoints_7var"
else
    echo "ERROR: No training manifest found in either location!"
    echo "  Tried: $STORAGE_ROOT/checkpoints/"
    echo "  Tried: $HOME/ts-sandbox/models/diffusion_tsf/checkpoints_7var/"
    exit 1
fi
VIZ_DIR="$STORAGE_ROOT/results/viz"

echo "Checkpoint dir: $CKPT_DIR"
echo "Output dir: $VIZ_DIR"

python -m models.diffusion_tsf.visualize_7var \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$VIZ_DIR" \
    --num-samples 3 \
    "$@"

echo "Done! Results in: $VIZ_DIR"

