#!/bin/bash
#SBATCH --job-name=diffusion-compare
#SBATCH --account=def-boyuwang
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Comparison visualization: iTransformer-only vs Diffusion, ~3 samples per dataset

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

if [ -z "$PROJECT" ]; then
    FIRST_PROJECT=$(ls -d $HOME/projects/def-* 2>/dev/null | head -1)
    if [ -n "$FIRST_PROJECT" ]; then
        export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
source "$STORAGE_ROOT/venv/bin/activate"

cd "$HOME/ts-sandbox"

# Find checkpoint dir
if ls "$STORAGE_ROOT/checkpoints"/*/metadata.json &>/dev/null; then
    CKPT_DIR="$STORAGE_ROOT/checkpoints"
else
    echo "ERROR: No subset checkpoints found in $STORAGE_ROOT/checkpoints/"
    exit 1
fi

OUT_DIR="$STORAGE_ROOT/results/viz/comparison"

echo "Checkpoint dir: $CKPT_DIR"
echo "Output dir: $OUT_DIR"

python -m models.diffusion_tsf.visualize_comparison \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUT_DIR" \
    --num-samples 3 \
    --vars 3 \
    --ensemble 3 \
    "$@"

echo "===================="
echo "Done! Sync to local:"
echo "  rsync -avz user@narval:~/projects/def-*/diffusion-tsf/results/viz/comparison/ ./synced_results/viz/comparison/"
echo "===================="
