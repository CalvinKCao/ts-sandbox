#!/bin/bash
# Run comparison plots on pulled Slurm ckpts. Requires utils/pull_results.sh first so
# *_itransformer_finetuned.pt sits next to subset dirs under each .../ckpts/.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

RESULTS_DIR="${RESULTS_DIR:-./results}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/viz/comparison}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
VARS="${VARS:-7}"

if [ ! -d "$RESULTS_DIR" ] || [ -z "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
    echo "No folders under $RESULTS_DIR. Run: ./utils/pull_results.sh '<MM-DD-JOBID-slug>'"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Scanning $RESULTS_DIR for .../ckpts directories..."

CKPT_DIRS=$(find "$RESULTS_DIR" -type d -name "ckpts" 2>/dev/null | sort || true)

if [ -z "${CKPT_DIRS:-}" ]; then
    echo "No 'ckpts' directories found under $RESULTS_DIR."
    exit 1
fi

for CKPT_DIR in $CKPT_DIRS; do
    echo "Processing: $CKPT_DIR"
    python -m models.diffusion_tsf.visualize_comparison \
        --checkpoint-dir "$CKPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --vars "$VARS" \
        --ensemble 1
done

echo ""
echo "Visualization complete. Plots saved to $OUTPUT_DIR"
