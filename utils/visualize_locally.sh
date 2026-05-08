#!/bin/bash
# Run comparison plots on pulled Slurm ckpts. Requires utils/pull_results.sh first so
# *_itransformer_finetuned.pt sits next to subset dirs under each .../ckpts/.
#
# Only runs visualize_comparison when this finds at least one direct child directory
# under .../ckpts/ that contains both metadata.json and best.pt (same layout run.sh writes).
# Empty or partial pulls (only pretrained_dim*, no subset folders) are skipped quietly.

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

# True if ckpts tree has a finetune subset folder (metadata + diffusion best.ckpt).
ckpt_dir_has_plottable_subset() {
    local dir="$1"
    local sub
    shopt -s nullglob
    for sub in "$dir"/*/; do
        [[ -d "$sub" ]] || continue
        if [[ -f "${sub}metadata.json" && -f "${sub}best.pt" ]]; then
            return 0
        fi
    done
    return 1
}

echo "Scanning $RESULTS_DIR for .../ckpts directories with subset checkpoints..."

CKPT_DIRS=$(find "$RESULTS_DIR" -type d -name "ckpts" 2>/dev/null | sort || true)

if [ -z "${CKPT_DIRS:-}" ]; then
    echo "No 'ckpts' directories found under $RESULTS_DIR."
    exit 1
fi

RAN=0
SKIPPED=0
while IFS= read -r CKPT_DIR; do
    [ -n "$CKPT_DIR" ] || continue
    if ! ckpt_dir_has_plottable_subset "$CKPT_DIR"; then
        echo "Skip (no subset with metadata.json + best.pt): $CKPT_DIR"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "Processing: $CKPT_DIR"
    python -m models.diffusion_tsf.visualize_comparison \
        --checkpoint-dir "$CKPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --vars "$VARS" \
        --ensemble 1
    RAN=$((RAN + 1))
done <<< "$CKPT_DIRS"

echo ""
if [ "$RAN" -eq 0 ]; then
    echo "ERROR: no ckpts directory contained a subset folder with both metadata.json and best.pt." >&2
    echo "Pull a full run first: ./utils/pull_results.sh '<job-folder>'" >&2
    exit 1
fi

echo "Done: ran visualization on $RAN checkpoint root(s) (skipped $SKIPPED without subset checkpoints)."
echo "Plots: $OUTPUT_DIR"
