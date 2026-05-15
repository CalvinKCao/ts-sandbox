#!/bin/bash
# Run comparison plots on pulled Slurm ckpts. Supports:
#   - Legacy: .../ckpts/<subset>/metadata.json + best.pt (+ *_itransformer_finetuned.pt).
#   - Joint e2e: .../ckpts/<subset>_joint_finetuned_gB.pt (or _gC.pt) with embedded config.
# Set JOB_GLOB (default *-ft-*) to limit which top-level result folders are scanned.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

RESULTS_DIR="${RESULTS_DIR:-./results}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/viz/comparison}"
# Shell glob relative to RESULTS_DIR; default matches joint finetune job folders.
JOB_GLOB="${JOB_GLOB:-*-ft-*}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
VARS="${VARS:-7}"
LOOKBACK_LENGTH="${LOOKBACK_LENGTH:-512}"
FORECAST_LENGTH="${FORECAST_LENGTH:-96}"

if [ ! -d "$RESULTS_DIR" ] || [ -z "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
    echo "No folders under $RESULTS_DIR. Run: ./utils/pull_results.sh '<MM-DD-JOBID-slug>'"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# True if ckpts tree has legacy finetune subset dirs (metadata + best.pt) or
# flat joint finetune checkpoints (*_joint_finetuned_g*.pt) next to subset dirs.
ckpt_dir_has_plottable_subset() {
    local dir="$1"
    local sub
    shopt -s nullglob
    for f in "$dir"/*_joint_finetuned_g*.pt; do
        [[ -f "$f" ]] && return 0
    done
    for sub in "$dir"/*/; do
        [[ -d "$sub" ]] || continue
        if [[ -f "${sub}metadata.json" && -f "${sub}best.pt" ]]; then
            return 0
        fi
    done
    return 1
}

echo "Scanning $RESULTS_DIR (job folder glob: $JOB_GLOB) for .../ckpts with legacy or joint finetune checkpoints..."

# Prefer .../<job>/ckpts where <job> matches JOB_GLOB, so we never fall back to unrelated
# experiment trees when pulled checkpoints are missing locally.
CKPT_DIRS=""
roots=()
shopt -s nullglob
for job in "$RESULTS_DIR"/*/; do
    [[ -d "${job}ckpts" ]] || continue
    bn=$(basename "${job%/}")
    [[ "$bn" == $JOB_GLOB ]] || continue
    roots+=("${job%/}/ckpts")
done
if ((${#roots[@]})); then
    CKPT_DIRS=$(printf '%s\n' "${roots[@]}" | sort -u)
elif [[ "$JOB_GLOB" == '*' ]]; then
    CKPT_DIRS=$(find "$RESULTS_DIR" -type d -name "ckpts" 2>/dev/null | sort || true)
else
    CKPT_DIRS=""
fi

if [ -z "${CKPT_DIRS:-}" ]; then
    echo "No .../ckpts directories under job folders matching $JOB_GLOB." >&2
    echo "Pull checkpoints first, or set JOB_GLOB='*' to scan every ckpts tree." >&2
    exit 1
fi

RAN=0
SKIPPED=0
while IFS= read -r CKPT_DIR; do
    [ -n "$CKPT_DIR" ] || continue
    if ! ckpt_dir_has_plottable_subset "$CKPT_DIR"; then
        echo "Skip (no legacy subset nor joint *_joint_finetuned_g*.pt): $CKPT_DIR"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "Processing: $CKPT_DIR"
    JOB_SLUG="$(basename "$(dirname "$CKPT_DIR")")"
    python -m models.diffusion_tsf.visualize_comparison \
        --checkpoint-dir "$CKPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --output-tag "$JOB_SLUG" \
        --num-samples "$NUM_SAMPLES" \
        --vars "$VARS" \
        --lookback-length "$LOOKBACK_LENGTH" \
        --forecast-length "$FORECAST_LENGTH" \
        --ensemble 1
    RAN=$((RAN + 1))
done <<< "$CKPT_DIRS"

echo ""
if [ "$RAN" -eq 0 ]; then
    echo "ERROR: no ckpts directory had legacy (metadata.json + best.pt) or joint (*_joint_finetuned_g*.pt) checkpoints." >&2
    echo "Pull a full run first: ./utils/pull_results.sh '<job-folder>'" >&2
    exit 1
fi

echo "Done: ran visualization on $RAN checkpoint root(s) (skipped $SKIPPED without plottable checkpoints)."
echo "Plots: $OUTPUT_DIR"
