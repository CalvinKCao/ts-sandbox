#!/usr/bin/env bash
# Run visualize_comparison only on *completed* window-norm ablation runs (see generate_completed_report.py).
# Requires: repo root, .venv, datasets under ./datasets, pulled ckpts with subset metadata.json + best.pt.
#
# Usage:
#   ./utils/window_norm_ablate/visualize_completed.sh
#   RESULTS_DIR=/path/to/results ./utils/window_norm_ablate/visualize_completed.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RESULTS_DIR="${RESULTS_DIR:-./results}"
OUT_BASE="${OUT_BASE:-./results/viz/window_norm_ablate}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
VARS="${VARS:-3}"
ENSEMBLE="${ENSEMBLE:-1}"
MIN_JOB_ID="${MIN_JOB_ID:-}"

mkdir -p "$OUT_BASE"

is_complete_log() {
  local log="$1"
  grep -q "PIPELINE COMPLETE" "$log" && grep -q "Job completed:" "$log"
}

n=0
for run_dir in "$RESULTS_DIR"/*wn-[abc]-*/; do
  [[ -d "$run_dir" ]] || continue
  name="$(basename "$run_dir")"
  if [[ -n "$MIN_JOB_ID" ]]; then
    jid="$(echo "$name" | cut -d'-' -f3)"
    if [[ "$jid" =~ ^[0-9]+$ ]] && [[ "$jid" -lt "$MIN_JOB_ID" ]]; then
      continue
    fi
  fi
  [[ "$name" == *wn-a-* || "$name" == *wn-b-* || "$name" == *wn-c-* ]] || continue
  log="$(ls "$run_dir"/logs/*.log 2>/dev/null | head -1 || true)"
  [[ -n "$log" ]] || continue
  is_complete_log "$log" || { echo "Skip (incomplete): $name"; continue; }
  ckpt="$run_dir/ckpts"
  [[ -d "$ckpt" ]] || { echo "Skip (no ckpts): $name"; continue; }

  out="$OUT_BASE/$name"
  mkdir -p "$out"
  echo "Visualizing: $name -> $out"
  python -m models.diffusion_tsf.visualize_comparison \
    --checkpoint-dir "$ckpt" \
    --output-dir "$out" \
    --num-samples "$NUM_SAMPLES" \
    --vars "$VARS" \
    --ensemble "$ENSEMBLE"
  n=$((n + 1))
done

if [[ "$n" -eq 0 ]]; then
  echo "No completed wn-* runs with ckpts found under $RESULTS_DIR." >&2
  exit 1
fi
echo "Done: $n run(s). Plots under $OUT_BASE"
