#!/bin/bash
# Submit paired, native-horizon descriptor statistics using existing forecast packs.
set -euo pipefail

PY_REL="temp/scripts/compute_shape_descriptor_stats_paired_native.py"
BINARY_ROOT="${BINARY_ROOT:-/scratch/ccao87/ts-sandbox-ordinal-fine}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/scratch/ccao87/ts-sandbox}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/ccao87/ts-sandbox-corrupt-20260729-013232/.venv/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    REPO="$(pwd)"
    [[ -f "$REPO/$PY_REL" ]] || { echo "ERROR: submit from repo root" >&2; exit 1; }
    mkdir -p "$REPO/results/shape_descriptor_paired_native/logs"
    sbatch \
        --chdir="$REPO" \
        --job-name="paired-native-shape" \
        --account=aip-boyuwang \
        --time=1:30:00 \
        --nodes=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --output="$REPO/results/shape_descriptor_paired_native/logs/paired-native-shape-%j.log" \
        --error="$REPO/results/shape_descriptor_paired_native/logs/paired-native-shape-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$0"
    exit 0
fi

REPO="${SLURM_SUBMIT_DIR:?ERROR: SLURM_SUBMIT_DIR unset}"
[[ -f "$REPO/$PY_REL" ]] || { echo "ERROR: missing $REPO/$PY_REL" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing $PYTHON_BIN" >&2; exit 1; }
[[ -d "$BINARY_ROOT/results/datasets" ]] || { echo "ERROR: missing binary root" >&2; exit 1; }
[[ -d "$REFERENCE_ROOT/results/datasets" ]] || { echo "ERROR: missing reference root" >&2; exit 1; }

OUT="$REPO/results/shape_descriptor_paired_native/datasets/${SLURM_JOB_ID}-paired-native"
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
"$PYTHON_BIN" -u "$PY_REL" \
    --binary-root "$BINARY_ROOT" \
    --reference-root "$REFERENCE_ROOT" \
    --output-dir "$OUT" \
    --lengths 8 96 \
    --normalizations zscore demean
echo "RESULTS=$OUT"
