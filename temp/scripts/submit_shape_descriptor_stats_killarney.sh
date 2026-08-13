#!/bin/bash
# Compute level/scale-invariant shape descriptors from existing forecast packs.
set -euo pipefail

PY_REL="temp/scripts/compute_shape_descriptor_stats.py"
BINARY_ROOT="${BINARY_ROOT:-/scratch/ccao87/ts-sandbox-ordinal-fine}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/scratch/ccao87/ts-sandbox}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/ccao87/ts-sandbox-corrupt-20260729-013232/.venv/bin/python}"
WINDOW_LENGTH=96

while [[ $# -gt 0 ]]; do
    case "$1" in
        --window-length) WINDOW_LENGTH="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument $1" >&2; exit 1 ;;
    esac
done
[[ "$WINDOW_LENGTH" =~ ^(8|96)$ ]] || { echo "ERROR: --window-length must be 8 or 96" >&2; exit 1; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    REPO="$(pwd)"
    [[ -f "$REPO/$PY_REL" ]] || { echo "ERROR: submit from the repo root" >&2; exit 1; }
    mkdir -p "$REPO/results/logs"
    sbatch \
        --chdir="$REPO" \
        --job-name="c128-shape-l${WINDOW_LENGTH}" \
        --account=aip-boyuwang \
        --time=1:00:00 \
        --nodes=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --output="$REPO/results/logs/c128-shape-l${WINDOW_LENGTH}-%j.out" \
        --error="$REPO/results/logs/c128-shape-l${WINDOW_LENGTH}-%j.out" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$0" --window-length "$WINDOW_LENGTH"
    exit 0
fi

REPO="${SLURM_SUBMIT_DIR:?ERROR: SLURM_SUBMIT_DIR unset}"

[[ -f "$REPO/$PY_REL" ]] || { echo "ERROR: missing $REPO/$PY_REL" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing scientific Python $PYTHON_BIN" >&2; exit 1; }
[[ -d "$BINARY_ROOT/results/datasets" ]] || { echo "ERROR: missing binary root $BINARY_ROOT" >&2; exit 1; }
[[ -d "$REFERENCE_ROOT/results/datasets" ]] || { echo "ERROR: missing reference root $REFERENCE_ROOT" >&2; exit 1; }

OUT="$REPO/results/shape_descriptor_c128_all/datasets/${SLURM_JOB_ID}-l${WINDOW_LENGTH}"
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
"$PYTHON_BIN" -u "$PY_REL" \
    --binary-root "$BINARY_ROOT" \
    --reference-root "$REFERENCE_ROOT" \
    --output-dir "$OUT" \
    --window-length "$WINDOW_LENGTH"
echo "RESULTS=$OUT"
