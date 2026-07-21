#!/usr/bin/env bash
# Narval A100 launcher for the boundary-centered oracle-coarse smoke test.
# Login node: ./submit_ordinal_patch_refinement_centered_narval_a100.sh --steps 200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=200
WALL="1:00:00"
WINDOW_INDEX=0
VARIATE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        --window-index) WINDOW_INDEX="$2"; shift 2 ;;
        --variate) VARIATE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ "$(hostname)" == *"narval"* ]] || { echo "ERROR: submit this from a Narval login node." >&2; exit 2; }
    REPO="${SCRATCH:-}/ts-sandbox"
    [[ -d "$REPO" ]] || { echo "ERROR: expected checkout at $REPO" >&2; exit 2; }
    mkdir -p "$REPO/results/logs"
    exec sbatch --job-name=ordinal-refine-centered --account=def-boyuwang \
        --nodes=1 --gpus=a100:1 --cpus-per-task=8 --mem=64G --time="$WALL" \
        --output="$REPO/results/logs/ordinal-refine-centered-%j.log" \
        --error="$REPO/results/logs/ordinal-refine-centered-%j.log" --mail-type=FAIL \
        --export=ALL,ORDINAL_REPO="$REPO",ORDINAL_STEPS="$STEPS",ORDINAL_WINDOW_INDEX="$WINDOW_INDEX",ORDINAL_VARIATE="$VARIATE" \
        "$SCRIPT_DIR/submit_ordinal_patch_refinement_centered_narval_a100.sh"
fi

REPO="${ORDINAL_REPO:-$SCRIPT_DIR}"
cd "$REPO"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" && -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: requirements file or SLURM_TMPDIR missing" >&2; exit 1; }
module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv unavailable after module load" >&2; exit 1; }
virtualenv --no-download "$SLURM_TMPDIR/ordinal-refine-env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/ordinal-refine-env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; assert torch.cuda.is_available(); print('[gpu]', torch.cuda.get_device_name(0))"
OUTPUT="$REPO/results/ordinal_patch_refinement_killtest/narval-centered-smoke-${SLURM_JOB_ID}"
python -u -m experiments.ordinal_patch_refinement_killtest.run_centered_narval_smoke \
    --steps "${ORDINAL_STEPS:-200}" --window-index "${ORDINAL_WINDOW_INDEX:-0}" \
    --variate "${ORDINAL_VARIATE:-0}" --output "$OUTPUT"
