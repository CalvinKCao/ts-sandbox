#!/usr/bin/env bash
# Narval A100 launcher for the full ordinal patch refinement + discriminator kill test.
# Login: ./submit_ordinal_patch_refinement_full_narval.sh --dataset ETTh1 --resolution 256
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLUTION=256
DATASET=ETTh1
WALL="8:00:00"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --time) WALL="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  [[ "$(hostname)" == *"narval"* ]] || { echo "ERROR: submit this from a Narval login node." >&2; exit 2; }
  REPO="${SCRATCH:-}/ts-sandbox"
  [[ -d "$REPO" ]] || { echo "ERROR: expected checkout at $REPO" >&2; exit 2; }
  mkdir -p "$REPO/results/logs"
  JOB_NAME="ord-full-${DATASET}-${RESOLUTION}"
  exec sbatch --job-name="$JOB_NAME" --account=def-boyuwang \
    --nodes=1 --gpus=a100:1 --cpus-per-task=8 --mem=80G --time="$WALL" \
    --output="$REPO/results/logs/${JOB_NAME}-%j.log" \
    --error="$REPO/results/logs/${JOB_NAME}-%j.log" --mail-type=FAIL \
    --export=ALL,ORDINAL_REPO="$REPO",ORDINAL_R="$RESOLUTION",ORDINAL_D="$DATASET",ORDINAL_SMOKE="$SMOKE" \
    "$SCRIPT_DIR/submit_ordinal_patch_refinement_full_narval.sh"
fi

REPO="${ORDINAL_REPO:-$SCRIPT_DIR}"
cd "$REPO"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" && -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: requirements file or SLURM_TMPDIR missing" >&2; exit 1; }
module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv unavailable after module load" >&2; exit 1; }
virtualenv --no-download "$SLURM_TMPDIR/ordinal-refine-full-env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/ordinal-refine-full-env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; assert torch.cuda.is_available(); print('[gpu]', torch.cuda.get_device_name(0))"
OUTPUT="$REPO/results/ordinal_patch_refinement_killtest/full-${ORDINAL_D}-${ORDINAL_R}-${SLURM_JOB_ID}"
SMOKE_FLAG=()
if [[ "${ORDINAL_SMOKE:-0}" == "1" ]]; then
  SMOKE_FLAG=(--smoke)
  OUTPUT="${OUTPUT}-smoke"
fi
python -u -m experiments.ordinal_patch_refinement_killtest.full_experiment \
  --dataset "${ORDINAL_D}" --resolution "${ORDINAL_R}" --output "$OUTPUT" "${SMOKE_FLAG[@]}"
