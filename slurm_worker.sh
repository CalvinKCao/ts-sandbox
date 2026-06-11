#!/bin/bash
# =============================================================================
# Pure compute-node worker script for diffusion pipeline.
#
# USAGE (do not call directly; use submit_grid.sh):
#   sbatch slurm_worker.sh --config configs/binary_anchor.yaml --dataset ETTh1
#
# Venv: node-local fast path — rebuilds on $SLURM_TMPDIR from
# setup/requirements-killarney.txt (generate via setup/killarney_freeze_requirements.sh).
# =============================================================================

set -euo pipefail

PY_ARGS=("$@")

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"
echo "=========================================="

STORE="${GRID_STORE:-$SCRATCH/ts-sandbox/results}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
REQ="$REPO/setup/requirements-killarney.txt"

[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch, optuna, wandb, einops, yaml; assert torch.cuda.is_available(), 'CUDA is not available (check driver compatibility)!'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1

cd "$REPO"
if [[ ! -f "models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: slurm_worker.sh must be submitted from repo root." >&2
    exit 1
fi

DATE_STR="${GRID_DATE_STR:-$(date +%m-%d)}"
DS="${GRID_DATASET:-unknown}"
CFG_NAME="${GRID_CFG_NAME:-run}"
RUN_STEM="${GRID_RUN_STEM:-${DATE_STR}-${SLURM_JOB_ID}-${DS}-${CFG_NAME}}"


CKPT_DIR="$STORE/ckpts/${RUN_STEM}"
DATA_DIR="$STORE/datasets/${RUN_STEM}"
mkdir -p "$CKPT_DIR" "$DATA_DIR"

# Benchmark CSVs live in the repo clone ($SCRATCH/ts-sandbox/datasets), not under $STORE/datasets.
BENCHMARK_DATASETS="${DATASETS_DIR:-$REPO/datasets}"
[[ -d "$BENCHMARK_DATASETS" ]] || {
    echo "ERROR: benchmark data directory not found at $BENCHMARK_DATASETS" >&2
    exit 1
}

echo "Repo: $PWD"
echo "Checkpoints: $CKPT_DIR"
echo "Results: $DATA_DIR"
echo "Benchmark CSVs: $BENCHMARK_DATASETS"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') ($(nvidia-smi -L 2>/dev/null | head -1 || echo none))"

PY_ARGS+=(
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$DATA_DIR"
    --datasets-dir "$BENCHMARK_DATASETS"
)

echo "[train] Starting pipeline: ${PY_ARGS[*]}"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
