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

_load_modules() {
    module purge 2>/dev/null || true
    module load StdEnv/2023 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || true
    module load cudnn/8.9 2>/dev/null || true
}

pip_retry() {
    local max_attempts=5 delay=20 attempt
    for attempt in $(seq 1 "$max_attempts"); do
        if "$@"; then
            return 0
        fi
        if [[ "$attempt" -lt "$max_attempts" ]]; then
            echo "[setup] pip failed (attempt ${attempt}/${max_attempts}), retry in ${delay}s..."
            sleep "$delay"
            delay=$((delay + 2))
        fi
    done
    echo "[setup] pip failed after ${max_attempts} attempts: $*" >&2
    return 1
}

setup_job_venv() {
    if [[ ! -f "$REQ" ]]; then
        echo "ERROR: missing $REQ" >&2
        echo "  On a Killarney login node run: ./setup/killarney_freeze_requirements.sh" >&2
        exit 1
    fi
    if [[ -z "${SLURM_TMPDIR:-}" ]]; then
        echo "ERROR: SLURM_TMPDIR is not set." >&2
        exit 1
    fi

    _load_modules
    if ! command -v virtualenv >/dev/null 2>&1; then
        echo "ERROR: virtualenv not available after module load." >&2
        exit 1
    fi

    echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
    virtualenv --no-download "$SLURM_TMPDIR/env"
    # shellcheck source=/dev/null
    source "$SLURM_TMPDIR/env/bin/activate"
    pip_retry pip install --no-index --upgrade pip -q
    pip_retry pip install --no-index -r "$REQ" -q

    python -c "import torch, optuna, wandb, einops, yaml" || {
        echo "[setup] ERROR: pipeline deps missing after install from $REQ" >&2
        exit 1
    }
}

setup_job_venv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

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

# Benchmark CSVs (shared). Not the per-job results dir under datasets/${RUN_STEM}.
_resolve_shared_datasets() {
    if [[ -n "${DATASETS_DIR:-}" ]]; then
        echo "$DATASETS_DIR"
        return
    fi
    if [[ -f "$REPO/datasets/ETT-small/ETTh1.csv" ]]; then
        echo "$REPO/datasets"
        return
    fi
    if [[ -f "$STORE/datasets/ETT-small/ETTh1.csv" ]]; then
        echo "$STORE/datasets"
        return
    fi
    echo "$REPO/datasets"
}
SHARED_DATASETS="$(_resolve_shared_datasets)"

echo "Repo: $PWD"
echo "Checkpoints: $CKPT_DIR"
echo "Results: $DATA_DIR"
echo "Datasets (benchmark CSVs): $SHARED_DATASETS"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

HAS_CKPT=0
HAS_DATASETS=0
for arg in "${PY_ARGS[@]}"; do
    [[ "$arg" == "--checkpoint-dir" ]] && HAS_CKPT=1
    [[ "$arg" == "--datasets-dir" ]] && HAS_DATASETS=1
done

if [[ "$HAS_CKPT" -eq 0 ]]; then
    PY_ARGS+=(--checkpoint-dir "$CKPT_DIR" --results-dir "$DATA_DIR")
fi
if [[ "$HAS_DATASETS" -eq 0 ]]; then
    PY_ARGS+=(--datasets-dir "$SHARED_DATASETS")
fi

if [[ "${GRID_RESUME:-0}" == "1" ]]; then
  has_resume=0
  for arg in "${PY_ARGS[@]}"; do
    [[ "$arg" == "--resume" ]] && has_resume=1
  done
  [[ "$has_resume" -eq 0 ]] && PY_ARGS+=(--resume)
fi

for ((i = 0; i < ${#PY_ARGS[@]}; i++)); do
    if [[ "${PY_ARGS[i]}" == "--config" ]]; then
        cfg="${PY_ARGS[i + 1]}"
        [[ "$cfg" == /* ]] || cfg="$REPO/$cfg"
        if [[ ! -f "$cfg" ]]; then
            echo "ERROR: config not found: $cfg" >&2
            echo "  Sync repo (git pull in $REPO) and resubmit." >&2
            exit 1
        fi
        break
    fi
done

echo "[train] Starting pipeline: ${PY_ARGS[*]}"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
