#!/bin/bash
# =============================================================================
# Pure compute-node worker script for diffusion pipeline.
#
# USAGE (do not call directly; use submit_grid.sh):
#   sbatch slurm_worker.sh --config configs/binary_anchor.yaml --dataset ETTh1
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

USER="${USER:-$(whoami)}"
STORE="${GRID_STORE:-${SLURM_SUBMIT_DIR:-$PWD}/results}"
STORE_VENV="$STORE/venv"

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

install_pipeline_deps() {
    pip_retry pip install --no-index --upgrade pip -q 2>/dev/null || pip_retry pip install -U pip -q

    if ! python -c "import torch" 2>/dev/null; then
        if ! pip_retry pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
            pip_retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
            pip_retry pip install numpy pandas scipy scikit-learn tqdm -q
        fi
    fi

    # Idempotent: persistent venv may exist with only torch from an older partial setup.
    pip_retry pip install optuna wandb einops pyyaml -q

    if ! pip_retry pip install --no-index matplotlib -q 2>/dev/null; then
        pip_retry pip install matplotlib -q 2>/dev/null || {
            echo "[setup] WARN: matplotlib install failed; continuing without viz wheels."
            export DIFFUSION_TSF_SKIP_VIZ=1
        }
    fi

    python -c "import optuna, wandb, einops, yaml" || {
        echo "[setup] ERROR: pipeline deps missing after pip reconcile." >&2
        return 1
    }
}

activate_venv() {
    if [[ -x "$STORE_VENV/bin/python" ]]; then
        echo "[setup] Using persistent venv: $STORE_VENV"
        # shellcheck source=/dev/null
        source "$STORE_VENV/bin/activate"
        echo "[setup] Reconciling packages in persistent venv..."
        install_pipeline_deps
        return 0
    fi

    echo "[setup] Building venv on \$SLURM_TMPDIR..."
    virtualenv --no-download "$SLURM_TMPDIR/env"
    # shellcheck source=/dev/null
    source "$SLURM_TMPDIR/env/bin/activate"
    install_pipeline_deps
}

activate_venv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: slurm_worker.sh must be submitted from repo root." >&2
    exit 1
fi

DATE_STR="${GRID_DATE_STR:-$(date +%m-%d)}"
DS="${GRID_DATASET:-unknown}"
CFG_NAME="${GRID_CFG_NAME:-run}"
RUN_STEM="${GRID_RUN_STEM:-${DATE_STR}-${SLURM_JOB_ID}-${DS}-${CFG_NAME}}"

# Legacy flat layout: results/ckpts/ETTh1/ (pre per-job dirs fix)
if [[ "${GRID_RESUME:-0}" == "1" && -z "${GRID_RUN_STEM:-}" ]]; then
    if [[ -f "$STORE/ckpts/${DS}/metadata.json" ]]; then
        RUN_STEM="$DS"
    fi
fi

CKPT_DIR="$STORE/ckpts/${RUN_STEM}"
DATA_DIR="$STORE/datasets/${RUN_STEM}"
mkdir -p "$CKPT_DIR" "$DATA_DIR"

# Benchmark CSVs (shared). Not the per-job results dir under datasets/${RUN_STEM}.
_resolve_shared_datasets() {
    if [[ -n "${DATASETS_DIR:-}" ]]; then
        echo "$DATASETS_DIR"
        return
    fi
    local repo="${SLURM_SUBMIT_DIR:-$PWD}"
    if [[ -f "$repo/datasets/ETT-small/ETTh1.csv" ]]; then
        echo "$repo/datasets"
        return
    fi
    if [[ -f "$STORE/datasets/ETT-small/ETTh1.csv" ]]; then
        echo "$STORE/datasets"
        return
    fi
    echo "$repo/datasets"
}
SHARED_DATASETS="$(_resolve_shared_datasets)"

echo "Repo: $PWD"
echo "Checkpoints: $CKPT_DIR"
echo "Results: $DATA_DIR"
echo "Datasets (benchmark CSVs): $SHARED_DATASETS"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

PY_ARGS=("$@")
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

echo "[train] Starting pipeline: ${PY_ARGS[*]}"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
