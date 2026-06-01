#!/bin/bash
# =============================================================================
# Slurm worker: one dataset × cfg-scale anchor eval (eval_mmpd_gaussian_anchor.py).
# Submit via submit_cfg_ablation.sh — do not use sbatch --wrap.
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME:-cfg}  ID: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"
echo "=========================================="

USER="${USER:-$(whoami)}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -f "$REPO/utils/eval_mmpd_gaussian_anchor.py" ]]; then
    echo "ERROR: submit from repo root (missing utils/eval_mmpd_gaussian_anchor.py)" >&2
    exit 1
fi

_resolve_store() {
    local cand
    for cand in \
        "${CFG_STORE:-}" \
        "${STORE:-}" \
        "${SCRATCH:-}/${USER}/ts-sandbox/results" \
        "${SCRATCH:-}/ts-sandbox/results" \
        "$REPO/results"; do
        if [[ -n "$cand" && -d "$cand" ]]; then
            echo "$cand"
            return 0
        fi
    done
    echo "$REPO/results"
}

STORE="$(_resolve_store)"

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

install_eval_deps() {
    pip_retry pip install --no-index --upgrade pip -q 2>/dev/null || pip_retry pip install -U pip -q
    if ! python -c "import torch" 2>/dev/null; then
        if ! pip_retry pip install --no-index 'torch==2.11.0+computecanada' torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
            pip_retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
            pip_retry pip install numpy pandas scipy scikit-learn tqdm -q
        fi
    fi
    pip_retry pip install optuna einops pyyaml scikit-learn -q
    if ! pip_retry pip install --no-index matplotlib -q 2>/dev/null; then
        pip_retry pip install matplotlib -q 2>/dev/null || true
    fi
    python -c "import torch, yaml, sklearn"
}

_find_venv() {
    local v
    for v in \
        "$STORE/venv" \
        "${SCRATCH:-}/${USER}/ts-sandbox/results/venv" \
        "${SCRATCH:-}/ts-sandbox/results/venv" \
        "$REPO/results/venv"; do
        if [[ -x "${v}/bin/python" ]]; then
            echo "$v"
            return 0
        fi
    done
    return 1
}

if venv_path="$(_find_venv)"; then
    echo "[setup] Using persistent venv: $venv_path"
    _load_modules
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    install_eval_deps
else
    echo "[setup] No persistent venv; building on \${SLURM_TMPDIR:-/tmp}..."
    _load_modules
    if ! command -v python >/dev/null 2>&1; then
        echo "[setup] ERROR: python unavailable. Create \$SCRATCH/ts-sandbox/results/venv on login node." >&2
        exit 1
    fi
    python -m venv "${SLURM_TMPDIR:-/tmp}/env"
    # shellcheck source=/dev/null
    source "${SLURM_TMPDIR:-/tmp}/env/bin/activate"
    install_eval_deps
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

exec python -u "$REPO/utils/eval_mmpd_gaussian_anchor.py" "$@"
