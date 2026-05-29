#!/bin/bash
# Write results/jobs/<RUN_STEM>/eval_preamble.sh (module/venv/pip retries for eval workers).

set -euo pipefail

JOB_DIR="${1:?JOB_DIR required}"
mkdir -p "$JOB_DIR"
PREAMBLE_FILE="$JOB_DIR/eval_preamble.sh"

cat > "$PREAMBLE_FILE" <<'PREAMBLE'
set -euo pipefail
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"

module_purge() {
    module --force purge 2>/dev/null || module purge 2>/dev/null || true
}

modules_ok=0
for _try in 1 2 3; do
    module_purge
    if module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null \
        && command -v virtualenv >/dev/null; then
        modules_ok=1
        break
    fi
    echo "[setup] module/virtualenv attempt ${_try}/3 failed (CVMFS/lmod?); sleep 30s..."
    sleep 30
done
if [[ "$modules_ok" -ne 1 ]]; then
    echo "[setup] ERROR: could not load modules or virtualenv on ${SLURMD_NODENAME:-unknown}" >&2
    exit 1
fi

echo "[setup] Building venv on $SLURM_TMPDIR..."
venv_ok=0
for _try in 1 2 3; do
    if virtualenv --no-download "$SLURM_TMPDIR/env"; then
        venv_ok=1
        break
    fi
    echo "[setup] virtualenv attempt ${_try}/3 failed; sleep 30s..."
    sleep 30
done
if [[ "$venv_ok" -ne 1 ]]; then
    echo "[setup] ERROR: virtualenv failed on ${SLURMD_NODENAME:-unknown}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"

pip_ok=0
for _try in 1 2 3; do
    if pip install --no-index --upgrade pip -q \
        && pip install --no-index \
            'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm optuna wandb einops \
            -q; then
        pip_ok=1
        break
    fi
    echo "[setup] pip install attempt ${_try}/3 failed (CVMFS I/O?); sleep 30s..."
    sleep 30
done
if [[ "$pip_ok" -ne 1 ]]; then
    echo "[setup] ERROR: pip install failed on ${SLURMD_NODENAME:-unknown}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
cd "$REPO"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY
PREAMBLE

echo "Wrote $PREAMBLE_FILE"
