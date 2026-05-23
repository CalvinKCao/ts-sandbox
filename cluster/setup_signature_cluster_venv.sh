# Shared venv bootstrap for signature+MSE Slurm jobs.
# Source from slurm_signature_tune.sh / slurm_signature_finalize.sh

signature_pip_retry() {
    local attempt=1
    local max="${SIGNATURE_PIP_RETRIES:-4}"
    while [ "$attempt" -le "$max" ]; do
        if "$@"; then
            return 0
        fi
        echo "[setup] pip failed (attempt ${attempt}/${max}): $*"
        sleep $((attempt * 15))
        attempt=$((attempt + 1))
    done
    return 1
}

# reformer-pytorch is not always in the wheelhouse; deps must include product-key-memory.
signature_install_reformer_stack() {
    local stack="reformer-pytorch==1.4.4 axial-positional-embedding local-attention product-key-memory"
    if signature_pip_retry pip install --no-index $stack -q 2>/dev/null; then
        echo "[setup] reformer stack from wheelhouse"
        return 0
    fi
    echo "[setup] wheelhouse missing reformer stack; using PyPI"
    signature_pip_retry pip install $stack -q
}

signature_verify_venv() {
    local skip_cuda="${SIGNATURE_SKIP_CUDA_CHECK:-0}"
    python - <<PY
import os
import torch
import signatory
from reformer_pytorch import LSHSelfAttention
import product_key_memory

skip = os.environ.get("SIGNATURE_SKIP_CUDA_CHECK", "0") == "1"
if not skip:
    assert torch.cuda.is_available(), "CUDA is required for this Slurm job"
print(
    "venv ok:",
    "torch", torch.__version__,
    "signatory", getattr(signatory, "__version__", "unknown"),
    "cuda_skip" if skip else ("gpu " + torch.cuda.get_device_name(0)),
)
PY
}

signature_install_core_packages() {
    signature_pip_retry pip install --no-index --upgrade pip -q
    signature_pip_retry pip install --no-index \
        'torch==2.11.0+computecanada' \
        numpy pandas scipy scikit-learn tqdm einops \
        sqlalchemy colorlog alembic packaging -q
    signature_pip_retry pip install --no-index optuna -q \
        || signature_pip_retry pip install optuna -q
    signature_install_reformer_stack
    signature_pip_retry pip install signatory --no-build-isolation -q
}

signature_cluster_venv() {
    local store="${STORE:-}"
    local shared_venv="${store}/venv"

    if [ "${SIGNATURE_USE_SHARED_VENV:-0}" = "1" ] && [ -n "$store" ] && [ -d "$shared_venv" ]; then
        echo "[setup] Activating shared venv: $shared_venv"
        # shellcheck source=/dev/null
        source "$shared_venv/bin/activate"
        if ! signature_verify_venv; then
            echo "[setup] shared venv failed import check; rebuild with BUILD_SHARED_VENV=1"
            return 1
        fi
        return 0
    fi

    echo "[setup] Building node-local venv on $SLURM_TMPDIR"
    virtualenv --no-download "$SLURM_TMPDIR/env"
    # shellcheck source=/dev/null
    source "$SLURM_TMPDIR/env/bin/activate"
    signature_install_core_packages
    signature_verify_venv
}

signature_build_shared_venv() {
    local store="${STORE:-}"
    if [ -z "$store" ]; then
        echo "ERROR: set STORE (e.g. \$PROJECT/\$USER/ts-sandbox-signature) before BUILD_SHARED_VENV=1"
        return 1
    fi
    mkdir -p "$store"
    export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp}"
    rm -rf "$store/venv"
    virtualenv --no-download "$store/venv"
    # shellcheck source=/dev/null
    source "$store/venv/bin/activate"
    signature_install_core_packages
    export SIGNATURE_SKIP_CUDA_CHECK=1
    signature_verify_venv
    echo "[setup] Shared venv ready: $store/venv"
}
