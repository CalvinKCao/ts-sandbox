# Shared venv bootstrap for signature+MSE Slurm jobs.
# Source from slurm_signature_tune.sh / slurm_signature_finalize.sh

signature_cluster_venv() {
    local store="${STORE:-}"
    local shared_venv="${store}/venv"

    if [ -n "$store" ] && [ -d "$shared_venv" ] && [ "${SIGNATURE_FORCE_NODE_VENV:-0}" != "1" ]; then
        echo "[setup] Activating shared venv: $shared_venv"
        # shellcheck source=/dev/null
        source "$shared_venv/bin/activate"
        return 0
    fi

    echo "[setup] Building node-local venv on $SLURM_TMPDIR"
    virtualenv --no-download "$SLURM_TMPDIR/env"
    # shellcheck source=/dev/null
    source "$SLURM_TMPDIR/env/bin/activate"

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

    signature_pip_retry pip install --no-index --upgrade pip -q

    # No matplotlib — training does not need it; matplotlib pulls fonttools from CVMFS
    # and triggers intermittent [Errno 5] Input/output error on some nodes.
    signature_pip_retry pip install --no-index \
        'torch==2.11.0+computecanada' \
        numpy pandas scipy scikit-learn tqdm einops \
        sqlalchemy colorlog alembic packaging -q

    signature_pip_retry pip install --no-index optuna -q \
        || signature_pip_retry pip install optuna -q

    if ! signature_pip_retry pip install --no-index reformer-pytorch -q 2>/dev/null; then
        signature_pip_retry pip install reformer-pytorch --no-deps -q
        signature_pip_retry pip install --no-index \
            axial-positional-embedding local-attention product-key-memory -q 2>/dev/null \
            || signature_pip_retry pip install axial-positional-embedding local-attention -q
    fi

    signature_pip_retry pip install signatory --no-build-isolation -q

    python - <<'PY'
import torch
import signatory
from reformer_pytorch import LSHSelfAttention
assert torch.cuda.is_available(), "CUDA is required for this Slurm job"
print("torch", torch.__version__, "signatory ok", "gpu", torch.cuda.get_device_name(0))
PY
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

    signature_pip_retry() {
        local attempt=1
        local max="${SIGNATURE_PIP_RETRIES:-4}"
        while [ "$attempt" -le "$max" ]; do
            if "$@"; then return 0; fi
            echo "[setup] pip failed (attempt ${attempt}/${max}): $*"
            sleep $((attempt * 15))
            attempt=$((attempt + 1))
        done
        return 1
    }

    signature_pip_retry pip install --no-index --upgrade pip -q
    signature_pip_retry pip install --no-index \
        'torch==2.11.0+computecanada' \
        numpy pandas scipy scikit-learn tqdm einops \
        sqlalchemy colorlog alembic packaging -q
    signature_pip_retry pip install --no-index optuna -q \
        || signature_pip_retry pip install optuna -q
    if ! signature_pip_retry pip install --no-index reformer-pytorch -q 2>/dev/null; then
        signature_pip_retry pip install reformer-pytorch --no-deps -q
        signature_pip_retry pip install --no-index \
            axial-positional-embedding local-attention product-key-memory -q 2>/dev/null \
            || signature_pip_retry pip install axial-positional-embedding local-attention -q
    fi
    signature_pip_retry pip install signatory --no-build-isolation -q
    echo "[setup] Shared venv ready: $store/venv"
}
