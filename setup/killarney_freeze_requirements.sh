#!/bin/bash
# =============================================================================
# One-shot login-node setup for the Slurm fast path.
#
# Installs pipeline wheels into a temporary venv, then freezes them to
# setup/requirements-killarney.txt. Jobs rebuild a node-local venv from that
# file on $SLURM_TMPDIR (see slurm_worker.sh).
#
# Run on Killarney login node (not inside sbatch):
#   cd "$SCRATCH/ts-sandbox"
#   ./setup/killarney_freeze_requirements.sh
#
# Re-run after adding dependencies or bumping package versions.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/setup/requirements-killarney.txt"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run on the login node, not inside a Slurm job." >&2
    exit 1
fi

if [[ -z "${PROJECT:-}" ]]; then
    if [[ -d "$HOME/projects" ]]; then
        _proj="$(ls -d "$HOME"/projects/aip-* "$HOME"/projects/def-* 2>/dev/null | head -1 || true)"
        [[ -n "$_proj" ]] && PROJECT="$(readlink -f "$_proj")"
    fi
fi
if [[ -z "${PROJECT:-}" ]]; then
    echo "ERROR: PROJECT not set. Example:" >&2
    echo "  export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)" >&2
    exit 1
fi

_load_modules() {
    module purge 2>/dev/null || true
    module load StdEnv/2023 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || true
}

echo "Repo:    $REPO_ROOT"
echo "PROJECT: $PROJECT"
echo "Output:  $OUT"
echo ""

_load_modules

BOOTSTRAP="$PROJECT/$USER/ts-sandbox/.venv-bootstrap-$$"
trap 'rm -rf "$BOOTSTRAP"' EXIT

echo "[1/4] Bootstrap venv at $BOOTSTRAP"
virtualenv --no-download "$BOOTSTRAP"
# shellcheck source=/dev/null
source "$BOOTSTRAP/bin/activate"
pip install --no-index --upgrade pip -q 2>/dev/null || pip install -U pip -q

echo "[2/4] Installing pipeline packages (wheel cache first)..."
_core=(torch torchvision numpy pandas scipy scikit-learn tqdm)
_pipeline=(optuna wandb einops pyyaml matplotlib)
if ! pip install --no-index "${_core[@]}" "${_pipeline[@]}" -q; then
    echo "  Wheel cache miss; falling back to PyTorch cu121 index..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install "${_core[@]:2}" "${_pipeline[@]}" -q
fi

echo "[3/4] Sanity check..."
python -c "
import torch, optuna, wandb, einops, yaml
print('  torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
"

echo "[4/4] Freezing to $OUT"
pip freeze --local > "$OUT"
deactivate

echo ""
echo "Done. Commit setup/requirements-killarney.txt and git pull on the cluster before submitting."
echo "Jobs will install from this file into \$SLURM_TMPDIR/env on each run."
