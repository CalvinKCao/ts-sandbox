#!/bin/bash
# One-shot login-node prep for classical baseline jobs.
#
# pyarrow: provided by the Arrow module (module load gcc arrow BEFORE any venv).
# statsforecast stack: PyPI wheels cached to PROJECT (login node has network).
#
# Run on Killarney login node (no venv active):
#   cd "$SCRATCH/ts-sandbox"
#   deactivate 2>/dev/null || true
#   ./setup/killarney_bootstrap_classical_wheels.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run on the login node, not inside a Slurm job." >&2
    exit 1
fi

if [[ -z "${PROJECT:-}" ]]; then
    shopt -s nullglob
    _m=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    if [[ "${#_m[@]}" -gt 0 ]]; then
        PROJECT="$(readlink -f "${_m[0]}")"
    fi
fi
if [[ -z "${PROJECT:-}" ]]; then
    echo "ERROR: PROJECT not set. Example:" >&2
    echo "  export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)" >&2
    exit 1
fi

WHEEL_DIR="$PROJECT/$USER/ts-sandbox/wheels-classical"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

deactivate 2>/dev/null || true
unset VIRTUAL_ENV PYTHONHOME

# shellcheck source=/dev/null
source "$REPO_ROOT/setup/killarney_classical_modules.sh"
killarney_classical_modules

mkdir -p "$WHEEL_DIR"
echo "Wheel cache: $WHEEL_DIR"
if command -v avail_wheels >/dev/null 2>&1; then
    echo "avail_wheels pyarrow: $(avail_wheels pyarrow 2>/dev/null | head -3 | tr '\n' ' ')"
fi
echo ""

# pyarrow is not cached: Alliance exposes a dummy wheel named
# pyarrow-9999+dummy; the real module comes from `module load gcc arrow`.
echo "[1/2] Verifying pyarrow from the Arrow module in a temp venv..."
BOOTSTRAP="$(mktemp -d "$PROJECT/$USER/ts-sandbox/.bootstrap-classical-XXXX")"
trap 'rm -rf "$BOOTSTRAP"' EXIT
virtualenv --no-download "$BOOTSTRAP"
# shellcheck source=/dev/null
source "$BOOTSTRAP/bin/activate"
pip install --no-index --upgrade pip -q
if ! python -c "import pyarrow; print('  pyarrow module:', pyarrow.__version__)"; then
    echo "ERROR: import pyarrow failed after loading gcc/arrow." >&2
    echo "Do not pip install pyarrow on Killarney; fix the Arrow module load first." >&2
    exit 1
fi
deactivate
unset VIRTUAL_ENV
rm -rf "$BOOTSTRAP"
trap - EXIT

echo "[2/2] statsforecast dependency wheels from PyPI..."
_classical_specs=(
    statsforecast==1.7.6
    statsmodels
    fugue
    cloudpickle
    threadpoolctl
    triad
    adagio
    patsy
    numba
    llvmlite
    pandas==2.3.3
    pytz
    tzdata
    plotly
    narwhals
)
for spec in "${_classical_specs[@]}"; do
    echo "  $spec"
    pip download \
        --index-url "$PYPI_INDEX" \
        --only-binary=:all: \
        --no-deps \
        -d "$WHEEL_DIR" \
        "$spec" \
        -q
done

echo ""
_n="$(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)"
echo "Done. $_n wheels in $WHEEL_DIR"
_required_wheels=(
    statsforecast-1.7.6
    statsmodels
    fugue
    cloudpickle
    threadpoolctl
    triad
    adagio
    patsy
    numba
    llvmlite
    pandas-2.3.3
    pytz
    tzdata
    plotly
    narwhals
)
for pkg in "${_required_wheels[@]}"; do
    if ! compgen -G "$WHEEL_DIR/$pkg"*.whl >/dev/null; then
        echo "ERROR: missing cached wheel matching $pkg*.whl in $WHEEL_DIR" >&2
        exit 1
    fi
done
echo "Submit: ./submit_classical_baselines.sh"
