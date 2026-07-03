#!/bin/bash
# One-shot login-node prep for classical baseline jobs.
#
# pyarrow: Alliance wheelhouse only (module load gcc arrow BEFORE any venv).
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

# pyarrow: install in a throwaway venv (arrow already loaded), wheel to PROJECT cache.
echo "[1/2] pyarrow via Alliance wheelhouse (temp venv)..."
BOOTSTRAP="$(mktemp -d "$PROJECT/$USER/ts-sandbox/.bootstrap-classical-XXXX")"
trap 'rm -rf "$BOOTSTRAP"' EXIT
virtualenv --no-download "$BOOTSTRAP"
# shellcheck source=/dev/null
source "$BOOTSTRAP/bin/activate"
pip install --no-index --upgrade pip -q
if ! pip install --no-index pyarrow -q; then
    echo "ERROR: pip install --no-index pyarrow failed." >&2
    echo "Ensure no venv was active and gcc/arrow loaded. See docs.alliancecan.ca/wiki/Arrow" >&2
    exit 1
fi
pip wheel --no-deps pyarrow -w "$WHEEL_DIR" -q
deactivate
unset VIRTUAL_ENV
rm -rf "$BOOTSTRAP"
trap - EXIT
echo "  cached: $(ls -1 "$WHEEL_DIR"/pyarrow*.whl 2>/dev/null | tail -1 | xargs basename 2>/dev/null || echo '?')"

echo "[2/2] statsforecast dependency wheels from PyPI..."
_classical_pkgs=(
    statsforecast
    statsmodels
    fugue
    utilsforecast
    coreforecast
    cloudpickle
    threadpoolctl
)
for pkg in "${_classical_pkgs[@]}"; do
    echo "  $pkg"
    pip download "$pkg" -d "$WHEEL_DIR" --no-deps -q
done

echo ""
_n="$(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)"
echo "Done. $_n wheels in $WHEEL_DIR"
echo "Submit: ./submit_classical_baselines.sh"
