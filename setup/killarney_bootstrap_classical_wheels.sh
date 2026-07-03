#!/bin/bash
# One-shot login-node prep for classical baseline jobs.
#
# statsforecast -> fugue -> pyarrow. On Alliance, pyarrow must come from the
# Arrow module (--no-index), not PyPI (pyarrow-noinstall dummy wheel).
#
# Run on Killarney login node:
#   cd "$SCRATCH/ts-sandbox"
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

_load_modules() {
    module purge 2>/dev/null || true
    module load StdEnv/2023 python/3.11 2>/dev/null || true
    module load gcc arrow 2>/dev/null || {
        echo "ERROR: could not load gcc/arrow (required for pyarrow)." >&2
        echo "Try: module spider arrow" >&2
        exit 1
    }
}

_load_modules

mkdir -p "$WHEEL_DIR"
echo "Wheel cache: $WHEEL_DIR"
echo ""

# pyarrow: Alliance wheel cache only (PyPI serves a dummy without arrow module).
echo "[1/2] Caching pyarrow from Alliance wheelhouse..."
pip download --no-index pyarrow -d "$WHEEL_DIR" -q

# Remaining statsforecast stack: PyPI on login is fine; install with --no-deps so
# pip does not re-resolve pyarrow from the dummy wheel.
echo "[2/2] Caching statsforecast dependency wheels from PyPI..."
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
echo "Done. $(ls -1 "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l) wheels in $WHEEL_DIR"
echo "Submit: ./submit_classical_baselines.sh"
