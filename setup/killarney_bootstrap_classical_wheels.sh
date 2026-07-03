#!/bin/bash
# One-shot login-node prep for classical baseline jobs.
#
# Downloads statsforecast + statsmodels wheels to PROJECT (PyPI allowed on login).
# Compute jobs install from that cache with --no-index (see submit_classical_baselines.sh).
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
REQ_CLASSICAL="$REPO_ROOT/setup/requirements-classical.txt"

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

mkdir -p "$WHEEL_DIR"
echo "Wheel cache: $WHEEL_DIR"
echo "Packages:    $(tr '\n' ' ' < "$REQ_CLASSICAL")"
echo ""

pip download -r "$REQ_CLASSICAL" -d "$WHEEL_DIR"

echo ""
echo "Done. Wheels cached under $WHEEL_DIR"
echo "Submit: ./submit_classical_baselines.sh"
