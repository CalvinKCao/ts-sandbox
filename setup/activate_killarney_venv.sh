#!/bin/bash
# Create or activate the persistent Killarney venv at $SCRATCH/ts-sandbox/.venv.
#
# Source after loading a Python module (e.g. module load python/3.11):
#   source setup/activate_killarney_venv.sh
#
# Override location:
#   KILLARNEY_VENV_DIR=/path/to/venv source setup/activate_killarney_venv.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO_ROOT/setup/requirements-killarney.txt"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: SCRATCH is unset (run on an Alliance cluster login/compute node)" >&2
    return 1 2>/dev/null || exit 1
fi

VENV_DIR="${KILLARNEY_VENV_DIR:-$SCRATCH/ts-sandbox/.venv}"

[[ -f "$REQ" ]] || {
    echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2
    return 1 2>/dev/null || exit 1
}

command -v virtualenv >/dev/null || {
    echo "ERROR: virtualenv missing — load a Python module first (e.g. module load python/3.11)" >&2
    return 1 2>/dev/null || exit 1
}

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[venv] creating $VENV_DIR from $REQ"
    mkdir -p "$(dirname "$VENV_DIR")"
    virtualenv --no-download "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --no-index --upgrade pip -q
    if ! pip install --no-index -r "$REQ" -q; then
        echo "ERROR: pip install failed — re-run ./setup/killarney_freeze_requirements.sh" >&2
        return 1 2>/dev/null || exit 1
    fi
else
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
