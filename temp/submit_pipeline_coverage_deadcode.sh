#!/bin/bash
# =============================================================================
# Coverage / dead-code probe for the ordinal patch-refine → MMPD → assert → disc
# phase graph. One L40S job, wall default 25 min (override with COVERAGE_WALL),
# tiny coverage_synth data, forced-fresh dirs.
#
# USAGE (from $SCRATCH/ts-sandbox on Killarney login node):
#   ./temp/submit_pipeline_coverage_deadcode.sh
#   ./temp/submit_pipeline_coverage_deadcode.sh --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
# Login-node path (may differ from Spool copy of this script).
LOGIN_REPO="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

DRY_RUN=0
RUN_NAME=""
EXCLUDE_NODES="${EXCLUDE_NODES:-kn010}"
WALL_TIME="${COVERAGE_WALL:-0:25:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --exclude) EXCLUDE_NODES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Login node: submit ourselves
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    REPO="$LOGIN_REPO"
    [[ -d "${SCRATCH:-}/ts-sandbox" ]] || {
        echo "ERROR: expected Killarney checkout at \$SCRATCH/ts-sandbox" >&2
        exit 1
    }
    [[ "$REPO" == "${SCRATCH}/ts-sandbox" ]] || {
        echo "ERROR: run from \$SCRATCH/ts-sandbox (got $REPO)" >&2
        exit 1
    }

    MMPD_REPO="$REPO/temp/MMPD"
    need_clone=0
    if [[ ! -d "$MMPD_REPO/.git" ]]; then
        need_clone=1
    elif [[ ! -f "$MMPD_REPO/metrics/prob_metrics.py" ]] \
        || [[ ! -f "$MMPD_REPO/models/backbones/decoder_only_transformer.py" ]]; then
        echo "MMPD checkout incomplete; re-cloning..."
        mv "$MMPD_REPO" "$REPO/temp/MMPD.corrupt-$(date +%m%d-%H%M)" || true
        need_clone=1
    fi
    if [[ "$need_clone" -eq 1 ]]; then
        echo "Cloning MMPD on login node (compute nodes cannot reach GitHub)..."
        mkdir -p "$REPO/temp"
        git clone https://github.com/Thinklab-SJTU/MMPD.git "$MMPD_REPO"
    fi
    [[ -f "$MMPD_REPO/metrics/prob_metrics.py" ]] || {
        echo "ERROR: MMPD metrics/prob_metrics.py missing" >&2
        exit 1
    }
    [[ -f "$MMPD_REPO/models/backbones/decoder_only_transformer.py" ]] || {
        echo "ERROR: MMPD decoder_only_transformer.py missing" >&2
        exit 1
    }
    MMPD_TOOLS="$MMPD_REPO/utils/tools.py"
    if [[ -f "$MMPD_TOOLS" ]] && grep -q 'np\.Inf' "$MMPD_TOOLS"; then
        sed -i 's/np\.Inf/np.inf/g' "$MMPD_TOOLS"
    fi
    python3 - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "$REPO")
from utils.eval_mmpd_gaussian_anchor import ensure_mmpd_repo
print("mmpd commit", ensure_mmpd_repo(Path("$MMPD_REPO"), update=False)[:12])
PY

    WHEEL="$(ls -1 "$REPO"/setup/coverage_wheels/coverage-*-cp311*.whl 2>/dev/null | head -1 || true)"
    [[ -n "$WHEEL" && -f "$WHEEL" ]] || {
        echo "ERROR: missing setup/coverage_wheels/coverage-*-cp311*.whl" >&2
        echo "  On a networked machine: mkdir -p setup/coverage_wheels && pip download coverage==7.15.2 -d setup/coverage_wheels --python-version 311 --only-binary=:all:" >&2
        exit 1
    }
    python3 "$REPO/temp/make_coverage_synth_dataset.py"
    [[ -f "$REPO/configs/coverage_deadcode_binary_patch_refine.yaml" ]] || {
        echo "ERROR: missing coverage binary config" >&2
        exit 1
    }

    mkdir -p "$REPO/results/logs"
    EXTRA=()
    [[ -z "$EXCLUDE_NODES" ]] || EXTRA+=(--exclude="$EXCLUDE_NODES")
    [[ -z "$RUN_NAME" ]] || EXTRA+=(--export=ALL,COVERAGE_RUN_NAME="$RUN_NAME")

    PASS_ARGS=()
    [[ "$DRY_RUN" -eq 1 ]] && PASS_ARGS+=(--dry-run)
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN (login): would sbatch L40S 30min coverage-deadcode with:"
        echo "  script=$SCRIPT_PATH"
        echo "  pass_args=${PASS_ARGS[*]:-}"
        echo "  exclude=${EXCLUDE_NODES:-none}"
        echo "  run_name=${RUN_NAME:-auto}"
        python3 "$REPO/temp/run_pipeline_coverage_deadcode.py" --dry-run \
            ${RUN_NAME:+--run-name "$RUN_NAME"} \
            --results-root "$REPO/results"
        exit 0
    fi
    echo "Submitting coverage deadcode probe (L40S, wall=$WALL_TIME)..."
    sbatch \
        --job-name=coverage-deadcode \
        --account=aip-boyuwang \
        --time="$WALL_TIME" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "${EXTRA[@]}" \
        "$SCRIPT_PATH" \
        "${PASS_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Inside the job — open run log ASAP (before any exit 1 after cd)
# ---------------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

RUN_NAME="${COVERAGE_RUN_NAME:-$(date +%m-%d-%H%M)-${SLURM_JOB_ID: -3}-coverage-deadcode}"
RUN_ROOT="$REPO/results/$RUN_NAME"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.log"
exec >>"$LOG" 2>&1

ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) =========================================="
echo "$(ts) Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) Run: $RUN_NAME"
echo "$(ts) Log: $LOG"
echo "$(ts) Started: $(date)"
echo "$(ts) =========================================="

[[ -f "$REPO/models/diffusion_tsf/train_multivariate_pipeline.py" ]] || {
    echo "ERROR: submit from repo root (SLURM_SUBMIT_DIR=$REPO)" >&2
    exit 1
}

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
WHEEL="$(ls -1 "$REPO"/setup/coverage_wheels/coverage-*-cp311*.whl 2>/dev/null | head -1 || true)"
[[ -n "$WHEEL" ]] || { echo "ERROR: coverage wheel missing under setup/coverage_wheels/" >&2; exit 1; }

echo "$(ts) [setup] Building venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
# Install by wheel path — Alliance wheelhouse has different coverage builds
# (e.g. 7.13.4+computecanada); --find-links + ==7.15.2 can miss the local file.
echo "$(ts) [setup] Installing coverage (Alliance wheelhouse preferred)"
pip install --no-index coverage -q || pip install --no-index "$WHEEL" -q

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
python -c "import coverage; print('coverage', coverage.__version__)"
python -u "$REPO/temp/make_coverage_synth_dataset.py"

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export COVERAGE_RUN_NAME="$RUN_NAME"

EXTRA_ARGS=()
[[ "${1:-}" == "--dry-run" ]] && EXTRA_ARGS+=(--dry-run)

echo "$(ts) [run] coverage deadcode orchestrator"
python -u "$REPO/temp/run_pipeline_coverage_deadcode.py" \
    --run-name "$RUN_NAME" \
    --results-root "$REPO/results" \
    "${EXTRA_ARGS[@]}"

echo "$(ts) =========================================="
echo "$(ts) Done: $(date)"
echo "$(ts) Summary: $RUN_ROOT/summary.json"
echo "$(ts) Coverage: $RUN_ROOT/coverage/html/index.html"
echo "$(ts) =========================================="
