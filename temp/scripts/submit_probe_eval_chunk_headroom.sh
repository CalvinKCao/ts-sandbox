#!/bin/bash
# Diagnostic: reserved-headroom unet_max_chunk_size probe for staged_eval
# unique-seg generate (ckpt loaded, n_samples=10, steps=4).
# Not a training wrapper.
#
# Narval A100 (from repo root on the login node):
#   ./temp/scripts/submit_probe_eval_chunk_headroom.sh \
#       --dataset electricity --n-variates 321 \
#       --config configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100.yaml \
#       --ckpt-dir results/ckpts/09-08-2650276-electricity-binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100 \
#       --lo 1024 --start 8192 --hi 20000 --max-reserved-gib 32 --headroom-gib 7.5
#
# Killarney L40S:
#   ./temp/scripts/submit_probe_eval_chunk_headroom.sh \
#       --dataset traffic --n-variates 862 \
#       --config configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv.yaml \
#       --ckpt-dir results/ckpts/09-08-2650287-traffic-binary_window_norm_patch_refine_canvas128_p32x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100 \
#       --lo 256 --start 1024 --max-reserved-gib 42 --headroom-gib 6 --fast-chunk-fwd --skip-full-window

set -euo pipefail

DATASET=""
N_VARIATES=""
CONFIG=""
CKPT_DIR=""
LO="512"
HI=""
START=""
STEPS="4"
N_SAMPLES="10"
HEADROOM="5"
MAX_RESERVED=""
MIN_OK="1024"
WALL="1:30:00"
CONFIRM_REPEATS="2"
SEARCH_REPEATS="1"
FAST_CHUNK=""
SKIP_WINDOW=""
DET_ONLY=""
NO_HOLD=""
SCRAPE="1.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        --lo) LO="$2"; shift 2 ;;
        --hi) HI="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --headroom-gib) HEADROOM="$2"; shift 2 ;;
        --max-reserved-gib) MAX_RESERVED="$2"; shift 2 ;;
        --min-ok-chunk) MIN_OK="$2"; shift 2 ;;
        --confirm-repeats) CONFIRM_REPEATS="$2"; shift 2 ;;
        --search-repeats) SEARCH_REPEATS="$2"; shift 2 ;;
        --step-down-scrape-gib) SCRAPE="$2"; shift 2 ;;
        --fast-chunk-fwd) FAST_CHUNK="1"; shift ;;
        --skip-full-window) SKIP_WINDOW="1"; shift ;;
        --det-only) DET_ONLY="1"; shift ;;
        --no-full-pack-hold) NO_HOLD="1"; shift ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$CONFIG" || -z "$CKPT_DIR" ]]; then
    echo "ERROR: --dataset --config --ckpt-dir required" >&2
    exit 1
fi

DS_SLUG="${DATASET//_/-}"
JOB_NAME="probe-${DS_SLUG}-headroom-n${N_SAMPLES}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    mkdir -p "$SCRIPT_DIR/results/logs"
    PY_FWD=(
        --dataset "$DATASET"
        --config "$CONFIG"
        --ckpt-dir "$CKPT_DIR"
        --lo "$LO"
        --steps "$STEPS"
        --n-samples "$N_SAMPLES"
        --headroom-gib "$HEADROOM"
        --min-ok-chunk "$MIN_OK"
        --confirm-repeats "$CONFIRM_REPEATS"
        --search-repeats "$SEARCH_REPEATS"
        --step-down-scrape-gib "$SCRAPE"
        --time "$WALL"
    )
    [[ -n "$N_VARIATES" ]] && PY_FWD+=(--n-variates "$N_VARIATES")
    [[ -n "$HI" ]] && PY_FWD+=(--hi "$HI")
    [[ -n "$START" ]] && PY_FWD+=(--start "$START")
    [[ -n "$MAX_RESERVED" ]] && PY_FWD+=(--max-reserved-gib "$MAX_RESERVED")
    [[ -n "$FAST_CHUNK" ]] && PY_FWD+=(--fast-chunk-fwd)
    [[ -n "$SKIP_WINDOW" ]] && PY_FWD+=(--skip-full-window)
    [[ -n "$DET_ONLY" ]] && PY_FWD+=(--det-only)
    [[ -n "$NO_HOLD" ]] && PY_FWD+=(--no-full-pack-hold)

    HOST="$(hostname)"
    SBATCH_EXTRA=()
    WALL_S=0
    rest="$WALL"
    days=0
    if [[ "$WALL" == *-* ]]; then
        days="${WALL%%-*}"
        rest="${WALL#*-}"
    fi
    IFS=':' read -r a b c <<< "$rest"
    if [[ -n "${c:-}" ]]; then WALL_S=$(( days * 86400 + a * 3600 + b * 60 + c ))
    elif [[ -n "${b:-}" ]]; then WALL_S=$(( days * 86400 + a * 3600 + b * 60 ))
    else WALL_S=$(( days * 86400 + a )); fi
    if [[ "$HOST" == *narval* ]]; then
        # b1=3h, b2=12h, b3=1d, b4=3d, b5=7d
        if [[ "$WALL_S" -le 10800 ]]; then A100_PART=gpubase_bygpu_b1
        elif [[ "$WALL_S" -le 43200 ]]; then A100_PART=gpubase_bygpu_b2
        elif [[ "$WALL_S" -le 86400 ]]; then A100_PART=gpubase_bygpu_b3
        elif [[ "$WALL_S" -le 259200 ]]; then A100_PART=gpubase_bygpu_b4
        else A100_PART=gpubase_bygpu_b5
        fi
        echo "Submitting ${JOB_NAME} on Narval A100 (${A100_PART}, ${WALL})..."
        SBATCH_EXTRA=(
            --account=def-boyuwang
            --partition="$A100_PART"
            --gpus=a100:1
        )
    else
        # b1 max wall is 3h; 4h+ probes must use b2 (12h).
        L40S_PART=gpubase_l40s_b1
        if [[ "$WALL_S" -gt 10800 ]]; then
            L40S_PART=gpubase_l40s_b2
        fi
        echo "Submitting ${JOB_NAME} on Killarney L40S (${L40S_PART}, ${WALL})..."
        SBATCH_EXTRA=(
            --account=aip-boyuwang
            --partition="$L40S_PART"
            --gres=gpu:l40s:1
        )
    fi
    sbatch \
        --job-name="$JOB_NAME" \
        --time="$WALL" \
        --nodes=1 \
        --cpus-per-task=8 \
        --mem=96G \
        --output="$SCRIPT_DIR/results/logs/probe-${DS_SLUG}-headroom-n${N_SAMPLES}-%j.log" \
        --error="$SCRIPT_DIR/results/logs/probe-${DS_SLUG}-headroom-n${N_SAMPLES}-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "${SBATCH_EXTRA[@]}" \
        "$SCRIPT_DIR/temp/scripts/submit_probe_eval_chunk_headroom.sh" \
        "${PY_FWD[@]}"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "dataset=$DATASET V=${N_VARIATES:-resolved} n_samples=$N_SAMPLES steps=$STEPS"
echo "lo=$LO hi=${HI:-n_items+1} start=${START:-default} cap=${MAX_RESERVED:-total-headroom}"
echo "fast_chunk_fwd=${FAST_CHUNK:-0} det_only=${DET_ONLY:-0} skip_full_window=${SKIP_WINDOW:-0} confirm_repeats=$CONFIRM_REPEATS"
echo "config=$CONFIG"
echo "ckpt=$CKPT_DIR"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    elif [ -f /etc/profile.d/z00_lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z00_lmod.sh
    fi
fi
type module >/dev/null 2>&1 || { echo "ERROR: Lmod unavailable after profile source" >&2; exit 127; }
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "$REPO_ROOT"
mkdir -p results/logs
[[ -f models/diffusion_tsf/dit.py ]] || { echo "ERROR: not repo root: $REPO_ROOT" >&2; exit 1; }
REQ="$REPO_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG" >&2; exit 1; }
PROBE="$REPO_ROOT/temp/scripts/probe_eval_chunk_headroom.py"
[[ -f "$PROBE" ]] || { echo "ERROR: missing $PROBE" >&2; exit 1; }
CKPT_ABS="$CKPT_DIR"
[[ "$CKPT_ABS" == /* ]] || CKPT_ABS="$REPO_ROOT/$CKPT_DIR"
[[ -d "$CKPT_ABS" ]] || { echo "ERROR: missing ckpt dir $CKPT_ABS" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/inductor"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "[setup] Building node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

PY_ARGS=(
    --config "$CONFIG"
    --dataset "$DATASET"
    --ckpt-dir "$CKPT_ABS"
    --n-samples "$N_SAMPLES"
    --lo "$LO"
    --steps "$STEPS"
    --headroom-gib "$HEADROOM"
    --min-ok-chunk "$MIN_OK"
    --confirm-repeats "$CONFIRM_REPEATS"
    --search-repeats "$SEARCH_REPEATS"
    --step-down-scrape-gib "$SCRAPE"
)
[[ -n "$N_VARIATES" ]] && PY_ARGS+=(--n-variates "$N_VARIATES")
[[ -n "$HI" ]] && PY_ARGS+=(--hi "$HI")
[[ -n "$START" ]] && PY_ARGS+=(--start "$START")
[[ -n "$MAX_RESERVED" ]] && PY_ARGS+=(--max-reserved-gib "$MAX_RESERVED")
[[ -n "$FAST_CHUNK" ]] && PY_ARGS+=(--fast-chunk-fwd)
[[ -n "$SKIP_WINDOW" ]] && PY_ARGS+=(--skip-full-window)
[[ -n "$DET_ONLY" ]] && PY_ARGS+=(--det-only)
[[ -n "$NO_HOLD" ]] && PY_ARGS+=(--no-full-pack-hold)

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u "$PROBE" "${PY_ARGS[@]}"

echo "Finished: $(date)"
