#!/bin/bash
# One-off Killarney util: L40S random-window 10-sample prob eval
# for the electricity 160/161 shard ckpts (jobs 4862421 / 4862422).
#
# Default: 40 of 162 stride-32 test windows, 8h wall (gpubase_l40s_b2; b1 max is 3h).
# Dense random (every test start): TEST_STRIDE=1 N_WINDOWS=80 TIME_LIM=12:00:00
# From $SCRATCH/ts-sandbox:
#   ./temp/scripts/submit_eval_electricity_shard_random_prob.sh
#   SHARD=v000_159 ./temp/scripts/submit_eval_electricity_shard_random_prob.sh
#   TIME_LIM=8:00:00 N_WINDOWS=40 N_SAMPLES=10 ./temp/scripts/submit_eval_electricity_shard_random_prob.sh
#   JOB_PREFIX=elec-randprob80-s1 N_WINDOWS=80 TEST_STRIDE=1 TIME_LIM=12:00:00 N_SAMPLES=10 \\
#     ./temp/scripts/submit_eval_electricity_shard_random_prob.sh
#
set -euo pipefail
export PATH="/opt/slurm/bin:/cm/shared/apps/slurm/current/bin:${PATH:-/usr/bin:/bin}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TIME_LIM="${TIME_LIM:-8:00:00}"
MEM="${MEM:-350G}"
CPUS="${CPUS:-8}"
N_SAMPLES="${N_SAMPLES:-10}"
N_WINDOWS="${N_WINDOWS:-40}"
TEST_STRIDE="${TEST_STRIDE:-32}"
SAMPLER="${SAMPLER:-quad_t}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
DRAIN_SECONDS="${DRAIN_SECONDS:-90}"
JOB_PREFIX="${JOB_PREFIX:-elec-randprob40}"

declare -A SHARD_CFG=(
  [v000_159]="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_s2_every4.yaml"
  [v160_320]="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity_v160_320_s2_every4.yaml"
)
declare -A SHARD_CKPT=(
  [v000_159]="results/ckpts/08-17-4854714-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity_v000_159_s2_every4"
  [v160_320]="results/ckpts/08-17-4854715-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity_v160_320_s2_every4"
)

slurm_time_to_seconds() {
    local t="$1" days=0 rest h=0 m=0 s=0 a b c
    if [[ "$t" == *-* ]]; then
        days="${t%%-*}"
        rest="${t#*-}"
    else
        rest="$t"
    fi
    IFS=':' read -r a b c <<< "$rest"
    if [[ -n "${c:-}" ]]; then
        h="$a"; m="$b"; s="$c"
    elif [[ -n "${b:-}" ]]; then
        h=0; m="$a"; s="$b"
    else
        h=0; m=0; s="$a"
    fi
    echo $(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

pick_l40s_partition() {
    local need_s="$1" part max_wall max_s best="" best_s=0
    if [[ -n "${PARTITION:-}" ]]; then
        echo "$PARTITION"
        return 0
    fi
    while read -r part max_wall; do
        [[ "$part" == gpubase_l40s_b* ]] || continue
        part="${part%\*}"
        max_s="$(slurm_time_to_seconds "$max_wall")"
        if [[ "$max_s" -ge "$need_s" ]]; then
            if [[ -z "$best" || "$max_s" -lt "$best_s" ]]; then
                best="$part"
                best_s="$max_s"
            fi
        fi
    done < <(sinfo -h -o "%P %l" 2>/dev/null || true)
    # 8h needs b2 (b1 max 3h)
    echo "${best:-gpubase_l40s_b2}"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/logs"
    SHARDS="${SHARD:-v000_159,v160_320}"
    IFS=',' read -ra WANT <<< "$SHARDS"
    PART="$(pick_l40s_partition "$(slurm_time_to_seconds "$TIME_LIM")")"
    for s in "${WANT[@]}"; do
        [[ -n "${SHARD_CFG[$s]:-}" ]] || { echo "ERROR: unknown SHARD=$s" >&2; exit 1; }
        echo "Submitting ${JOB_PREFIX}-$s (L40S part=${PART} ${TIME_LIM} mem=${MEM} n_windows=${N_WINDOWS} n_samples=${N_SAMPLES}) from $REPO_ROOT"
        sbatch \
            --job-name="${JOB_PREFIX}-${s}" \
            --account=aip-boyuwang \
            --partition="$PART" \
            --time="$TIME_LIM" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --export=ALL,SHARD="$s",N_SAMPLES="$N_SAMPLES",N_WINDOWS="$N_WINDOWS",TEST_STRIDE="$TEST_STRIDE",SAMPLER="$SAMPLER",STEPS="$STEPS",SEED="$SEED",DRAIN_SECONDS="$DRAIN_SECONDS",JOB_PREFIX="$JOB_PREFIX" \
            --output=/dev/null \
            --error=/dev/null \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/submit_eval_electricity_shard_random_prob.sh"
    done
    exit 0
fi

SHARD="${SHARD:?SHARD must be set in the batch env}"
CFG="${SHARD_CFG[$SHARD]}"
CKPT="${SHARD_CKPT[$SHARD]}"
N_SAMPLES="${N_SAMPLES:-10}"
N_WINDOWS="${N_WINDOWS:-40}"
TEST_STRIDE="${TEST_STRIDE:-32}"
SAMPLER="${SAMPLER:-quad_t}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
DRAIN_SECONDS="${DRAIN_SECONDS:-90}"
JOB_PREFIX="${JOB_PREFIX:-elec-randprob40}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$REPO_ROOT"
fi
case "$PROJECT_ROOT" in
    "${SCRATCH}"/*) ;;
    *)
        if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under \$SCRATCH" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-${JOB_PREFIX}-${SHARD}"
mkdir -p ./results/logs
LOG="./results/logs/${STEM}.log"
exec >>"$LOG" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "SHARD=$SHARD CFG=$CFG"
echo "CKPT=$CKPT"
echo "N_WINDOWS=$N_WINDOWS N_SAMPLES=$N_SAMPLES TEST_STRIDE=$TEST_STRIDE SAMPLER=$SAMPLER STEPS=$STEPS"
echo "LOG=$LOG"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        export FORCE_CC_CVMFS="${FORCE_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    elif [ -f /etc/profile.d/z00_lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z00_lmod.sh
    fi
fi
type module >/dev/null 2>&1 || {
    echo "ERROR: Lmod 'module' unavailable after profile source" >&2
    exit 127
}
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/$CFG" ]] || { echo "ERROR: missing config $CFG" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/$CKPT" ]] || { echo "ERROR: missing ckpt $CKPT" >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in job env"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

JSONL="./results/logs/${STEM}.jsonl"

# Remaining wall after venv; python drain-seconds (~90s) stops before SIGTERM.
MAX_SECONDS="${MAX_SECONDS:-0}"
if [[ "$MAX_SECONDS" -le 0 ]]; then
    NOW=$(date +%s)
    END_STR=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2; exit}')
    END=0
    if [[ -n "${END_STR:-}" && "$END_STR" != "Unknown" ]]; then
        END=$(date -d "$END_STR" +%s 2>/dev/null || echo 0)
    fi
    if [[ "$END" -gt "$NOW" ]]; then
        MAX_SECONDS=$((END - NOW))
    else
        MAX_SECONDS=27900
    fi
fi
echo "MAX_SECONDS=$MAX_SECONDS DRAIN_SECONDS=$DRAIN_SECONDS n_windows=$N_WINDOWS"

python -u temp/scripts/eval_electricity_shard_random_prob.py \
    --config "$CFG" \
    --ckpt-dir "$CKPT" \
    --out-jsonl "$JSONL" \
    --n-samples "$N_SAMPLES" \
    --n-windows "$N_WINDOWS" \
    --test-stride "$TEST_STRIDE" \
    --sampler "$SAMPLER" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --max-seconds "$MAX_SECONDS" \
    --drain-seconds "$DRAIN_SECONDS"

echo "========================================"
echo "Job complete: $(date)"
echo "jsonl: $JSONL"
echo "log:   $LOG"
echo "========================================"
