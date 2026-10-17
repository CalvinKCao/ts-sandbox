#!/bin/bash
# Ablation L8/L16 candidate-only disc. Ladder height comes from each run's
# patch_refine_canvas_height (256 legacy / 128 canvas128 leaf).
#
# USAGE (Killarney login, repo root):
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --dataset ETTh1
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --datasets ETTh1,weather
#   ./temp/scripts/submit_ablation_disc_l8_l16.sh --dataset weather --time 5:59:00
#
# Existing preds only (--reuse-forecast-cache). One job per dataset.
# Fake sources: binary sample0, MMPD sample, iTransformer, PatchTST.
# Lean disc: --disc-arch mlp (override with --disc-arch).
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_DATASETS=(
    ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic exchange_rate solar_Alabama PeMS
)

resolve_disc_ckpt() {
    local ds="$1"
    case "$ds" in
        ETTh1)
            CKPT="results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
            ;;
        ETTh2)
            CKPT="results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
            ;;
        ETTm1)
            CKPT="results/ckpts/08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml"
            ;;
        ETTm2)
            CKPT="results/ckpts/08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml"
            ;;
        electricity)
            CKPT="results/ckpts/08-04-4597054-electricity-binary_window_norm_patch_refine_canvas128_p64x6_electricity"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_electricity.yaml"
            ;;
        traffic)
            CKPT="results/ckpts/08-04-4597055-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_traffic.yaml"
            ;;
        exchange_rate)
            CKPT="results/ckpts/08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate.yaml"
            ;;
        PeMS)
            CKPT="results/ckpts/08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml"
            ;;
        solar_Alabama)
            CKPT="results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml"
            ;;
        weather)
            CKPT="results/ckpts/08-17-4849780-weather-binary_window_norm_patch_refine_canvas128_p64x6"
            CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6_weather.yaml"
            ;;
        *)
            echo "ERROR: no disc ckpt map for dataset=$ds" >&2
            return 1
            ;;
    esac
    RUN_NAME="${DISC_RUN_NAME:-window_norm_c128}"
}

wall_for() {
    case "$1" in
        traffic|solar_Alabama|weather|electricity) echo "5:59:00" ;;
        *) echo "8:00:00" ;;
    esac
}

slurm_time_to_seconds() {
    local t="$1" days=0 rest h=0 m=0 s=0
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
    echo $(( days * 86400 + h * 3600 + m * 60 + s ))
}

pick_l40s_partition() {
    local need_s="$1" part max_wall max_s best="" best_s=0
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
    if [[ -z "$best" ]]; then
        echo "ERROR: no gpubase_l40s_b* partition allows --time $WALL" >&2
        return 1
    fi
    echo "$best"
}

mem_for() {
    case "$1" in
        traffic|solar_Alabama|electricity) echo "64G" ;;
        *) echo "50G" ;;
    esac
}

DATASETS=()
DISC_ARCH="${DISC_ARCH:-mlp}"
WALL_OVERRIDE=""
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASETS+=("$2")
            shift 2
            ;;
        --datasets)
            IFS=',' read -ra _ds <<< "$2"
            DATASETS+=("${_ds[@]}")
            shift 2
            ;;
        --disc-arch)
            DISC_ARCH="$2"
            shift 2
            ;;
        --time)
            WALL_OVERRIDE="$2"
            shift 2
            ;;
        *)
            PASSTHRU+=("$1")
            shift
            ;;
    esac
done
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    JOBS=()
    for ds in "${DATASETS[@]}"; do
        resolve_disc_ckpt "$ds"
        if [[ ! -d "$REPO_ROOT/$CKPT" ]]; then
            echo "SKIP $ds: missing ckpt $CKPT" >&2
            continue
        fi
        IS_SMOKE=0
        IS_VIZ=0
        for arg in "${PASSTHRU[@]+"${PASSTHRU[@]}"}"; do
            [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
            [ "$arg" = "--viz-only" ] && IS_VIZ=1
        done
        if [ "$IS_SMOKE" -eq 1 ] || [ "$IS_VIZ" -eq 1 ]; then
            WALL=0:45:00
            MEM=32G
        else
            WALL="${WALL_OVERRIDE:-$(wall_for "$ds")}"
            MEM="$(mem_for "$ds")"
        fi
        NAME="disc-${ds}"
        PART="$(pick_l40s_partition "$(slurm_time_to_seconds "$WALL")")"
        echo "Submitting $NAME partition=$PART wall=$WALL mem=$MEM ckpt=$(basename "$CKPT")" >&2
        JOB=$(sbatch --parsable \
            --chdir="$REPO_ROOT" \
            --job-name="$NAME" \
            --account=aip-boyuwang \
            --partition="$PART" \
            --time="$WALL" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem="$MEM" \
            --exclude=kn010 \
            --export=ALL \
            --output="$REPO_ROOT/results/slurm/%x-%j.out" \
            --error="$REPO_ROOT/results/slurm/%x-%j.err" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/submit_ablation_disc_l8_l16.sh" \
            --dataset "$ds" --disc-arch "$DISC_ARCH" \
            "${PASSTHRU[@]+"${PASSTHRU[@]}"}")
        echo "  -> $ds $JOB"
        JOBS+=("$JOB:$ds")
    done
    echo "queued ${#JOBS[@]} disc jobs: ${JOBS[*]:-none}"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

if [[ ${#DATASETS[@]} -ne 1 ]]; then
    echo "ERROR: compute job expects exactly one --dataset, got ${DATASETS[*]:-none}" >&2
    exit 1
fi
DATASET="${DATASETS[0]}"
resolve_disc_ckpt "$DATASET"
CFG="${DISC_CONFIG:-$CFG}"
CKPT="${CKPT_OVERRIDE:-$CKPT}"

# Non-interactive / bare --export=ALL submits may lack Lmod; init before module.
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

# Prefer SLURM_SUBMIT_DIR (directory where sbatch was invoked) so spool copies
# of this script do not fall back to \$SCRATCH/ts-sandbox (often on main).
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
case "$PROJECT_ROOT" in
    "${SCRATCH}"/*) ;;
    *)
        if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox"
        elif [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox-ordinal-fine" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under \$SCRATCH" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT DATASET=$DATASET CKPT=$CKPT CFG=$CFG"

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing after module load" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/$CKPT" ]] || { echo "ERROR: missing ckpt $PROJECT_ROOT/$CKPT" >&2; exit 1; }

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

STAMP="$(date +%m-%d-%H%M)"
OUT_TAG="${OUT_TAG:-valtest80-${DATASET}}"
OUT_DIR="results/datasets/${STAMP}-ablation-disc-l8-l16-${OUT_TAG}"
mkdir -p "$OUT_DIR" results/slurm

RUN_SPEC="${RUN_NAME}:${CKPT}:${CFG}"
echo "RUN_SPEC=$RUN_SPEC disc_arch=$DISC_ARCH"

# Existing preds only. unique_abs / bin-center / 80/20 defaults in Python+flags.
python temp/scripts/eval_ablation_disc_l8_l16.py \
    --dataset "$DATASET" \
    --output-dir "$OUT_DIR" \
    --lookback 336 \
    --horizon 96 \
    --pack-test-stride 4 \
    --pack-splits val,test \
    --train-fraction 0.8 \
    --val-fraction 0 \
    --fake-agg sample0 \
    --slice-lengths 8 16 \
    --candidate-only \
    --disc-bin-center-shift \
    --no-offset-embedding \
    --disc-arch "$DISC_ARCH" \
    --reuse-forecast-cache \
    --no-viz \
    --fake-sources binary_staged mmpd itransformer patchtst \
    --num-sampling-steps 20 \
    --mmpd-output-root results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd \
    --runs "$RUN_SPEC" \
    "${PASSTHRU[@]+"${PASSTHRU[@]}"}"

echo "Finished: $(date)"
echo "output_dir=$OUT_DIR"
