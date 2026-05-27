#!/bin/bash
# =============================================================================
# MMPD vs Gaussian + binary-anchor matrix (parallel fan-out on Killarney L40S).
#
# Reuses MMPD train/eval from a prior run (default: 05-26-0688-mmpd-anchor-eval)
# for ETTh1/ETTh2/exchange_rate when artifacts already exist.
#
# USAGE (repo root on Killarney login node, $SCRATCH/ts-sandbox):
#   ./slurm_mmpd_anchor_matrix_eval.sh
#   ./slurm_mmpd_anchor_matrix_eval.sh --smoke-test
#   MMPD_SHARED=./results/datasets/05-26-0688-mmpd-anchor-eval \
#     MATRIX_OUT=./results/datasets/05-26-mmpd-anchor-matrix \
#     ./slurm_mmpd_anchor_matrix_eval.sh
#
# Continue only the in-flight Gaussian resume (0688) separately:
#   ./slurm_mmpd_gaussian_anchor_eval.sh --continue-from ./results/datasets/05-26-0688-mmpd-anchor-eval
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
SEED=2026
MATRIX_OUT="${MATRIX_OUT:-./results/datasets/05-26-mmpd-anchor-matrix}"
MMPD_SHARED="${MMPD_SHARED:-./results/datasets/05-26-0688-mmpd-anchor-eval}"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate illness)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --matrix-out) MATRIX_OUT="$2"; shift 2 ;;
        --mmpd-shared) MMPD_SHARED="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Worker body is submitted via heredoc; see submit_worker below.
    echo "ERROR: run this script from the login node (not inside a worker job)." >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/utils/eval_mmpd_gaussian_anchor.py" ]]; then
    echo "ERROR: submit from the ts-sandbox repo root." >&2
    exit 1
fi
if [[ "$SCRIPT_DIR" == /home/* ]]; then
    echo "ERROR: Killarney GPU jobs should run from a scratch/project checkout, not /home." >&2
    exit 1
fi

REPO="$SCRIPT_DIR"
MATRIX_OUT="$(cd "$REPO" && realpath -m "$MATRIX_OUT")"
MMPD_SHARED="$(cd "$REPO" && realpath -m "$MMPD_SHARED")"
LOG_DIR="$REPO/results/logs"
mkdir -p "$LOG_DIR" "$MATRIX_OUT/metrics_partial" "$MMPD_SHARED/raw" "$MMPD_SHARED/mmpd_out"

if [[ "$SMOKE" -eq 1 ]]; then
    DATASETS=(ETTh1)
    WALL_IDX="0:25:00"
    WALL_MMPD_TRAIN="0:45:00"
    WALL_MMPD_EVAL="0:30:00"
    WALL_ANCHOR="0:45:00"
    WALL_MERGE="0:15:00"
    MEM="24G"
    CPUS=4
    SUFFIX="-smoke"
    EXTRA_PY=(
        --test-fraction 0.02
        --test-max-items 32
        --sample-num 5
        --num-sampling-steps 5
        --gmm-components 5
        --gmm-iterations 3
        --mmpd-train-epochs 1
        --mmpd-patience 1
        --mmpd-batch-size 16
        --mmpd-eval-batch-size 4
        --anchor-batch-size 4
    )
else
    WALL_IDX="0:25:00"
    WALL_MMPD_TRAIN="2:00:00"
    WALL_MMPD_EVAL="1:30:00"
    WALL_ANCHOR_ETTH="3:00:00"
    WALL_ANCHOR_ETTM="5:00:00"
    WALL_ANCHOR_SMALL="2:00:00"
    WALL_MERGE="0:20:00"
    MEM="60G"
    CPUS=8
    SUFFIX=""
    EXTRA_PY=(
        --test-fraction 0.5
        --sample-num 100
        --num-sampling-steps 20
        --gmm-components 10
        --gmm-iterations 10
        --mmpd-train-epochs 20
        --mmpd-patience 5
        --mmpd-batch-size 32
        --mmpd-eval-batch-size 16
        --anchor-batch-size 16
    )
fi

PREAMBLE_FILE="$REPO/results/logs/mmpd-matrix-preamble.sh"
cat >"$PREAMBLE_FILE" <<'PREAMBLE'
set -euo pipefail
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops optuna \
    -q
export PYTHONUNBUFFERED=1
PREAMBLE

# 92d3 matrix checkpoint dirs (see reports/3769033_binary_gaussian_anchor_matrix_report.md)
declare -A GAUSS_ROOT=(
    [ETTh1]="05-26-3037-gauss-anchor-etth1"
    [ETTh2]="05-26-3038-gauss-anchor-etth2"
    [ETTm1]="05-26-9040-gauss-anchor-ettm1"
    [ETTm2]="05-26-9041-gauss-anchor-ettm2"
    [exchange_rate]="05-26-3257-gauss-anchor-exchange_rate"
    [illness]="05-26-9043-gauss-anchor-illness"
)
declare -A BIN_ROOT=(
    [ETTh1]="05-26-9033-binary-anchor-etth1"
    [ETTh2]="05-26-9034-binary-anchor-etth2"
    [ETTm1]="05-26-9035-binary-anchor-ettm1"
    [ETTm2]="05-26-9036-binary-anchor-ettm2"
    [exchange_rate]="05-26-9038-binary-anchor-exchange_rate"
    [illness]="05-26-9039-binary-anchor-illness"
)

anchor_wall_for_dataset() {
    local ds="$1"
    if [[ "$SMOKE" -eq 1 ]]; then
        echo "$WALL_ANCHOR"
        return
    fi
    case "$ds" in
        ETTm1|ETTm2) echo "$WALL_ANCHOR_ETTM" ;;
        ETTh1|ETTh2) echo "$WALL_ANCHOR_ETTH" ;;
        *) echo "$WALL_ANCHOR_SMALL" ;;
    esac
}

mmpd_ckpt_exists() {
    local ds="$1"
    local stem="data${ds}_il96_ol96_backboneDecoder_lossMMPD_weightedTrue_patch12_pointW0.01_diffH256_diffLayer1_radius3_diffStep1000_betalinear"
    [[ -f "$MMPD_SHARED/mmpd_out/checkpoints/Decoder-MMPD/${stem}/model_checkpoint.pth" ]]
}

mmpd_raw_exists() {
    [[ -f "$MMPD_SHARED/raw/mmpd_${1}.npz" ]] || [[ -f "$MATRIX_OUT/raw/mmpd_${1}.npz" ]]
}

anchor_partial_exists() {
    local variant="$1" ds="$2"
    local key="gaussian_anchor"
    [[ "$variant" == "binary" ]] && key="binary_anchor"
    [[ -f "$MATRIX_OUT/metrics_partial/${ds}__${key}.json" ]]
}

submit_worker() {
  # Do not use bash variable name "phase" — conflicts with Lmod/modules on Alliance nodes.
    local job_name="$1" wall="$2" dep="${3:-}" run_phase="$4" dataset="${5:-}" variant="${6:-}"
    local sbatch_dep=()
    [[ -n "$dep" ]] && sbatch_dep=(--dependency="$dep")

    local root_args=""
    if [[ -n "$dataset" ]]; then
        local gr="${GAUSS_ROOT[$dataset]:-}"
        local br="${BIN_ROOT[$dataset]:-}"
        [[ -n "$gr" && -d "$REPO/results/ckpts/$gr" ]] && root_args+=" --anchor-root $REPO/results/ckpts/$gr"
        [[ -n "$br" && -d "$REPO/results/ckpts/$br" ]] && root_args+=" --binary-anchor-root $REPO/results/ckpts/$br"
    else
        local ds gr br
        for ds in "${DATASETS[@]}"; do
            gr="${GAUSS_ROOT[$ds]:-}"
            br="${BIN_ROOT[$ds]:-}"
            [[ -n "$gr" && -d "$REPO/results/ckpts/$gr" ]] && root_args+=" --anchor-root $REPO/results/ckpts/$gr"
            [[ -n "$br" && -d "$REPO/results/ckpts/$br" ]] && root_args+=" --binary-anchor-root $REPO/results/ckpts/$br"
        done
    fi

    local dataset_flag=""
    if [[ -n "$dataset" ]]; then
        dataset_flag="--datasets $dataset"
    else
        dataset_flag="--datasets ${DATASETS[*]}"
    fi

    local variant_flag=""
    if [[ "$run_phase" == "anchor" && -n "$variant" ]]; then
        variant_flag="--anchor-variant $variant"
    fi

    local extra_py="${EXTRA_PY[*]}"

    sbatch --parsable \
        --job-name="$job_name" \
        --account=aip-boyuwang \
        --time="$wall" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output="$LOG_DIR/${job_name}-%j.out" \
        --error="$LOG_DIR/${job_name}-%j.err" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "${sbatch_dep[@]}" \
        <<EOF
#!/bin/bash
set -euo pipefail
source "$PREAMBLE_FILE"
cd "$REPO"
python -u utils/eval_mmpd_gaussian_anchor.py \\
    --output-dir "$MATRIX_OUT" \\
    --mmpd-output-root "$MMPD_SHARED/mmpd_out" \\
    --mmpd-raw-dir "$MMPD_SHARED/raw" \\
    --mmpd-raw-fallback "$MMPD_SHARED/raw" \\
    --reuse-anchor-raw-from "$MMPD_SHARED/raw" \\
    --ckpt-base "$REPO/results/ckpts" \\
    --mmpd-repo "$REPO/temp/MMPD" \\
    --mmpd-data-dir "$REPO/temp/mmpd_datasets" \\
    --seed "$SEED" \\
    --no-update-mmpd \\
    --phase $run_phase \\
    $dataset_flag \\
    $variant_flag \\
    $root_args \\
    $extra_py
echo "Done: \$(date)"
EOF
}

echo "Matrix output: $MATRIX_OUT"
echo "Shared MMPD:   $MMPD_SHARED"
echo "Datasets:      ${DATASETS[*]}"

JOB_IDX=$(submit_worker "mmpd-mx-idx${SUFFIX}" "$WALL_IDX" "" "indices")
echo "  indices -> $JOB_IDX"

MERGE_DEP_IDS=("$JOB_IDX")
MMPD_EVAL_JOBS=()
ANCHOR_JOBS=()

for ds in "${DATASETS[@]}"; do
    dep_train="afterok:${JOB_IDX}"
    if ! mmpd_ckpt_exists "$ds"; then
        j=$(submit_worker "mmpd-mx-tr-${ds}${SUFFIX}" "$WALL_MMPD_TRAIN" "$dep_train" "mmpd-train" "$ds")
        echo "  mmpd-train $ds -> $j"
        dep_train="afterok:$j"
        MERGE_DEP_IDS+=("$j")
    else
        echo "  [skip] mmpd-train $ds (checkpoint in $MMPD_SHARED/mmpd_out)"
    fi

    if ! mmpd_raw_exists "$ds"; then
        j=$(submit_worker "mmpd-mx-ev-${ds}${SUFFIX}" "$WALL_MMPD_EVAL" "$dep_train" "mmpd-eval" "$ds")
        echo "  mmpd-eval $ds -> $j"
        MMPD_EVAL_JOBS+=("$j")
        MERGE_DEP_IDS+=("$j")
    else
        echo "  [skip] mmpd-eval $ds (raw/mmpd_${ds}.npz exists)"
    fi

    aw=$(anchor_wall_for_dataset "$ds")
    for variant in gaussian binary; do
        if anchor_partial_exists "$variant" "$ds"; then
            echo "  [skip] anchor-$variant $ds (metrics partial exists)"
            continue
        fi
        j=$(submit_worker "mmpd-mx-a-${variant:0:1}-${ds}${SUFFIX}" "$aw" "afterok:${JOB_IDX}" "anchor" "$ds" "$variant")
        echo "  anchor-$variant $ds -> $j"
        ANCHOR_JOBS+=("$j")
        MERGE_DEP_IDS+=("$j")
    done
done

dep_merge="afterok:$(IFS=:; echo "${MERGE_DEP_IDS[*]}")"
JOB_MERGE=$(submit_worker "mmpd-mx-merge${SUFFIX}" "$WALL_MERGE" "$dep_merge" "merge")
echo "  merge -> $JOB_MERGE"

echo ""
echo "=================================================================="
echo "  indices     $JOB_IDX"
echo "  mmpd-eval   ${MMPD_EVAL_JOBS[*]:-(skipped — cached)}"
echo "  anchor jobs ${ANCHOR_JOBS[*]:-(skipped — cached)}"
echo "  merge       $JOB_MERGE"
echo ""
echo "  Metrics:    $MATRIX_OUT/metrics.json"
echo "  Partials:   $MATRIX_OUT/metrics_partial/"
echo "  Monitor:    squeue -u \$USER"
echo "=================================================================="
