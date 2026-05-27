#!/bin/bash
# =============================================================================
# Train binary-anchor DiT at image heights 64 and 128 (matrix datasets; no electricity), then
# run the full MMPD + gaussian + binary matrix eval (9-sample, texture-per-sample,
# deterministic + probabilistic metrics). Skips prob-100 eval.
#
# H=128 matrix reuses MMPD outputs from the H=64 matrix run.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   export WANDB_API_KEY=...   # optional
#   ./slurm_binary_h64_h128_matrix_chain.sh
#   ./slurm_binary_h64_h128_matrix_chain.sh --smoke-test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
DATE_TAG="$(date +%m-%d)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run from the login node only (this script submits jobs)." >&2
    exit 1
fi

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$(cd "$SCRIPT_DIR" && pwd)"
fi
if [[ "$REPO" == /home/* ]]; then
    echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
    exit 1
fi
cd "$REPO"

# Same set as slurm_mmpd_gaussian_anchor_eval.sh full run (electricity excluded).
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate)
HEIGHTS=(64 128)
TRAIN_SCRIPT="$REPO/slurm_binary_anchor_92d3.sh"
MATRIX_SCRIPT="$REPO/slurm_mmpd_gaussian_anchor_eval.sh"
SMOKE_SUFFIX=""
[[ "$SMOKE" -eq 1 ]] && SMOKE_SUFFIX="-smoke"

chmod +x "$TRAIN_SCRIPT" "$MATRIX_SCRIPT"

declare -A TRAIN_JOB
declare -A CKPT_STEM

echo "Repo:     $REPO"
echo "Date tag: $DATE_TAG"
echo "Heights:  ${HEIGHTS[*]}"
echo "Datasets: ${DATASETS[*]}"

for H in "${HEIGHTS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        stem="${DATE_TAG}-bin-h${H}-${ds,,}"
        CKPT_STEM["${H}:${ds}"]="$stem"
        train_extra=(--dataset "$ds" --image-height "$H" --fresh --run-stem "$stem")
        [[ "$SMOKE" -eq 1 ]] && train_extra+=(--smoke-test)
        echo "Submitting train bin-h${H}-${ds} (stem=$stem)..."
        jid=$(sbatch --parsable \
            --job-name="bin-h${H}-${ds,,}" \
            --account=aip-boyuwang \
            --time="$([[ "$SMOKE" -eq 1 ]] && echo 0:30:00 || echo 1-00:00:00)" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task="$([[ "$SMOKE" -eq 1 ]] && echo 4 || echo 8)" \
            --mem="$([[ "$SMOKE" -eq 1 ]] && echo 24G || echo 60G)" \
            --output=/dev/null \
            --error=/dev/null \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            "$TRAIN_SCRIPT" "${train_extra[@]}")
        TRAIN_JOB["${H}:${ds}"]="$jid"
        echo "  -> $jid"
    done
done

train_dep_for_height() {
    local H="$1"
    local dep="afterok:${TRAIN_JOB[$H:${DATASETS[0]}]}"
    local ds
    for ds in "${DATASETS[@]:1}"; do
        dep+=":${TRAIN_JOB[$H:$ds]}"
    done
    echo "$dep"
}

binary_roots_for_height() {
    local H="$1"
    local roots=() ds stem
    for ds in "${DATASETS[@]}"; do
        stem="${CKPT_STEM[$H:$ds]}"
        roots+=("results/ckpts/${stem}")
    done
    echo "${roots[*]}"
}

submit_matrix() {
    local H="$1"
    local dep="$2"
    local matrix_out="$REPO/results/datasets/${DATE_TAG}-bin-h${H}-mmpd-anchor-matrix${SMOKE_SUFFIX}"
    local bin_roots
    bin_roots="$(binary_roots_for_height "$H")"

    echo "Submitting matrix eval (H=${H}) -> ${matrix_out}"
    echo "  dependency: $dep"
    echo "  binary roots: ${bin_roots}"

    local -a matrix_env=(
        "BINARY_ANCHOR_ROOTS=$bin_roots"
        "MATRIX_OUTPUT_DIR=$matrix_out"
        "MATRIX_DATASETS=${DATASETS[*]}"
    )
    local -a matrix_args=()
    [[ "$H" == "128" ]] && matrix_env+=(
        "SKIP_MMPD_TRAIN=1"
        "MMPD_REUSE_DIR=$REPO/results/datasets/${DATE_TAG}-bin-h64-mmpd-anchor-matrix${SMOKE_SUFFIX}"
    )
    [[ "$SMOKE" -eq 1 ]] && matrix_args+=(--smoke-test)

    local wrap_log="$REPO/results/logs/${DATE_TAG}-bin-h${H}-matrix-wrap-%j.out"
    local wrap_cmd="cd '$REPO' && ${matrix_env[*]} '$MATRIX_SCRIPT' ${matrix_args[*]}"

    MATRIX_WRAP_JOB=$(sbatch --parsable \
        --job-name="matrix-h${H}" \
        --account=aip-boyuwang \
        --nodes=1 \
        --cpus-per-task=2 \
        --mem=4G \
        --time=00:10:00 \
        --dependency="$dep" \
        --output="$wrap_log" \
        --error="$wrap_log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        --wrap="$wrap_cmd")
    echo "  -> matrix wrapper: $MATRIX_WRAP_JOB"
}

DEP64=$(train_dep_for_height 64)
submit_matrix 64 "$DEP64"
MATRIX64_WRAP_JOB="$MATRIX_WRAP_JOB"

DEP128=$(train_dep_for_height 128)
DEP128="${DEP128}:${MATRIX64_WRAP_JOB}"
submit_matrix 128 "$DEP128"

MATRIX64_OUT="$REPO/results/datasets/${DATE_TAG}-bin-h64-mmpd-anchor-matrix${SMOKE_SUFFIX}"
MATRIX128_OUT="$REPO/results/datasets/${DATE_TAG}-bin-h128-mmpd-anchor-matrix${SMOKE_SUFFIX}"

echo ""
echo "=================================================================="
echo "  Chain submitted"
echo "  Train ckpts: results/ckpts/${DATE_TAG}-bin-h{64,128}-{dataset}"
echo "  Matrix H=64:  $MATRIX64_OUT"
echo "  Matrix H=128: $MATRIX128_OUT  (reuses MMPD from H=64)"
echo "  Datasets: ${DATASETS[*]} (electricity excluded)"
echo "  Metrics: full profile + --texture-per-sample (not prob-100)"
echo "  Monitor: squeue -u \$USER"
echo "=================================================================="
