#!/bin/bash
# Login-node submitter for MMPD train+eval campaigns (flat subsets / paper Decoder).
#
# USAGE (Killarney or Narval login node, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd.sh --smoke-test
#   ./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_paper_lb336_hz720 \
#       --output-dir results/datasets/$(date +%m-%d)-mmpd-paper-lb336-hz720 --time 24:00:00
#   ./submit_mmpd.sh --datasets ETTh1,traffic --mmpd-run-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720.yaml
#   ./submit_mmpd.sh --mmpd-backbone MaskAE --output-dir results/datasets/06-15-mmpd-maskae-subset
#
# Cluster auto-detect: Killarney -> gpu:l40s (aip-boyuwang); Narval -> a100 (def-boyuwang).
# Clones temp/MMPD on the login node before submit (compute nodes have no GitHub egress).
# Override GPU: --gpu a100_1g.5gb (Narval) or --gpu l40s (Killarney).
#   ./submit_mmpd.sh --use-anchor-ckpts --anchor-config binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo
#   ./submit_mmpd.sh --resume --output-dir results/datasets/... --datasets PeMS,dynamic --skip-mmpd-train
#
# --mmpd-run-config accepts a path or bare stem under configs/*.yaml.
# Do NOT add new submit_*.sh wrappers for minor YAML variants — use this script.
# Default: --subset-config (no binary ckpts required). Legacy ckpt mode: --use-anchor-ckpts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
RESUME=0
FORCE=0
FORCE_INIT=0
SKIP_MMPD_TRAIN=0
OUTPUT_DIR=""
SUBSET_CONFIG="configs/binary_anchor_stationary_flat_subsets.yaml"
USE_ANCHOR_CKPTS=0
ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets"
DATASETS_CSV="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
DEPENDENCY=""
SEED=2026
LOOKBACK=96
HORIZON=96
WALL_MMPD="3:00:00"
WALL_INIT="0:45:00"
WALL_MERGE="0:30:00"
MMPD_RUN_CONFIG=""
MMPD_BACKBONE="Decoder"
MMPD_TUNE_TRIALS=0
MMPD_TUNE_EPOCHS=10
MMPD_TUNE_PATIENCE=3
ORDINAL_UPSCALE=0
DATASETS_EXPLICIT=0
GPU_TYPE=""
MMPD_INSTANCE_NORM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --force-init) FORCE_INIT=1; shift ;;
        --force) FORCE=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --subset-config) SUBSET_CONFIG="$2"; shift 2 ;;
        --use-anchor-ckpts) USE_ANCHOR_CKPTS=1; shift ;;
        --anchor-config) ANCHOR_CONFIG="$2"; USE_ANCHOR_CKPTS=1; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; DATASETS_EXPLICIT=1; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --lookback) LOOKBACK="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --time) WALL_MMPD="$2"; shift 2 ;;
        --mmpd-backbone) MMPD_BACKBONE="$2"; shift 2 ;;
        --mmpd-run-config) MMPD_RUN_CONFIG="$2"; shift 2 ;;
        --mmpd-tune-trials) MMPD_TUNE_TRIALS="$2"; shift 2 ;;
        --mmpd-tune-epochs) MMPD_TUNE_EPOCHS="$2"; shift 2 ;;
        --mmpd-tune-patience) MMPD_TUNE_PATIENCE="$2"; shift 2 ;;
        --force-mmpd-tune) FORCE=1; shift ;;
        --skip-mmpd-train) SKIP_MMPD_TRAIN=1; shift ;;
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        --mmpd-instance-norm) MMPD_INSTANCE_NORM=1; shift ;;
        --no-mmpd-instance-norm) MMPD_INSTANCE_NORM=0; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run from login node, not inside a Slurm job." >&2
    exit 1
fi

if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
    CLUSTER="narval"
    [[ -z "$GPU_TYPE" ]] && GPU_TYPE="a100"
elif [[ "$(hostname)" == *"killarney"* || "$(hostname)" == kl* ]]; then
    ACCOUNT="aip-boyuwang"
    CLUSTER="killarney"
    [[ -z "$GPU_TYPE" ]] && GPU_TYPE="l40s"
else
    ACCOUNT="aip-boyuwang"
    CLUSTER="killarney"
    [[ -z "$GPU_TYPE" ]] && GPU_TYPE="l40s"
fi

if [[ "$GPU_TYPE" == a100* || "$GPU_TYPE" == h100* ]]; then
    GPU_SBATCH=(--gpus="${GPU_TYPE}:1")
else
    GPU_SBATCH=(--gres="gpu:${GPU_TYPE}:1")
fi

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$SCRIPT_DIR"
fi
if [[ "$REPO" == /home/* ]]; then
    echo "ERROR: submit from \$SCRATCH/ts-sandbox on ${CLUSTER}, not /home." >&2
    exit 1
fi
cd "$REPO"
# shellcheck source=utils/mmpd_submit_helpers.sh
source "$REPO/utils/mmpd_submit_helpers.sh"

MMPD_REPO="$REPO/temp/MMPD"
MMPD_URL="https://github.com/Thinklab-SJTU/MMPD.git"
mkdir -p "$REPO/temp"
if [[ ! -d "$MMPD_REPO/.git" ]]; then
    echo "Cloning MMPD on login node (compute nodes cannot reach GitHub)..."
    git clone "$MMPD_URL" "$MMPD_REPO"
fi
MMPD_TOOLS="$MMPD_REPO/utils/tools.py"
if [[ -f "$MMPD_TOOLS" ]] && grep -q 'np\.Inf' "$MMPD_TOOLS"; then
    sed -i 's/np\.Inf/np.inf/g' "$MMPD_TOOLS"
fi

IFS=',' read -ra DATASETS <<< "$DATASETS_CSV"

filter_datasets_available() {
    local ds path
    local -a available=()
    for ds in "${DATASETS[@]}"; do
        path="$(mmpd_dataset_file_path "$ds" "$REPO" 2>/dev/null || true)"
        if [[ -n "$path" && -f "$path" ]]; then
            available+=("$ds")
        else
            echo "WARN: skipping ${ds} — dataset file missing (${path:-unknown path})" >&2
        fi
    done
    if [[ ${#available[@]} -eq 0 ]]; then
        echo "ERROR: no datasets with local files under ${REPO}/datasets" >&2
        exit 1
    fi
    DATASETS=("${available[@]}")
    DATASETS_CSV=$(IFS=,; echo "${DATASETS[*]}")
}

filter_datasets_available

pick_anchor_root() {
    local ds="$1"
    local matches=()
    shopt -s nullglob
    matches=( "$REPO/results/ckpts"/*-"${ds}"-"${ANCHOR_CONFIG}" )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        return 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

pick_resume_output_dir() {
    local matches=()
    shopt -s nullglob
    matches=(
        "$REPO/results/datasets"/*-sweep-subset-mmpd
        "$REPO/results/datasets"/*-sweep-subset-mmpd-smoke
        "$REPO/results/datasets"/*-mmpd-decoder-paper-lb336-hz96-subset
        "$REPO/results/datasets"/*-mmpd-decoder-grad-accum-200-lr-lo-subset
    )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: --resume but no results/datasets/*-sweep-subset-mmpd found; pass --output-dir" >&2
        exit 1
    fi
    printf '%s\n' "${matches[@]}" | sort | tail -1
}

ANCHOR_ROOTS=()
if [[ "$USE_ANCHOR_CKPTS" -eq 1 ]]; then
    for ds in "${DATASETS[@]}"; do
        if root="$(pick_anchor_root "$ds")"; then
            ANCHOR_ROOTS+=( "$root" )
        else
            ANCHOR_ROOTS+=( "(pending: *-${ds}-${ANCHOR_CONFIG})" )
        fi
    done
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
        OUTPUT_DIR="$(pick_resume_output_dir)"
    else
        RUN_STEM="$(date +%m-%d)-$$-sweep-subset-mmpd$([[ "$SMOKE" -eq 1 ]] && echo -smoke)"
        OUTPUT_DIR="$REPO/results/datasets/${RUN_STEM}"
    fi
else
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO/$OUTPUT_DIR"
fi
RUN_STEM="$(basename "$OUTPUT_DIR")"
LOG_DIR="$REPO/results/logs/${RUN_STEM}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

resolve_repo_yaml() {
    local raw="$1" cand
    raw="${raw#./}"
    if [[ "$raw" == /* && -f "$raw" ]]; then
        echo "$raw"
        return 0
    fi
    if [[ -f "$REPO/$raw" ]]; then
        echo "$REPO/$raw"
        return 0
    fi
    if [[ "$raw" != configs/* && -f "$REPO/configs/$raw" ]]; then
        echo "$REPO/configs/$raw"
        return 0
    fi
    cand="${raw%.yaml}"
    cand="${cand%.yml}"
    if [[ -f "$REPO/configs/${cand}.yaml" ]]; then
        echo "$REPO/configs/${cand}.yaml"
        return 0
    fi
    echo "ERROR: YAML not found for: $1" >&2
    return 1
}

if [[ -n "$MMPD_RUN_CONFIG" ]]; then
    MMPD_RUN_CONFIG="$(resolve_repo_yaml "$MMPD_RUN_CONFIG")" || exit 1
    if grep -Eq '^[[:space:]]*task:[[:space:]]*ordinal_upscale([[:space:]]|$)' "$MMPD_RUN_CONFIG"; then
        ORDINAL_UPSCALE=1
        echo "MMPD task: ordinal_upscale (custom 1D 16-bin -> 256-bin worker)"
    fi
    EVAL_BASE=(
        "$REPO/utils/eval_mmpd_gaussian_anchor.py"
        --mmpd-run-config "$MMPD_RUN_CONFIG"
        --output-dir "$OUTPUT_DIR"
        --ckpt-base "$REPO/results/ckpts"
        --mmpd-repo "$REPO/temp/MMPD"
        --mmpd-data-dir "$REPO/temp/mmpd_datasets"
        --seed "$SEED"
        --no-update-mmpd
        --force-mmpd-eval
        --force-indices
    )
else
EVAL_BASE=(
    "$REPO/utils/eval_mmpd_gaussian_anchor.py"
    --output-dir "$OUTPUT_DIR"
    --ckpt-base "$REPO/results/ckpts"
    --lookback "$LOOKBACK"
    --horizon "$HORIZON"
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --seed "$SEED"
    --mmpd-backbone "$MMPD_BACKBONE"
    --no-update-mmpd
    --force-mmpd-eval
    --force-indices
)
if [[ "$USE_ANCHOR_CKPTS" -eq 1 ]]; then
    EVAL_BASE+=(--anchor-config "$ANCHOR_CONFIG")
else
    if ! grep -q -- '--subset-config' "$REPO/utils/eval_mmpd_gaussian_anchor.py"; then
        echo "ERROR: eval script missing --subset-config; git pull (need ed65c6f+) on cluster." >&2
        exit 1
    fi
    EVAL_BASE+=(
        --subset-config "$REPO/$SUBSET_CONFIG"
        --mmpd-only
    )
fi
fi

if [[ "$SKIP_MMPD_TRAIN" -eq 1 ]]; then
    EVAL_BASE+=(--skip-mmpd-train)
elif [[ "$FORCE" -eq 1 || "$RESUME" -eq 0 ]]; then
    EVAL_BASE+=(--force-mmpd-train)
fi

if [[ "$MMPD_INSTANCE_NORM" -eq 1 ]]; then
    MMPD_NORM_FLAG=(--mmpd-instance-norm)
else
    MMPD_NORM_FLAG=(--no-mmpd-instance-norm)
fi

if [[ "$SMOKE" -eq 1 ]]; then
    WALL_MMPD="0:45:00"
    WALL_INIT="0:25:00"
    WALL_MERGE="0:15:00"
    MEM="24G"
    CPUS=4
    EVAL_EXTRA=(
        --smoke-test
        --mmpd-train-epochs 1
        --mmpd-patience 1
        --mmpd-tune-trials 1
        --mmpd-tune-epochs 1
        --mmpd-tune-patience 1
        --test-fraction 1.0
        --test-max-items 1
        --sample-num 2
        --num-sampling-steps 2
        --topk-max 3
        --gmm-components 3
        --gmm-iterations 2
        --mmpd-batch-size 8
        --mmpd-eval-batch-size 2
        --force-mmpd-tune
        "${MMPD_NORM_FLAG[@]}"
    )
    DATASETS=(ETTh1)
    if [[ -n "$MMPD_RUN_CONFIG" ]]; then
        EVAL_BASE=(
            "$REPO/utils/eval_mmpd_gaussian_anchor.py"
            --mmpd-run-config "$MMPD_RUN_CONFIG"
            --output-dir "$OUTPUT_DIR"
            --ckpt-base "$REPO/results/ckpts"
            --mmpd-repo "$REPO/temp/MMPD"
            --mmpd-data-dir "$REPO/temp/mmpd_datasets"
            --seed "$SEED"
            --no-update-mmpd
            --force-mmpd-eval
            --force-indices
        )
    else
        EVAL_BASE=(
            "$REPO/utils/eval_mmpd_gaussian_anchor.py"
            --output-dir "$OUTPUT_DIR"
            --ckpt-base "$REPO/results/ckpts"
            --lookback "$LOOKBACK"
            --horizon "$HORIZON"
            --mmpd-repo "$REPO/temp/MMPD"
            --mmpd-data-dir "$REPO/temp/mmpd_datasets"
            --seed "$SEED"
            --mmpd-backbone "$MMPD_BACKBONE"
            --no-update-mmpd
            --force-mmpd-eval
            --force-indices
        )
        if [[ "$USE_ANCHOR_CKPTS" -eq 1 ]]; then
            EVAL_BASE+=(--anchor-config "$ANCHOR_CONFIG")
        else
            EVAL_BASE+=(--subset-config "$REPO/$SUBSET_CONFIG" --mmpd-only)
        fi
    fi
    if [[ "$SKIP_MMPD_TRAIN" -eq 1 ]]; then
        EVAL_BASE+=(--skip-mmpd-train)
    elif [[ "$FORCE" -eq 1 || "$RESUME" -eq 0 ]]; then
        EVAL_BASE+=(--force-mmpd-train)
    fi
else
    MEM="60G"
    CPUS=8
    if [[ -n "$MMPD_RUN_CONFIG" ]]; then
        # YAML mmpd: block owns train/eval hyperparams (e.g. paper N=100, tune_trials=0).
        EVAL_EXTRA=(
            --test-fraction 1.0
            --metrics-profile anchor-compat
            "${MMPD_NORM_FLAG[@]}"
            --topk-max 3
            --mmpd-eval-batch-size 16
        )
    else
        EVAL_EXTRA=(
            --mmpd-train-epochs 20
            --mmpd-patience 5
            --test-fraction 1.0
            --eval-test-stride 4
            --sample-num 20
            --num-sampling-steps 20
            --metrics-profile anchor-compat
            "${MMPD_NORM_FLAG[@]}"
            --topk-max 3
            --gmm-components 10
            --gmm-iterations 10
            --mmpd-batch-size 32
            --mmpd-eval-batch-size 16
        )
    fi
fi

if [[ "$MMPD_TUNE_TRIALS" -gt 0 ]]; then
    EVAL_EXTRA+=(
        --mmpd-tune-trials "$MMPD_TUNE_TRIALS"
        --mmpd-tune-epochs "$MMPD_TUNE_EPOCHS"
        --mmpd-tune-patience "$MMPD_TUNE_PATIENCE"
    )
    if [[ "$FORCE" -eq 1 ]]; then
        EVAL_EXTRA+=(--force-mmpd-tune)
    fi
fi

PREAMBLE_FILE="$REPO/results/job_preamble_mmpd_sweep_subset.sh"
cat > "$PREAMBLE_FILE" << PREAMBLE
set -euo pipefail
echo "Job: \$SLURM_JOB_NAME  ID: \$SLURM_JOB_ID  Node: \${SLURMD_NODENAME:-unknown}"
echo "GPU: \$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: \$(date)"

REPO="$REPO"
REQ="\$REPO/setup/requirements-killarney.txt"
[[ -f "\$REQ" ]] || { echo "ERROR: missing \$REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "\${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from \$REQ"
virtualenv --no-download "\$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "\$SLURM_TMPDIR/env/bin/activate"
export PYTHON="\$SLURM_TMPDIR/env/bin/python"
pip install --no-index --upgrade pip -q
pip install --no-index -r "\$REQ" -q
"\$PYTHON" -c "import torch, optuna, wandb, einops, yaml; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ -f "\$REPO/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "\$REPO/.env"
    set +a
fi
if [[ -n "\${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    echo "[wandb] WANDB_API_KEY set — leaderboard rows log when mmpd YAML has leaderboard: true"
else
    echo "[wandb] WARN: WANDB_API_KEY unset — mmpd_eval leaderboard logging will be skipped"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="\$REPO"
export PYTHONPATH="\$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
cd "\$REPO"
PREAMBLE

write_worker_script() {
    local path="$1"
    shift
    cat > "$path" << SCRIPT
#!/bin/bash
source "$PREAMBLE_FILE"
exec "\$PYTHON" -u $(printf '%q ' "$@")
SCRIPT
    chmod +x "$path"
}

SBATCH_COMMON=(
    --account="$ACCOUNT"
    --nodes=1
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    "${GPU_SBATCH[@]}"
    --export=ALL
    --mail-type=FAIL
    --mail-user=ccao87@uwo.ca
)

echo "Cluster:       $CLUSTER"
echo "GPU:           $GPU_TYPE"
echo "Repo:          $REPO"
echo "Output:        $OUTPUT_DIR"
if [[ "$USE_ANCHOR_CKPTS" -eq 1 ]]; then
    echo "Subset source: anchor ckpts (*-<dataset>-${ANCHOR_CONFIG})"
else
    echo "Subset source: $SUBSET_CONFIG (YAML data_subset)"
fi
echo "Lookback/horizon: $LOOKBACK / $HORIZON"
echo "MMPD backbone:   $MMPD_BACKBONE"
echo "MMPD tune:       trials=$MMPD_TUNE_TRIALS epochs=$MMPD_TUNE_EPOCHS patience=$MMPD_TUNE_PATIENCE"
echo "Resume:        $RESUME  Force: $FORCE  Skip train: $SKIP_MMPD_TRAIN"
echo "Datasets:      ${DATASETS[*]}"
if [[ "$USE_ANCHOR_CKPTS" -eq 1 ]]; then
    for i in "${!DATASETS[@]}"; do
        echo "  ${DATASETS[$i]} <- ${ANCHOR_ROOTS[$i]}"
    done
fi

if [[ "$ORDINAL_UPSCALE" -eq 1 ]]; then
    ORDINAL_SBATCH_EXTRA=()
    if [[ "$DATASETS_EXPLICIT" -eq 0 ]]; then
        ORDINAL_DATASETS=$(sed -n -E 's/^[[:space:]]*datasets:[[:space:]]*\[([^]]+)\][[:space:]]*$/\1/p' "$MMPD_RUN_CONFIG" | head -n 1)
        if [[ -z "$ORDINAL_DATASETS" ]]; then
            echo "ERROR: ordinal_upscale config must use an inline mmpd.datasets list or pass --datasets." >&2
            exit 1
        fi
        DATASETS_CSV="${ORDINAL_DATASETS//[[:space:]]/}"
        IFS=',' read -ra DATASETS <<< "$DATASETS_CSV"
        filter_datasets_available
        echo "Ordinal datasets from config: ${DATASETS[*]}"
    fi
    [[ -n "$DEPENDENCY" ]] && ORDINAL_SBATCH_EXTRA=(--dependency="$DEPENDENCY")
    ORDINAL_EXTRA=()
    [[ "$SMOKE" -eq 1 ]] && ORDINAL_EXTRA+=(--smoke)
    ORDINAL_IDS=()
    for ds in "${DATASETS[@]}"; do
        case "$ds" in
            ETTh1|exchange_rate|electricity|traffic) ;;
            *)
                echo "ERROR: ordinal_upscale supports ETTh1, exchange_rate, electricity, or traffic; got $ds" >&2
                exit 1
                ;;
        esac
        ORDINAL_OUT="$OUTPUT_DIR/$ds"
        ORDINAL_WORKER="$LOG_DIR/submit-ordinal-upscale-${ds}.sh"
        mkdir -p "$ORDINAL_OUT"
        write_worker_script "$ORDINAL_WORKER" \
            -m experiments.ordinal_patch_refinement_killtest.run_mmpd_ordinal_upscale_tpe_ema \
            --config "$MMPD_RUN_CONFIG" \
            --dataset "$ds" \
            --output "$ORDINAL_OUT" \
            --seed "$SEED" \
            "${ORDINAL_EXTRA[@]}"
        echo "Submitting ordinal-upscale-${ds} wall=${WALL_MMPD}..."
        JOB_ORDINAL=$(sbatch --parsable \
            --job-name="mmpd-ord-${ds}$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
            "${SBATCH_COMMON[@]}" \
            --time="$WALL_MMPD" \
            "${ORDINAL_SBATCH_EXTRA[@]}" \
            --output="$LOG_DIR/ordinal-upscale-${ds}-%j.out" \
            --error="$LOG_DIR/ordinal-upscale-${ds}-%j.err" \
            "$ORDINAL_WORKER")
        echo "  -> ordinal-upscale-${ds}: $JOB_ORDINAL"
        ORDINAL_IDS+=("$JOB_ORDINAL")
    done
    echo "Submitted ordinal-upscale jobs: ${ORDINAL_IDS[*]}"
    exit 0
fi
SKIP_INIT=0
if [[ "$FORCE_INIT" -eq 1 ]]; then
    echo "Force-init: will rerun init (ignoring existing manifest)"
elif [[ "$RESUME" -eq 1 && -f "$OUTPUT_DIR/run_manifest.json" ]]; then
    SKIP_INIT=1
    echo "Resume: reusing $OUTPUT_DIR/run_manifest.json"
elif [[ "$RESUME" -eq 1 ]]; then
    echo "Resume: no run_manifest.json — will submit init"
fi

JOB_INIT=""
WORKER_DEP=()
INIT_SBATCH_EXTRA=()
if [[ -n "$DEPENDENCY" ]]; then
    INIT_SBATCH_EXTRA=(--dependency="$DEPENDENCY")
fi
if [[ "$SKIP_INIT" -eq 0 ]]; then
    INIT_SCRIPT="$LOG_DIR/submit-init.sh"
    write_worker_script "$INIT_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
        --phase init --datasets "${DATASETS[@]}"
    echo "Init script: $INIT_SCRIPT"
    grep -E 'subset-config|anchor-config' "$INIT_SCRIPT" || true
    echo "Submitting init..."
    JOB_INIT=$(sbatch --parsable \
        --job-name="mmpd-sw-init$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --time="$WALL_INIT" \
        "${INIT_SBATCH_EXTRA[@]}" \
        --output="$LOG_DIR/init-%j.out" \
        --error="$LOG_DIR/init-%j.err" \
        "$INIT_SCRIPT")
    echo "  -> init: $JOB_INIT"
    WORKER_DEP=(--dependency="afterok:$JOB_INIT")
fi

WORKER_IDS=()
PENDING_DATASETS=()
for ds in "${DATASETS[@]}"; do
    partial="$OUTPUT_DIR/partials/${ds}_mmpd.json"
    if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$partial" ]]; then
        echo "Skip mmpd-${ds}: partial exists ($partial)"
        continue
    fi
    PENDING_DATASETS+=("$ds")

    WORKER_SCRIPT="$LOG_DIR/submit-mmpd-${ds}.sh"
    DS_EXTRA=()
    while IFS= read -r _arg; do
        [[ -n "$_arg" ]] && DS_EXTRA+=("$_arg")
    done < <(mmpd_dataset_worker_extra_args "$ds")
    DS_WALL="$(mmpd_dataset_wall_time "$ds" "$WALL_MMPD")"
    write_worker_script "$WORKER_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
        "${DS_EXTRA[@]}" \
        --phase mmpd --datasets "$ds"
    echo "Submitting mmpd-${ds} wall=${DS_WALL} ${WORKER_DEP[*]}..."
    JOB_MMPD=$(sbatch --parsable \
        --job-name="mmpd-sw-${ds}$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
        "${SBATCH_COMMON[@]}" \
        --time="$DS_WALL" \
        "${WORKER_DEP[@]}" \
        --output="$LOG_DIR/mmpd-${ds}-%j.out" \
        --error="$LOG_DIR/mmpd-${ds}-%j.err" \
        "$WORKER_SCRIPT")
    echo "  -> mmpd-${ds}: $JOB_MMPD"
    WORKER_IDS+=("$JOB_MMPD")
done

if [[ ${#WORKER_IDS[@]} -eq 0 && ${#PENDING_DATASETS[@]} -eq 0 ]]; then
    echo "All dataset partials present; submitting merge only."
    MERGE_DEP_ARGS=()
elif [[ ${#WORKER_IDS[@]} -eq 0 ]]; then
    echo "ERROR: nothing to submit (no pending datasets and no workers)." >&2
    exit 1
else
    MERGE_DEP="afterok:${WORKER_IDS[0]}"
    for wid in "${WORKER_IDS[@]:1}"; do
        MERGE_DEP+=":$wid"
    done
    MERGE_DEP_ARGS=(--dependency="$MERGE_DEP")
fi

MERGE_SCRIPT="$LOG_DIR/submit-merge.sh"
write_worker_script "$MERGE_SCRIPT" "${EVAL_BASE[@]}" "${EVAL_EXTRA[@]}" \
    --phase merge --datasets "${DATASETS[@]}" --cpu

echo "Submitting merge ${MERGE_DEP_ARGS[*]}..."
JOB_MERGE=$(sbatch --parsable \
    --job-name="mmpd-sw-merge$([[ "$SMOKE" -eq 1 ]] && echo -smoke)" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time="$WALL_MERGE" \
    "${MERGE_DEP_ARGS[@]}" \
    --output="$LOG_DIR/merge-%j.out" \
    --error="$LOG_DIR/merge-%j.err" \
    --mail-type=FAIL \
    --mail-user=ccao87@uwo.ca \
    "$MERGE_SCRIPT")
echo "  -> merge: $JOB_MERGE"

echo ""
echo "=================================================================="
echo "  MMPD sweep-subset submitted"
if [[ -n "$JOB_INIT" ]]; then echo "  init:  $JOB_INIT"; fi
echo "  workers: ${#WORKER_IDS[@]} pending dataset(s): ${PENDING_DATASETS[*]:-none}"
echo "  merge: $JOB_MERGE"
echo "  Output: $OUTPUT_DIR"
echo "  Logs:   $LOG_DIR/"
echo "  Monitor: squeue -u \$USER"
echo "=================================================================="
