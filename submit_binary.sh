#!/bin/bash
# =============================================================================
# Login-node submitter for binary / patch-decoder diffusion pipeline jobs.
# Compute worker: slurm_worker.sh → models.diffusion_tsf.train_multivariate_pipeline
#
# Each job gets isolated checkpoint/results dirs:
#   ./results/ckpts/MM-DD-<jobid>-<dataset>-<config>/
#
# USAGE (run from login node, repo root / $SCRATCH/ts-sandbox):
#   ./submit_binary.sh --configs binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm \
#       --datasets ETTh1,traffic --time 10:00:00
#   ./submit_binary.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
#   ./submit_binary.sh --smoke
#   ./submit_binary.sh --resume --configs binary_dual_scale_staged --datasets ETTh1
#   ./submit_binary.sh --parallel-optuna 4 --configs binary_dual_scale_staged --datasets ETTh1
#
# --configs accepts comma-separated paths, globs, or bare stems under configs/*.yaml.
# Do NOT add new submit_*.sh wrappers for minor YAML variants — use this script.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=""
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama"
SEEDS="42"
SMOKE=0
RESUME=0
CKPT_CONFIG=""
DEPENDENCY=""
WANDB_PROJECT=""
WANDB_PROJECT_EXPLICIT=0
WALL_OVERRIDE=""
PARALLEL_OPTUNA=""
EVAL_EXISTING_PATCH_REFINE=0
EXISTING_CKPT_ROOTS=""
DISC_RUN=""
if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
    GPU_TYPE="a100"
else
    ACCOUNT="aip-boyuwang"
    GPU_TYPE="l40s"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs|--config) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --ckpt-config) CKPT_CONFIG="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; WANDB_PROJECT_EXPLICIT=1; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        --parallel-optuna) PARALLEL_OPTUNA="$2"; shift 2 ;;
        --eval-existing-patch-refine) EVAL_EXISTING_PATCH_REFINE=1; shift ;;
        --existing-ckpt-roots) EXISTING_CKPT_ROOTS="$2"; shift 2 ;;
        --disc-run) DISC_RUN="$2"; shift 2 ;;
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Fixed-checkpoint h96 discriminator mode.  This deliberately bypasses the
# training pipeline: the only model fit is the discriminator inside the eval.
if [[ "$EVAL_EXISTING_PATCH_REFINE" -eq 1 ]]; then
    [[ -n "$EXISTING_CKPT_ROOTS" ]] || {
        echo "ERROR: --eval-existing-patch-refine requires --existing-ckpt-roots dataset=/absolute/or/relative/run,..." >&2
        exit 1
    }
    [[ -z "$CONFIGS" ]] || {
        echo "ERROR: --configs is not used with --eval-existing-patch-refine" >&2
        exit 1
    }
    [[ "$RESUME" -eq 0 && "$SMOKE" -eq 0 && -z "$PARALLEL_OPTUNA" ]] || {
        echo "ERROR: --resume, --smoke, and --parallel-optuna are not valid with --eval-existing-patch-refine" >&2
        exit 1
    }

    STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
    LOG_DIR="$STORE/logs"
    mkdir -p "$LOG_DIR"
    IFS=',' read -ra EVAL_DATASETS <<< "$DATASETS"
    IFS=',' read -ra ROOT_PAIRS <<< "$EXISTING_CKPT_ROOTS"
    declare -A ROOT_BY_DATASET=()
    for pair in "${ROOT_PAIRS[@]}"; do
        [[ "$pair" == *=* ]] || {
            echo "ERROR: invalid --existing-ckpt-roots entry: $pair (expected dataset=path)" >&2
            exit 1
        }
        dataset_key="${pair%%=*}"
        checkpoint_root="${pair#*=}"
        [[ -n "$dataset_key" && -n "$checkpoint_root" ]] || {
            echo "ERROR: invalid --existing-ckpt-roots entry: $pair" >&2
            exit 1
        }
        [[ "$checkpoint_root" == /* ]] || checkpoint_root="$SCRIPT_DIR/$checkpoint_root"
        ROOT_BY_DATASET["$dataset_key"]="$checkpoint_root"
    done

    DISC_RUN="${DISC_RUN:-$(date +%m-%d)-patch-refine-h96-existing-disc}"
    [[ "$DISC_RUN" == /* ]] && {
        echo "ERROR: --disc-run must be a relative results/datasets run name" >&2
        exit 1
    }
    WALL="${WALL_OVERRIDE:-2:00:00}"
    USER_NAME="$(whoami)"
    if [[ "$GPU_TYPE" == a100* || "$GPU_TYPE" == h100* ]]; then
        GPU_ARG="--gpus=${GPU_TYPE}:1"
    else
        GPU_ARG="--gres=gpu:${GPU_TYPE}:1"
    fi

    echo "Submitting fixed h96 patch-refine discriminator jobs (forecast checkpoints are read-only)."
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        checkpoint_root="${ROOT_BY_DATASET[$dataset_name]:-}"
        [[ -n "$checkpoint_root" ]] || {
            echo "ERROR: no checkpoint root provided for dataset $dataset_name" >&2
            exit 1
        }
        subset_name="$dataset_name"
        [[ "$dataset_name" == traffic ]] && subset_name="traffic_4v_s1"
        [[ -f "$checkpoint_root/$subset_name/coarse/best.pt" ]] || {
            echo "ERROR: missing coarse checkpoint: $checkpoint_root/$subset_name/coarse/best.pt" >&2
            exit 1
        }
        [[ -f "$checkpoint_root/$subset_name/patch_refine/best.pt" ]] || {
            echo "ERROR: missing patch_refine checkpoint: $checkpoint_root/$subset_name/patch_refine/best.pt" >&2
            exit 1
        }
        [[ -f "$checkpoint_root/$subset_name/patch_refine/metadata.json" ]] || {
            echo "ERROR: missing patch_refine metadata: $checkpoint_root/$subset_name/patch_refine/metadata.json" >&2
            exit 1
        }
        output_dir="$STORE/datasets/$DISC_RUN/$dataset_name"
        job_id=$(sbatch --parsable \
            --job-name="disc-pr96-${dataset_name}" \
            --account="$ACCOUNT" \
            --time="$WALL" \
            --nodes=1 \
            "$GPU_ARG" \
            --cpus-per-task=8 \
            --mem=50G \
            --output="$LOG_DIR/disc-pr96-${dataset_name}-%j.log" \
            --error="$LOG_DIR/disc-pr96-${dataset_name}-%j.log" \
            --mail-type=FAIL \
            --mail-user="${USER_NAME}@uwo.ca" \
            --export=ALL,GRID_EVAL_PATCH_REFINE=1,GRID_DATASET="$dataset_name",GRID_EXISTING_CKPT="$checkpoint_root",GRID_DISC_OUTPUT="$output_dir" \
            "$SCRIPT_DIR/slurm_worker.sh")
        echo "  -> $dataset_name: $job_id ($output_dir)"
    done
    echo "Monitor with: squeue -u $USER_NAME"
    exit 0
fi

if [[ "$SMOKE" -eq 1 ]]; then
    CONFIGS="${CONFIGS:-configs/smoke_test.yaml}"
    WALL_DEFAULT="0:30:00"
    MEM="24G"
    CPUS=4
    GPUS=1
    JOB_PREFIX="smoke"
else
    CONFIGS="${CONFIGS:-configs/binary_dual_scale_staged.yaml}"
    WALL_DEFAULT="3:00:00"
    MEM="60G"
    CPUS=8
    GPUS=1
    JOB_PREFIX="grid"
fi

# Leaderboard default when wandb is enabled and caller did not pass --wandb-project.
if [[ -n "${WANDB_API_KEY:-}" && "$WANDB_PROJECT_EXPLICIT" -eq 0 ]]; then
    WANDB_PROJECT="ts-sandbox-leaderboard"
    WANDB_PROJECT_EXPLICIT=1
fi

if [[ -n "$PARALLEL_OPTUNA" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        echo "ERROR: --parallel-optuna is not supported with --smoke" >&2
        exit 1
    fi
    if ! [[ "$PARALLEL_OPTUNA" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --parallel-optuna requires a positive integer (got: $PARALLEL_OPTUNA)" >&2
        exit 1
    fi
    GPUS="$PARALLEL_OPTUNA"
    MEM="$((PARALLEL_OPTUNA * 20))G"
    CPUS="$((PARALLEL_OPTUNA * 2))"
fi

IFS=',' read -ra CONF_ARR <<< "$CONFIGS"
IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

# Resolve a single config token to a repo-relative path under configs/.
resolve_config_token() {
    local raw="$1" cand
    raw="${raw#./}"
    if [[ -f "$SCRIPT_DIR/$raw" ]]; then
        echo "$raw"
        return 0
    fi
    if [[ "$raw" != configs/* && -f "$SCRIPT_DIR/configs/$raw" ]]; then
        echo "configs/$raw"
        return 0
    fi
    cand="${raw%.yaml}"
    cand="${cand%.yml}"
    if [[ -f "$SCRIPT_DIR/configs/${cand}.yaml" ]]; then
        echo "configs/${cand}.yaml"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/${cand}.yaml" ]]; then
        echo "${cand}.yaml"
        return 0
    fi
    echo "ERROR: config not found for token: $1 (tried path and configs/${cand}.yaml)" >&2
    return 1
}

expand_config_globs() {
    local expanded=() CFG matches m rel
    for CFG in "${CONF_ARR[@]}"; do
        if [[ "$CFG" == *"*"* || "$CFG" == *"?"* || "$CFG" == *"["* ]]; then
            shopt -s nullglob
            matches=( "$SCRIPT_DIR"/$CFG )
            if [[ ${#matches[@]} -eq 0 && "$CFG" != configs/* ]]; then
                matches=( "$SCRIPT_DIR"/configs/$CFG )
            fi
            shopt -u nullglob
            if [[ ${#matches[@]} -eq 0 ]]; then
                echo "ERROR: no config yml files matched glob: $CFG" >&2
                exit 1
            fi
            for m in "${matches[@]}"; do
                expanded+=( "${m#$SCRIPT_DIR/}" )
            done
        else
            rel="$(resolve_config_token "$CFG")" || exit 1
            expanded+=( "$rel" )
        fi
    done
    CONF_ARR=( "${expanded[@]}" )
}
expand_config_globs

USER=$(whoami)
STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
LOG_DIR="$STORE/logs"
CKPT_ROOT="$STORE/ckpts"
DATA_ROOT="$STORE/datasets"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$DATA_ROOT"

# Resume: find newest ckpts/*-<dataset>-<config> dir; stem is reused for logs/ckpts/datasets.
pick_resume_stem() {
    local ds="$1" cfg="$2"
    local best="" best_mtime=0 d m
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*-"${ds}"-"${cfg}"; do
        [[ -d "$d" ]] || continue
        m=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [[ "$m" -gt "$best_mtime" ]]; then
            best_mtime="$m"
            best="$(basename "$d")"
        fi
    done
    shopt -u nullglob
    echo "$best"
}

for CFG in "${CONF_ARR[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/$CFG" ]]; then
        echo "ERROR: config yml file that was specified was not found: $SCRIPT_DIR/$CFG" >&2
        exit 1
    fi
done

echo "Submitting grid... (Storage: $STORE)"
[[ "$RESUME" -eq 1 ]] && echo "Resume: requires existing *-<dataset>-<config> checkpoint dir (newest mtime wins)."
[[ -n "$CKPT_CONFIG" ]] && echo "Checkpoint stem matcher: *-<dataset>-${CKPT_CONFIG}"
printf "%-10s %-15s %-25s %-8s %s\n" "JOB ID" "DATASET" "CONFIG" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

for CFG in "${CONF_ARR[@]}"; do
    CFG_NAME=$(basename "$CFG" .yaml)
    for DS in "${DATA_ARR[@]}"; do
        for SD in "${SEED_ARR[@]}"; do

            JOB_NAME="${JOB_PREFIX}-${DS}-${CFG_NAME}"
            DATE_STR=$(date +%m-%d)

            if [[ -n "$WALL_OVERRIDE" ]]; then
                WALL="$WALL_OVERRIDE"
            else
                WALL="$WALL_DEFAULT"
            fi

            RUN_STEM=""
            CKPT_MATCH="${CKPT_CONFIG:-$CFG_NAME}"
            if [[ "$RESUME" -eq 1 ]]; then
                RUN_STEM=$(pick_resume_stem "$DS" "$CKPT_MATCH")
                if [[ -z "$RUN_STEM" ]]; then
                    echo "ERROR: --resume but no checkpoint dir matching ${CKPT_ROOT}/*-${DS}-${CKPT_MATCH}" >&2
                    exit 1
                fi
            fi

            if [[ -n "$RUN_STEM" ]]; then
                LOG_FILE="$LOG_DIR/${RUN_STEM}.log"
            else
                LOG_FILE="$LOG_DIR/${DATE_STR}-%j-${DS}-${CFG_NAME}.log"
            fi

            if [[ "$GPU_TYPE" == a100* || "$GPU_TYPE" == h100* ]]; then
                GPU_ARG="--gpus=${GPU_TYPE}:${GPUS}"
            else
                GPU_ARG="--gres=gpu:${GPU_TYPE}:${GPUS}"
            fi

            S_ARGS=(
                --parsable
                --job-name="$JOB_NAME"
                --account="$ACCOUNT"
                --time="$WALL"
                --nodes=1
                "$GPU_ARG"
                --cpus-per-task="$CPUS"
                --mem="$MEM"
                --output="$LOG_FILE"
                --error="$LOG_FILE"
                --mail-type=FAIL
                --mail-user="${USER}@uwo.ca"
                --export=ALL,GRID_DATE_STR="$DATE_STR",GRID_DATASET="$DS",GRID_CFG_NAME="$CFG_NAME",GRID_STORE="$STORE",GRID_RESUME="$RESUME",GRID_RUN_STEM="$RUN_STEM"
            )

            if [[ -n "$DEPENDENCY" ]]; then
                S_ARGS+=(--dependency="$DEPENDENCY")
            fi

            PY_ARGS=(
                --config "$CFG"
                --dataset "$DS"
                --seed "$SD"
            )

            if [[ -n "${N_VARIATES:-}" ]]; then
                PY_ARGS+=(--n-variates "$N_VARIATES")
            fi

            if [[ -n "${WANDB_API_KEY:-}" ]]; then
                PY_ARGS+=(--wandb)
                if [[ "$WANDB_PROJECT_EXPLICIT" -eq 1 ]]; then
                    PY_ARGS+=(--wandb-project "$WANDB_PROJECT")
                fi
            fi

            if [[ "$SMOKE" -eq 1 ]]; then
                PY_ARGS+=(--smoke-test)
            fi

            if [[ "$RESUME" -eq 1 && -n "$RUN_STEM" ]]; then
                PY_ARGS+=(--resume)
            fi

            if [[ -n "$PARALLEL_OPTUNA" ]]; then
                PY_ARGS+=(--parallel-optuna-workers "$PARALLEL_OPTUNA")
            fi

            JOB_ID=$(sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/slurm_worker.sh" "${PY_ARGS[@]}")

            if [[ -z "$RUN_STEM" ]]; then
                RUN_STEM="${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}"
            fi
            ACTUAL_LOG="$LOG_DIR/${RUN_STEM}.log"
            if [[ "$LOG_FILE" == *'%j'* ]]; then
                ACTUAL_LOG="$LOG_DIR/${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}.log"
            fi
            printf "%-10s %-15s %-25s %-8s %s\n" "$JOB_ID" "$DS" "$CFG_NAME" "$SD" "$ACTUAL_LOG"

        done
    done
done
echo "--------------------------------------------------------------------------------"
echo "Monitor with: squeue -u $USER"
