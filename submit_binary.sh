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
#   ./submit_binary.sh --gpu h100 --configs <stem> --datasets electricity --time 1-00:00:00
#   ./submit_binary.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
#   ./submit_binary.sh --smoke
#   ./submit_binary.sh --resume --configs binary_window_norm_patch_refine_canvas128_p64x6 --datasets ETTh1
#
# --configs accepts comma-separated paths, globs, or bare stems under configs/*.yaml.
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
EXISTING_CKPT_ROOTS=""
DISC_RUN=""
ORDINAL_SLICE_LENGTHS="8;16;32"
SBATCH_EXCLUDE_NODES=""
RAW_RUN=""
JOB_MANIFEST=""
EVAL_ORDINAL_PATCH_REFINE_MMPD=0
MMPD_ROOT=""
ORDINAL_DISC_EVALUATOR="temp/scripts/eval_univariate_patch_refine_ordinal_vs_mmpd.py"
ORDINAL_BINARY_CONFIG="configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
DEFER_CHECKPOINT_CHECK=0
N_VARIATES=""

if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
    GPU_TYPE="a100"
else
    ACCOUNT="aip-boyuwang"
    GPU_TYPE="l40s"
fi
PARTITION_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs|--config) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --ckpt-config) CKPT_CONFIG="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; WANDB_PROJECT_EXPLICIT=1; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        --existing-ckpt-roots) EXISTING_CKPT_ROOTS="$2"; shift 2 ;;
        --disc-run) DISC_RUN="$2"; shift 2 ;;
        --raw-run) RAW_RUN="$2"; shift 2 ;;
        --job-manifest) JOB_MANIFEST="$2"; shift 2 ;;
        --eval-ordinal-patch-refine-vs-mmpd) EVAL_ORDINAL_PATCH_REFINE_MMPD=1; shift ;;
        --mmpd-root) MMPD_ROOT="$2"; shift 2 ;;
        --ordinal-disc-evaluator) ORDINAL_DISC_EVALUATOR="$2"; shift 2 ;;
        --ordinal-binary-config) ORDINAL_BINARY_CONFIG="$2"; shift 2 ;;
        --defer-checkpoint-check) DEFER_CHECKPOINT_CHECK=1; shift ;;
        --slice-lengths) ORDINAL_SLICE_LENGTHS="${2//,/;}" ; ORDINAL_SLICE_LENGTHS="${ORDINAL_SLICE_LENGTHS// /;}"; shift 2 ;;
        --exclude) SBATCH_EXCLUDE_NODES="$2"; shift 2 ;;
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

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

pick_gpubase_partition() {
    local prefix="$1" need_s="$2" part max_wall max_s best="" best_s=0
    if [[ -n "$PARTITION_OVERRIDE" ]]; then
        echo "$PARTITION_OVERRIDE"
        return 0
    fi
    while read -r part max_wall; do
        [[ "$part" == ${prefix}* ]] || continue
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
        echo "ERROR: no ${prefix}* partition allows --time wall ($need_s s). Check sinfo." >&2
        return 1
    fi
    echo "$best"
}

gpu_sbatch_args() {
    local gpus="${1:-1}"
    GPU_SBATCH_ARGS=()
    if [[ "$GPU_TYPE" == h100* ]]; then
        local wall="${WALL:-${WALL_DEFAULT:-1:00:00}}"
        local part
        part="$(pick_gpubase_partition "gpubase_h100_b" "$(slurm_time_to_seconds "$wall")")" || return 1
        GPU_SBATCH_ARGS=(--partition="$part" --gpus-per-node=h100:"$gpus")
        echo "H100 request: partition=$part gpus-per-node=h100:$gpus wall=$wall" >&2
    elif [[ "$GPU_TYPE" == a100* ]]; then
        GPU_SBATCH_ARGS=(--gpus="${GPU_TYPE}:${gpus}")
    elif [[ "$GPU_TYPE" == l40s* ]]; then
        local wall="${WALL:-${WALL_DEFAULT:-1:00:00}}"
        local part
        part="$(pick_gpubase_partition "gpubase_l40s_b" "$(slurm_time_to_seconds "$wall")")" || return 1
        GPU_SBATCH_ARGS=(--partition="$part" --gres=gpu:l40s:"$gpus")
        echo "L40S request: partition=$part gres=gpu:l40s:$gpus wall=$wall" >&2
    else
        GPU_SBATCH_ARGS=(--gres=gpu:${GPU_TYPE}:${gpus})
    fi
}

manifest_tool() {
    python3 "$SCRIPT_DIR/temp/scripts/submission_manifest.py" "$@"
}

# Deferred h96 ordinal patch-refine discriminator mode.
if [[ "$EVAL_ORDINAL_PATCH_REFINE_MMPD" -eq 1 ]]; then
    [[ -n "$EXISTING_CKPT_ROOTS" ]] || {
        echo "ERROR: --eval-ordinal-patch-refine-vs-mmpd requires --existing-ckpt-roots dataset=run,..." >&2
        exit 1
    }
    [[ -n "$MMPD_ROOT" ]] || {
        echo "ERROR: --eval-ordinal-patch-refine-vs-mmpd requires --mmpd-root" >&2
        exit 1
    }
    [[ -n "$ORDINAL_BINARY_CONFIG" ]] || {
        echo "ERROR: --eval-ordinal-patch-refine-vs-mmpd requires --ordinal-binary-config" >&2
        exit 1
    }
    [[ -z "$CONFIGS" && "$RESUME" -eq 0 && "$SMOKE" -eq 0 ]] || {
        echo "ERROR: ordinal patch-refine discriminator mode does not accept --configs, --resume, or --smoke" >&2
        exit 1
    }
    [[ "$DEFER_CHECKPOINT_CHECK" -eq 1 ]] || {
        echo "ERROR: ordinal patch-refine discriminator mode requires --defer-checkpoint-check for its explicit compute-node validation contract" >&2
        exit 1
    }

    STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
    [[ "$STORE" == /* ]] || STORE="$SCRIPT_DIR/$STORE"
    LOG_DIR="$STORE/logs"
    mkdir -p "$LOG_DIR"
    [[ "$MMPD_ROOT" == /* ]] || MMPD_ROOT="$SCRIPT_DIR/$MMPD_ROOT"
    [[ "$ORDINAL_DISC_EVALUATOR" == /* ]] || ORDINAL_DISC_EVALUATOR="$SCRIPT_DIR/$ORDINAL_DISC_EVALUATOR"
    [[ "$ORDINAL_BINARY_CONFIG" == /* ]] || ORDINAL_BINARY_CONFIG="$SCRIPT_DIR/$ORDINAL_BINARY_CONFIG"
    [[ -f "$ORDINAL_DISC_EVALUATOR" ]] || {
        echo "ERROR: ordinal discriminator evaluator not found: $ORDINAL_DISC_EVALUATOR" >&2
        exit 1
    }
    [[ -f "$ORDINAL_BINARY_CONFIG" ]] || {
        echo "ERROR: ordinal binary config not found: $ORDINAL_BINARY_CONFIG" >&2
        exit 1
    }
    IFS=',' read -ra EVAL_DATASETS <<< "$DATASETS"
    IFS=',' read -ra ROOT_PAIRS <<< "$EXISTING_CKPT_ROOTS"
    declare -A ROOT_BY_DATASET=()
    for pair in "${ROOT_PAIRS[@]}"; do
        [[ "$pair" == *=* ]] || { echo "ERROR: invalid --existing-ckpt-roots entry: $pair" >&2; exit 1; }
        dataset_key="${pair%%=*}"
        checkpoint_root="${pair#*=}"
        [[ -n "$dataset_key" && -n "$checkpoint_root" ]] || { echo "ERROR: invalid --existing-ckpt-roots entry: $pair" >&2; exit 1; }
        [[ "$checkpoint_root" == /* ]] || checkpoint_root="$SCRIPT_DIR/$checkpoint_root"
        ROOT_BY_DATASET["$dataset_key"]="$checkpoint_root"
    done
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        [[ -n "${ROOT_BY_DATASET[$dataset_name]:-}" ]] || {
            echo "ERROR: no checkpoint root provided for dataset $dataset_name" >&2
            exit 1
        }
    done

    DISC_RUN="${DISC_RUN:-$(date +%m-%d)-ordinal-patch-refine-h96-disc}"
    RAW_RUN="${RAW_RUN:-${DISC_RUN}-raw}"
    [[ "$DISC_RUN" != /* && "$RAW_RUN" != /* ]] || {
        echo "ERROR: --disc-run/--raw-run must be relative results/datasets names" >&2
        exit 1
    }
    DISC_OUTPUT="$STORE/datasets/$DISC_RUN"
    RAW_OUTPUT="$STORE/datasets/$RAW_RUN"
    MANIFEST_PATH="${JOB_MANIFEST:-$DISC_OUTPUT/submission_manifest.json}"
    [[ "$MANIFEST_PATH" == /* ]] || MANIFEST_PATH="$SCRIPT_DIR/$MANIFEST_PATH"
    manifest_tool init --path "$MANIFEST_PATH" --component "ordinal_patch_refine_discriminator" \
        --repo "$SCRIPT_DIR" --datasets "$DATASETS" \
        --set "mmpd_root=$MMPD_ROOT" --set "output_dir=$DISC_OUTPUT" --set "raw_eval_dir=$RAW_OUTPUT" \
        --set "evaluator=$ORDINAL_DISC_EVALUATOR" --set "binary_config=$ORDINAL_BINARY_CONFIG"

    WALL="${WALL_OVERRIDE:-2:00:00}"
    USER_NAME="$(whoami)"
    gpu_sbatch_args 1 || exit 1
    DEP_ARGS=()
    [[ -n "$DEPENDENCY" ]] && DEP_ARGS=(--dependency="$DEPENDENCY")
    EXCLUDE_ARGS=()
    [[ -n "${SBATCH_EXCLUDE_NODES:-}" ]] && EXCLUDE_ARGS=(--exclude="$SBATCH_EXCLUDE_NODES")
    DISC_JOB_IDS=()
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        checkpoint_root="${ROOT_BY_DATASET[$dataset_name]}"
        job_label="disc-opr96"
        job_id=$(sbatch --parsable \
            --job-name="${job_label}-${dataset_name}" \
            --account="$ACCOUNT" --time="$WALL" --nodes=1 "${GPU_SBATCH_ARGS[@]}" \
            --cpus-per-task=8 --mem=50G \
            "${DEP_ARGS[@]}" \
            "${EXCLUDE_ARGS[@]}" \
            --output="$LOG_DIR/${job_label}-${dataset_name}-%j.log" \
            --error="$LOG_DIR/${job_label}-${dataset_name}-%j.log" \
            --mail-type=FAIL --mail-user="${USER_NAME}@uwo.ca" \
            --export=ALL,GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1,GRID_DATASET="$dataset_name",GRID_EXISTING_CKPT="$checkpoint_root",GRID_MMPD_ROOT="$MMPD_ROOT",GRID_DISC_OUTPUT="$DISC_OUTPUT",GRID_RAW_DISC_OUTPUT="$RAW_OUTPUT",GRID_ORDINAL_DISC_EVALUATOR="$ORDINAL_DISC_EVALUATOR",GRID_ORDINAL_BINARY_CONFIG="$ORDINAL_BINARY_CONFIG",GRID_SLICE_LENGTHS="$ORDINAL_SLICE_LENGTHS" \
            "$SCRIPT_DIR/slurm_worker.sh")
        manifest_tool record --path "$MANIFEST_PATH" --role "ordinal_disc" --dataset "$dataset_name" --job-id "$job_id" \
            --set "checkpoint_root=$checkpoint_root"
        DISC_JOB_IDS+=("$job_id")
    done

    disc_dep="afterok:${DISC_JOB_IDS[0]}"
    for job_id in "${DISC_JOB_IDS[@]:1}"; do disc_dep+=":$job_id"; done
    merge_id=$(sbatch --parsable \
        --job-name="disc-opr96-merge" --account="$ACCOUNT" --nodes=1 \
        --cpus-per-task=2 --mem=8G --time=0:30:00 --dependency="$disc_dep" \
        --output="$LOG_DIR/disc-opr96-merge-%j.log" --error="$LOG_DIR/disc-opr96-merge-%j.log" \
        --mail-type=FAIL --mail-user="${USER_NAME}@uwo.ca" \
        --export=ALL,GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1,GRID_ORDINAL_DISC_MERGE=1,GRID_DISC_OUTPUT="$DISC_OUTPUT",GRID_RAW_DISC_OUTPUT="$RAW_OUTPUT",GRID_ORDINAL_DISC_EVALUATOR="$ORDINAL_DISC_EVALUATOR",GRID_ORDINAL_BINARY_CONFIG="$ORDINAL_BINARY_CONFIG" \
        "$SCRIPT_DIR/slurm_worker.sh")
    manifest_tool record --path "$MANIFEST_PATH" --role ordinal_disc_merge --job-id "$merge_id"
    echo "ordinal patch-refine discriminator manifest: $MANIFEST_PATH"
    exit 0
fi

if [[ "$SMOKE" -eq 1 ]]; then
    # Legacy configs/smoke_test.yaml (residual fine) was removed; canvas128 leaf
    # or temp/scripts/smoke_patch_refine.py for a few-second geometry+step smoke.
    CONFIGS="${CONFIGS:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
    WALL_DEFAULT="0:30:00"
    MEM="24G"
    CPUS=4
    GPUS=1
    JOB_PREFIX="smoke"
else
    CONFIGS="${CONFIGS:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
    WALL_DEFAULT="3:00:00"
    MEM="60G"
    CPUS=8
    GPUS=1
    JOB_PREFIX="grid"
fi

if [[ "$GPU_TYPE" == h100* ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        MEM="48G"
        CPUS=8
    else
        MEM="80G"
        CPUS=16
    fi
fi

if [[ -n "${WANDB_API_KEY:-}" && "$WANDB_PROJECT_EXPLICIT" -eq 0 ]]; then
    WANDB_PROJECT="ts-sandbox-leaderboard"
    WANDB_PROJECT_EXPLICIT=1
fi

IFS=',' read -ra CONF_ARR <<< "$CONFIGS"
IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

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

if [[ -n "$JOB_MANIFEST" ]]; then
    [[ "$JOB_MANIFEST" == /* ]] || JOB_MANIFEST="$SCRIPT_DIR/$JOB_MANIFEST"
    manifest_tool init --path "$JOB_MANIFEST" --component binary_training \
        --repo "$SCRIPT_DIR" --datasets "$DATASETS" \
        --set "configs=$(IFS=,; echo "${CONF_ARR[*]}")" --set "store=$STORE"
fi

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

            gpu_sbatch_args "$GPUS" || exit 1

            S_ARGS=(
                --parsable
                --job-name="$JOB_NAME"
                --account="$ACCOUNT"
                --time="$WALL"
                --nodes=1
                "${GPU_SBATCH_ARGS[@]}"
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
            if [[ -n "${SBATCH_EXCLUDE_NODES:-}" ]]; then
                S_ARGS+=(--exclude="$SBATCH_EXCLUDE_NODES")
            fi

            PY_ARGS=(
                --config "$CFG"
                --dataset "$DS"
                --seed "$SD"
            )

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

            JOB_ID=$(sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/slurm_worker.sh" "${PY_ARGS[@]}")

            if [[ -z "$RUN_STEM" ]]; then
                RUN_STEM="${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}"
            fi
            ACTUAL_LOG="$LOG_DIR/${RUN_STEM}.log"
            if [[ "$LOG_FILE" == *'%j'* ]]; then
                ACTUAL_LOG="$LOG_DIR/${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}.log"
            fi
            printf "%-10s %-15s %-25s %-8s %s\n" "$JOB_ID" "$DS" "$CFG_NAME" "$SD" "$ACTUAL_LOG"
            if [[ -n "$JOB_MANIFEST" ]]; then
                manifest_tool record --path "$JOB_MANIFEST" --role binary_train --dataset "$DS" --job-id "$JOB_ID" \
                    --set "config=$CFG" --set "seed=$SD" \
                    --set "checkpoint_root=$CKPT_ROOT/$RUN_STEM" \
                    --set "results_root=$DATA_ROOT/$RUN_STEM"
            fi

        done
    done
done
echo "--------------------------------------------------------------------------------"
echo "Monitor with: squeue -u $USER"
