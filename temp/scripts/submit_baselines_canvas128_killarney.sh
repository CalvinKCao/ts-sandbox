#!/bin/bash
# =============================================================================
# Train iTransformer + PatchTST on canvas128 leaderboard subsets.
#
# One sbatch per dataset. Node-local venv on $SLURM_TMPDIR. Patches + PeMS
# 60/20/20 + published per-dataset script HPs in
# temp/scripts/{apply,run}_baselines_canvas128_*.py.
#
# USAGE (from repo root on login node, e.g. $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --smoke-test
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --model itransformer --datasets ETTh1
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --gpu a100 \
#       --subset-yaml configs/compare_baselines_allv_seed42_frac10.yaml \
#       --eval-subsets binary_10pct,test_100pct \
#       --seq-len 336 --pred-len 720 --datasets ETTh1 --time 12:00:00 --force
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --model itransformer \
#       --lr 1e-4 --d-model 512 --e-layers 2 --batch-size 32 --dropout 0.1 \
#       --train-epochs 10 --datasets ETTh1 --force
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --model patchtst \
#       --datasets traffic --seq-len 512 --batch-size 8 --force
# =============================================================================

set -euo pipefail

MODEL="both"
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,electricity,traffic,exchange_rate,weather,solar_Alabama,PeMS,illness,dynamic"
SMOKE=0
TIME_FULL="2:00:00"
TIME_SMOKE="0:45:00"
WALL_OVERRIDE=""
FORCE=0
SEQ_LEN=336
PRED_LEN=96
SUBSET_YAML="configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml"
GPU_TYPE=""
EVAL_SUBSETS=""
ITRANS_LR=""
ITRANS_D_MODEL=""
ITRANS_D_FF=""
ITRANS_E_LAYERS=""
ITRANS_N_HEADS=""
BATCH_SIZE=""
ITRANS_DROPOUT=""
ITRANS_EPOCHS=""
PRETRAIN_SYNTH="none"

# Conservative walls by dataset size (iTrans/PatchTST faster than diffusion).
# --time override wins for every dataset.
ds_wall() {
  local ds="$1" default="$2"
  if [ -n "$WALL_OVERRIDE" ] || [ "$SMOKE" -eq 1 ]; then
    echo "$default"
    return
  fi
  # Walls from measured iTrans/PatchTST elapsed + slight leeway.
  case "$ds" in
    illness|ETTh1|ETTh2|ETTm1|ETTm2|exchange_rate|weather) echo "1:00:00" ;;
    PeMS) echo "3:00:00" ;;
    solar_Alabama) echo "4:00:00" ;;
    traffic) echo "5:00:00" ;;
    dynamic) echo "6:00:00" ;;
    electricity) echo "0-08:00:00" ;;
    *) echo "$default" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --smoke-test|--smoke) SMOKE=1; shift ;;
    --time) WALL_OVERRIDE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --eval-subsets)
      EVAL_SUBSETS="$2"
      shift 2
      ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --pred-len) PRED_LEN="$2"; shift 2 ;;
    --subset-yaml) SUBSET_YAML="$2"; shift 2 ;;
    --gpu) GPU_TYPE="$2"; shift 2 ;;
    --lr) ITRANS_LR="$2"; shift 2 ;;
    --d-model) ITRANS_D_MODEL="$2"; shift 2 ;;
    --d-ff) ITRANS_D_FF="$2"; shift 2 ;;
    --e-layers) ITRANS_E_LAYERS="$2"; shift 2 ;;
    --n-heads) ITRANS_N_HEADS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dropout) ITRANS_DROPOUT="$2"; shift 2 ;;
    --train-epochs) ITRANS_EPOCHS="$2"; shift 2 ;;
    --pretrain-synth) PRETRAIN_SYNTH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

canon_dataset() {
  case "$1" in
    exchange) echo exchange_rate ;;
    *) echo "$1" ;;
  esac
}
_ds_new=""
IFS=',' read -r -a _ds_arr <<< "$DATASETS"
for _d in "${_ds_arr[@]}"; do
  _d="$(echo "$_d" | xargs)"
  [ -n "$_d" ] || continue
  _d="$(canon_dataset "$_d")"
  _ds_new="${_ds_new:+$_ds_new,}$_d"
done
DATASETS="$_ds_new"

if [[ "$PRETRAIN_SYNTH" == "pwb_linear" ]]; then
  if [[ "$SUBSET_YAML" == "configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_randwin_lr10.yaml" ]]; then
    SUBSET_YAML="configs/compare_baselines_allv_fullT.yaml"
  fi
fi

ITRANS_HP_SET="${ITRANS_LR}${ITRANS_D_MODEL}${ITRANS_D_FF}${ITRANS_E_LAYERS}${ITRANS_N_HEADS}${ITRANS_DROPOUT}${ITRANS_EPOCHS}"
if [[ -n "$ITRANS_HP_SET" && "$MODEL" == "patchtst" ]]; then
  echo "ERROR: iTransformer HP flags require --model itransformer or both (--batch-size is ok)" >&2
  exit 1
fi

if [[ "$(hostname)" == *"narval"* ]]; then
  ACCOUNT="def-boyuwang"
  [[ -z "$GPU_TYPE" ]] && GPU_TYPE="a100"
elif [[ "$(hostname)" == *"killarney"* || "$(hostname)" == kl* ]]; then
  ACCOUNT="aip-boyuwang"
  [[ -z "$GPU_TYPE" ]] && GPU_TYPE="l40s"
else
  ACCOUNT="aip-boyuwang"
  [[ -z "$GPU_TYPE" ]] && GPU_TYPE="l40s"
fi

if [[ "$GPU_TYPE" == a100* || "$GPU_TYPE" == h100* ]]; then
  GPU_SBATCH=(--gpus="${GPU_TYPE}:1")
else
  GPU_SBATCH=(--gres="gpu:${GPU_TYPE}:1")
fi

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

# Shortest gpubase_l40s_b* whose MaxTime covers the job wall (Killarney).
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
    echo "ERROR: no gpubase_l40s_b* partition allows wall ${need_s}s" >&2
    return 1
  fi
  echo "$best"
}

# ---------------------------------------------------------------------------
# Login-node: submit one job per dataset
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  REPO_ROOT="$(pwd)"
  if [ ! -f "$REPO_ROOT/temp/scripts/run_baselines_canvas128_subset.py" ]; then
    echo "ERROR: submit from repo root (missing temp/scripts/run_baselines_canvas128_subset.py)" >&2
    exit 1
  fi

  if [ -n "$WALL_OVERRIDE" ]; then
    WALL="$WALL_OVERRIDE"
  elif [ "$SMOKE" -eq 1 ]; then
    WALL="$TIME_SMOKE"
  else
    WALL="$TIME_FULL"
  fi

  mkdir -p "$REPO_ROOT/temp" "$REPO_ROOT/results/baselines_canvas128_subset/logs"
  if [ ! -d "$REPO_ROOT/temp/iTransformer/.git" ]; then
    echo "Cloning iTransformer on login node (compute nodes cannot reach GitHub)..."
    git clone --depth 1 https://github.com/thuml/iTransformer.git "$REPO_ROOT/temp/iTransformer"
  fi
  if [ ! -d "$REPO_ROOT/temp/PatchTST/.git" ]; then
    echo "Cloning PatchTST on login node (compute nodes cannot reach GitHub)..."
    git clone --depth 1 https://github.com/yuqinie98/PatchTST.git "$REPO_ROOT/temp/PatchTST"
  fi

  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  for ds in "${DS_ARR[@]}"; do
    ds="$(echo "$ds" | xargs)"
    [ -n "$ds" ] || continue
    JOB_NAME="base-c128-h${PRED_LEN}-${MODEL}-${ds}"
    [ "$MODEL" = "both" ] && JOB_NAME="base-c128-h${PRED_LEN}-${ds}"
    [ "$SMOKE" -eq 1 ] && JOB_NAME="base-c128-smoke-h${PRED_LEN}-${ds}"
    EXTRA=(
      --model "$MODEL"
      --datasets "$ds"
      --seq-len "$SEQ_LEN"
      --pred-len "$PRED_LEN"
      --subset-yaml "$SUBSET_YAML"
      --gpu "$GPU_TYPE"
    )
    [ -n "$EVAL_SUBSETS" ] && EXTRA+=(--eval-subsets "$EVAL_SUBSETS")
    [ "$SMOKE" -eq 1 ] && EXTRA+=(--smoke-test)
    [ "$FORCE" -eq 1 ] && EXTRA+=(--force)
    [ -n "$ITRANS_LR" ] && EXTRA+=(--lr "$ITRANS_LR")
    [ -n "$ITRANS_D_MODEL" ] && EXTRA+=(--d-model "$ITRANS_D_MODEL")
    [ -n "$ITRANS_D_FF" ] && EXTRA+=(--d-ff "$ITRANS_D_FF")
    [ -n "$ITRANS_E_LAYERS" ] && EXTRA+=(--e-layers "$ITRANS_E_LAYERS")
    [ -n "$ITRANS_N_HEADS" ] && EXTRA+=(--n-heads "$ITRANS_N_HEADS")
    [ -n "$BATCH_SIZE" ] && EXTRA+=(--batch-size "$BATCH_SIZE")
    [ -n "$ITRANS_DROPOUT" ] && EXTRA+=(--dropout "$ITRANS_DROPOUT")
    [ -n "$ITRANS_EPOCHS" ] && EXTRA+=(--train-epochs "$ITRANS_EPOCHS")
    [ -n "$PRETRAIN_SYNTH" ] && EXTRA+=(--pretrain-synth "$PRETRAIN_SYNTH")
    DS_WALL="$(ds_wall "$ds" "$WALL")"
    PART_ARGS=()
    if [[ "$GPU_TYPE" == l40s* ]] && { [[ "$(hostname)" == *"killarney"* ]] || [[ "$(hostname)" == kl* ]]; }; then
      PART_ARGS=(--partition="$(pick_l40s_partition "$(slurm_time_to_seconds "$DS_WALL")")")
    fi
    echo "[submit] $ds model=$MODEL wall=$DS_WALL gpu=$GPU_TYPE ${PART_ARGS[*]+"${PART_ARGS[*]} "}pred_len=$PRED_LEN seq_len=$SEQ_LEN subset=$SUBSET_YAML"
    sbatch \
      --job-name="$JOB_NAME" \
      --account="$ACCOUNT" \
      --time="$DS_WALL" \
      --nodes=1 \
      --cpus-per-task=8 \
      --mem=50G \
      "${PART_ARGS[@]}" \
      "${GPU_SBATCH[@]}" \
      --output="$REPO_ROOT/results/baselines_canvas128_subset/logs/%x-%j.out" \
      --mail-type=END,FAIL \
      --mail-user=ccao87@uwo.ca \
      "$SCRIPT_PATH" \
        "${EXTRA[@]}"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Compute-node body
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "seq_len=$SEQ_LEN pred_len=$PRED_LEN subset_yaml=$SUBSET_YAML"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:?}"
echo "PROJECT_ROOT=$PWD"

if ! type module >/dev/null 2>&1; then
  if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
    export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
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
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing" >&2; exit 1; }

REQ="$PWD/setup/requirements-killarney.txt"
[ -f "$REQ" ] || { echo "ERROR: missing $REQ" >&2; exit 1; }

echo "[setup] node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
# Optional; iTransformer import is stubbed if missing.
pip install --no-index reformer-pytorch 2>/dev/null \
  || pip install reformer-pytorch 2>/dev/null \
  || echo "[warn] reformer-pytorch not installed; using import stub"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1

mkdir -p temp results/baselines_canvas128_subset
CLONE_LOCK="$PWD/results/baselines_canvas128_subset/clone.lock"
(
  flock 9
  if [ ! -d temp/iTransformer/.git ]; then
    rm -rf temp/iTransformer
    git clone --depth 1 https://github.com/thuml/iTransformer.git temp/iTransformer
  fi
  if [ ! -d temp/PatchTST/.git ]; then
    rm -rf temp/PatchTST
    git clone --depth 1 https://github.com/yuqinie98/PatchTST.git temp/PatchTST
  fi
  find temp/iTransformer temp/PatchTST -name '*.py' -print0 | xargs -0 sed -i 's/np\.Inf/np.inf/g' || true
  python -u temp/scripts/apply_baseline_canvas128_patches.py
  python -u temp/scripts/export_canvas128_subset_csvs.py \
    --subset-yaml "$SUBSET_YAML" \
    --datasets "$DATASETS"
) 9>"$CLONE_LOCK"

EXTRA_FLAGS=()
[ "$SMOKE" -eq 1 ] && EXTRA_FLAGS+=(--smoke-test)
[ "${FORCE:-0}" -eq 1 ] && EXTRA_FLAGS+=(--force)
[ -n "$ITRANS_LR" ] && EXTRA_FLAGS+=(--lr "$ITRANS_LR")
[ -n "$ITRANS_D_MODEL" ] && EXTRA_FLAGS+=(--d-model "$ITRANS_D_MODEL")
[ -n "$ITRANS_D_FF" ] && EXTRA_FLAGS+=(--d-ff "$ITRANS_D_FF")
[ -n "$ITRANS_E_LAYERS" ] && EXTRA_FLAGS+=(--e-layers "$ITRANS_E_LAYERS")
[ -n "$ITRANS_N_HEADS" ] && EXTRA_FLAGS+=(--n-heads "$ITRANS_N_HEADS")
[ -n "$BATCH_SIZE" ] && EXTRA_FLAGS+=(--batch-size "$BATCH_SIZE")
[ -n "$ITRANS_DROPOUT" ] && EXTRA_FLAGS+=(--dropout "$ITRANS_DROPOUT")
[ -n "$ITRANS_EPOCHS" ] && EXTRA_FLAGS+=(--train-epochs "$ITRANS_EPOCHS")
[ -n "$PRETRAIN_SYNTH" ] && EXTRA_FLAGS+=(--pretrain-synth "$PRETRAIN_SYNTH")
[ -n "$SUBSET_YAML" ] && EXTRA_FLAGS+=(--subset-yaml "$SUBSET_YAML")

python -u temp/scripts/run_baselines_canvas128_subset.py \
  --model "$MODEL" \
  --dataset "$DATASETS" \
  --seq-len "$SEQ_LEN" \
  --pred-len "$PRED_LEN" \
  --force \
  "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
  ${EVAL_SUBSETS:+--eval-subsets "$EVAL_SUBSETS"}

echo "Finished: $(date)"
