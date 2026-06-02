#!/bin/bash
# =============================================================================
# Probabilistic sampler ablation (eval-only) on finished binary_dual_scale ckpts.
#
# Same metrics path as CFG ablation (eval_mmpd_gaussian_anchor.py):
#   50% test windows, 1× anchor det, 100× stochastic → CRPS/top1/top3/texture
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   CKPT_SUFFIX=binary_dual_scale_patch48 ./submit_sampler_ablation.sh
#   CKPT_SUFFIX=binary_dual_scale_patch48 SAMPLERS=dpmpp,ddim ./submit_sampler_ablation.sh
#   ./submit_sampler_ablation.sh --merge-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLERS="${SAMPLERS:-dpmpp,ddim}"
DATASETS="${DATASETS:-ETTm1,ETTm2,dalia,electricity,exchange_rate,solar_Alabama,traffic,weather}"
CKPT_SUFFIX="${CKPT_SUFFIX:-binary_dual_scale_patch48}"
# Optional: only match stems containing this substring (e.g. 06-02-384445 for patch48 grid).
CKPT_STEM_PREFIX="${CKPT_STEM_PREFIX:-}"
GPU="${GPU:-l40s}"
EXCLUDE_NODES="${SAMPLER_EXCLUDE_NODES:-kn120,kn132}"
SEED=42
SMOKE=0
RUN_STEM=""
MERGE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --samplers) SAMPLERS="$2"; shift 2 ;;
        --ckpt-suffix) CKPT_SUFFIX="$2"; shift 2 ;;
        --ckpt-stem-prefix) CKPT_STEM_PREFIX="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --exclude-nodes) EXCLUDE_NODES="$2"; shift 2 ;;
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --merge-only) MERGE_ONLY=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$GPU" in
    l40s) GPU_SBATCH=(--gres=gpu:l40s:1) ;;
    h100) GPU_SBATCH=(--partition=gpubase_h100_b4 --gpus-per-node=h100:1) ;;
    *) echo "ERROR: --gpu must be l40s or h100" >&2; exit 1 ;;
esac

USER=$(whoami)
REPO="$SCRIPT_DIR"
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/${USER}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/${USER}/ts-sandbox"
elif [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
fi

STORE="${RESULTS_ROOT:-$REPO/results}"
CKPT_ROOT="$STORE/ckpts"
CFG_VENV="${CFG_VENV:-$STORE/venv}"
DATE_TAG="$(date +%m-%d)"
RUN_STEM="${RUN_STEM:-${DATE_TAG}-patch48-sampler}"
LOG_DIR="$STORE/logs/sampler_ablation"
mkdir -p "$LOG_DIR"

IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SAMPLER_ARR <<< "$SAMPLERS"

if [[ ! -x "$CFG_VENV/bin/python" ]]; then
    echo "ERROR: venv missing: $CFG_VENV" >&2
    exit 1
fi

pick_ckpt_dir() {
    local ds="$1"
    local best="" best_mtime=0 d m
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*-"${ds}"-"${CKPT_SUFFIX}"; do
        [[ -d "$d" ]] || continue
        if [[ -n "$CKPT_STEM_PREFIX" && "$(basename "$d")" != *"${CKPT_STEM_PREFIX}"* ]]; then
            continue
        fi
        if ! compgen -G "${d}"/*/best.pt >/dev/null; then
            continue
        fi
        m=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [[ "$m" -gt "$best_mtime" ]]; then
            best_mtime="$m"
            best="$d"
        fi
    done
    shopt -u nullglob
    echo "$best"
}

merge_sampler() {
    local out_dir="$1" sampler="$2"
    python3 - <<'PY' "$out_dir" "$sampler"
import json, sys
from pathlib import Path
out_dir, sampler = Path(sys.argv[1]), sys.argv[2]
partials = sorted((out_dir / "partials").glob("*.json"))
if not partials:
    print(f"[merge] no partials in {out_dir}/partials", flush=True)
    sys.exit(1)
results = {}
for path in partials:
    dataset = path.name.split("_", 1)[0]
    with path.open() as f:
        results[dataset] = {"binary_anchor": json.load(f)}
(out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
rows = [
    "dataset,sampler,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,"
    "det_texture_mse,prob_texture_mse,det_n_samples,prob_n_samples"
]
for ds in sorted(results):
    m = results[ds]["binary_anchor"]
    rows.append(
        f"{ds},{sampler},"
        f"{m.get('mse','')},{m.get('mae','')},{m.get('crps','')},"
        f"{m.get('top1_mse','')},{m.get('top1_mae','')},"
        f"{m.get('top3_mse','')},{m.get('top3_mae','')},"
        f"{m.get('texture_mse','')},{m.get('prob_texture_mse','')},"
        f"{m.get('det_n_samples',1)},{m.get('prob_n_samples', m.get('n_samples',''))}"
    )
(out_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
print(f"[merge] {out_dir}/metrics.csv ({len(results)} datasets, sampler={sampler})", flush=True)
PY
}

if [[ "$MERGE_ONLY" -eq 1 ]]; then
    for SAMPLER in "${SAMPLER_ARR[@]}"; do
        merge_sampler "$STORE/datasets/${RUN_STEM}-${SAMPLER}" "$SAMPLER"
    done
    exit 0
fi

if [[ "$SMOKE" -eq 1 ]]; then
    DATA_ARR=(ETTm1)
    SAMPLER_ARR=(dpmpp)
    WALL="0:45:00"
    MEM="24G"
    CPUS=4
    EVAL_EXTRA=(
        --test-fraction 0.05
        --test-max-items 16
        --sample-num 5
        --num-sampling-steps 5
        --anchor-batch-size 4
        --gmm-components 5
        --gmm-iterations 3
    )
else
    # ETTm-scale 50%×100 prob draws need ~4h+; 3h wall caused timeouts on patch48 redo.
    WALL="${EVAL_WALL:-6:00:00}"
    MEM="60G"
    CPUS=8
    EVAL_EXTRA=(
        --test-fraction 0.5
        --sample-num 100
        --num-sampling-steps 20
        --anchor-batch-size 16
        --gmm-components 10
        --gmm-iterations 10
        --texture-per-sample
        --topk-max 3
    )
fi

WORKER="$REPO/slurm_cfg_ablation_worker.sh"
[[ -x "$WORKER" ]] || chmod +x "$WORKER"

echo "Sampler ablation (eval_mmpd_gaussian_anchor — CFG-ablation metrics)"
echo "  run_stem=$RUN_STEM  ckpt_suffix=$CKPT_SUFFIX  samplers=${SAMPLER_ARR[*]}"
[[ -n "$CKPT_STEM_PREFIX" ]] && echo "  ckpt_stem_prefix=$CKPT_STEM_PREFIX"
printf "%-10s %-12s %-8s %-6s %s\n" "JOB" "DATASET" "SAMPLER" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

JOB_IDS=()
for SAMPLER in "${SAMPLER_ARR[@]}"; do
    OUT_DIR="$STORE/datasets/${RUN_STEM}-${SAMPLER}"
    mkdir -p "$OUT_DIR/partials"

    for DS in "${DATA_ARR[@]}"; do
        CKPT_DIR=$(pick_ckpt_dir "$DS")
        if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
            echo "ERROR: no ckpt for $DS matching *-${CKPT_SUFFIX} under $CKPT_ROOT" >&2
            exit 1
        fi
        LOG_FILE="$LOG_DIR/${RUN_STEM}-${SAMPLER}-${DS}.log"
        JOB_NAME="smp-${DS}-${SAMPLER}"

        JOB_ID=$(sbatch --parsable \
            --job-name="$JOB_NAME" \
            --account=aip-boyuwang \
            --time="$WALL" \
            --nodes=1 \
            "${GPU_SBATCH[@]}" \
            ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --chdir="$REPO" \
            --output="$LOG_FILE" \
            --error="$LOG_FILE" \
            --mail-type=FAIL \
            --mail-user="${USER}@uwo.ca" \
            --export=ALL,CFG_STORE="$STORE",CFG_VENV="$CFG_VENV" \
            "$WORKER" \
            --phase anchor \
            --anchor-variant binary \
            --datasets "$DS" \
            --binary-anchor-root "$CKPT_DIR" \
            --ckpt-base "$CKPT_DIR" \
            --output-dir "$OUT_DIR" \
            --seed "$SEED" \
            --no-use-cfg-inference \
            --metrics-profile full \
            --force-anchor-eval \
            --skip-mmpd-train \
            --anchor-prob-sampler "$SAMPLER" \
            "${EVAL_EXTRA[@]}")
        printf "%-10s %-12s %-8s %-6s %s\n" "$JOB_ID" "$DS" "$SAMPLER" "$SEED" "$LOG_FILE"
        JOB_IDS+=("$JOB_ID")
    done
done

MERGE_DEP="afterok:${JOB_IDS[0]}"
for jid in "${JOB_IDS[@]:1}"; do
    MERGE_DEP="${MERGE_DEP}:${jid}"
done

MERGE_LOG="$LOG_DIR/${RUN_STEM}-merge.log"
MERGE_ID=$(sbatch --parsable \
    --job-name="${RUN_STEM}-merge" \
    --account=aip-boyuwang \
    --time="0:15:00" \
    --cpus-per-task=2 \
    --mem="4G" \
    --chdir="$REPO" \
    --output="$MERGE_LOG" \
    --error="$MERGE_LOG" \
    ${MERGE_DEP:+--dependency="$MERGE_DEP"} \
    <<ENDSCRIPT
#!/bin/bash
set -euo pipefail
cd "$REPO"
STORE="$STORE"
RUN_STEM="$RUN_STEM"
SAMPLERS="$SAMPLERS"
IFS=',' read -ra SMP <<< "\$SAMPLERS"
for s in "\${SMP[@]}"; do
  OUT="\$STORE/datasets/\${RUN_STEM}-\${s}"
  python3 - <<PY "\$OUT" "\$s"
import json, sys
from pathlib import Path
out_dir, sampler = Path(sys.argv[1]), sys.argv[2]
partials = sorted((out_dir / "partials").glob("*.json"))
results = {}
for path in partials:
    ds = path.name.split("_", 1)[0]
    with path.open() as f:
        results[ds] = {"binary_anchor": json.load(f)}
(out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
rows = ["dataset,sampler,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,det_texture_mse,prob_texture_mse,det_n_samples,prob_n_samples"]
for ds in sorted(results):
    m = results[ds]["binary_anchor"]
    rows.append(f"{ds},{sampler},{m.get('mse','')},{m.get('mae','')},{m.get('crps','')},{m.get('top1_mse','')},{m.get('top1_mae','')},{m.get('top3_mse','')},{m.get('top3_mae','')},{m.get('texture_mse','')},{m.get('prob_texture_mse','')},{m.get('det_n_samples',1)},{m.get('prob_n_samples', m.get('n_samples',''))}")
(out_dir / "metrics.csv").write_text("\\n".join(rows) + "\\n")
print(f"[merge] {out_dir}/metrics.csv ({len(results)} datasets)", flush=True)
PY
done
ENDSCRIPT
)

echo "--------------------------------------------------------------------------------"
echo "Merge: $MERGE_ID  logs: $MERGE_LOG"
echo "Results: $STORE/datasets/${RUN_STEM}-{dpmpp,ddim}/metrics.csv"
