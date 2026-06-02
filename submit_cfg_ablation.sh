#!/bin/bash
# =============================================================================
# CFG scale ablation — eval-only on finished binary_dual_scale checkpoints.
#
# Uses utils/eval_mmpd_gaussian_anchor.py (same path as MMPD matrix binary eval):
#   - 50% seeded test windows (not full stride-1 test set)
#   - 1× anchor decode → deterministic MSE/MAE + texture
#   - 100× dpmpp stochastic draws → CRPS, top1/top3, prob_texture (first 3)
#   - No MMPD, no iTrans retrain, no pipeline viz
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   ./submit_cfg_ablation.sh --smoke-test
#   GPU=l40s ./submit_cfg_ablation.sh   # or GPU=h100
#   ./submit_cfg_ablation.sh \
#     --datasets ETTh1,ETTh2,exchange_rate,weather,traffic,PeMS,dalia \
#     --cfg-scales 4,10
#
# Cancel slow pipeline-based cfg jobs (if any):
#   scancel -u $USER -n cfg-
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_SCALES="${CFG_SCALES:-4,10}"
DATASETS="${DATASETS:-ETTh1,ETTh2,exchange_rate,weather,traffic,PeMS,dalia}"
GPU="${GPU:-l40s}"
EXCLUDE_NODES="${CFG_EXCLUDE_NODES:-}"
SEED=42
SMOKE=0
RUN_STEM=""
MERGE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --cfg-scales) CFG_SCALES="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --exclude-nodes) EXCLUDE_NODES="$2"; shift 2 ;;
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --merge-only) MERGE_ONLY=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$GPU" in
    l40s)
        GPU_SBATCH=(--gres=gpu:l40s:1)
        # kn120 exposed no CUDA devices; kn132 could not reliably see the shared venv.
        if [[ -z "$EXCLUDE_NODES" ]]; then
            EXCLUDE_NODES="kn120,kn132"
        fi
        ;;
    h100)
        GPU_SBATCH=(--partition=gpubase_h100_b4 --gpus-per-node=h100:1)
        ;;
    *)
        echo "ERROR: --gpu must be l40s or h100 (got $GPU)" >&2
        exit 1
        ;;
esac

IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SCALE_ARR <<< "$CFG_SCALES"

USER=$(whoami)
REPO="$SCRIPT_DIR"
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/${USER}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/${USER}/ts-sandbox"
elif [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
fi
if [[ "$REPO" == /home/* ]]; then
    echo "WARN: submitting from /home; prefer \$SCRATCH/ts-sandbox on Killarney" >&2
fi

STORE="${RESULTS_ROOT:-$REPO/results}"
CKPT_ROOT="$STORE/ckpts"
CFG_VENV="${CFG_VENV:-$STORE/venv}"
DATE_TAG="$(date +%m-%d)"
if [[ -z "$RUN_STEM" ]]; then
    RUN_STEM="${DATE_TAG}-cfg-ablation"
fi
LOG_DIR="$STORE/logs/cfg_ablation"
mkdir -p "$LOG_DIR"

if [[ ! -x "$CFG_VENV/bin/python" ]]; then
    echo "ERROR: required CFG venv missing or not executable: $CFG_VENV" >&2
    echo "Build/fix the shared venv first; this script no longer rebuilds venvs inside Slurm jobs." >&2
    exit 1
fi

pick_ckpt_dir() {
    local ds="$1"
    if [[ -d "$CKPT_ROOT/${ds}" && -f "$CKPT_ROOT/${ds}/metadata.json" ]]; then
        echo "$CKPT_ROOT/${ds}"
        return
    fi
    local best="" best_mtime=0 d m
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*-"${ds}"-binary_dual_scale; do
        [[ -d "$d" ]] || continue
        m=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [[ "$m" -gt "$best_mtime" ]]; then
            best_mtime="$m"
            best="$d"
        fi
    done
    shopt -u nullglob
    echo "$best"
}

merge_cfg_scale() {
    local out_dir="$1"
    local scale="$2"
    python3 - <<'PY' "$out_dir" "$scale"
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
scale = sys.argv[2]
partials = sorted((out_dir / "partials").glob("*.json"))
if not partials:
    print(f"[merge] no partials under {out_dir}/partials", flush=True)
    sys.exit(1)
results = {}
for path in partials:
    dataset = path.name.split("_", 1)[0]
    with path.open() as f:
        results[dataset] = {"binary_anchor": json.load(f)}
manifest = {
    "cfg_scale": float(scale),
    "partials": [str(p) for p in partials],
}
(out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
(out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
rows = ["dataset,cfg_scale,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,det_n_samples,prob_n_samples"]
for ds in sorted(results):
    m = results[ds]["binary_anchor"]
    rows.append(
        f"{ds},{scale},"
        f"{m.get('mse', '')},{m.get('mae', '')},{m.get('crps', '')},"
        f"{m.get('top1_mse', '')},{m.get('top1_mae', '')},"
        f"{m.get('top3_mse', '')},{m.get('top3_mae', '')},"
        f"{m.get('det_n_samples', 1)},{m.get('prob_n_samples', m.get('n_samples', ''))}"
    )
(out_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
print(f"[merge] wrote {out_dir}/metrics.csv ({len(results)} datasets)", flush=True)
PY
}

if [[ "$MERGE_ONLY" -eq 1 ]]; then
    for SCALE in "${SCALE_ARR[@]}"; do
        OUT="$STORE/datasets/${RUN_STEM}-cfg${SCALE}"
        merge_cfg_scale "$OUT" "$SCALE"
    done
    exit 0
fi

if [[ "$SMOKE" -eq 1 ]]; then
    DATA_ARR=(ETTh1)
    SCALE_ARR=(4)
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
    WALL="3:00:00"
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

echo "CFG ablation (eval_mmpd anchor path)"
echo "  run_stem=$RUN_STEM  gpu=$GPU  scales=${SCALE_ARR[*]}  storage=$STORE"
echo "  venv=$CFG_VENV"
if [[ -n "$EXCLUDE_NODES" ]]; then
    echo "  exclude_nodes=$EXCLUDE_NODES"
fi
printf "%-10s %-12s %-8s %-6s %s\n" "JOB" "DATASET" "CFG" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

JOB_IDS=()
for SCALE in "${SCALE_ARR[@]}"; do
    OUT_DIR="$STORE/datasets/${RUN_STEM}-cfg${SCALE}"
    mkdir -p "$OUT_DIR/partials"
    USE_CFG_FLAG=()
    if python3 -c "import sys; s=float('$SCALE'); sys.exit(0 if abs(s-1.0)>1e-6 else 1)"; then
        USE_CFG_FLAG=(--use-cfg-inference)
    else
        USE_CFG_FLAG=(--no-use-cfg-inference)
    fi

    for DS in "${DATA_ARR[@]}"; do
        CKPT_DIR=$(pick_ckpt_dir "$DS")
        if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
            echo "ERROR: no binary_dual_scale ckpt for $DS under $CKPT_ROOT" >&2
            exit 1
        fi
        LOG_FILE="$LOG_DIR/${RUN_STEM}-cfg${SCALE}-${DS}.log"
        JOB_NAME="cfg-${DS}-w${SCALE}"

        WORKER="$REPO/slurm_cfg_ablation_worker.sh"
        if [[ ! -x "$WORKER" ]]; then
            chmod +x "$WORKER"
        fi

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
            --cfg-scale "$SCALE" \
            "${USE_CFG_FLAG[@]}" \
            --metrics-profile full \
            --force-anchor-eval \
            --skip-mmpd-train \
            "${EVAL_EXTRA[@]}")
        printf "%-10s %-12s %-8s %-6s %s\n" "$JOB_ID" "$DS" "$SCALE" "$SEED" "$LOG_FILE"
        JOB_IDS+=("$JOB_ID")
    done
done

MERGE_DEP=""
if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
    MERGE_DEP="afterok:${JOB_IDS[0]}"
    for jid in "${JOB_IDS[@]:1}"; do
        MERGE_DEP="${MERGE_DEP}:${jid}"
    done
fi

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
REPO="$REPO"
STORE="$STORE"
RUN_STEM="$RUN_STEM"
CFG_SCALES="$CFG_SCALES"
cd "\$REPO"
IFS=',' read -ra SCALES <<< "\$CFG_SCALES"
for SCALE in "\${SCALES[@]}"; do
  OUT="\$STORE/datasets/\${RUN_STEM}-cfg\${SCALE}"
  python3 - <<PY "\$OUT" "\$SCALE"
import json, sys
from pathlib import Path
out_dir, scale = Path(sys.argv[1]), sys.argv[2]
partials = sorted((out_dir / "partials").glob("*.json"))
results = {}
for path in partials:
    dataset = path.name.split("_", 1)[0]
    with path.open() as f:
        results[dataset] = {"binary_anchor": json.load(f)}
(out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
rows = ["dataset,cfg_scale,mse,mae,crps,top1_mse,top1_mae,top3_mse,top3_mae,det_n_samples,prob_n_samples"]
for ds in sorted(results):
    m = results[ds]["binary_anchor"]
    rows.append(f"{ds},{scale},{m.get('mse','')},{m.get('mae','')},{m.get('crps','')},{m.get('top1_mse','')},{m.get('top1_mae','')},{m.get('top3_mse','')},{m.get('top3_mae','')},{m.get('det_n_samples',1)},{m.get('prob_n_samples', m.get('n_samples',''))}")
(out_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
print(f"[merge] {out_dir}/metrics.csv ({len(results)} datasets)", flush=True)
PY
done
echo "[merge] done: \$(date)"
ENDSCRIPT
)

echo "--------------------------------------------------------------------------------"
echo "Merge job: $MERGE_ID (logs: $MERGE_LOG)"
echo "Monitor: squeue -u $USER | grep '^cfg-'"
echo "Results: $STORE/datasets/${RUN_STEM}-cfg{scale}/metrics.csv"
