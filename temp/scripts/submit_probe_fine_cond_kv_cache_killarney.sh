#!/bin/bash
# Diagnostic: fine unique-seg eval generate, cond K/V cache vs 228-token baseline.
# Not a training wrapper. From repo root on the Killarney login node:
#   ./temp/scripts/submit_probe_fine_cond_kv_cache_killarney.sh

set -euo pipefail

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
        best="gpubase_l40s_b1"
    fi
    echo "$best"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
    mkdir -p "$SCRIPT_DIR/results/fine-cond-kv-cache/logs" "$SCRIPT_DIR/results/logs"
    WALL="${WALL:-0:50:00}"
    part="$(pick_l40s_partition "$(slurm_time_to_seconds "$WALL")")"
    echo "Submitting fine cond-KV cache probe (L40S $part $WALL)"
    sbatch \
        --job-name=probe-cond-kv \
        --account=aip-boyuwang \
        --partition="$part" \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --output="$SCRIPT_DIR/results/logs/probe-cond-kv-%j.log" \
        --error="$SCRIPT_DIR/results/logs/probe-cond-kv-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/temp/scripts/submit_probe_fine_cond_kv_cache_killarney.sh"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "$REPO_ROOT"
[[ -f models/diffusion_tsf/dit.py ]] || { echo "ERROR: not repo root: $REPO_ROOT" >&2; exit 1; }
REQ="$REPO_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }
[[ -f temp/scripts/probe_fine_cond_kv_cache.py ]] || {
    echo "ERROR: missing temp/scripts/probe_fine_cond_kv_cache.py" >&2
    exit 1
}

STEM="$(date +%m-%d)-${SLURM_JOB_ID}-fine-cond-kv-cache"
OUT_DIR="$REPO_ROOT/results/fine-cond-kv-cache/logs/${STEM}"
mkdir -p "$OUT_DIR" "$REPO_ROOT/results/logs"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/inductor"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "[setup] Building node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "NOTE: cache_cond_kv is not numerically identical to bidirectional 228-token attn."
echo "Running temp/scripts/probe_fine_cond_kv_cache.py"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u temp/scripts/probe_fine_cond_kv_cache.py \
    --n-variates 4 \
    --n-windows 1 \
    --n-samples 10 \
    --n-warm 1 \
    --n-meas 3 \
    --chunk 0 \
    --sampler quad_t \
    --steps 4 \
    --out-json "$OUT_DIR/fine_cond_kv_cache.json"

echo "Wrote $OUT_DIR"
echo "Finished: $(date)"
exit 0
