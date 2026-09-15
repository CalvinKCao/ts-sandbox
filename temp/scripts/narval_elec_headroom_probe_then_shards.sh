#!/bin/bash
# Narval login node: submit electricity reserved-headroom probe, poll until
# it finishes, then submit 10 eval shards at the probed fine chunk.
# Run from $SCRATCH/ts-sandbox after rsync. Do not cancel solar/PeMS/dynamic.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f models/diffusion_tsf/dit.py ]] || { echo "ERROR: run from repo root" >&2; exit 1; }

CFG_PROBE="configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100.yaml"
CFG_SHARD="configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv_a100_fulltest_elec_headroom.yaml"
CKPT=""
for d in results/ckpts/*-2650276-electricity-*; do
    [[ -d "$d" ]] && CKPT="$d" && break
done
[[ -n "$CKPT" ]] || { echo "ERROR: no 2650276 electricity ckpt under results/ckpts" >&2; exit 1; }
[[ -f "$CFG_PROBE" ]] || { echo "ERROR: missing $CFG_PROBE" >&2; exit 1; }
[[ -f "$CFG_SHARD" ]] || { echo "ERROR: missing $CFG_SHARD" >&2; exit 1; }

echo "ckpt=$CKPT"
echo "Submitting electricity reserved-headroom probe..."
PROBE_OUT="$(./temp/scripts/submit_probe_eval_chunk_headroom.sh \
    --dataset electricity \
    --n-variates 321 \
    --config "$CFG_PROBE" \
    --ckpt-dir "$CKPT" \
    --lo 512 \
    --start 2048 \
    --hi 25000 \
    --max-reserved-gib 34.5 \
    --headroom-gib 5 \
    --min-ok-chunk 1024 \
    --time 1:30:00)"
echo "$PROBE_OUT"
PROBE_JID="$(echo "$PROBE_OUT" | grep -Eo 'Submitted batch job [0-9]+' | awk '{print $4}' | tail -1)"
[[ -n "$PROBE_JID" ]] || { echo "ERROR: could not parse probe job id" >&2; exit 1; }
LOG="results/logs/probe-electricity-headroom-n10-${PROBE_JID}.log"
echo "probe_jid=$PROBE_JID log=$LOG"

echo "Polling probe $PROBE_JID ..."
while true; do
    st="$(sacct -j "$PROBE_JID" -X --format=State -P -n 2>/dev/null | head -1 | tr -d ' ')"
    case "$st" in
        COMPLETED)
            echo "probe COMPLETED"
            break
            ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
            echo "ERROR: probe $PROBE_JID ended $st"
            [[ -f "$LOG" ]] && tail -40 "$LOG"
            exit 1
            ;;
        *)
            echo "$(date +%H:%M:%S) probe $PROBE_JID $st"
            sleep 45
            ;;
    esac
done

[[ -f "$LOG" ]] || { echo "ERROR: missing probe log $LOG" >&2; exit 1; }
if grep -q STOP_NO_HEADROOM "$LOG"; then
    echo "ERROR: probe found no chunk >=1024 with reserved headroom; not submitting shards"
    grep -E "TRIAL|FINE_MAX|STOP_|SUMMARY|reserved=" "$LOG" | tail -40
    exit 1
fi
FINE="$(grep -E '^FINE_MAX_CHUNK ' "$LOG" | tail -1 | awk '{print $2}')"
WIN="$(grep -E '^WINDOW_SECONDS ' "$LOG" | tail -1 | awk '{print $2}')"
RES="$(grep -E '^FINE_RESERVED_GIB ' "$LOG" | tail -1 | awk '{print $2}')"
WIN_RES="$(grep -E '^WINDOW_RESERVED_GIB ' "$LOG" | tail -1 | awk '{print $2}')"
[[ -n "$FINE" ]] || { echo "ERROR: no FINE_MAX_CHUNK in $LOG" >&2; tail -50 "$LOG"; exit 1; }
[[ -n "$WIN" ]] || { echo "ERROR: no WINDOW_SECONDS in $LOG" >&2; tail -50 "$LOG"; exit 1; }

echo "winning fine chunk=$FINE reserved_GiB=${RES:-?} window_s=$WIN window_reserved=${WIN_RES:-?}"
if [[ "$FINE" == "26624" || "$FINE" == "39350" ]]; then
    echo "ERROR: refusing banned chunk $FINE"
    exit 1
fi
if awk -v c="$FINE" 'BEGIN { exit !(c < 1024) }'; then
    echo "ERROR: chunk $FINE < 1024; not submitting 10 doomed jobs"
    exit 1
fi

# 454 windows/shard * s/window * 1.4 buffer, min 12h, prefer ~24h if close
NEED_S="$(python3 - << PY
win=float("$WIN")
need=int(win * 454 * 1.4 + 1800)
print(max(need, 12 * 3600))
PY
)"
if (( NEED_S <= 24 * 3600 )); then
    WALL="1-00:00:00"
elif (( NEED_S <= 48 * 3600 )); then
    WALL="2-00:00:00"
elif (( NEED_S <= 72 * 3600 )); then
    WALL="3-00:00:00"
else
    echo "ERROR: estimated shard wall ${NEED_S}s too long at ${WIN}s/window; not submitting"
    exit 1
fi
echo "shard wall=$WALL (need_s=$NEED_S from ${WIN}s/window * 454 * 1.4)"

echo "Submitting 10 electricity shards at chunk=$FINE partition=gpubase_bygpu_b4"
for i in $(seq 0 9); do
    ./submit_binary.sh \
        --configs "$CFG_SHARD" \
        --datasets electricity \
        --eval-only \
        --eval-source-checkpoint-dir "$CKPT" \
        --eval-shard "${i}/10" \
        --unet-max-chunk-size "$FINE" \
        --time "$WALL" \
        --partition gpubase_bygpu_b4
done
echo "Done $(date). Monitor: squeue -u \$USER"
