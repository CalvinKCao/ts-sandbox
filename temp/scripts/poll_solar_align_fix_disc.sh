#!/usr/bin/env bash
# Poll solar MMPD restage jobs, then submit+poll mlp disc; pull AUROC+viz.
set -u
MMPD_IDS=(4631186 4631187 4631188)
DISC_IDS=()
LOCAL=/home/cao/ts-sandbox/temp/lean_disc_c128_results
LOG=$LOCAL/poll_solar_align_fix.log
STATUS=$LOCAL/poll_solar_align_fix_status.txt
MERGE_PY=/home/cao/ts-sandbox/temp/scripts/merge_four_mlp_auroc_into_table.py
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
REMOTE_ROOT='$SCRATCH/ts-sandbox-ordinal-fine'
MMPD_ROOT=results/datasets/08-06-mmpd-solar-binary-aligned-fix
INTERVAL=180
MAX=$((5*3600))
START=$(date +%s)
PHASE=mmpd
DISC_SUBMITTED=0

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mkdir -p "$LOCAL"
: > "$LOG"
echo "poll_start $(date -Is) mmpd=${MMPD_IDS[*]} max=${MAX}s" | tee "$STATUS"

get_states() {
  local ids=("$@")
  local idlist="${ids[*]}"
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE
for j in $idlist; do
  st=\$(sacct -j \$j -X -n -o State | head -1 | tr -d ' ')
  el=\$(sacct -j \$j -X -n -o Elapsed | head -1 | tr -d ' ')
  ec=\$(sacct -j \$j -X -n -o ExitCode | head -1 | tr -d ' ')
  echo \"\$j \$st \$el \$ec\"
done"
}

all_completed() {
  local ok=1 line jid st el ec
  while read -r jid st el ec; do
    [ -z "${jid:-}" ] && continue
    [ "$st" = "COMPLETED" ] || ok=0
  done <<< "$(get_states "$@")"
  [ "$ok" -eq 1 ]
}

any_failed() {
  local line jid st el ec
  while read -r jid st el ec; do
    [ -z "${jid:-}" ] && continue
    case "$st" in
      FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) return 0 ;;
    esac
  done <<< "$(get_states "$@")"
  return 1
}

submit_disc() {
  log "Submitting solar mlp disc (exclude kn002,kn010) using $MMPD_ROOT"
  local out new
  out=$(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
CKPT=results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama
CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml
OUT_TAG=solar_Alabama-c128-wn128grid-valtest80-byvar
JID=\$(sbatch \
  --job-name=ablation-mlp-solar_Alabama \
  --account=aip-boyuwang \
  --time=4:00:00 \
  --nodes=1 \
  --gres=gpu:l40s:1 \
  --cpus-per-task=8 \
  --mem=50G \
  --exclude=kn002,kn010 \
  --export=ALL,CKPT=\"\$CKPT\",DISC_CONFIG=\"\$CFG\",DISC_RUN_NAME=window_norm_c128,OUT_TAG=\"\$OUT_TAG\",SCRATCH,HOME,USER,PATH,SLURM_CONF \
  --output=\$PWD/results/slurm/%x-%j.out \
  --error=\$PWD/results/slurm/%x-%j.err \
  --mail-type=END,FAIL \
  --mail-user=ccao87@uwo.ca \
  \$PWD/temp/scripts/submit_ablation_disc_l8_l16.sh \
  --dataset solar_Alabama \
  --disc-arch mlp \
  --viz-n-windows 1 \
  --mmpd-output-root \"$MMPD_ROOT\" \
  | awk '{print \$4}')
echo NEWJID=\$JID
echo \"\$JID solar_Alabama \$OUT_TAG align_fix\" >> temp/lean_disc_c128_jobs/four_mlp_disc_submitted.txt
")
  log "$out"
  new=$(echo "$out" | sed -n 's/.*NEWJID=\([0-9]*\).*/\1/p' | tail -1)
  if [ -z "$new" ]; then
    log "ERROR: disc submit failed"
    return 1
  fi
  DISC_IDS=("$new")
  DISC_SUBMITTED=1
  PHASE=disc
  log "tracking disc job $new"
}

pull_solar() {
  log "PULL solar auroc+viz"
  mapfile -t PACK_LINES < <(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
for p in \$(ls -dt results/datasets/*-ablation-disc-l8-l16-solar_Alabama-c128-wn128grid-valtest80-byvar 2>/dev/null); do
  if [ -f \"\$p/auroc_table.json\" ]; then
    echo \"\$PWD/\$p\"
    break
  fi
done
")
  if [ "${#PACK_LINES[@]}" -lt 1 ]; then
    log "ERROR: no solar auroc pack"
    return 1
  fi
  rpath="${PACK_LINES[0]}"
  mkdir -p "$LOCAL/solar_Alabama" "$LOCAL/viz/solar_Alabama"
  rsync -az "killarney:$rpath/auroc_table.json" "killarney:$rpath/summary.json" "$LOCAL/solar_Alabama/" || true
  rsync -az "killarney:$rpath/auroc_by_variate.json" "$LOCAL/solar_Alabama/" 2>/dev/null || true
  rsync -az "killarney:$rpath/viz/" "$LOCAL/viz/solar_Alabama/" || true
  python3 "$MERGE_PY" | tee -a "$LOG" || true
  python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
p = Path("/home/cao/ts-sandbox/temp/lean_disc_c128_results/solar_Alabama/auroc_table.json")
t = json.loads(p.read_text())
print("\n=== SOLAR mlp AUROC ===")
rows = t if isinstance(t, list) else t.get("rows") or t.get("table") or []
if isinstance(t, dict) and not rows:
    for k,v in t.items():
        if isinstance(v, list):
            rows = v
            break
print(json.dumps(t, indent=2)[:4000])
PY
  echo "DONE $(date -Is) disc=${DISC_IDS[*]:-none}" | tee "$STATUS"
}

while true; do
  now=$(date +%s)
  elapsed=$((now-START))
  if [ "$elapsed" -ge "$MAX" ]; then
    log "TIMEOUT after ${elapsed}s phase=$PHASE"
    echo "TIMEOUT $(date -Is) phase=$PHASE" | tee "$STATUS"
    exit 2
  fi

  if [ "$PHASE" = "mmpd" ]; then
    if ! states=$(get_states "${MMPD_IDS[@]}"); then
      log "ssh fail; retry in ${INTERVAL}s"
      sleep "$INTERVAL"
      continue
    fi
    log "mmpd states elapsed=${elapsed}s:"$'\n'"$states"
    printf '%s\n' "$states" > "$STATUS"
    if any_failed "${MMPD_IDS[@]}"; then
      log "MMPD failed — abort"
      echo "FAIL_MMPD $(date -Is)" | tee "$STATUS"
      exit 1
    fi
    if all_completed "${MMPD_IDS[@]}"; then
      log "MMPD complete — submitting disc"
      submit_disc || exit 1
    else
      sleep "$INTERVAL"
      continue
    fi
  fi

  if [ "$PHASE" = "disc" ]; then
    if ! states=$(get_states "${DISC_IDS[@]}"); then
      log "ssh fail; retry in ${INTERVAL}s"
      sleep "$INTERVAL"
      continue
    fi
    log "disc states elapsed=${elapsed}s:"$'\n'"$states"
    printf '%s\n' "$states" > "$STATUS"
    # progress tail
    prog=$(ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE
cd $REMOTE_ROOT
for j in ${DISC_IDS[*]}; do
  echo === \$j ===
  rg -n 'align|AUROC|Finished|Error|Traceback|generate |materialize|MMPD→binary' results/slurm/*-\$j.out results/slurm/*-\$j.err 2>/dev/null | tail -n 8 || tail -n 5 results/slurm/*-\$j.out 2>/dev/null || true
done
" 2>/dev/null || true)
    [ -n "$prog" ] && log "progress:"$'\n'"$prog"
    if any_failed "${DISC_IDS[@]}"; then
      log "disc FAILED"
      echo "FAIL_DISC $(date -Is)" | tee "$STATUS"
      exit 1
    fi
    if all_completed "${DISC_IDS[@]}"; then
      pull_solar
      exit 0
    fi
    sleep "$INTERVAL"
  fi
done
