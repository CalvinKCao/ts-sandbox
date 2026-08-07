#!/usr/bin/env bash
# Poll solar/ETTm1/ETTm2 mlp disc; PeMS already done. Cap 2h. Exclude kn002 on resub.
set -u
IDS=(4628324 4628348 4628388)
declare -A DS=(
  [4628324]=ETTm1
  [4628348]=solar_Alabama
  [4628388]=ETTm2
)
LOCAL=/home/cao/ts-sandbox/temp/lean_disc_c128_results
LOG=$LOCAL/poll_three_mlp.log
STATUS=$LOCAL/poll_three_mlp_status.txt
MERGE_PY=/home/cao/ts-sandbox/temp/scripts/merge_four_mlp_auroc_into_table.py
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
REMOTE_ROOT='$SCRATCH/ts-sandbox-ordinal-fine'
INTERVAL=300
MAX=$((2*3600))
START=$(date +%s)
RESUBBED=()

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mkdir -p "$LOCAL"
: > "$LOG"
echo "poll_start $(date -Is) ids=${IDS[*]} max=${MAX}s" | tee "$STATUS"

get_states() {
  local idlist="${IDS[*]}"
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE
for j in $idlist; do
  st=\$(sacct -j \$j -X -n -o State | head -1 | tr -d ' ')
  el=\$(sacct -j \$j -X -n -o Elapsed | head -1 | tr -d ' ')
  ec=\$(sacct -j \$j -X -n -o ExitCode | head -1 | tr -d ' ')
  echo \"\$j \$st \$el \$ec\"
done"
}

tail_progress() {
  local idlist="${IDS[*]}"
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "$SSH_PRE
cd $REMOTE_ROOT
for j in $idlist; do
  echo === \$j ===
  # last generate line or Finished/AUROC
  rg -n 'generate |AUROC table|Finished:|Error|Traceback|materialize done|disc_auroc|Wrote' results/slurm/*-\$j.out 2>/dev/null | tail -n 6 || tail -n 6 results/slurm/*-\$j.out 2>/dev/null || echo no-out
done
"
}

diagnose() {
  local jid="$1"
  ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
echo '=== sacct ==='
sacct -j $jid -X -o JobID,JobName%40,State,ExitCode,Elapsed,End,NodeList
echo '=== tail out ==='
tail -n 100 results/slurm/*-$jid.out 2>/dev/null || true
echo '=== tail err ==='
tail -n 40 results/slurm/*-$jid.err 2>/dev/null || true
"
}

resub_once() {
  local jid="$1" ds="${DS[$jid]}"
  log "RESUB once for $ds (old=$jid) exclude=kn002,kn010"
  local out new
  out=$(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
case $ds in
  solar_Alabama)
    CKPT=results/ckpts/08-05-4623006-solar_Alabama-binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama
    CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_solar_alabama.yaml
    OUT_TAG=solar_Alabama-c128-wn128grid-valtest80-byvar
    ;;
  ETTm1)
    CKPT=results/ckpts/08-05-4623007-ETTm1-binary_window_norm_patch_refine_canvas128_p64x6_ettm1
    CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm1.yaml
    OUT_TAG=ETTm1-c128-wn128grid-valtest80-byvar
    ;;
  ETTm2)
    CKPT=results/ckpts/08-05-4623008-ETTm2-binary_window_norm_patch_refine_canvas128_p64x6_ettm2
    CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_ettm2.yaml
    OUT_TAG=ETTm2-c128-wn128grid-valtest80-byvar
    ;;
esac
MMPD_ROOT=results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four
JID=\$(sbatch \
  --job-name=ablation-mlp-\${ds} \
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
  --dataset \$ds \
  --disc-arch mlp \
  --viz-n-windows 1 \
  --mmpd-output-root \"\$MMPD_ROOT\" \
  --reuse-forecast-cache \
  | awk '{print \$4}')
echo NEWJID=\$JID
echo \"\$JID \$ds \$OUT_TAG resub_of_$jid\" >> temp/lean_disc_c128_jobs/four_mlp_disc_submitted.txt
")
  log "$out"
  new=$(echo "$out" | sed -n 's/.*NEWJID=\([0-9]*\).*/\1/p' | tail -1)
  if [ -n "$new" ]; then
    for i in "${!IDS[@]}"; do
      if [ "${IDS[$i]}" = "$jid" ]; then IDS[$i]="$new"; fi
    done
    DS[$new]="$ds"
    RESUBBED+=("$ds:$jid->$new")
    log "tracking new job $new for $ds"
  else
    log "ERROR: resub failed for $ds"
  fi
}

pull_three() {
  log "PULL starting (solar/ETTm1/ETTm2; PeMS already local)"
  mapfile -t PACK_LINES < <(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
for ds in solar_Alabama ETTm1 ETTm2; do
  for p in \$(ls -dt results/datasets/*-ablation-disc-l8-l16-\${ds}-c128-wn128grid-valtest80-byvar 2>/dev/null); do
    if [ -f \"\$p/auroc_table.json\" ]; then
      echo \"\$ds|\$PWD/\$p\"
      break
    fi
  done
done
")
  if [ "${#PACK_LINES[@]}" -lt 3 ]; then
    log "WARN: expected 3 packs, got ${#PACK_LINES[@]}: ${PACK_LINES[*]}"
  fi
  for line in "${PACK_LINES[@]}"; do
    ds="${line%%|*}"
    rpath="${line#*|}"
    mkdir -p "$LOCAL/$ds" "$LOCAL/viz/$ds"
    log "rsync metrics $ds <- $rpath"
    rsync -az "killarney:$rpath/auroc_table.json" "killarney:$rpath/summary.json" "$LOCAL/$ds/" 2>>"$LOG" || log "WARN metrics partial $ds"
    rsync -az "killarney:$rpath/auroc_by_variate.json" "$LOCAL/$ds/" 2>>"$LOG" || true
    log "rsync viz $ds"
    rsync -az "killarney:$rpath/viz/" "$LOCAL/viz/$ds/" 2>>"$LOG" || log "WARN: no viz for $ds"
  done
  python3 "$MERGE_PY" | tee -a "$LOG"
  # print single-table rows for the 3
  python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
p = Path("/home/cao/ts-sandbox/temp/lean_disc_c128_results/full_metrics_table.json")
t = json.loads(p.read_text())
want = {"solar_Alabama", "ETTm1", "ETTm2"}
print("\n=== UPDATED SINGLE-TABLE ROWS (mlp) ===")
print(f"{'dataset':16} {'L':>3} {'binary':>10} {'mmpd':>10} {'stamp':>16}")
for d in t["datasets"]:
    if d["dataset"] not in want:
        continue
    da = d.get("disc_auroc") or {}
    stamp = da.get("pack_stamp", "")
    rows = da.get("rows") or []
    if not rows:
        print(f"{d['dataset']:16}  — empty —")
        continue
    for r in rows:
        if r.get("arch") != "mlp":
            continue
        b, m = r.get("binary_auroc"), r.get("mmpd_auroc")
        bf = f"{b:.6f}" if b is not None else "None"
        mf = f"{m:.6f}" if m is not None else "None"
        print(f"{d['dataset']:16} {r['L']:3d} {bf:>10} {mf:>10} {stamp:>16}")
PY
  log "PULL done"
  echo "DONE $(date -Is) resubs=${RESUBBED[*]:-none}" | tee "$STATUS"
}

while true; do
  now=$(date +%s)
  elapsed=$((now-START))
  if [ "$elapsed" -ge "$MAX" ]; then
    log "TIMEOUT after ${elapsed}s"
    get_states | tee -a "$LOG" || true
    echo "TIMEOUT $(date -Is)" | tee "$STATUS"
    exit 2
  fi
  if ! states=$(get_states); then
    log "ssh fail; retry in ${INTERVAL}s"
    sleep "$INTERVAL"
    continue
  fi
  log "states elapsed=${elapsed}s:"$'\n'"$states"
  printf '%s\n' "$states" > "$STATUS"
  prog=$(tail_progress 2>/dev/null || true)
  if [ -n "$prog" ]; then
    log "progress:"$'\n'"$prog"
  fi

  all_done=1
  while read -r jid st el ec; do
    [ -z "${jid:-}" ] && continue
    case "$st" in
      COMPLETED) ;;
      RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED)
        all_done=0
        ;;
      FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
        all_done=0
        already=0
        for r in "${RESUBBED[@]:-}"; do
          case "$r" in ${DS[$jid]}:*) already=1 ;; esac
        done
        if [ "$already" -eq 0 ]; then
          log "FAIL $jid ${DS[$jid]} $st ec=$ec — diagnose+resub (exclude kn002)"
          diagnose "$jid" | tee -a "$LOG"
          resub_once "$jid"
        else
          log "FAIL again for ${DS[$jid]} after resub — giving up that ds"
        fi
        ;;
      *)
        all_done=0
        log "unknown state $jid $st"
        ;;
    esac
  done <<< "$states"

  if [ "$all_done" -eq 1 ]; then
    ok=1
    while read -r jid st el ec; do
      [ -z "${jid:-}" ] && continue
      [ "$st" = "COMPLETED" ] || ok=0
    done <<< "$(get_states)"
    if [ "$ok" -eq 1 ]; then
      pull_three
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
