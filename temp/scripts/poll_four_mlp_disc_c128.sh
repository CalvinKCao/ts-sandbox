#!/usr/bin/env bash
# Poll PeMS/solar/ETTm1/ETTm2 mlp disc jobs; on all COMPLETE pull AUROC+viz.
set -u
IDS=(4628322 4628324 4628348 4628388)
declare -A DS=(
  [4628322]=PeMS
  [4628324]=ETTm1
  [4628348]=solar_Alabama
  [4628388]=ETTm2
)
LOCAL=/home/cao/ts-sandbox/temp/lean_disc_c128_results
LOG=$LOCAL/poll_four_mlp.log
STATUS=$LOCAL/poll_four_mlp_status.txt
MERGE_PY=/home/cao/ts-sandbox/temp/scripts/merge_four_mlp_auroc_into_table.py
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
REMOTE_ROOT='$SCRATCH/ts-sandbox-ordinal-fine'
INTERVAL=600
MAX=$((3*3600))
START=$(date +%s)
RESUBBED=()

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mkdir -p "$LOCAL"
: > "$LOG"
echo "poll_start $(date -Is) ids=${IDS[*]}" | tee "$STATUS"

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

diagnose() {
  local jid="$1"
  ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
echo '=== sacct ==='
sacct -j $jid -X -o JobID,JobName%40,State,ExitCode,Elapsed,End
echo '=== tail out ==='
tail -n 80 results/slurm/*-$jid.out 2>/dev/null || true
echo '=== tail err ==='
tail -n 40 results/slurm/*-$jid.err 2>/dev/null || true
"
}

resub_once() {
  local jid="$1" ds="${DS[$jid]}"
  log "RESUB once for $ds (old=$jid)"
  local out new
  out=$(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
case $ds in
  PeMS)
    CKPT=results/ckpts/08-05-4623005-PeMS-binary_window_norm_patch_refine_canvas128_p64x6_pems
    CFG=configs/binary_window_norm_patch_refine_canvas128_p64x6_pems.yaml
    OUT_TAG=PeMS-c128-wn128grid-valtest80-byvar
    ;;
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

pull_all() {
  log "PULL starting"
  mapfile -t PACK_LINES < <(ssh -o BatchMode=yes killarney "$SSH_PRE
cd $REMOTE_ROOT
for ds in PeMS solar_Alabama ETTm1 ETTm2; do
  for p in \$(ls -dt results/datasets/*-ablation-disc-l8-l16-\${ds}-c128-wn128grid-valtest80-byvar 2>/dev/null); do
    if [ -f \"\$p/auroc_table.json\" ]; then
      echo \"\$ds|\$PWD/\$p\"
      break
    fi
  done
done
")
  if [ "${#PACK_LINES[@]}" -eq 0 ]; then
    log "ERROR: no packs with auroc_table.json found"
    return 1
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
  log "PULL done"
  echo "DONE $(date -Is) resubs=${RESUBBED[*]:-none}" | tee "$STATUS"
}

while true; do
  now=$(date +%s)
  elapsed=$((now-START))
  if [ "$elapsed" -ge "$MAX" ]; then
    log "TIMEOUT after ${elapsed}s"
    echo "TIMEOUT $(date -Is)" | tee "$STATUS"
    exit 2
  fi
  if ! states=$(get_states); then
    log "ssh fail; retry in ${INTERVAL}s"
    sleep "$INTERVAL"
    continue
  fi
  log "states:"$'\n'"$states"
  printf '%s\n' "$states" > "$STATUS"

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
          log "FAIL $jid ${DS[$jid]} $st ec=$ec — diagnose+resub"
          diagnose "$jid" | tee -a "$LOG"
          resub_once "$jid"
        else
          log "FAIL again for ${DS[$jid]} after resub — stop tracking success"
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
      pull_all
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
