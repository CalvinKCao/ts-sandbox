#!/usr/bin/env bash
set -u
JID=$(cat /home/cao/ts-sandbox/temp/lean_disc_c128_results/solar_disc_jid.txt)
LOCAL=/home/cao/ts-sandbox/temp/lean_disc_c128_results
LOG=$LOCAL/poll_solar_disc_only.log
STATUS=$LOCAL/poll_solar_disc_only_status.txt
MERGE_PY=/home/cao/ts-sandbox/temp/scripts/merge_four_mlp_auroc_into_table.py
INTERVAL=180
MAX=$((4*3600))
START=$(date +%s)
log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mkdir -p "$LOCAL"
echo "poll_start $(date -Is) disc=$JID" | tee "$STATUS" | tee "$LOG"

kssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:\$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf; $*"
}

while true; do
  now=$(date +%s)
  elapsed=$((now-START))
  if [ "$elapsed" -ge "$MAX" ]; then
    log "TIMEOUT after ${elapsed}s"
    echo TIMEOUT | tee "$STATUS"
    exit 2
  fi
  if ! states=$(kssh "sacct -j $JID -X -n -o JobID,State,Elapsed,ExitCode | head -1"); then
    log "ssh_fail; sleep $INTERVAL"
    sleep "$INTERVAL"
    continue
  fi
  log "elapsed=${elapsed}s $states"
  echo "$states" > "$STATUS"
  st=$(echo "$states" | awk '{print $2}')
  case "$st" in
    COMPLETED)
      log "COMPLETED — pulling"
      rpath=$(kssh "cd \$SCRATCH/ts-sandbox-ordinal-fine; for p in \$(ls -dt results/datasets/*-ablation-disc-l8-l16-solar_Alabama-c128-wn128grid-valtest80-byvar 2>/dev/null); do [ -f \"\$p/auroc_table.json\" ] && echo \"\$PWD/\$p\" && break; done")
      log "rpath=$rpath"
      mkdir -p "$LOCAL/solar_Alabama" "$LOCAL/viz/solar_Alabama"
      rsync -az "killarney:$rpath/auroc_table.json" "killarney:$rpath/summary.json" "$LOCAL/solar_Alabama/" || true
      rsync -az "killarney:$rpath/auroc_by_variate.json" "$LOCAL/solar_Alabama/" 2>/dev/null || true
      rsync -az "killarney:$rpath/viz/" "$LOCAL/viz/solar_Alabama/" || true
      python3 "$MERGE_PY" | tee -a "$LOG" || true
      python3 -c 'import json;from pathlib import Path;print(Path("/home/cao/ts-sandbox/temp/lean_disc_c128_results/solar_Alabama/auroc_table.json").read_text()[:6000])' | tee -a "$LOG"
      echo DONE | tee "$STATUS"
      exit 0
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
      log "FAIL $states"
      kssh "cd \$SCRATCH/ts-sandbox-ordinal-fine; tail -80 results/slurm/*-$JID.err; tail -40 results/slurm/*-$JID.out" | tee -a "$LOG" || true
      echo FAIL | tee "$STATUS"
      exit 1
      ;;
    RUNNING|PENDING|CONFIGURING|COMPLETING)
      kssh "cd \$SCRATCH/ts-sandbox-ordinal-fine; rg -n 'align|AUROC|Finished|Error|Traceback|generate |materialize|MMPD|target_rmse' results/slurm/*-$JID.out results/slurm/*-$JID.err 2>/dev/null | tail -n 8" | tee -a "$LOG" || true
      ;;
  esac
  sleep "$INTERVAL"
done
