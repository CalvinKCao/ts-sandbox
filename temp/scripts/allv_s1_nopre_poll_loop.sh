#!/bin/bash
# 90m poll loop for full all-var s1 nopretrain electricity + solar on Killarney.
# Resubmits on FAILED/OUT_OF_MEMORY/TIMEOUT with higher --mem (elec up to 500G).
set -euo pipefail

REPO=/home/cao/ts-sandbox
WORKTREE=/scratch/ccao87/ts-sandbox-main-fullhp
LOG="$REPO/temp/scripts/allv_s1_nopre_poll_loop.log"
PIDFILE="$REPO/temp/scripts/allv_s1_nopre_poll_loop.pid"
JOBSFILE="$REPO/temp/scripts/allv_s1_nopre_jobs.txt"
INTERVAL_SEC=5400
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "$old" && "$old" != "$$" ]] && kill -0 "$old" 2>/dev/null; then
    echo "poll loop already running pid=$old" >&2
    exit 0
  fi
fi
echo $$ > "$PIDFILE"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

ssh_k() {
  ssh -o BatchMode=yes -o ConnectTimeout=60 -o ServerAliveInterval=30 killarney "$SSH_PRE; $*"
}

job_state() {
  local jid="$1"
  ssh_k "sacct -j ${jid} -X --format=State -P -n 2>/dev/null | head -1 | tr -d '\"'" 2>/dev/null || echo UNKNOWN
}

job_exit() {
  local jid="$1"
  ssh_k "sacct -j ${jid} -X --format=ExitCode -P -n 2>/dev/null | head -1 | tr -d '\"'" 2>/dev/null || echo "?:?"
}

parse_jobsfile() {
  # lines: JOBID DATASET CONFIG_STEM MEM TIME [REPLACED_JOBID]
  mapfile -t ROWS < <(grep -v '^#' "$JOBSFILE" | grep -v '^[[:space:]]*$' || true)
}

write_jobsfile() {
  : > "$JOBSFILE"
  for row in "${ROWS[@]}"; do
    echo "$row" >> "$JOBSFILE"
  done
}

bump_mem() {
  local mem="$1"
  case "$mem" in
    *G) local n=${mem%G}; echo "$(( n < 500 ? (n < 200 ? 200 : (n < 300 ? 300 : 500)) : 500))G" ;;
    *) echo "500G" ;;
  esac
}

resubmit_row() {
  local idx="$1" reason="$2"
  IFS=$' ' read -r jid dataset cfg mem time replaced <<<"${ROWS[$idx]}"
  local new_mem
  new_mem="$(bump_mem "$mem")"
  log "RESUBMIT $dataset ($cfg) was=$jid reason=$reason mem $mem->$new_mem"
  local out
  out=$(ssh_k "cd $WORKTREE && git pull -q && ./submit_binary.sh \
    --configs $cfg \
    --datasets $dataset \
    --mem $new_mem \
    --time $time" 2>&1) || true
  log "$out"
  local newjid
  newjid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | tail -1 | awk '{print $NF}')
  if [[ -z "$newjid" ]]; then
    log "RESUBMIT_FAILED $dataset could not parse job id"
    return 1
  fi
  ROWS[$idx]="$newjid $dataset $cfg $new_mem $time ${jid}"
  write_jobsfile
  log "RESUBMIT_OK $dataset new_job=$newjid replaced=$jid"
}

all_terminal_ok() {
  local ok=1
  for row in "${ROWS[@]}"; do
    IFS=$' ' read -r jid _rest <<<"$row"
    local st
    st=$(job_state "$jid")
    if [[ "$st" == "COMPLETED" ]]; then
      continue
    fi
    if [[ "$st" == "RUNNING" || "$st" == "PENDING" ]]; then
      ok=0
      continue
    fi
    ok=0
  done
  [[ "$ok" -eq 1 ]]
}

log "POLL_START pid=$$ interval=${INTERVAL_SEC}s jobsfile=$JOBSFILE"
while true; do
  parse_jobsfile
  if [[ ${#ROWS[@]} -eq 0 ]]; then
    log "NO_JOBS in $JOBSFILE — sleeping"
    sleep "$INTERVAL_SEC"
    continue
  fi

  log "=== TICK watch ${#ROWS[@]} jobs ==="
  ssh_k "squeue -u ccao87 -o '%.18i %.40j %.10T %.10M %.8m %R'" >>"$LOG" 2>&1 || log "squeue failed"

  local_idx=0
  while [[ $local_idx -lt ${#ROWS[@]} ]]; do
    IFS=$' ' read -r jid dataset cfg mem time replaced <<<"${ROWS[$local_idx]}"
    st=$(job_state "$jid")
    ec=$(job_exit "$jid")
    log "JOB $jid $dataset state=$st exit=$ec mem=$mem cfg=$cfg"
  if [[ "$st" == "FAILED" || "$st" == "OUT_OF_MEMORY" || "$st" == "TIMEOUT" ]]; then
      if [[ "$ec" == "0:125" || "$st" == "OUT_OF_MEMORY" || "$st" == "TIMEOUT" || "$st" == "FAILED" ]]; then
        resubmit_row "$local_idx" "$st" || true
      fi
    fi
    local_idx=$((local_idx + 1))
  done

  parse_jobsfile
  done=1
  for row in "${ROWS[@]}"; do
    IFS=$' ' read -r jid _rest <<<"$row"
    st=$(job_state "$jid")
    if [[ "$st" != "COMPLETED" ]]; then
      done=0
    fi
  done
  if [[ "$done" -eq 1 ]]; then
    log "ALL_DONE ${ROWS[*]}"
    rm -f "$PIDFILE"
    exit 0
  fi

  sleep "$INTERVAL_SEC"
done
