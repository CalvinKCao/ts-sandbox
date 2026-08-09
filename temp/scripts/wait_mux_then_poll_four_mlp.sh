#!/usr/bin/env bash
set -u
shopt -s nullglob
LOCAL=/home/cao/ts-sandbox/temp/lean_disc_c128_results
LOG=$LOCAL/mux_wait.log
SOCK_DIR=/home/cao/.ssh/sockets
POLL=/home/cao/ts-sandbox/temp/scripts/poll_four_mlp_disc_c128.sh
mkdir -p "$LOCAL" "$SOCK_DIR"
: > "$LOG"
log(){ printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
log "waiting for ControlMaster under $SOCK_DIR (need interactive MFA ssh killarney)"
DEADLINE=$(($(date +%s)+10800))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for s in "$SOCK_DIR"/*; do
    [ -e "$s" ] || continue
    [ -S "$s" ] || continue
    if ssh -o BatchMode=yes -o ControlPath="$s" -o ConnectTimeout=8 killarney 'true' 2>/dev/null; then
      log "MUX_OK path=$s"
      SSH='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
      ssh -o BatchMode=yes -o ControlPath="$s" killarney "$SSH
cd \$SCRATCH/ts-sandbox-ordinal-fine
echo '=== squeue ==='
squeue -u ccao87 -o '%.18i %.40j %.2t %.10M %.10l %R' | grep -E 'JOBID|4628322|4628324|4628348|4628388' || true
echo '=== sacct ==='
for j in 4628322 4628324 4628348 4628388; do sacct -j \$j -X -n -o JobID,JobName%30,State,ExitCode,Elapsed,Timelimit; done
echo '=== progress ==='
for j in 4628322 4628324 4628348 4628388; do
  echo --- \$j ---
  grep -E 'generate |materialize done|auroc_table|Error|Traceback' results/slurm/*-\$j.out 2>/dev/null | tail -n 8
  tail -n 2 results/slurm/*-\$j.out 2>/dev/null
done
" | tee -a "$LOG" | tee "$LOCAL/eta_snapshot.txt"
      exec bash "$POLL"
    fi
  done
  if ssh -o BatchMode=yes -o ConnectTimeout=5 killarney 'true' 2>/dev/null; then
    log "DIRECT_OK"
    exec bash "$POLL"
  fi
  sleep 15
done
log "TIMEOUT waiting for mux"
exit 2
