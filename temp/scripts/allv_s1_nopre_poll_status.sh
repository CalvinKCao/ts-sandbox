#!/bin/bash
# One-shot status dump for allv s1 nopre jobs (used by poll loop ticks).
set -euo pipefail
REPO=/home/cao/ts-sandbox
JOBSFILE="$REPO/temp/scripts/allv_s1_nopre_jobs.txt"
SSH_PRE='source /etc/profile >/dev/null 2>&1; export PATH=/cm/shared/apps/slurm/current/bin:$PATH; [ -f /cm/shared/apps/slurm/var/etc/killarney/slurm.conf ] && export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf'
WATCH=$(grep -v '^#' "$JOBSFILE" | awk '{print $1}' | paste -sd, -)
echo "=== $(date -Is) allv_s1_nopre status watch=$WATCH ==="
ssh -o BatchMode=yes -o ConnectTimeout=60 killarney "$SSH_PRE
echo '--- squeue ---'
squeue -u ccao87 -o '%.18i %.40j %.10T %.10M %.8m %R' || true
echo '--- sacct ---'
sacct -j $WATCH -X --format=JobID,JobName%50,State,ExitCode,Elapsed,ReqMem -P || true
" 2>&1
