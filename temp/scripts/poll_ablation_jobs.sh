#!/usr/bin/env bash
# Poll submitted ablation jobs; print status; exit 0 when all terminal.
# Tracked jobs (override with JOBS=...):
#   disc resub 4570208 (max_batches fix, reuse 08-03-2105 out) + traffic + viz
set -euo pipefail
JOBS="${JOBS:-4571065,4594230,4594231,4563093,4563094,4563095,4569025,4569752}"
IFS=',' read -r -a IDS <<< "$JOBS"

ssh -o BatchMode=yes -o ConnectTimeout=40 killarney "bash -l -c $(printf '%q' "
set -euo pipefail
IDS=($(printf '%s ' "${IDS[@]}"))
all_done=1
any_fail=0
echo \"=== \$(date -Is) job poll ===\"
for j in \"\${IDS[@]}\"; do
  line=\$(sacct -j \"\$j\" --format=JobID,JobName%45,State,ExitCode,Elapsed,Timelimit -n -P 2>/dev/null | head -1 || true)
  if [[ -z \"\$line\" ]]; then
    echo \"\$j|MISSING\"
    all_done=0
    continue
  fi
  echo \"\$line\"
  state=\$(echo \"\$line\" | cut -d'|' -f3)
  case \"\$state\" in
    COMPLETED) ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
      any_fail=1
      ;;
    RUNNING|PENDING|COMPLETING|CONFIGURING|REQUEUED)
      all_done=0
      ;;
    *)
      # treat unknown non-terminal as still going
      if [[ \"\$state\" != COMPLETED ]]; then all_done=0; fi
      ;;
  esac
done
echo \"ALL_DONE=\$all_done ANY_FAIL=\$any_fail\"
# brief live queue
squeue -u ccao87 -o '%.10i %.8T %.10M %.40j %R' 2>/dev/null | head -15 || true
")"
