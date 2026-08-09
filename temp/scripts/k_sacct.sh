#!/bin/bash
export PATH=/cm/shared/apps/slurm/current/bin:/usr/bin:/bin
JOBS="${1:?}"
sacct -j "$JOBS" -X -o JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,NodeList,End
echo ---
squeue -u ccao87 -o '%.18i %.8T %.10M %R %j' | head -40
