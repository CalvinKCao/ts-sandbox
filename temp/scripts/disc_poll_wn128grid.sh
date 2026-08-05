#!/bin/bash
# Poll wn128grid disc ablation jobs. Re-auth first if needed: ssh killarney
set -euo pipefail
cd "$(dirname "$0")/../.."
ssh killarney 'bash -s' <<'REMOTE'
set -euo pipefail
cd "$SCRATCH/ts-sandbox-ordinal-fine"
echo "=== sacct ==="
sacct -j 4604919,4604920,4604921,4604922,4604923 --format=JobID,JobName,State,Elapsed,ExitCode -P
echo
echo "=== auroc / snap ==="
while read -r DS JOB DIR; do
  [ -z "${DS:-}" ] && continue
  echo "--- $DS job=$JOB dir=$DIR ---"
  if [ -f "$DIR/auroc_table.json" ]; then
    python3 -c "
import json, glob
rows=json.load(open('$DIR/auroc_table.json'))
for r in rows:
    print(f\"  L={r['L']} {r['source']}: auroc={r['disc_auroc']:.4f} acc={r['disc_acc']:.4f}\")
partials=glob.glob('$DIR/partials/lattice_*.json')
if partials:
    lat=json.load(open(partials[0]))
    bs=(lat.get('binary_snap') or {})
    print(f\"  snap_mode={lat.get('snap_mode')} canvas={lat.get('canvas_height')} binary_meanΔ={bs.get('mean_abs_snap_delta')}\")
"
  else
    echo "  (no auroc yet)"
    rg -n 'snap mode=|Traceback|Finished:' "results/slurm/ablation-disc-l8l16-${JOB}.out" 2>/dev/null | tail -8 || true
  fi
done < temp/killarney_disc_wn128grid_ids.txt
REMOTE
