#!/bin/bash
sleep 300
echo 'AGENT_LOOP_WAKE_baselines {"prompt":"Poll Killarney base-c128 fixed-HP jobs 4694792-4694800. Check both itransformer_summary.json and patchtst_summary.json under $SCRATCH/ts-sandbox/temp/baselines_canvas128_subset/results. Resub FAILED missing summaries via ./temp/scripts/submit_baselines_canvas128_killarney.sh --force --datasets NAME from $SCRATCH/ts-sandbox. When 9/9 have both, print MSE/MAE table and stop; else re-arm 5m."}'
