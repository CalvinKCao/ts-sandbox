#!/usr/bin/env python3
import os
import re
from datetime import datetime

RESULTS_DIR = 'results'
REPORTS_DIR = 'reports'
OUT = os.path.join(REPORTS_DIR, 'mse_runs_summary.md')

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# find mse run logs
mse_logs = []
for d in sorted(os.listdir(RESULTS_DIR)):
    if 'mse' in d.lower():
        log_dir = os.path.join(RESULTS_DIR, d, 'logs')
        if os.path.isdir(log_dir):
            for lf in os.listdir(log_dir):
                if lf.endswith('.log') and 'mse' in lf.lower():
                    mse_logs.append(os.path.join(log_dir, lf))

# fallback: also search for files named mse_ablation_sweep.log anywhere under results
if not mse_logs:
    for root, _, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.lower().startswith('mse_ablation') and f.endswith('.log'):
                mse_logs.append(os.path.join(root, f))

# patterns
itrans_pattern = re.compile(r"\[([^\]]+)\] iTransformer baseline: MSE=([\d\.]+), MAE=([\d\.]+)")
diff_pattern = re.compile(r"\[([^\]]+)\] Avg: MSE=([\d\.]+), MAE=([\d\.]+)")
time_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
started_pattern = re.compile(r"^Started:\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

rows = []
skipped = []
for path in sorted(mse_logs):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        skipped.append((path, f'read error: {e}'))
        continue
    tail100 = ''.join(lines[-100:])
    lower_tail = tail100.lower()
    if any(m in lower_tail for m in ['traceback', 'exception', 'fatal error', '\nerror\n']):
        skipped.append((path, 'failed/incomplete in tail'))
        continue
    last_lines = ''.join(lines[-200:])
    dataset_metrics = {}
    for line in re.split(r"\n", last_lines):
        im = itrans_pattern.search(line)
        if im:
            ds = im.group(1)
            dataset_metrics.setdefault(ds, {})['itrans_mse'] = im.group(2)
            dataset_metrics[ds]['itrans_mae'] = im.group(3)
        dm = diff_pattern.search(line)
        if dm:
            ds = dm.group(1)
            dataset_metrics.setdefault(ds, {})['diff_mse'] = dm.group(2)
            dataset_metrics[ds]['diff_mae'] = dm.group(3)
    # job id from parent dir
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(path))) if path.count(os.sep)>=2 else os.path.basename(path)
    job_match = re.search(r"-(\d+)-", run_dir)
    job_id = job_match.group(1) if job_match else ''
    # duration
    start_time = None
    end_time = None
    for line in lines[:200]:
        s = started_pattern.search(line)
        if s:
            try:
                start_time = datetime.strptime('2026-'+s.group(1), '%Y-%m-%d %H:%M:%S')
                break
            except Exception:
                pass
        t = time_pattern.search(line)
        if t:
            try:
                start_time = datetime.strptime(t.group(1), '%Y-%m-%d %H:%M:%S')
                break
            except Exception:
                pass
    for line in reversed(lines):
        t = time_pattern.search(line)
        if t:
            try:
                end_time = datetime.strptime(t.group(1), '%Y-%m-%d %H:%M:%S')
                break
            except Exception:
                pass
    duration = 'Unknown'
    if start_time and end_time and end_time>=start_time:
        d = end_time - start_time
        h,m = divmod(d.total_seconds(),3600)
        m,s = divmod(m,60)
        duration = f"{int(h)}h {int(m)}m {int(s)}s"
    if not dataset_metrics:
        rows.append((os.path.basename(path), job_id, 'None', None, None, None, None, 'N/A', duration))
    else:
        for ds, m in dataset_metrics.items():
            it_mse = m.get('itrans_mse')
            it_mae = m.get('itrans_mae')
            df_mse = m.get('diff_mse')
            df_mae = m.get('diff_mae')
            imp = 'N/A'
            try:
                if it_mse and df_mse:
                    imp = f"{(float(it_mse)-float(df_mse))/float(it_mse)*100:.2f}%"
            except Exception:
                imp = 'N/A'
            rows.append((os.path.basename(path), job_id, ds, it_mse, it_mae, df_mse, df_mae, imp, duration))

with open(OUT, 'w', encoding='utf-8') as out:
    out.write('# MSE ablation runs summary\n\n')
    out.write('Consolidated summary of recently pulled MSE ablation runs. Skipped runs with detected failures in the last 100 lines.\n\n')
    out.write('| Log file | Job ID | Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) | Duration |\n')
    out.write('|---|---:|---|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:
        out.write(f"| {r[0]} | {r[1] or 'N/A'} | {r[2]} | {r[3] or 'N/A'} | {r[4] or 'N/A'} | {r[5] or 'N/A'} | {r[6] or 'N/A'} | {r[7]} | {r[8]} |\n")
    if skipped:
        out.write('\n## Skipped runs (failed/incomplete detected)\n')
        for p, reason in skipped:
            out.write(f'- {p}: {reason}\n')

print('Wrote MSE summary:', OUT)
