#!/usr/bin/env python3
import os
import re
from datetime import datetime

RESULTS_DIR = "results"
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
REPORTS_DIR = "reports"

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

log_files = []
for fn in sorted(os.listdir(LOG_DIR)):
    if "-joint-" in fn and fn.endswith('.log'):
        log_files.append(os.path.join(LOG_DIR, fn))

# patterns
itrans_pattern = re.compile(r"\[([^\]]+)\] iTransformer baseline: MSE=([\d\.]+), MAE=([\d\.]+)")
diff_pattern = re.compile(r"\[([^\]]+)\] Avg: MSE=([\d\.]+), MAE=([\d\.]+)")
started_pattern = re.compile(r"^Started:\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
time_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

reports_created = []
for log_path in log_files:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    last_lines = ''.join(lines[-150:])
    tail100 = ''.join(lines[-100:])

    # detect failure
    lower_tail = tail100.lower()
    failed_markers = ['traceback', 'exception', 'error', 'failed']
    failed = any(m in lower_tail for m in failed_markers)
    if failed:
        # skip failed runs
        continue

    # parse job id from filename if possible
    fn = os.path.basename(log_path)
    job_match = re.search(r"-(\d+)-", fn)
    job_id = None
    if job_match:
        job_id = job_match.group(1)
    else:
        # fallback: find digits in filename
        dmatch = re.search(r"(\d{3,})", fn)
        if dmatch:
            job_id = dmatch.group(1)

    # parse start/end times
    start_time = None
    end_time = None
    for line in lines[:200]:
        s = started_pattern.search(line)
        if s:
            try:
                start_time = datetime.strptime('2026-' + s.group(1), "%Y-%m-%d %H:%M:%S")
                break
            except Exception:
                pass
        t = time_pattern.search(line)
        if t:
            try:
                start_time = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
                break
            except Exception:
                pass
    for line in reversed(lines):
        t = time_pattern.search(line)
        if t:
            try:
                end_time = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
                break
            except Exception:
                pass

    duration_str = "Unknown"
    if start_time and end_time and end_time >= start_time:
        dur = end_time - start_time
        hours, rem = divmod(dur.total_seconds(), 3600)
        mins, secs = divmod(rem, 60)
        duration_str = f"{int(hours)}h {int(mins)}m {int(secs)}s"

    # extract dataset metrics from last lines
    dataset_metrics = {}
    for line in re.split(r"\n", last_lines):
        im = itrans_pattern.search(line)
        if im:
            ds = im.group(1)
            if ds not in dataset_metrics:
                dataset_metrics[ds] = {}
            dataset_metrics[ds]['itrans_mse'] = im.group(2)
            dataset_metrics[ds]['itrans_mae'] = im.group(3)
        dm = diff_pattern.search(line)
        if dm:
            ds = dm.group(1)
            if ds not in dataset_metrics:
                dataset_metrics[ds] = {}
            dataset_metrics[ds]['diff_mse'] = dm.group(2)
            dataset_metrics[ds]['diff_mae'] = dm.group(3)

    # write per-run markdown
    run_name = fn.replace('.log','')
    md_path = os.path.join(REPORTS_DIR, f"{run_name}.md")
    with open(md_path, 'w', encoding='utf-8') as out:
        out.write(f"# Run report: {run_name}\n\n")
        out.write(f"- Job ID: {job_id or 'Unknown'}\n")
        out.write(f"- Log: {log_path}\n")
        out.write(f"- Duration: {duration_str}\n\n")

        if dataset_metrics:
            out.write("| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |\n")
            out.write("|---|---:|---:|---:|---:|---:|\n")
            for ds, m in dataset_metrics.items():
                it_mse = m.get('itrans_mse')
                df_mse = m.get('diff_mse')
                improvement = 'N/A'
                if it_mse and df_mse:
                    try:
                        improvement = f"{(float(it_mse)-float(df_mse))/float(it_mse)*100:.2f}%"
                    except Exception:
                        improvement = 'N/A'
                out.write(f"| {ds} | {it_mse or 'N/A'} | {m.get('itrans_mae') or 'N/A'} | {df_mse or 'N/A'} | {m.get('diff_mae') or 'N/A'} | {improvement} |\n")
        else:
            out.write("No final dataset metrics parsed from the log.\n\n")

        out.write("## Last 100 lines (stats and errors)\n\n```")
        out.write(tail100)
        if not tail100.endswith('\n'):
            out.write('\n')
        out.write("```\n")

    reports_created.append(md_path)
    print(f"Wrote report: {md_path}")

# create index
index_path = os.path.join(REPORTS_DIR, "index.md")
with open(index_path, 'w', encoding='utf-8') as idx:
    idx.write("# Generated run reports\n\n")
    for p in sorted(reports_created):
        idx.write(f"- [{os.path.basename(p)}]({os.path.basename(p)})\n")

print(f"Created {len(reports_created)} reports. Index: {index_path}")
