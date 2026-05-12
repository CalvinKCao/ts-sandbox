import os
import re
from datetime import datetime

results_dir = "results"
report_path = "reports/recent_experiment_report.md"

if not os.path.exists("reports"):
    os.makedirs("reports")

all_results = []
dir_pattern = re.compile(r"^\d{2}-\d{2}-(\d+)-(.*)$")

itrans_pattern = re.compile(r"\[([^\]]+)\] iTransformer baseline: MSE=([\d\.]+), MAE=([\d\.]+)")
diff_pattern = re.compile(r"\[([^\]]+)\] Avg: MSE=([\d\.]+), MAE=([\d\.]+)")
time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
started_pattern = re.compile(r"^Started:\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

if os.path.exists(results_dir):
    for d in os.listdir(results_dir):
        match = dir_pattern.match(d)
        if match:
            job_id = int(match.group(1))
            if job_id >= 3515961:
                log_dir = os.path.join(results_dir, d, "logs")
                if not os.path.exists(log_dir):
                    continue
                
                for log_file in os.listdir(log_dir):
                    if not log_file.endswith(".log"):
                        continue
                    
                    log_path = os.path.join(log_dir, log_file)
                    
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        start_time = None
                        end_time = None
                        
                        # Find start time
                        for line in lines[:200]:
                            s_match = started_pattern.search(line)
                            if s_match:
                                # Assume year 2026
                                dt_str = "2026-" + s_match.group(1)
                                start_time = parse_time(dt_str)
                                break
                            
                            t_match = time_pattern.search(line)
                            if t_match:
                                start_time = parse_time(t_match.group(1))
                                break
                                
                        # Find end time
                        for line in reversed(lines):
                            t_match = time_pattern.search(line)
                            if t_match:
                                end_time = parse_time(t_match.group(1))
                                break
                        
                        duration_str = "Unknown"
                        if start_time and end_time and end_time >= start_time:
                            duration = end_time - start_time
                            hours, remainder = divmod(duration.total_seconds(), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

                        # Only look at the last ~150 lines for metrics
                        last_lines = lines[-150:]
                        
                        dataset_metrics = {}
                        
                        for line in reversed(last_lines):
                            itrans_match = itrans_pattern.search(line)
                            if itrans_match:
                                ds = itrans_match.group(1)
                                if ds not in dataset_metrics:
                                    dataset_metrics[ds] = {}
                                dataset_metrics[ds]['itrans_mse'] = itrans_match.group(2)
                                dataset_metrics[ds]['itrans_mae'] = itrans_match.group(3)
                            
                            diff_match = diff_pattern.search(line)
                            if diff_match:
                                ds = diff_match.group(1)
                                if ds not in dataset_metrics:
                                    dataset_metrics[ds] = {}
                                dataset_metrics[ds]['diff_mse'] = diff_match.group(2)
                                dataset_metrics[ds]['diff_mae'] = diff_match.group(3)
                        
                        # If we couldn't parse any datasets from the end, we still want to show the run exists
                        if not dataset_metrics:
                            all_results.append({
                                "run_dir": d,
                                "job_id": job_id,
                                "dataset": "None (Failed/Incomplete)",
                                "itrans_mse": None,
                                "itrans_mae": None,
                                "diff_mse": None,
                                "diff_mae": None,
                                "duration": duration_str
                            })
                        else:
                            for ds, metrics in dataset_metrics.items():
                                all_results.append({
                                    "run_dir": d,
                                    "job_id": job_id,
                                    "dataset": ds,
                                    "itrans_mse": metrics.get('itrans_mse'),
                                    "itrans_mae": metrics.get('itrans_mae'),
                                    "diff_mse": metrics.get('diff_mse'),
                                    "diff_mae": metrics.get('diff_mae'),
                                    "duration": duration_str
                                })
                            
                    except Exception as e:
                        print(f"Error reading {log_path}: {e}")

with open(report_path, "w") as f:
    f.write("# Recent Experiment Comparison Report (Job ID >= 3515961)\n\n")
    f.write("Comparing iTransformer Baseline vs Diffusion (Avg Ensemble)\n\n")
    
    # Sort by job_id, then run_dir, then dataset
    all_results.sort(key=lambda x: (x["job_id"], x["run_dir"], x["dataset"]))
    
    current_dir = None
    for r in all_results:
        if r["run_dir"] != current_dir:
            current_dir = r["run_dir"]
            # To get duration we can take it from the current record (they should be all same for one run)
            dur = r.get("duration", "Unknown")
            f.write(f"### Run: {current_dir} *(Duration: {dur})*\n\n")
            f.write("| Dataset | iTrans MSE | iTrans MAE | Diffusion MSE | Diffusion MAE | Improvement (MSE) |\n")
            f.write("|---------|------------|------------|---------------|---------------|-------------------|\n")
            
        it_mse = r["itrans_mse"]
        df_mse = r["diff_mse"]
        
        improvement = "N/A"
        if it_mse is not None and df_mse is not None:
            imp = (float(it_mse) - float(df_mse)) / float(it_mse) * 100
            improvement = f"{imp:.2f}%"
            
        it_mse_str = it_mse if it_mse is not None else "N/A"
        it_mae_str = r['itrans_mae'] if r['itrans_mae'] is not None else "N/A"
        df_mse_str = df_mse if df_mse is not None else "N/A"
        df_mae_str = r['diff_mae'] if r['diff_mae'] is not None else "N/A"
            
        f.write(f"| {r['dataset']} | {it_mse_str} | {it_mae_str} | {df_mse_str} | {df_mae_str} | {improvement} |\n")
        
    # List missing or incomplete runs
    processed_dirs = set(r["run_dir"] for r in all_results)
    all_target_dirs = []
    for d in os.listdir(results_dir):
        match = dir_pattern.match(d)
        if match and int(match.group(1)) >= 3515961:
            all_target_dirs.append(d)
            
    missing_dirs = set(all_target_dirs) - processed_dirs
    if missing_dirs:
        f.write("\n## Missing Runs entirely\n\n")
        for md in sorted(list(missing_dirs)):
            f.write(f"- {md}\n")

print(f"Report generated at {report_path}")