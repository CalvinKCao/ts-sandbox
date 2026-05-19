#!/usr/bin/env python3
import os
import re
import sys
import hashlib
import datetime
from collections import defaultdict

def is_log_file(filename):
    lower = filename.lower()
    base = os.path.basename(filename)
    ext = os.path.splitext(base)[1]
    if ext in {'.log', '.out', '.txt'}:
        return True
    if any(x in lower for x in ['log', 'stdout', 'stderr']):
        return True
    if ext == '':
        return True
    return False

def find_numeric_tokens(s):
    return re.findall(r'(\d{5,})', s)

def pick_run_id(path_parts):
    candidates = []
    for part in path_parts:
        nums = find_numeric_tokens(part)
        for n in nums:
            candidates.append(n)
    if candidates:
        # Pick longest, then largest
        candidates.sort(key=lambda x: (-len(x), -int(x)))
        return candidates[0]
    # No numeric token, fallback
    rel_path = os.path.relpath(os.path.join(*path_parts), start=repo_root)
    sha1 = hashlib.sha1(rel_path.encode()).hexdigest()
    return f'NP-{sha1}'

def safe_count_lines(path):
    try:
        with open(path, 'rb') as f:
            # Read in binary, count newlines
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
        return None

def safe_tail_lines(path, n):
    try:
        with open(path, 'rb') as f:
            # Efficient tail
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b''
            lines = []
            while size > 0 and len(lines) <= n:
                read_size = min(block, size)
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
                lines = data.splitlines()
            # decode lines
            lines = [l.decode(errors='replace') for l in lines[-n:]]
            return lines
    except Exception as e:
        print(f"[WARN] Could not tail {path}: {e}", file=sys.stderr)
        return []

def main():
    global repo_root
    repo_root = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(repo_root, 'results')
    if not os.path.isdir(results_dir):
        print(f"[ERROR] No results directory found at {results_dir}", file=sys.stderr)
        sys.exit(1)
    log_files = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if is_log_file(f):
                log_files.append(full_path)
    scanned = len(log_files)
    file_line_counts = {}
    short_filtered = 0
    for path in log_files:
        nlines = safe_count_lines(path)
        if nlines is None or nlines < 100:
            short_filtered += 1
            continue
        file_line_counts[path] = nlines
    # Assign run ids
    runid_to_files = defaultdict(list)
    for path, nlines in file_line_counts.items():
        rel = os.path.relpath(path, repo_root)
        parts = rel.split(os.sep)
        runid = pick_run_id(parts)
        runid_to_files[runid].append((path, nlines))
    distinct_ids = len(runid_to_files)
    # Pick representative log per run id
    runid_representative = {}
    for runid, files in runid_to_files.items():
        files.sort(key=lambda x: -x[1])
        runid_representative[runid] = files[0]
    # Analyze logs
    failed = []
    possible_success = []
    fail_keywords = re.compile(r'(error|exception|traceback|failed|segmentation fault|oom|killed)', re.I)
    metric_regex = re.compile(r'(final|metric|mae|mse|loss|accuracy|acc|eval|score|best|val_)', re.I)
    runid_info = {}
    for runid, (path, nlines) in runid_representative.items():
        tail = safe_tail_lines(path, 1000)
        joined_tail = '\n'.join(tail)
        if fail_keywords.search(joined_tail):
            failed.append((runid, path))
            runid_info[runid] = {'status': 'FAILED', 'path': path, 'lines': nlines}
        else:
            # Extract metrics
            metrics = [line for line in tail if metric_regex.search(line)]
            if metrics:
                excerpt = '\n'.join(metrics[-20:])
            else:
                excerpt = '\n'.join(tail[-60:])
            possible_success.append((runid, path, nlines, excerpt))
            runid_info[runid] = {'status': 'POSSIBLE_SUCCESS', 'path': path, 'lines': nlines, 'excerpt': excerpt}
    # Write markdown report
    utcnow = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    report_dir = os.path.join(repo_root, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'updated_runs_summary_manual_{utcnow}.md')
    with open(report_path, 'w', encoding='utf-8') as out:
        out.write(f"# Run Summary Report\n\n")
        out.write(f"Generated: {utcnow} UTC\n\n")
        out.write(f"- Candidate logs scanned: {scanned}\n")
        out.write(f"- Filtered out by short length: {short_filtered}\n")
        out.write(f"- Distinct run ids: {distinct_ids}\n")
        out.write(f"- POSSIBLE_SUCCESS runs: {len(possible_success)}\n\n")
        out.write(f"## POSSIBLE_SUCCESS Runs\n\n")
        for runid, path, nlines, excerpt in possible_success:
            rel = os.path.relpath(path, repo_root)
            out.write(f"### Run ID: {runid}\n")
            out.write(f"- Log: `{rel}`\n")
            out.write(f"- Lines: {nlines}\n")
            out.write(f"- Excerpt:\n")
            out.write('```\n')
            out.write(excerpt.strip() + '\n')
            out.write('```\n\n')
        out.write(f"## FAILED Runs\n\n")
        for runid, path in failed:
            rel = os.path.relpath(path, repo_root)
            out.write(f"- Run ID: {runid}, Log: `{rel}` — **FAILED**\n")
    print(f"{os.path.relpath(report_path, repo_root)} scanned={scanned} short-filtered={short_filtered} distinct-ids={distinct_ids} possible-success={len(possible_success)} failed={len(failed)}")

if __name__ == '__main__':
    main()
