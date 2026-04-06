#!/usr/bin/env bash
# Comprehensive search for training results on the remote cluster (Traffic & Electricity).
# Looks for: per-subset results (diffusion+iTransformer), full-dim iTransformer baseline,
# and any training logs/checkpoints related to traffic or electricity.
# Usage: ssh ccao87@killarney.alliancecan.ca 'bash -s' < find_results.sh

set -euo pipefail

SEP="========================================================"
USER=${USER:-ccao87}

# ── candidate storage roots ───────────────────────────────────────────────────
ROOTS=()

# Scratch (job working dir)
[ -d "/scratch/$USER/ts-sandbox" ]                     && ROOTS+=("/scratch/$USER/ts-sandbox")

# Project storage — try both account prefixes and the known lustre path
for base in \
    "$HOME/projects/aip-boyuwang/$USER" \
    "$HOME/projects/def-boyuwang/$USER" \
    "/lustre06/project/6054110/$USER" \
    "/lustre07/project/6054110/$USER" \
    "/project/6054110/$USER" \
; do
    [ -d "$base" ] && ROOTS+=("$base")
done

# Home copy of the repo (fallback)
[ -d "$HOME/ts-sandbox" ] && ROOTS+=("$HOME/ts-sandbox")

echo "$SEP"
echo "Searching as user: $USER"
echo "Candidate storage roots found:"
for r in "${ROOTS[@]:-}"; do echo "  $r"; done
echo "$SEP"

# ── helper: pretty-print a results.json ──────────────────────────────────────
print_result() {
    local f="$1"
    python3 - "$f" <<'PYEOF'
import json, sys
f = sys.argv[1]
try:
    d = json.load(open(f))
    sid    = d.get("subset_id", d.get("dataset", "?"))
    nvar   = len(d.get("variate_indices", []))
    em     = d.get("eval_metrics", {})
    avg    = em.get("averaged", em.get("avg", {}))
    single = em.get("single", {})
    it     = d.get("itransformer_metrics", {})
    bl     = d.get("baseline_metrics", {})  # full-dim iTransformer baseline if present
    parts = [f"subset={sid}  vars={nvar}"]
    if avg:
        parts.append(f"diff_avg_mse={avg.get('mse','?'):.4f}  diff_avg_mae={avg.get('mae','?'):.4f}")
    if single:
        parts.append(f"diff_single_mse={single.get('mse','?'):.4f}")
    if it:
        parts.append(f"itrans_mse={it.get('mse','?'):.4f}  itrans_mae={it.get('mae','?'):.4f}")
    if bl:
        parts.append(f"BASELINE_full_itrans_mse={bl.get('mse','?'):.4f}")
    print("  " + " | ".join(parts))
except Exception as e:
    print(f"  [parse error: {e}]  file: {f}")
PYEOF
}

# ── 1. results.json files ────────────────────────────────────────────────────
echo ""
echo "── 1. results.json files mentioning traffic or electricity ──────────"
found_results=0
for root in "${ROOTS[@]:-}"; do
    while IFS= read -r -d '' f; do
        dir=$(dirname "$f")
        # only include files where the path or content relates to traffic or electricity
        if echo "$dir" | grep -qiE "traffic|electricity" || python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    s = str(d).lower()
    sys.exit(0 if ('traffic' in s or 'electricity' in s) else 1)
except: sys.exit(1)
" "$f" 2>/dev/null; then
            echo "  $f"
            print_result "$f"
            found_results=1
        fi
    done < <(find "$root" -name "results.json" -print0 2>/dev/null)
done
[ $found_results -eq 0 ] && echo "  (none found)"

# ── 2. full-dim iTransformer baseline checkpoints ────────────────────────────
echo ""
echo "── 2. full-dim iTransformer baseline checkpoints ─────────────────────"
found_baseline=0
for root in "${ROOTS[@]:-}"; do
    while IFS= read -r -d '' f; do
        echo "  $f  ($(du -sh "$f" 2>/dev/null | cut -f1))"
        found_baseline=1
    done < <(find "$root" -type f \( \
        -path "*/traffic-baseline*" \
        -o -path "*/electricity-baseline*" \
        -o -name "itransformer_full*" \
        -o -name "*baseline*traffic*" \
        -o -name "*traffic*baseline*" \
        -o -name "*baseline*electricity*" \
        -o -name "*electricity*baseline*" \
    \) -print0 2>/dev/null)
done
[ $found_baseline -eq 0 ] && echo "  (none found)"

# ── 3. subset checkpoint directories ─────────────────────────────────────────
echo ""
echo "── 3. traffic/electricity subset checkpoint dirs ────────────────────"
found_ckpt=0
for root in "${ROOTS[@]:-}"; do
    while IFS= read -r -d '' d; do
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        nvar=""
        meta="$d/metadata.json"
        if [ -f "$meta" ]; then
            nvar=$(python3 -c "
import json
try:
    d=json.load(open('$meta'))
    print(f\"{len(d.get('variate_indices',[]))}var\")
except: print('')
" 2>/dev/null)
        fi
        echo "  $d  [$nvar, $size]"
        found_ckpt=1
    done < <(find "$root" -type d \( -name "traffic-*" -o -name "electricity-*" \) -print0 2>/dev/null)
done
[ $found_ckpt -eq 0 ] && echo "  (none found)"

# ── 4. training manifests ─────────────────────────────────────────────────────
echo ""
echo "── 4. training_manifest.json files ──────────────────────────────────"
found_manifest=0
for root in "${ROOTS[@]:-}"; do
    while IFS= read -r -d '' f; do
        echo "  $f"
        python3 - "$f" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    subs = d.get("subsets", {})
    relevant_subs = {k: v for k, v in subs.items() if any(x in k.lower() for x in ["traffic", "electricity"])}
    if relevant_subs:
        print(f"    relevant subsets in manifest: {list(relevant_subs.keys())}")
        for k, v in relevant_subs.items():
            print(f"      {k}: status={v.get('status','?')}  mse={v.get('metrics',{}).get('mse','?')}")
    else:
        non_relevant = list(subs.keys())[:5]
        print(f"    no traffic/electricity entries (first 5 keys: {non_relevant})")
except Exception as e:
    print(f"    [parse error: {e}]")
PYEOF
        found_manifest=1
    done < <(find "$root" -name "training_manifest.json" -print0 2>/dev/null)
done
[ $found_manifest -eq 0 ] && echo "  (none found)"

# ── 5. Slurm output logs ──────────────────────────────────────────────────────
echo ""
echo "── 5. Slurm .out logs with traffic/electricity results ──────────────"
found_logs=0
for root in "${ROOTS[@]:-}"; do
    while IFS= read -r -d '' f; do
        if grep -qiE "traffic|electricity" "$f" 2>/dev/null; then
            echo "  $f  ($(wc -l < "$f") lines)"
            # Show any lines with results
            grep -iE "(traffic|electricity).*(mse|mae|complete|baseline)|\[(traffic|electricity)" "$f" 2>/dev/null \
                | tail -20 \
                | sed 's/^/    /'
            found_logs=1
        fi
    done < <(find "$root" /scratch/$USER "$HOME" -maxdepth 3 -name "*.out" -newer /etc/hostname -print0 2>/dev/null | head -z -20)
done
# also check home dir directly
while IFS= read -r -d '' f; do
    if grep -qiE "traffic|electricity" "$f" 2>/dev/null; then
        echo "  $f  ($(wc -l < "$f") lines)"
        grep -iE "(traffic|electricity).*(mse|mae|complete|baseline)|\[(traffic|electricity)" "$f" 2>/dev/null \
            | tail -10 | sed 's/^/    /'
        found_logs=1
    fi
done < <(find "$HOME" -maxdepth 2 -name "*.out" -print0 2>/dev/null)
[ $found_logs -eq 0 ] && echo "  (none found)"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo "Done. Paths searched:"
for r in "${ROOTS[@]:-}"; do echo "  $r"; done
echo "$SEP"
