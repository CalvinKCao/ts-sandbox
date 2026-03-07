#!/bin/bash
#SBATCH --job-name=diffusion-baseline
#SBATCH --account=def-boyuwang
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca

# iTransformer baseline evaluation job
# Runs the pretrained iTransformer on the same test splits as diffusion eval,
# writing itransformer_baseline.json so summarize_results.py shows the comparison.
#
# Usage:
#   sbatch slurm_baseline_eval.sh                  # all completed subsets
#   sbatch slurm_baseline_eval.sh --subset ETTh1   # single subset

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

# Auto-detect PROJECT
if [ -z "$PROJECT" ]; then
    FIRST_PROJECT=$(ls -d $HOME/projects/def-* 2>/dev/null | head -1)
    [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

export STORAGE_ROOT="$PROJECT/diffusion-tsf"
source "$STORAGE_ROOT/venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0

cd "$HOME/ts-sandbox"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Checkpoint dir: $STORAGE_ROOT/checkpoints"
echo "Results dir:    $STORAGE_ROOT/results"
echo "=========================================="

python -m models.diffusion_tsf.evaluate_7var \
    --baseline \
    --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
    --results-dir "$STORAGE_ROOT/results" \
    "$@"

# Migrate any leftover flat *_results.json files to per-subset subdirs
python3 - <<'PYEOF'
import json, os
from pathlib import Path
root = Path(os.environ["STORAGE_ROOT"] + "/results")
migrated = 0
for f in list(root.glob("*_results.json")):
    try:
        with open(f) as fh:
            data = json.load(fh)
        subset_id = data.get("subset_id") or f.name.replace("_results.json", "")
        if "metrics" in data and "eval_metrics" not in data:
            data["eval_metrics"] = data.pop("metrics")
        subdir = root / subset_id
        subdir.mkdir(exist_ok=True)
        dest = subdir / "results.json"
        if not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime:
            with open(dest, "w") as fh:
                json.dump(data, fh, indent=2)
        f.unlink()
        migrated += 1
    except Exception as e:
        print(f"Warning: could not migrate {f}: {e}")
if migrated:
    print(f"Migrated {migrated} flat JSON files to per-subset subdirs")
PYEOF

EXIT_CODE=$?

echo ""
echo "=========================================="
[ $EXIT_CODE -eq 0 ] && echo "Done: $(date)" || echo "FAILED (exit $EXIT_CODE): $(date)"
echo "Baseline JSON: $STORAGE_ROOT/results/itransformer_baseline.json"
echo "=========================================="
exit $EXIT_CODE
