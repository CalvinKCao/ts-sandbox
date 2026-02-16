"""
Summarize 7-Variate Pipeline Results.

Reads synced JSON results and produces a nicely formatted markdown summary.

Usage:
    python summarize_results.py                          # Default: ./synced_results/
    python summarize_results.py --results-dir ./my_results/
    python summarize_results.py --output report.md       # Save to file
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

def load_results(results_dir: str) -> list:
    """Load all *_results.json files."""
    results = []
    for f in sorted(Path(results_dir).glob("*_results.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return results


def format_report(results: list) -> str:
    """Generate a markdown report from results."""
    lines = []
    lines.append("# Diffusion TSF — 7-Variate Pipeline Results\n")
    
    if not results:
        lines.append("**No results found.**\n")
        return "\n".join(lines)
    
    lines.append(f"**Total models evaluated:** {len(results)}\n")
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results:
        dataset = r.get('dataset', 'unknown')
        by_dataset[dataset].append(r)
    
    # ==========================================
    # Per-Dataset Summary Table
    # ==========================================
    lines.append("## Per-Dataset Summary\n")
    lines.append("| Dataset | Subsets | Avg MSE (single) | Avg MAE (single) | Avg MSE (30-avg) | Avg MAE (30-avg) | Avg Trend Acc |")
    lines.append("|---------|---------|-------------------|-------------------|-------------------|-------------------|---------------|")
    
    all_single_mse = []
    all_single_mae = []
    all_avg_mse = []
    all_avg_mae = []
    all_trend_acc = []
    
    dataset_summaries = []
    
    for dataset in sorted(by_dataset.keys()):
        subsets = by_dataset[dataset]
        n = len(subsets)
        
        s_mse = [r['eval_metrics']['single']['mse'] for r in subsets if 'eval_metrics' in r]
        s_mae = [r['eval_metrics']['single']['mae'] for r in subsets if 'eval_metrics' in r]
        a_mse = [r['eval_metrics']['averaged']['mse'] for r in subsets if 'eval_metrics' in r]
        a_mae = [r['eval_metrics']['averaged']['mae'] for r in subsets if 'eval_metrics' in r]
        t_acc = [r['eval_metrics']['averaged'].get('trend_accuracy', 0) for r in subsets if 'eval_metrics' in r]
        
        if not s_mse:
            continue
        
        avg_s_mse = sum(s_mse) / len(s_mse)
        avg_s_mae = sum(s_mae) / len(s_mae)
        avg_a_mse = sum(a_mse) / len(a_mse)
        avg_a_mae = sum(a_mae) / len(a_mae)
        avg_t_acc = sum(t_acc) / len(t_acc) if t_acc else 0
        
        lines.append(f"| {dataset} | {n} | {avg_s_mse:.4f} | {avg_s_mae:.4f} | {avg_a_mse:.4f} | {avg_a_mae:.4f} | {avg_t_acc:.3f} |")
        
        all_single_mse.extend(s_mse)
        all_single_mae.extend(s_mae)
        all_avg_mse.extend(a_mse)
        all_avg_mae.extend(a_mae)
        all_trend_acc.extend(t_acc)
        
        dataset_summaries.append({
            'dataset': dataset, 'n': n,
            'single_mse': avg_s_mse, 'single_mae': avg_s_mae,
            'avg_mse': avg_a_mse, 'avg_mae': avg_a_mae,
            'trend_acc': avg_t_acc,
        })
    
    # Overall average
    if all_single_mse:
        lines.append(f"| **OVERALL** | **{len(results)}** | **{sum(all_single_mse)/len(all_single_mse):.4f}** | **{sum(all_single_mae)/len(all_single_mae):.4f}** | **{sum(all_avg_mse)/len(all_avg_mse):.4f}** | **{sum(all_avg_mae)/len(all_avg_mae):.4f}** | **{sum(all_trend_acc)/len(all_trend_acc):.3f}** |")
    
    lines.append("")
    
    # ==========================================
    # Best / Worst models
    # ==========================================
    lines.append("## Best & Worst Models (by averaged MSE)\n")
    
    scored = [(r['subset_id'], r['eval_metrics']['averaged']['mse']) 
              for r in results if 'eval_metrics' in r]
    scored.sort(key=lambda x: x[1])
    
    lines.append("### Top 5 Best")
    lines.append("| Rank | Subset | MSE |")
    lines.append("|------|--------|-----|")
    for i, (name, mse) in enumerate(scored[:5], 1):
        lines.append(f"| {i} | {name} | {mse:.4f} |")
    
    lines.append("")
    lines.append("### Top 5 Worst")
    lines.append("| Rank | Subset | MSE |")
    lines.append("|------|--------|-----|")
    for i, (name, mse) in enumerate(scored[-5:], 1):
        lines.append(f"| {i} | {name} | {mse:.4f} |")
    
    lines.append("")
    
    # ==========================================
    # Per-Dataset Detailed Breakdown
    # ==========================================
    lines.append("## Detailed Per-Dataset Breakdown\n")
    
    for dataset in sorted(by_dataset.keys()):
        subsets = by_dataset[dataset]
        lines.append(f"### {dataset} ({len(subsets)} subset{'s' if len(subsets) > 1 else ''})\n")
        
        if len(subsets) <= 10:
            # Show all subsets
            lines.append("| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |")
            lines.append("|--------|------------|------------|---------|---------|-----------|----------|")
            
            for r in sorted(subsets, key=lambda x: x.get('subset_id', '')):
                sid = r.get('subset_id', '?')
                em = r.get('eval_metrics', {})
                tm = r.get('train_metrics', {})
                
                s = em.get('single', {})
                a = em.get('averaged', {})
                val_loss = tm.get('best_val_loss', 0)
                
                lines.append(f"| {sid} | {s.get('mse', 0):.4f} | {s.get('mae', 0):.4f} | "
                           f"{a.get('mse', 0):.4f} | {a.get('mae', 0):.4f} | "
                           f"{a.get('trend_accuracy', 0):.3f} | {val_loss:.4f} |")
        else:
            # Too many subsets, show summary stats
            mses = [r['eval_metrics']['averaged']['mse'] for r in subsets if 'eval_metrics' in r]
            maes = [r['eval_metrics']['averaged']['mae'] for r in subsets if 'eval_metrics' in r]
            
            if mses:
                lines.append(f"- **MSE:** min={min(mses):.4f}, max={max(mses):.4f}, "
                           f"mean={sum(mses)/len(mses):.4f}, std={_std(mses):.4f}")
                lines.append(f"- **MAE:** min={min(maes):.4f}, max={max(maes):.4f}, "
                           f"mean={sum(maes)/len(maes):.4f}, std={_std(maes):.4f}")
        
        lines.append("")
    
    # ==========================================
    # HP Tuning Summary (if available)
    # ==========================================
    hp_results = [r for r in results if r.get('train_metrics', {}).get('tuned_params')]
    if hp_results:
        lines.append("## Hyperparameter Summary\n")
        
        lrs = [r['train_metrics']['tuned_params']['learning_rate'] for r in hp_results 
               if 'learning_rate' in r.get('train_metrics', {}).get('tuned_params', {})]
        bss = [r['train_metrics']['tuned_params']['batch_size'] for r in hp_results
               if 'batch_size' in r.get('train_metrics', {}).get('tuned_params', {})]
        
        if lrs:
            lines.append(f"- **Learning rates:** min={min(lrs):.2e}, max={max(lrs):.2e}, median={sorted(lrs)[len(lrs)//2]:.2e}")
        if bss:
            from collections import Counter
            bs_counts = Counter(bss)
            lines.append(f"- **Batch sizes:** {dict(bs_counts)}")
        lines.append("")
    
    # ==========================================
    # iTransformer Baseline Comparison
    # ==========================================
    baseline_path = os.path.join(results_dir, 'itransformer_baseline.json')
    
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        lines.append("## Diffusion vs iTransformer Baseline\n")
        lines.append("| Dataset | Subsets | Diff MSE | iTrans MSE | Δ MSE | Winner |")
        lines.append("|---------|---------|----------|------------|-------|--------|")
        
        by_ds = defaultdict(lambda: {'diff': [], 'itrans': []})
        
        for subset_id, bdata in baseline.items():
            dataset = bdata.get('dataset', 'unknown')
            itrans_mse = bdata['itransformer_metrics']['mse']
            
            diff_result = next((r for r in results if r.get('subset_id') == subset_id), None)
            if diff_result and 'eval_metrics' in diff_result:
                diff_mse = diff_result['eval_metrics']['averaged']['mse']
                by_ds[dataset]['diff'].append(diff_mse)
                by_ds[dataset]['itrans'].append(itrans_mse)
        
        total_diff = []
        total_itrans = []
        
        for dataset in sorted(by_ds.keys()):
            d = by_ds[dataset]
            n = len(d['diff'])
            avg_diff = sum(d['diff']) / n
            avg_itrans = sum(d['itrans']) / n
            delta = avg_diff - avg_itrans
            winner = "**Diffusion**" if delta < 0 else "iTransformer"
            
            lines.append(f"| {dataset} | {n} | {avg_diff:.4f} | {avg_itrans:.4f} | {delta:+.4f} | {winner} |")
            total_diff.extend(d['diff'])
            total_itrans.extend(d['itrans'])
        
        if total_diff:
            avg_d = sum(total_diff) / len(total_diff)
            avg_i = sum(total_itrans) / len(total_itrans)
            delta = avg_d - avg_i
            winner = "**Diffusion**" if delta < 0 else "iTransformer"
            lines.append(f"| **OVERALL** | **{len(total_diff)}** | **{avg_d:.4f}** | **{avg_i:.4f}** | **{delta:+.4f}** | {winner} |")
        
        lines.append("")
    
    return "\n".join(lines)


def _std(values):
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description='Summarize 7-Variate pipeline results')
    parser.add_argument('--results-dir', type=str, default='./synced_results',
                       help='Directory containing *_results.json files')
    parser.add_argument('--output', type=str, default=None,
                       help='Save report to file (default: print to stdout)')
    args = parser.parse_args()
    
    # Try multiple possible locations
    results_dir = args.results_dir
    if not any(Path(results_dir).glob("*_results.json")):
        # Try subdirectories
        for subdir in ['results', 'results_7var']:
            candidate = os.path.join(results_dir, subdir)
            if any(Path(candidate).glob("*_results.json")):
                results_dir = candidate
                break
    
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        print(f"Sync first: rsync -avz ccao87@narval.alliancecan.ca:/lustre06/project/6054110/diffusion-tsf/results/ ./synced_results/")
        return
    
    report = format_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()

