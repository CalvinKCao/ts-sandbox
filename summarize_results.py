"""
Summarize 7-Variate Pipeline Results.

Reads per-subset results.json files from subdirectories and produces
a markdown report including diffusion metrics and iTransformer baseline
comparison side-by-side.

Usage:
    python summarize_results.py                          # Default: ./synced_results/
    python summarize_results.py --results-dir ./my_results/results_7var
    python summarize_results.py --output report.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: str) -> list:
    """Load results.json from each per-subset subdirectory."""
    results = []
    root = Path(results_dir)
    for subset_dir in sorted(root.iterdir()):
        if not subset_dir.is_dir():
            continue
        rfile = subset_dir / 'results.json'
        if not rfile.exists():
            continue
        try:
            with open(rfile) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"Warning: could not load {rfile}: {e}", file=sys.stderr)
    return results


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def format_report(results: list) -> str:
    lines = []
    lines.append("# Diffusion TSF — 7-Variate Pipeline Results\n")

    if not results:
        lines.append("**No results found.**\n")
        return "\n".join(lines)

    has_baseline = any('itransformer_metrics' in r for r in results)
    n_with_eval = sum(1 for r in results if 'eval_metrics' in r)
    n_with_baseline = sum(1 for r in results if 'itransformer_metrics' in r)

    lines.append(f"**Models with diffusion eval:** {n_with_eval}  ")
    lines.append(f"**Models with iTransformer baseline:** {n_with_baseline}\n")

    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r.get('dataset', 'unknown')].append(r)

    # =========================================================
    # Per-Dataset Summary
    # =========================================================
    if has_baseline:
        lines.append("## Per-Dataset Summary\n")
        lines.append("| Dataset | N | Diff MSE (avg) | Diff MAE (avg) | iTrans MSE | iTrans MAE | ΔMSE | Winner |")
        lines.append("|---------|---|----------------|----------------|------------|------------|------|--------|")
    else:
        lines.append("## Per-Dataset Summary\n")
        lines.append("| Dataset | N | Avg MSE (single) | Avg MAE (single) | Avg MSE (30-avg) | Avg MAE (30-avg) | Avg Trend Acc |")
        lines.append("|---------|---|------------------|------------------|------------------|------------------|---------------|")

    all_diff_mse, all_diff_mae = [], []
    all_itrans_mse, all_itrans_mae = [], []

    for dataset in sorted(by_dataset.keys()):
        subsets = by_dataset[dataset]
        n = len(subsets)

        diff_mse = [r['eval_metrics']['averaged']['mse'] for r in subsets if 'eval_metrics' in r]
        diff_mae = [r['eval_metrics']['averaged']['mae'] for r in subsets if 'eval_metrics' in r]
        s_mse    = [r['eval_metrics']['single']['mse']   for r in subsets if 'eval_metrics' in r]
        s_mae    = [r['eval_metrics']['single']['mae']   for r in subsets if 'eval_metrics' in r]
        t_acc    = [r['eval_metrics']['averaged'].get('trend_accuracy', 0) for r in subsets if 'eval_metrics' in r]
        i_mse    = [r['itransformer_metrics']['mse'] for r in subsets if 'itransformer_metrics' in r]
        i_mae    = [r['itransformer_metrics']['mae'] for r in subsets if 'itransformer_metrics' in r]

        if not diff_mse:
            continue

        avg_dm = sum(diff_mse) / len(diff_mse)
        avg_dma = sum(diff_mae) / len(diff_mae)

        all_diff_mse.extend(diff_mse)
        all_diff_mae.extend(diff_mae)

        if has_baseline and i_mse:
            avg_im = sum(i_mse) / len(i_mse)
            avg_ima = sum(i_mae) / len(i_mae)
            delta = avg_dm - avg_im
            winner = "**Diffusion**" if delta < 0 else "iTransformer"
            lines.append(f"| {dataset} | {n} | {avg_dm:.4f} | {avg_dma:.4f} | {avg_im:.4f} | {avg_ima:.4f} | {delta:+.4f} | {winner} |")
            all_itrans_mse.extend(i_mse)
            all_itrans_mae.extend(i_mae)
        elif has_baseline:
            lines.append(f"| {dataset} | {n} | {avg_dm:.4f} | {avg_dma:.4f} | — | — | — | — |")
        else:
            avg_sm = sum(s_mse) / len(s_mse)
            avg_sma = sum(s_mae) / len(s_mae)
            avg_ta = sum(t_acc) / len(t_acc) if t_acc else 0
            lines.append(f"| {dataset} | {n} | {avg_sm:.4f} | {avg_sma:.4f} | {avg_dm:.4f} | {avg_dma:.4f} | {avg_ta:.3f} |")

    # Overall row
    if all_diff_mse:
        n_total = len(all_diff_mse)
        odm = sum(all_diff_mse) / n_total
        odma = sum(all_diff_mae) / n_total
        if has_baseline and all_itrans_mse:
            oim = sum(all_itrans_mse) / len(all_itrans_mse)
            oima = sum(all_itrans_mae) / len(all_itrans_mae)
            od = odm - oim
            winner = "**Diffusion**" if od < 0 else "iTransformer"
            lines.append(f"| **OVERALL** | **{n_total}** | **{odm:.4f}** | **{odma:.4f}** | **{oim:.4f}** | **{oima:.4f}** | **{od:+.4f}** | {winner} |")
        else:
            lines.append(f"| **OVERALL** | **{n_total}** | | | **{odm:.4f}** | **{odma:.4f}** | |")

    lines.append("")

    # =========================================================
    # Best / Worst (diffusion averaged MSE)
    # =========================================================
    lines.append("## Best & Worst Models (diffusion avg MSE)\n")
    scored = [(r['subset_id'], r['eval_metrics']['averaged']['mse'])
              for r in results if 'eval_metrics' in r]
    scored.sort(key=lambda x: x[1])

    lines.append("### Top 5 Best")
    lines.append("| Rank | Subset | Diff MSE |")
    lines.append("|------|--------|----------|")
    for i, (name, mse) in enumerate(scored[:5], 1):
        lines.append(f"| {i} | {name} | {mse:.4f} |")

    lines.append("")
    lines.append("### Top 5 Worst")
    lines.append("| Rank | Subset | Diff MSE |")
    lines.append("|------|--------|----------|")
    for i, (name, mse) in enumerate(scored[-5:], 1):
        lines.append(f"| {i} | {name} | {mse:.4f} |")

    lines.append("")

    # =========================================================
    # Detailed Per-Dataset Breakdown
    # =========================================================
    lines.append("## Detailed Per-Dataset Breakdown\n")

    for dataset in sorted(by_dataset.keys()):
        subsets = by_dataset[dataset]
        lines.append(f"### {dataset} ({len(subsets)} subset{'s' if len(subsets) > 1 else ''})\n")

        if len(subsets) <= 10:
            if has_baseline:
                lines.append("| Subset | Diff MSE | Diff MAE | iTrans MSE | iTrans MAE | ΔMSE | Trend Acc | Val Loss |")
                lines.append("|--------|----------|----------|------------|------------|------|-----------|----------|")
            else:
                lines.append("| Subset | Single MSE | Single MAE | Avg MSE | Avg MAE | Trend Acc | Val Loss |")
                lines.append("|--------|------------|------------|---------|---------|-----------|----------|")

            for r in sorted(subsets, key=lambda x: x.get('subset_id', '')):
                sid = r.get('subset_id', '?')
                em = r.get('eval_metrics', {})
                tm = r.get('train_metrics', {})
                s = em.get('single', {})
                a = em.get('averaged', {})
                val_loss = tm.get('best_val_loss', 0) or 0
                it = r.get('itransformer_metrics', {})

                if has_baseline:
                    diff_mse = a.get('mse', 0)
                    diff_mae = a.get('mae', 0)
                    i_mse = it.get('mse')
                    i_mae = it.get('mae')
                    if i_mse is not None:
                        delta = f"{diff_mse - i_mse:+.4f}"
                        lines.append(f"| {sid} | {diff_mse:.4f} | {diff_mae:.4f} | "
                                     f"{i_mse:.4f} | {i_mae:.4f} | {delta} | "
                                     f"{a.get('trend_accuracy', 0):.3f} | {val_loss:.4f} |")
                    else:
                        lines.append(f"| {sid} | {diff_mse:.4f} | {diff_mae:.4f} | "
                                     f"— | — | — | "
                                     f"{a.get('trend_accuracy', 0):.3f} | {val_loss:.4f} |")
                else:
                    lines.append(f"| {sid} | {s.get('mse', 0):.4f} | {s.get('mae', 0):.4f} | "
                                 f"{a.get('mse', 0):.4f} | {a.get('mae', 0):.4f} | "
                                 f"{a.get('trend_accuracy', 0):.3f} | {val_loss:.4f} |")
        else:
            mses = [r['eval_metrics']['averaged']['mse'] for r in subsets if 'eval_metrics' in r]
            maes = [r['eval_metrics']['averaged']['mae'] for r in subsets if 'eval_metrics' in r]
            i_mses = [r['itransformer_metrics']['mse'] for r in subsets if 'itransformer_metrics' in r]
            if mses:
                lines.append(f"- **Diffusion MSE:** min={min(mses):.4f}, max={max(mses):.4f}, "
                             f"mean={sum(mses)/len(mses):.4f}, std={_std(mses):.4f}")
                lines.append(f"- **Diffusion MAE:** min={min(maes):.4f}, max={max(maes):.4f}, "
                             f"mean={sum(maes)/len(maes):.4f}, std={_std(maes):.4f}")
            if i_mses:
                avg_dm = sum(mses) / len(mses)
                avg_im = sum(i_mses) / len(i_mses)
                lines.append(f"- **iTransformer MSE:** min={min(i_mses):.4f}, max={max(i_mses):.4f}, "
                             f"mean={avg_im:.4f} ({'better' if avg_im < avg_dm else 'worse'} than diffusion by {abs(avg_dm - avg_im):.4f})")

        lines.append("")

    # =========================================================
    # HP Summary
    # =========================================================
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
            lines.append(f"- **Batch sizes:** {dict(Counter(bss))}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize 7-Variate pipeline results')
    parser.add_argument('--results-dir', type=str, default='./synced_results',
                        help='Directory containing per-subset subdirs with results.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Save report to file (default: print to stdout)')
    args = parser.parse_args()

    results_dir = args.results_dir

    # Auto-locate: check common subdirectory names
    if not any(p.is_dir() for p in Path(results_dir).iterdir() if (p / 'results.json').exists()):
        for subdir in ['results_7var', 'results']:
            candidate = os.path.join(results_dir, subdir)
            if Path(candidate).exists() and any(
                (p / 'results.json').exists() for p in Path(candidate).iterdir() if p.is_dir()
            ):
                results_dir = candidate
                break

    results = load_results(results_dir)

    if not results:
        print(f"No per-subset results.json files found in {results_dir}")
        print(f"Expected structure: {results_dir}/{{subset_id}}/results.json")
        print(f"\nSync first:")
        print(f"  ./sync_results.sh ccao87@narval.alliancecan.ca /lustre06/project/6054110/diffusion-tsf")
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
