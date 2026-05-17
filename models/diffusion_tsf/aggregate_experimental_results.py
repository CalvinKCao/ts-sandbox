import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_experimental")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} not found.")
        return
        
    all_files = [os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir) if f.startswith('results_') and f.endswith('.csv')]
    
    if not all_files:
        print("No results files found.")
        return
        
    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates if any (e.g. if a run was repeated)
    df = df.drop_duplicates(subset=['dataset', 'experiment'], keep='last')
    
    print("="*60)
    print("Experimental Results Aggregation")
    print("="*60)
    
    # Pivot for MSE
    mse_pivot = df.pivot(index='dataset', columns='experiment', values='mse')
    print("\\nMean Squared Error (MSE) Comparison:")
    print("-" * 60)
    print(mse_pivot.to_string())
    
    # Pivot for MAE
    mae_pivot = df.pivot(index='dataset', columns='experiment', values='mae')
    print("\\nMean Absolute Error (MAE) Comparison:")
    print("-" * 60)
    print(mae_pivot.to_string())
    
    # Breakdown for A and A+B
    if 'res_mse' in df.columns and 'trend_mse' in df.columns:
        print("\\nResidual and Trend breakdown for Experiments A and A+B:")
        print("-" * 60)
        breakdown_df = df[df['experiment'].isin(['A', 'A+B'])].dropna(subset=['res_mse', 'trend_mse'])
        if not breakdown_df.empty:
            print(breakdown_df[['dataset', 'experiment', 'mse', 'res_mse', 'trend_mse']].to_string(index=False))

if __name__ == "__main__":
    main()
