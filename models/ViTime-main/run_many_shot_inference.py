import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from local_model_predictor import InferenceInterface

def main():
    # 1. Setup paths
    model_path = os.path.join(current_dir, 'ViTime_Model.pth')
    dataset_path = os.path.join(current_dir, '../../datasets/ETT-small/ETTh1.csv')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    


    # 2. Load Data
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    data = df['OT'].values
    
    # Parameters
    seq_len = 512
    pred_len = 96
    num_samples = 4
    
    total_len = len(data)
    # We need at least seq_len + pred_len
    valid_start_indices = range(0, total_len - seq_len - pred_len)
    
    # Select evenly spaced indices
    indices = np.linspace(0, len(valid_start_indices) - 1, num_samples, dtype=int)
    selected_starts = [valid_start_indices[i] for i in indices]
    
    print(f"Selected start indices: {selected_starts}")

    # 3. Initialize Model
    print("Initializing ViTime model...")
    try:
        predictor = InferenceInterface(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 4. Run Inference and Plot
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    results = []

    for i, start_idx in enumerate(selected_starts):
        end_history = start_idx + seq_len
        end_pred = end_history + pred_len
        
        history_data = data[start_idx:end_history]
        ground_truth = data[end_history:end_pred]
        
        print(f"Processing sample {i+1}/{num_samples} (Index {start_idx})...")
        
        try:
            # Run inference and get probability map
            forecast, prob_map = predictor.inference(
                history_data,
                prediction_length=pred_len,
                return_prob_map=True
            )

            # Extract prediction
            # forecast shape: (total_length, 1) or (total_length, 1, 1)
            if len(forecast.shape) == 2:
                pred_data = forecast[-pred_len:, 0]
            elif len(forecast.shape) == 3:
                pred_data = forecast[-pred_len:, 0, 0]
            else:
                pred_data = forecast[-pred_len:]
            
            # Calculate Metrics
            mse = np.mean((pred_data - ground_truth) ** 2)
            mae = np.mean(np.abs(pred_data - ground_truth))
            results.append({'index': start_idx, 'mse': mse, 'mae': mae})
            
            # Plot forecast
            ax = axes[i]
            # Plot last 100 points of history
            ax.plot(range(seq_len-100, seq_len), history_data[-100:], label='History (Last 100)', color='blue')
            # Plot Ground Truth
            ax.plot(range(seq_len, seq_len + pred_len), ground_truth, label='Ground Truth', color='green')
            # Plot Prediction
            ax.plot(range(seq_len, seq_len + pred_len), pred_data, label=f'Prediction (MAE: {mae:.2f})', color='red', linestyle='--')
            
            ax.set_title(f'Sample {i+1} (Start Index: {start_idx})')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

            # Save probability map for this sample (time on x-axis, value bins on y-axis)
            prob_viz_file = os.path.join(current_dir, f'probability_map_sample_{i+1}.png')
            plt.figure(figsize=(10, 3))
            plt.imshow(prob_map.T, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(label='Probability')
            plt.xlabel('Time')
            plt.ylabel('Value bin')
            plt.title(f'Probability map - sample {i+1} (start {start_idx})')
            plt.tight_layout()
            plt.savefig(prob_viz_file)
            plt.close()
            print(f"Probability map saved to {prob_viz_file}")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()

    plt.tight_layout()
    viz_file = os.path.join(current_dir, 'many_shot_visualization.png')
    plt.savefig(viz_file)
    print(f"Visualization saved to {viz_file}")
    
    # Print Summary
    print("\nSummary Results:")
    for res in results:
        print(f"Index {res['index']}: MSE={res['mse']:.4f}, MAE={res['mae']:.4f}")

if __name__ == "__main__":
    main()
