import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(results_dir='results', output_dir='visualizations_generated'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all subdirectories in results_dir
    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        pred_path = os.path.join(model_path, 'pred.npy')
        true_path = os.path.join(model_path, 'true.npy')
        input_path = os.path.join(model_path, 'input.npy')

        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print(f"Skipping {model_name}: pred.npy or true.npy not found.")
            continue
        
        has_input = os.path.exists(input_path)
        if not has_input:
            print(f"Warning: input.npy not found for {model_name}. Lookback will not be visualized. Please re-run test phase.")

        print(f"Processing {model_name}...")
        
        try:
            preds = np.load(pred_path)
            trues = np.load(true_path)
            if has_input:
                inputs = np.load(input_path)
        except Exception as e:
            print(f"Error loading files for {model_name}: {e}")
            continue

        # preds shape: [samples, pred_len, features]
        num_samples = preds.shape[0]
        
        # Choose 5 evenly spaced indices
        indices = np.linspace(0, num_samples - 1, 5, dtype=int)
        
        # Create a subdirectory for this model's visualizations
        model_output_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        for i in indices:
            plt.figure(figsize=(15, 6))
            
            # Plot the last feature (often the target)
            feature_idx = -1 
            
            if has_input:
                input_seq = inputs[i, :, feature_idx]
                seq_len = len(input_seq)
                pred_len = preds.shape[1]
                
                # X-axis for lookback: 0 to seq_len
                # X-axis for prediction: seq_len to seq_len + pred_len
                
                plt.plot(range(seq_len), input_seq, label='Lookback', color='blue', linewidth=2)
                plt.plot(range(seq_len, seq_len + pred_len), trues[i, :, feature_idx], label='GroundTruth', color='green', linewidth=2)
                plt.plot(range(seq_len, seq_len + pred_len), preds[i, :, feature_idx], label='Prediction', color='red', linestyle='--', linewidth=2)
                
                # Add a vertical line at the boundary
                plt.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.5)
                
            else:
                plt.plot(trues[i, :, feature_idx], label='GroundTruth', color='green', linewidth=2)
                plt.plot(preds[i, :, feature_idx], label='Prediction', color='red', linestyle='--', linewidth=2)

            plt.title(f'{model_name}\nSample {i} - Feature {feature_idx}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            
            save_path = os.path.join(model_output_dir, f'sample_{i}.png')
            plt.savefig(save_path)
            plt.close()
            
        print(f"Saved visualizations to {model_output_dir}")

if __name__ == "__main__":
    visualize_results()
