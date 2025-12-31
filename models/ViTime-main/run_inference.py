import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from local_model_predictor import InferenceInterface

def main():
    # 1. Check for the pretrained model
    model_path = os.path.join(current_dir, 'ViTime_Model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Pretrained model not found at {model_path}")
        print("Please download the model from the link in README.md and place it in this directory.")
        print("Download Link: https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view?usp=sharing")
        return

    # 2. Load the dataset (ETTh1.csv)
    # We use ETTh1 as an example of a dataset the model was not trained on (zero-shot)
    dataset_path = os.path.join(current_dir, '../../datasets/ETT-small/ETTh1.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Use the 'OT' column for univariate forecasting
    # Take the last 512 points as history context
    seq_len = 512
    pred_len = 96 # Forecast horizon
    
    data = df['OT'].values
    if len(data) < seq_len:
        print(f"Error: Dataset length {len(data)} is shorter than sequence length {seq_len}")
        return

    # Select a sample: let's take a window that allows us to have ground truth for verification
    # We need seq_len for history + pred_len for ground truth
    total_needed = seq_len + pred_len
    
    if len(data) < total_needed:
        print(f"Error: Dataset length {len(data)} is shorter than required {total_needed}")
        return

    # Take the last window
    history_data = data[-(total_needed):-pred_len]
    ground_truth = data[-pred_len:]
    
    print(f"Input history shape: {history_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # 3. Initialize the model
    print("Initializing ViTime model...")
    try:
        predictor = InferenceInterface(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 4. Run inference
    print("Running inference...")
    try:
        # inference(self, x, prediction_length=None, sampleNumber=None, tempature=1)
        # x should be (seq_len, 1) or (seq_len,)
        forecast, prob_map = predictor.inference(history_data, prediction_length=pred_len, return_prob_map=True)
        
        # The output shape from inference is (total_length, channels, samples)
        # We need to extract the prediction part
        # total_length = seq_len + pred_len
        
        print("Inference complete.")
        print(f"Output shape: {forecast.shape}")
        
        # Extract the predicted part (last pred_len points)
        # forecast shape is (total_length, 1)
        if len(forecast.shape) == 2:
             pred_data = forecast[-pred_len:, 0]
        elif len(forecast.shape) == 3:
             pred_data = forecast[-pred_len:, 0, 0]
        else:
             pred_data = forecast[-pred_len:]

        print("Predicted values (first 5):")
        print(pred_data[:5])
        
        # Save results
        output_file = os.path.join(current_dir, 'inference_result.npy')
        np.save(output_file, pred_data)
        print(f"Prediction saved to {output_file}")

        # Visualization
        print("Generating visualization...")
        plt.figure(figsize=(12, 6))
        
        # Plot history (last 100 points for clarity)
        plt.plot(range(seq_len-100, seq_len), history_data[-100:], label='History (Last 100)', color='blue')
        
        # Plot Ground Truth
        plt.plot(range(seq_len, seq_len + pred_len), ground_truth, label='Ground Truth', color='green')
        
        # Plot Prediction
        plt.plot(range(seq_len, seq_len + pred_len), pred_data, label='Prediction', color='red', linestyle='--')
        
        plt.title(f'ViTime Zero-Shot Forecast on ETTh1 (OT)\nSeq Len: {seq_len}, Pred Len: {pred_len}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        viz_file = os.path.join(current_dir, 'forecast_visualization.png')
        plt.savefig(viz_file)
        print(f"Visualization saved to {viz_file}")
        plt.close()

        # Probability map visualization (time on x-axis, value bins on y-axis)
        prob_viz_file = os.path.join(current_dir, 'probability_map.png')
        plt.figure(figsize=(12, 4))
        plt.imshow(prob_map.T, aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(label='Probability')
        plt.xlabel('Time')
        plt.ylabel('Value bin')
        plt.title('ViTime probability map')
        plt.tight_layout()
        plt.savefig(prob_viz_file)
        print(f"Probability map saved to {prob_viz_file}")
        plt.close()



    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
