
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.realts import RealTS

def visualize():
    # Parameters
    NUM_SAMPLES = 5
    NUM_VARS = 4
    LOOKBACK = 128
    FORECAST = 64
    TOTAL_LEN = LOOKBACK + FORECAST
    
    print(f"Generating {NUM_SAMPLES} samples with {NUM_VARS} variables each...")
    
    # Initialize Dataset
    dataset = RealTS(
        num_samples=NUM_SAMPLES, 
        num_variables=NUM_VARS, 
        lookback_length=LOOKBACK, 
        forecast_length=FORECAST
    )
    
    # Create output directory
    output_dir = os.path.join(current_dir, 'synthetic_vis')
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(NUM_SAMPLES):
        past, future = dataset[i]
        # past: (num_vars, lookback)
        # future: (num_vars, forecast)
        
        # Concatenate for full view
        full_seq = np.concatenate([past.numpy(), future.numpy()], axis=1) # (num_vars, total_len)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Main plot with all variables
        plt.subplot(2, 1, 1)
        for v in range(NUM_VARS):
            plt.plot(full_seq[v], label=f'Var {v}', alpha=0.8)
            
        plt.axvline(x=LOOKBACK, color='k', linestyle='--', label='Forecast Start', alpha=0.5)
        plt.title(f'Multivariate Synthetic Sample {i+1} (Stacked)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        
        # Subplots for individual variables to see details better
        plt.subplot(2, 1, 2)
        offset = 0
        for v in range(NUM_VARS):
            # Normalize to 0-1 range for offset plotting or just offset by constant
            # The data is already normalized by RealTS, so it's roughly N(0,1)
            # We offset them by 4 units to separate them
            series = full_seq[v] + (v * 4)
            plt.plot(series, label=f'Var {v}')
            plt.text(0, v * 4, f'Var {v}', va='center', fontweight='bold')
            
        plt.axvline(x=LOOKBACK, color='k', linestyle='--', alpha=0.5)
        plt.title('Separated Channels')
        plt.yticks([]) # Hide y ticks as they are arbitrary due to offset
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'sample_{i+1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    print("\nDone! Check the 'models/diffusion_tsf/synthetic_vis' directory.")

if __name__ == "__main__":
    visualize()
