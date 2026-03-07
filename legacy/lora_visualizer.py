import matplotlib
matplotlib.use('QtAgg')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Set random seed for reproducible "kinks"
torch.manual_seed(42)

# ==========================================
# 1. Define the Base Piecewise Linear Network
# ==========================================
class SimpleMLP(nn.Module):
    """A 1D -> 1D network with ReLU to create a piecewise linear function."""
    def __init__(self, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ==========================================
# 2. Define the LoRA Wrapper
# ==========================================
class LoRAMLP(nn.Module):
    """Wraps the base model and injects low-rank trainable matrices."""
    def __init__(self, base_model, rank=2, alpha=1.0):
        super().__init__()
        self.base = base_model
        self.rank = rank
        self.alpha = alpha
        
        hidden_dim = base_model.fc1.out_features
        
        # LoRA matrices for Layer 1
        # B1: (hidden_dim, rank), A1: (rank, 1)
        self.lora_B1 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.5)
        self.lora_A1 = nn.Parameter(torch.randn(rank, 1) * 0.5)
        
        # LoRA matrices for Layer 2
        # B2: (1, rank), A2: (rank, hidden_dim)
        self.lora_B2 = nn.Parameter(torch.randn(1, rank) * 0.5)
        self.lora_A2 = nn.Parameter(torch.randn(rank, hidden_dim) * 0.5)

    def forward(self, x, current_alpha=None):
        alpha_val = current_alpha if current_alpha is not None else self.alpha
        scale = alpha_val / self.rank

        # Compute Layer 1 with LoRA
        W1_base = self.base.fc1.weight
        b1_base = self.base.fc1.bias
        W1_lora = W1_base + scale * (self.lora_B1 @ self.lora_A1)
        hidden = torch.relu(x @ W1_lora.T + b1_base)

        # Compute Layer 2 with LoRA
        W2_base = self.base.fc2.weight
        b2_base = self.base.fc2.bias
        W2_lora = W2_base + scale * (self.lora_B2 @ self.lora_A2)
        out = hidden @ W2_lora.T + b2_base
        
        return out

# ==========================================
# 3. Setup the Visualization & UI
# ==========================================
def main():
    # Initialize models
    base_model = SimpleMLP(hidden_dim=15)
    lora_model = LoRAMLP(base_model, rank=2)

    # Generate input data space
    X_numpy = np.linspace(-5, 5, 500)
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32).unsqueeze(1)

    # Get base function output (Frozen)
    with torch.no_grad():
        Y_base = base_model(X_tensor).numpy().flatten()

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25) # Make room for the slider
    
    # Plot Base Model
    ax.plot(X_numpy, Y_base, label='Base Model (Frozen)', color='blue', linestyle='--', linewidth=2)
    
    # Plot LoRA Model (Initial state)
    with torch.no_grad():
        Y_lora_init = lora_model(X_tensor, current_alpha=0.0).numpy().flatten()
    lora_line, = ax.plot(X_numpy, Y_lora_init, label='With LoRA Applied', color='red', linewidth=2)

    ax.set_title("Impact of LoRA on a Network's Piecewise Linear Function")
    ax.set_xlabel("Input Space (X)")
    ax.set_ylabel("Network Output (Y)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 10)

    # Setup Slider
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    alpha_slider = Slider(
        ax=ax_slider,
        label=r'LoRA Alpha ($\alpha$)',
        valmin=-10.0,
        valmax=10.0,
        valinit=0.0,
    )

    # Update function for the slider
    def update(val):
        current_alpha = alpha_slider.val
        with torch.no_grad():
            Y_new = lora_model(X_tensor, current_alpha=current_alpha).numpy().flatten()
        lora_line.set_ydata(Y_new)
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update)

    # Reset Button
    ax_reset = plt.axes([0.85, 0.1, 0.1, 0.03])
    reset_btn = Button(ax_reset, 'Reset')
    def reset(event):
        alpha_slider.reset()
    reset_btn.on_clicked(reset)

    plt.show()

if __name__ == "__main__":
    main()