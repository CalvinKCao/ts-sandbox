import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
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
    batch_size = 4
    epochs = 5
    learning_rate = 1e-4
    
    # 3. Initialize Model
    print("Initializing ViTime model...")
    try:
        predictor = InferenceInterface(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    model = predictor.model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Simple MSE loss for fine-tuning

    # 4. Prepare Training Data
    # Create sliding windows
    total_len = len(data)
    # Use first 80% for training (or just a subset for "few shot" / "train a bit")
    # Let's use a small subset to simulate "few shot" or quick fine-tuning
    train_len = int(total_len * 0.1) # 10% as mentioned in paper summary
    train_data = data[:train_len]
    
    windows = []
    for i in range(0, len(train_data) - seq_len - pred_len, pred_len): # Stride = pred_len
        windows.append(train_data[i : i + seq_len + pred_len])
    
    windows = np.array(windows)
    print(f"Created {len(windows)} training samples.")
    
    # 5. Training Loop
    print("Starting fine-tuning...")
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(windows)
        
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            if len(batch) == 0: continue
            
            # Prepare batch data
            # We need to replicate the preprocessing steps from inference
            
            # 1. Interpolate/Reshape
            # Input x is (B, seq_len + pred_len)
            # We treat the whole sequence as "history" for the purpose of image generation, 
            # but we mask the future part in the model if needed.
            # However, for training, we want the model to reconstruct the whole image or predict the future.
            # The model takes the whole sequence as input image.
            
            batch_x = batch[:, :seq_len]
            batch_y = batch # Full sequence
            
            # Interpolation (simplified, assuming fixed size for now or using predictor's method)
            # predictor._interpolate_sequence expects (T, C)
            # We loop over batch
            
            batch_imgs = []
            
            for j in range(len(batch)):
                sample_x = batch_x[j].reshape(-1, 1)
                sample_y = batch_y[j].reshape(-1, 1)
                
                # Interpolate to model's expected input size
                # target_total_length = predictor.seq_len + predictor.pred_len
                # But here our data is already seq_len + pred_len.
                # Let's assume the model's seq_len/pred_len matches our data config (512/96)
                # If not, we might need to adjust.
                # predictor.seq_len is likely 512. predictor.pred_len is likely 720 (default).
                # We need to match the model's expected dimensions.
                
                target_total_length = predictor.seq_len + predictor.pred_len
                
                # We need to stretch our (512+96) data to (512+720) or whatever the model expects?
                # Or we just use the model's native resolution.
                # Let's check predictor.args.size
                
                # Actually, let's just use the predictor's helper to convert to pixels
                # We need to normalize first.
                
                # Interpolate sample_y to target_total_length
                sample_y_interp = predictor._interpolate_sequence(sample_y, target_total_length)
                
                # Normalize
                mu = np.mean(sample_y_interp, axis=0)
                std = np.std(sample_y_interp, axis=0) + 1e-7
                sample_y_norm = (sample_y_interp - mu) / std
                
                # Convert to pixel
                # _data2pixel(dataX, dataY)
                # It uses dataY for the full image. dataX is just for shape?
                # Let's look at _data2pixel again. It uses dataY to create imgY0.
                
                _, imgY0, _ = predictor._data2pixel(sample_y_norm, sample_y_norm)
                # imgY0 shape: (C, T, H) -> (1, T, H)
                
                # Apply Gaussian Blur
                kernel_size = (31, 31)
                imgY0[0] = cv2.GaussianBlur(imgY0[0], kernel_size, 0) * kernel_size[0]
                
                batch_imgs.append(imgY0)
            
            batch_imgs = np.array(batch_imgs) # (B, C, T, H)
            batch_tensor = torch.from_numpy(batch_imgs).float().to(predictor.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # The model expects (B, C, T, H)
            # output = model(x, temparture=1)
            output = model(batch_tensor, temparture=1)
            
            # Loss
            # We want the model to reconstruct the image.
            # In "masked" training, we would mask parts.
            # Here, we are fine-tuning. We can use the reconstruction loss on the whole image 
            # or just the future part.
            # Let's use the whole image reconstruction for simplicity/stability.
            
            loss = criterion(output, batch_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(windows)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 6. Save Fine-tuned Model
    finetuned_path = os.path.join(current_dir, 'ViTime_Model_Finetuned.pth')
    # We need to save it in a way that InferenceInterface can load it.
    # InferenceInterface loads: checkpoint = torch.load(model_path); args = checkpoint['args']
    # So we need to wrap it.
    
    checkpoint = {
        'args': predictor.args,
        'model': model.state_dict() # Wait, InferenceInterface loads state_dict into model?
        # Let's check local_model_predictor.py again.
        # It does: model.load_state_dict(state_dict, strict=False)
        # But where does it get state_dict?
        # checkpoint = torch.load(...)
        # if 'model' in checkpoint: state_dict = checkpoint['model']
    }
    torch.save(checkpoint, finetuned_path)
    print(f"Fine-tuned model saved to {finetuned_path}")
    
    # 7. Visualize Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(current_dir, 'finetune_loss.png'))
    print("Loss plot saved.")

if __name__ == "__main__":
    main()
