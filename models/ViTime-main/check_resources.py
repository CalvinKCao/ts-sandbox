import os
import sys
import torch
import numpy as np

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from local_model_predictor import InferenceInterface

def check_model_size():
    model_path = os.path.join(current_dir, 'ViTime_Model.pth')
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Load checkpoint directly to inspect args without initializing the full interface if possible,
        # but InferenceInterface does it conveniently.
        checkpoint = torch.load(model_path, map_location='cpu')
        args = checkpoint['args']
        
        print("\n--- Model Configuration ---")
        # print(f"Model Name: {args.model}") # args.model might not exist
        print(f"Input Size (Seq/Label/Pred): {args.size}")
        print(f"Patch Size: {args.patch_size}")
        print(f"Embed Dim: {args.embed_dim}")
        # print(f"Depth: {args.depth}")
        # print(f"Num Heads: {args.num_heads}")
        # print(f"Decoder Embed Dim: {args.decoder_embed_dim}")
        # print(f"Decoder Depth: {args.decoder_depth}")
        # print(f"Decoder Num Heads: {args.decoder_num_heads}")
        
        # Initialize model to count parameters
        predictor = InferenceInterface(model_path=model_path, device='cpu')
        model = predictor.model
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n--- Parameter Count ---")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Estimate Memory
        # 4 bytes per parameter (float32)
        model_mem_mb = total_params * 4 / (1024**2)
        grad_mem_mb = trainable_params * 4 / (1024**2)
        optimizer_mem_mb = trainable_params * 8 / (1024**2) # Adam keeps 2 states per param
        
        total_static_mem_mb = model_mem_mb + grad_mem_mb + optimizer_mem_mb
        
        print("\n--- Estimated Memory Requirements (Static) ---")
        print(f"Model Weights: {model_mem_mb:.2f} MB")
        print(f"Gradients: {grad_mem_mb:.2f} MB")
        print(f"Optimizer States (Adam): {optimizer_mem_mb:.2f} MB")
        print(f"Total Static Memory (Training): {total_static_mem_mb:.2f} MB")
        print(f"Total Static Memory (Inference): {model_mem_mb:.2f} MB")
        
        print("\n--- Note ---")
        print("This does not include activation memory, which depends on batch size and image size.")
        print("For a 3050 Ti (4GB VRAM), you have limited headroom.")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    check_model_size()
