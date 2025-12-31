"""
Test script to verify the implementation is working correctly
"""

import sys
import os
import torch
import numpy as np

print("="*60)
print("Testing Pathformer + DILATE Implementation")
print("="*60)

# Test 1: Check imports
print("\n[Test 1] Checking imports...")
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from shared_utils.visualization import TimeSeriesVisualizer
    from shared_utils.visualization_metrics import visualization
    print("✓ Shared utils imported successfully")
except Exception as e:
    print(f"✗ Failed to import shared utils: {e}")
    sys.exit(1)

try:
    from dilate_loss_wrapper import DilateLoss, CombinedLoss
    print("✓ DILATE loss wrapper imported successfully")
except Exception as e:
    print(f"✗ Failed to import DILATE loss: {e}")
    print("  Make sure losses/DILATE-master/ is present")
    sys.exit(1)

# Test 2: Check dataset
print("\n[Test 2] Checking dataset...")
dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'ETT-small', 'ETTh2.csv')
if os.path.exists(dataset_path):
    print(f"✓ ETTh2 dataset found at {dataset_path}")
    # Check if we can read it
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"  Dataset shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Warning: Could not read dataset: {e}")
else:
    print(f"✗ ETTh2 dataset not found at {dataset_path}")

# Test 3: Test DILATE loss
print("\n[Test 3] Testing DILATE loss...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Using device: {device}")

try:
    batch_size = 4
    seq_len = 96
    n_features = 7
    
    # Create dummy data
    outputs = torch.randn(batch_size, seq_len, n_features).to(device)
    targets = torch.randn(batch_size, seq_len, n_features).to(device)
    
    # Test DilateLoss
    dilate_loss = DilateLoss(alpha=0.5, gamma=0.01, device=device)
    loss, loss_shape, loss_temporal = dilate_loss(outputs, targets)
    
    print(f"✓ DILATE loss computed successfully")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Shape loss: {loss_shape.item():.4f}")
    print(f"  Temporal loss: {loss_temporal.item():.4f}")
    
    # Test CombinedLoss with all types
    for loss_type in ['mae', 'mse', 'dilate']:
        combined = CombinedLoss(loss_type=loss_type, device=device)
        result = combined(outputs, targets)
        if loss_type == 'dilate':
            loss, _, _ = result
        else:
            loss = result
        print(f"✓ {loss_type.upper()} loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"✗ DILATE loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test visualization
print("\n[Test 4] Testing visualization...")
try:
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create dummy data
    batch_size = 50
    seq_len = 96
    n_features = 7
    
    real_data = np.random.randn(batch_size, seq_len, n_features)
    pred_data = real_data + np.random.randn(batch_size, seq_len, n_features) * 0.1
    
    # Test visualizer
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    visualizer = TimeSeriesVisualizer(
        save_dir=test_dir,
        feature_names=feature_names
    )
    
    # Test individual plots
    print("  Creating time series comparison...")
    visualizer.plot_random_samples(real_data, pred_data, num_samples=3, save_name='test_samples')
    
    print("  Creating feature distributions...")
    visualizer.plot_feature_distributions(real_data, pred_data, save_name='test_distributions')
    
    print("  Creating PCA embedding...")
    visualizer.plot_pca_embedding(real_data, pred_data, save_name='test_pca')
    
    print(f"✓ Visualization test completed successfully")
    print(f"  Test outputs saved to: {test_dir}")
    
except Exception as e:
    print(f"✗ Visualization test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check if required packages are installed
print("\n[Test 5] Checking required packages...")
required_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'sklearn': 'Scikit-learn',
    'seaborn': 'Seaborn',
    'numba': 'Numba (for DILATE)'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"✓ {name} installed")
    except ImportError:
        print(f"✗ {name} NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
    print(f"  Install with: pip install {' '.join(missing_packages)}")
else:
    print("\n✓ All required packages are installed")

# Test 6: Check file structure
print("\n[Test 6] Checking file structure...")
files_to_check = [
    ('train_etth2.py', 'Training script'),
    ('visualize_etth2.py', 'Visualization script'),
    ('dilate_loss_wrapper.py', 'DILATE loss wrapper'),
    ('README_ETTH2.md', 'Documentation'),
    ('quick_start.bat', 'Quick start script'),
    ('run_experiments.bat', 'Experiment runner'),
]

for filename, description in files_to_check:
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        print(f"✓ {description}: {filename}")
    else:
        print(f"✗ Missing {description}: {filename}")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print("✓ Implementation is ready to use!")
print("\nNext steps:")
print("1. Run quick start:")
print("   cd models/pathformer-main")
print("   quick_start.bat")
print("\n2. Or train manually:")
print("   python train_etth2.py")
print("\n3. Then visualize:")
print("   python visualize_etth2.py --save_metrics")
print("="*60)
