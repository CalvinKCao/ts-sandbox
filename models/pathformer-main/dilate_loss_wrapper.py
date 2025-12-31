"""
DILATE Loss Wrapper for Pathformer
Integrates the DILATE loss (DIstortion and shapeLATEness) for time series forecasting
"""

import torch
import torch.nn as nn
import sys
import os

# Add DILATE loss path to system path
dilate_path = os.path.join(os.path.dirname(__file__), '..', '..', 'losses', 'DILATE-master')
if dilate_path not in sys.path:
    sys.path.insert(0, dilate_path)

# Import DILATE loss modules
try:
    from loss import soft_dtw
    from loss import path_soft_dtw
except ImportError:
    # Try alternative import path
    import importlib.util
    
    # Load soft_dtw
    soft_dtw_path = os.path.join(dilate_path, 'loss', 'soft_dtw.py')
    spec = importlib.util.spec_from_file_location("soft_dtw", soft_dtw_path)
    soft_dtw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soft_dtw)
    
    # Load path_soft_dtw
    path_soft_dtw_path = os.path.join(dilate_path, 'loss', 'path_soft_dtw.py')
    spec = importlib.util.spec_from_file_location("path_soft_dtw", path_soft_dtw_path)
    path_soft_dtw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(path_soft_dtw)


class DilateLoss(nn.Module):
    """
    DILATE Loss for time series forecasting
    
    DILATE = α * Shape Loss + (1-α) * Temporal Loss
    - Shape Loss: Measures shape similarity using soft-DTW
    - Temporal Loss: Penalizes temporal misalignment
    
    Args:
        alpha: Weight for shape loss vs temporal loss (0 to 1)
        gamma: Smoothing parameter for soft-DTW
        device: Device to run computations on
    """
    
    def __init__(self, alpha=0.5, gamma=0.01, device='cuda'):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
    def forward(self, outputs, targets):
        """
        Compute DILATE loss
        
        Args:
            outputs: Predicted sequences (batch_size, seq_len, features)
            targets: Ground truth sequences (batch_size, seq_len, features)
            
        Returns:
            loss: Total DILATE loss
            loss_shape: Shape component
            loss_temporal: Temporal component
        """
        batch_size, N_output, n_features = outputs.shape
        
        # Initialize losses
        loss_shape = 0
        loss_temporal = 0
        
        # Compute loss for each feature dimension
        for feature_idx in range(n_features):
            # Extract single feature: (batch_size, seq_len, 1)
            outputs_feat = outputs[:, :, feature_idx:feature_idx+1]
            targets_feat = targets[:, :, feature_idx:feature_idx+1]
            
            # Compute pairwise distances for soft-DTW
            D = torch.zeros((batch_size, N_output, N_output)).to(self.device)
            for k in range(batch_size):
                Dk = soft_dtw.pairwise_distances(
                    targets_feat[k, :, :].view(-1, 1),
                    outputs_feat[k, :, :].view(-1, 1)
                )
                D[k:k+1, :, :] = Dk
            
            # Shape loss using Soft-DTW
            softdtw_batch = soft_dtw.SoftDTWBatch.apply
            loss_shape += softdtw_batch(D, self.gamma)
            
            # Temporal loss using Path-DTW
            path_dtw = path_soft_dtw.PathDTWBatch.apply
            path = path_dtw(D, self.gamma)
            Omega = soft_dtw.pairwise_distances(
                torch.arange(1, N_output + 1).view(N_output, 1).float()
            ).to(self.device)
            loss_temporal += torch.sum(path * Omega) / (N_output * N_output)
        
        # Average across features
        loss_shape = loss_shape / n_features
        loss_temporal = loss_temporal / n_features
        
        # Combined DILATE loss
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        
        return loss, loss_shape, loss_temporal


class FrequencySelectiveDilateLoss(nn.Module):
    """
    Frequency-Selective DILATE Loss
    
    Applies DILATE loss only to high-frequency components extracted via FFT,
    and regular loss to low-frequency components.
    
    Args:
        base_loss_type: Base loss for low frequencies ('mse' or 'mae')
        alpha: DILATE alpha parameter for high frequencies
        gamma: DILATE gamma parameter for high frequencies
        freq_threshold: Percentile threshold for high frequency (0-100)
                       e.g., 80 means top 20% frequencies are considered high
        use_fft_dilate: If True, apply DILATE to high freq; if False, use regular loss
        device: Device for computation
    """
    
    def __init__(self, base_loss_type='mae', alpha=0.5, gamma=0.01, 
                 freq_threshold=80.0, use_fft_dilate=True, device='cuda'):
        super(FrequencySelectiveDilateLoss, self).__init__()
        self.base_loss_type = base_loss_type.lower()
        self.freq_threshold = freq_threshold
        self.use_fft_dilate = use_fft_dilate
        self.device = device
        
        # Base loss for low frequencies
        if self.base_loss_type == 'mse':
            self.base_criterion = nn.MSELoss()
        elif self.base_loss_type == 'mae':
            self.base_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown base loss type: {base_loss_type}")
        
        # DILATE loss for high frequencies
        self.dilate_criterion = DilateLoss(alpha=alpha, gamma=gamma, device=device)
        
    def extract_frequency_components(self, signal, mask=None):
        """
        Extract high and low frequency components using FFT
        
        Args:
            signal: Input signal (batch_size, seq_len, features)
            mask: Optional boolean mask for frequencies. If None, computed based on frequency index.
            
        Returns:
            high_freq: High frequency component
            low_freq: Low frequency component
            mask: Boolean mask indicating high frequency indices
        """
        batch_size, seq_len, n_features = signal.shape
        
        # Perform FFT along time dimension
        fft_result = torch.fft.rfft(signal, dim=1)
        num_freqs = fft_result.shape[1]
        
        if mask is None:
            # Create mask based on frequency index (strictly correct meaning)
            # freq_threshold is percentile (e.g. 80). 
            # We interpret this as the split point in the spectrum.
            # If freq_threshold is 80, then the first 80% of frequencies are Low, and top 20% are High.
            
            cutoff_idx = int(num_freqs * (self.freq_threshold / 100.0))
            
            mask = torch.zeros(num_freqs, dtype=torch.bool, device=self.device)
            # Mark high frequencies as True
            mask[cutoff_idx:] = True 
        
        # Create filtered signals
        high_freq_fft = fft_result.clone()
        low_freq_fft = fft_result.clone()
        
        # Zero out frequencies based on mask
        # mask is True for High Freq
        high_freq_fft[:, ~mask, :] = 0  # Keep only high frequencies (mask=True)
        low_freq_fft[:, mask, :] = 0    # Keep only low frequencies (mask=False)
        
        # Inverse FFT to get time-domain signals
        high_freq = torch.fft.irfft(high_freq_fft, n=seq_len, dim=1)
        low_freq = torch.fft.irfft(low_freq_fft, n=seq_len, dim=1)
        
        return high_freq, low_freq, mask
    
    def forward(self, outputs, targets):
        """
        Compute frequency-selective loss
        
        Returns:
            total_loss: Combined loss
            low_freq_loss: Loss on low frequencies
            high_freq_loss: Loss on high frequencies (DILATE or base)
            freq_info: Dictionary with frequency analysis info
        """
        # Extract frequency components
        # We generate the mask from targets (though it's index-based now, so it's constant)
        # and apply it to outputs to ensure consistency.
        targets_high, targets_low, freq_mask = self.extract_frequency_components(targets)
        outputs_high, outputs_low, _ = self.extract_frequency_components(outputs, mask=freq_mask)
        
        # Compute loss on low frequencies (base loss)
        low_freq_loss = self.base_criterion(outputs_low, targets_low)
        
        # Compute loss on high frequencies
        if self.use_fft_dilate:
            # Use DILATE loss for high frequencies
            high_freq_loss, shape_loss, temporal_loss = self.dilate_criterion(
                outputs_high, targets_high
            )
            freq_info = {
                'shape_loss': shape_loss.item(),
                'temporal_loss': temporal_loss.item(),
                'high_freq_ratio': freq_mask.float().mean().item()
            }
        else:
            # Use base loss for high frequencies too
            high_freq_loss = self.base_criterion(outputs_high, targets_high)
            freq_info = {
                'high_freq_ratio': freq_mask.float().mean().item()
            }
        
        # Combine losses (weighted by signal energy)
        outputs_high_energy = (outputs_high ** 2).mean()
        outputs_low_energy = (outputs_low ** 2).mean()
        total_energy = outputs_high_energy + outputs_low_energy + 1e-8
        
        high_weight = outputs_high_energy / total_energy
        low_weight = outputs_low_energy / total_energy
        
        total_loss = low_weight * low_freq_loss + high_weight * high_freq_loss
        
        return total_loss, low_freq_loss, high_freq_loss, freq_info


class CombinedLoss(nn.Module):
    """
    Combined loss that can switch between MSE, MAE, DILATE, and FFT-DILATE
    
    Args:
        loss_type: Type of loss ('mse', 'mae', 'dilate', 'fft_dilate')
        alpha: DILATE alpha parameter (only used if loss_type='dilate' or 'fft_dilate')
        gamma: DILATE gamma parameter (only used if loss_type='dilate' or 'fft_dilate')
        freq_threshold: Percentile for high frequency threshold (only for 'fft_dilate')
        device: Device for computation
    """
    
    def __init__(self, loss_type='mae', alpha=0.5, gamma=0.01, 
                 freq_threshold=80.0, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.device = device
        
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'dilate':
            self.criterion = DilateLoss(alpha=alpha, gamma=gamma, device=device)
        elif self.loss_type == 'fft_dilate':
            self.criterion = FrequencySelectiveDilateLoss(
                base_loss_type='mae',
                alpha=alpha,
                gamma=gamma,
                freq_threshold=freq_threshold,
                use_fft_dilate=True,
                device=device
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose 'mse', 'mae', 'dilate', or 'fft_dilate'")
    
    def forward(self, outputs, targets):
        """
        Compute loss
        
        Returns:
            If loss_type is 'dilate': (total_loss, shape_loss, temporal_loss)
            If loss_type is 'fft_dilate': (total_loss, low_freq_loss, high_freq_loss, freq_info)
            Otherwise: loss
        """
        if self.loss_type in ['dilate', 'fft_dilate']:
            return self.criterion(outputs, targets)
        else:
            return self.criterion(outputs, targets)


if __name__ == '__main__':
    # Test the DILATE loss
    print("Testing DILATE Loss...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy data
    batch_size = 4
    seq_len = 96
    n_features = 7
    
    outputs = torch.randn(batch_size, seq_len, n_features).to(device)
    targets = torch.randn(batch_size, seq_len, n_features).to(device)
    
    # Test DILATE loss
    dilate_loss = DilateLoss(alpha=0.5, gamma=0.01, device=device)
    loss, loss_shape, loss_temporal = dilate_loss(outputs, targets)
    
    print(f"\nDILATE Loss: {loss.item():.4f}")
    print(f"Shape Loss: {loss_shape.item():.4f}")
    print(f"Temporal Loss: {loss_temporal.item():.4f}")
    
    # Test Combined Loss with different types
    for loss_type in ['mae', 'mse', 'dilate', 'fft_dilate']:
        print(f"\n--- Testing {loss_type.upper()} loss ---")
        
        if loss_type == 'fft_dilate':
            # Test with different frequency thresholds
            for freq_thresh in [70.0, 80.0, 90.0]:
                print(f"  Frequency threshold: {freq_thresh}%")
                combined_loss = CombinedLoss(
                    loss_type=loss_type, 
                    freq_threshold=freq_thresh,
                    device=device
                )
                result = combined_loss(outputs, targets)
                total, low_loss, high_loss, freq_info = result
                print(f"    Total: {total.item():.4f}, Low-freq: {low_loss.item():.4f}, "
                      f"High-freq: {high_loss.item():.4f}")
                print(f"    High-freq ratio: {freq_info['high_freq_ratio']:.2%}")
                if 'shape_loss' in freq_info:
                    print(f"    Shape loss: {freq_info['shape_loss']:.4f}, "
                          f"Temporal loss: {freq_info['temporal_loss']:.4f}")
        else:
            combined_loss = CombinedLoss(loss_type=loss_type, device=device)
            result = combined_loss(outputs, targets)
            
            if loss_type == 'dilate':
                loss, loss_shape, loss_temporal = result
                print(f"  Total: {loss.item():.4f}, Shape: {loss_shape.item():.4f}, "
                      f"Temporal: {loss_temporal.item():.4f}")
            else:
                print(f"  Loss: {result.item():.4f}")
    
    print("\n" + "="*60)
    print("DILATE Loss test completed!")
    print("="*60)
    print("\nNew FFT-DILATE loss options:")
    print("  --loss_type fft_dilate")
    print("  --freq_threshold <percentile>  (e.g., 80 for top 20% frequencies)")
    print("\nThis applies DILATE loss only to high frequencies,")
    print("and regular MAE loss to low frequencies.")
