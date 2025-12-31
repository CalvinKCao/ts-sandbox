"""
ETTm2 Dataset Loader for TTS-GAN
Loads the ETTm2 time series dataset and prepares it for GAN training
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ETTm2Dataset(Dataset):
    """
    ETTm2 Dataset for time series generation with TTS-GAN
    
    Args:
        data_path: Path to ETTm2.csv file
        seq_len: Length of each time series sequence
        data_mode: 'Train', 'Val', or 'Test'
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.1)
        test_ratio: Ratio of data for testing (default: 0.2)
        normalize: Whether to normalize the data
        stride: Stride for creating overlapping sequences
        features: Which features to use ('M' for multivariate, 'S' for single target)
        target: Target column name for single variate (default: 'OT')
    """
    
    def __init__(self, 
                 data_path='../../datasets/ETT-small/ETTm2.csv',
                 seq_len=96,
                 data_mode='Train',
                 train_ratio=0.7,
                 val_ratio=0.1,
                 test_ratio=0.2,
                 normalize=True,
                 stride=1,
                 features='M',
                 target='OT'):
        
        self.data_path = data_path
        self.seq_len = seq_len
        self.data_mode = data_mode
        self.normalize = normalize
        self.stride = stride
        self.features = features
        self.target = target
        
        # Load data
        self._load_data(train_ratio, val_ratio, test_ratio)
        
    def _load_data(self, train_ratio, val_ratio, test_ratio):
        """Load and preprocess the ETTm2 dataset"""
        
        # Read CSV file
        df = pd.read_csv(self.data_path)
        
        # Remove date column
        df = df.drop(columns=['date'])
        
        # Select features
        if self.features == 'M':
            # Multivariate: use all features
            data = df.values
            self.channels = data.shape[1]
        elif self.features == 'S':
            # Single variate: use only target
            data = df[[self.target]].values
            self.channels = 1
        else:
            raise ValueError(f"Unknown features type: {self.features}")
        
        # Split data
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # IMPORTANT: Fit scaler only on training data
        if self.normalize:
            self.scaler = StandardScaler()
            train_data = data[:train_end]
            self.scaler.fit(train_data)
        
        # Select data split
        if self.data_mode == 'Train':
            data = data[:train_end]
        elif self.data_mode == 'Val':
            data = data[train_end:val_end]
        elif self.data_mode == 'Test':
            data = data[val_end:]
        else:
            raise ValueError(f"Unknown data_mode: {self.data_mode}")
        
        # Apply normalization (using training statistics for all splits)
        if self.normalize:
            data = self.scaler.transform(data)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(data) - self.seq_len + 1, self.stride):
            seq = data[i:i + self.seq_len]
            self.sequences.append(seq)
        
        self.sequences = np.array(self.sequences)
        
        print(f"{self.data_mode} - Number of sequences: {len(self.sequences)}")
        print(f"{self.data_mode} - Sequence shape: {self.sequences.shape}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: Shape (channels, 1, seq_len) to match TTS-GAN format
            label: Dummy label (0) for compatibility with GAN training loop
        """
        seq = self.sequences[idx]  # Shape: (seq_len, channels)
        
        # Transpose to (channels, seq_len)
        seq = seq.T
        
        # Add height dimension: (channels, 1, seq_len)
        seq = seq[:, np.newaxis, :]
        
        # Convert to tensor
        seq = torch.FloatTensor(seq)
        
        # Dummy label for compatibility
        label = 0
        
        return seq, label
    
    def get_data_for_visualization(self, num_samples=100):
        """
        Get a subset of data for visualization purposes
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            numpy array of shape (num_samples, channels, 1, seq_len)
        """
        indices = np.random.choice(len(self.sequences), min(num_samples, len(self.sequences)), replace=False)
        samples = []
        
        for idx in indices:
            seq, _ = self.__getitem__(idx)
            samples.append(seq.numpy())
        
        return np.array(samples)
    
    def inverse_transform(self, data):
        """
        Inverse transform normalized data back to original scale
        
        Args:
            data: Normalized data array
            
        Returns:
            Data in original scale
        """
        if self.normalize and hasattr(self, 'scaler'):
            # data shape: (batch, channels, 1, seq_len)
            # Need to reshape for scaler
            original_shape = data.shape
            
            # Reshape to (batch * seq_len, channels)
            if len(data.shape) == 4:
                batch, channels, _, seq_len = data.shape
                data = data.squeeze(2).transpose(0, 2, 1)  # (batch, seq_len, channels)
                data = data.reshape(-1, channels)
                data = self.scaler.inverse_transform(data)
                data = data.reshape(batch, seq_len, channels)
                data = data.transpose(0, 2, 1)[:, :, np.newaxis, :]
            else:
                # Handle 2D or 3D cases
                data = self.scaler.inverse_transform(data.reshape(-1, self.channels))
                data = data.reshape(original_shape)
            
            return data
        else:
            return data


def get_ettm2_dataloader(data_path='../../datasets/ETT-small/ETTm2.csv',
                         seq_len=96,
                         batch_size=64,
                         data_mode='Train',
                         num_workers=4,
                         shuffle=True,
                         train_ratio=0.7,
                         val_ratio=0.1,
                         test_ratio=0.2,
                         normalize=True,
                         stride=1,
                         features='M',
                         target='OT'):
    """
    Get DataLoader for ETTm2 dataset
    
    Returns:
        DataLoader instance
    """
    dataset = ETTm2Dataset(
        data_path=data_path,
        seq_len=seq_len,
        data_mode=data_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=normalize,
        stride=stride,
        features=features,
        target=target
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader, dataset


if __name__ == '__main__':
    # Test the dataloader
    print("Testing ETTm2 DataLoader...")
    
    train_loader, train_dataset = get_ettm2_dataloader(
        seq_len=96,
        batch_size=32,
        data_mode='Train',
        features='M'
    )
    
    print(f"\nDataset channels: {train_dataset.channels}")
    print(f"Dataset sequence length: {train_dataset.seq_len}")
    
    # Test one batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nBatch shape: {data.shape}")
        print(f"Data min: {data.min():.4f}, max: {data.max():.4f}, mean: {data.mean():.4f}")
        break
    
    print("\nDataLoader test completed successfully!")
