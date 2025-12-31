"""
Visualization Module for TTS-GAN Time Series Generation
Provides reusable methods for visualizing generated vs real time series data
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Optional, List, Tuple, Union
import os
from datetime import datetime


class TimeSeriesVisualizer:
    """
    A comprehensive visualization toolkit for time series generation with GANs
    
    Features:
    - Compare real vs generated time series
    - Visualize specific test horizons
    - Random sampling with seed control
    - PCA and t-SNE embeddings
    - Feature-wise comparisons
    - Statistical analysis plots
    """
    
    def __init__(self, save_dir='visualizations', feature_names=None):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save visualization outputs
            feature_names: List of feature names for labeling
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.feature_names = feature_names
        
    def plot_time_series_comparison(self,
                                    real_data: np.ndarray,
                                    generated_data: np.ndarray,
                                    num_samples: int = 5,
                                    num_features: Optional[int] = None,
                                    save_name: str = 'ts_comparison',
                                    random_seed: Optional[int] = None,
                                    specific_indices: Optional[List[int]] = None,
                                    title_prefix: str = ''):
        """
        Plot comparison between real and generated time series
        
        Args:
            real_data: Real time series data (N, channels, 1, seq_len) or (N, seq_len, channels)
            generated_data: Generated time series data (same shape as real_data)
            num_samples: Number of samples to visualize
            num_features: Number of features to plot (None = all)
            save_name: Name for saved figure
            random_seed: Random seed for sampling (None = no seeding)
            specific_indices: Specific sample indices to plot (overrides random sampling)
            title_prefix: Prefix for the plot title
        """
        # Reshape data if needed
        real_data = self._prepare_data(real_data)
        generated_data = self._prepare_data(generated_data)
        
        batch_size, seq_len, num_channels = real_data.shape
        
        if num_features is None:
            num_features = num_channels
        else:
            num_features = min(num_features, num_channels)
        
        # Select samples
        if specific_indices is not None:
            indices = specific_indices[:num_samples]
        else:
            if random_seed is not None:
                np.random.seed(random_seed)
            indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
        
        # Create subplots
        fig, axes = plt.subplots(num_samples, num_features, figsize=(4*num_features, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_features == 1:
            axes = axes.reshape(-1, 1)
        
        for i, idx in enumerate(indices):
            for j in range(num_features):
                ax = axes[i, j]
                
                # Plot real and generated
                ax.plot(real_data[idx, :, j], label='Real', alpha=0.7, linewidth=2)
                ax.plot(generated_data[idx, :, j], label='Generated', alpha=0.7, linewidth=2, linestyle='--')
                
                # Labels
                if i == 0:
                    feature_name = self.feature_names[j] if self.feature_names else f'Feature {j}'
                    ax.set_title(feature_name, fontsize=12, fontweight='bold')
                
                if j == 0:
                    ax.set_ylabel(f'Sample {idx}', fontsize=10)
                
                if i == num_samples - 1:
                    ax.set_xlabel('Time Step', fontsize=10)
                
                if i == 0 and j == 0:
                    ax.legend(loc='upper right')
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title_prefix}Real vs Generated Time Series', fontsize=16, fontweight='bold', y=1.002)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series comparison to {save_path}")
        plt.close()
        
        return fig
    
    def plot_random_samples(self,
                           real_data: np.ndarray,
                           generated_data: np.ndarray,
                           num_samples: int = 10,
                           random_seed: int = 42,
                           save_name: str = 'random_samples'):
        """
        Plot random samples from real and generated data
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            num_samples: Number of random samples to plot
            random_seed: Random seed for reproducibility
            save_name: Name for saved figure
        """
        return self.plot_time_series_comparison(
            real_data=real_data,
            generated_data=generated_data,
            num_samples=num_samples,
            save_name=save_name,
            random_seed=random_seed,
            title_prefix='Random Samples: '
        )
    
    def plot_specific_horizons(self,
                              real_data: np.ndarray,
                              generated_data: np.ndarray,
                              horizon_indices: List[int],
                              save_name: str = 'specific_horizons'):
        """
        Plot specific test horizon samples
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            horizon_indices: List of specific indices to visualize
            save_name: Name for saved figure
        """
        return self.plot_time_series_comparison(
            real_data=real_data,
            generated_data=generated_data,
            num_samples=len(horizon_indices),
            save_name=save_name,
            specific_indices=horizon_indices,
            title_prefix='Specific Horizons: '
        )
    
    def plot_feature_distributions(self,
                                   real_data: np.ndarray,
                                   generated_data: np.ndarray,
                                   num_features: Optional[int] = None,
                                   save_name: str = 'feature_distributions'):
        """
        Plot distribution comparison for each feature
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            num_features: Number of features to plot (None = all)
            save_name: Name for saved figure
        """
        real_data = self._prepare_data(real_data)
        generated_data = self._prepare_data(generated_data)
        
        _, _, num_channels = real_data.shape
        
        if num_features is None:
            num_features = num_channels
        else:
            num_features = min(num_features, num_channels)
        
        # Create subplots
        ncols = min(3, num_features)
        nrows = int(np.ceil(num_features / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        
        if num_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for j in range(num_features):
            ax = axes[j]
            
            # Flatten the data for this feature
            real_feature = real_data[:, :, j].flatten()
            gen_feature = generated_data[:, :, j].flatten()
            
            # Plot histograms
            ax.hist(real_feature, bins=50, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(gen_feature, bins=50, alpha=0.5, label='Generated', density=True, color='orange')
            
            feature_name = self.feature_names[j] if self.feature_names else f'Feature {j}'
            ax.set_title(feature_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(num_features, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Feature Distribution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature distributions to {save_path}")
        plt.close()
        
        return fig
    
    def plot_pca_embedding(self,
                          real_data: np.ndarray,
                          generated_data: np.ndarray,
                          n_components: int = 2,
                          save_name: str = 'pca_embedding'):
        """
        Plot PCA embedding of real vs generated data
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            n_components: Number of PCA components (2 or 3)
            save_name: Name for saved figure
        """
        real_data = self._prepare_data(real_data)
        generated_data = self._prepare_data(generated_data)
        
        # Flatten sequences for PCA
        real_flat = real_data.reshape(real_data.shape[0], -1)
        gen_flat = generated_data.reshape(generated_data.shape[0], -1)
        
        # Combine data
        combined = np.vstack([real_flat, gen_flat])
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        embedded = pca.fit_transform(combined)
        
        # Split back
        n_real = real_flat.shape[0]
        real_embedded = embedded[:n_real]
        gen_embedded = embedded[n_real:]
        
        # Plot
        if n_components == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.scatter(real_embedded[:, 0], real_embedded[:, 1], 
                      alpha=0.5, label='Real', s=20)
            ax.scatter(gen_embedded[:, 0], gen_embedded[:, 1], 
                      alpha=0.5, label='Generated', s=20)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        else:  # 3D
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(real_embedded[:, 0], real_embedded[:, 1], real_embedded[:, 2],
                      alpha=0.5, label='Real', s=20)
            ax.scatter(gen_embedded[:, 0], gen_embedded[:, 1], gen_embedded[:, 2],
                      alpha=0.5, label='Generated', s=20)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
            ax.legend(fontsize=12)
        
        plt.title('PCA Embedding: Real vs Generated', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA embedding to {save_path}")
        plt.close()
        
        return fig
    
    def plot_tsne_embedding(self,
                           real_data: np.ndarray,
                           generated_data: np.ndarray,
                           perplexity: int = 30,
                           n_iter: int = 1000,
                           random_seed: int = 42,
                           save_name: str = 'tsne_embedding'):
        """
        Plot t-SNE embedding of real vs generated data
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for t-SNE
            random_seed: Random seed for reproducibility
            save_name: Name for saved figure
        """
        real_data = self._prepare_data(real_data)
        generated_data = self._prepare_data(generated_data)
        
        # Flatten sequences for t-SNE
        real_flat = real_data.reshape(real_data.shape[0], -1)
        gen_flat = generated_data.reshape(generated_data.shape[0], -1)
        
        # Combine data
        combined = np.vstack([real_flat, gen_flat])
        
        # Apply t-SNE (use max_iter for newer scikit-learn versions)
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_seed)
        except TypeError:
            # Fallback for older versions that use n_iter
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
        embedded = tsne.fit_transform(combined)
        
        # Split back
        n_real = real_flat.shape[0]
        real_embedded = embedded[:n_real]
        gen_embedded = embedded[n_real:]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(real_embedded[:, 0], real_embedded[:, 1], 
                  alpha=0.5, label='Real', s=20, color='blue')
        ax.scatter(gen_embedded[:, 0], gen_embedded[:, 1], 
                  alpha=0.5, label='Generated', s=20, color='orange')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.title('t-SNE Embedding: Real vs Generated', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE embedding to {save_path}")
        plt.close()
        
        return fig
    
    def plot_statistical_summary(self,
                                real_data: np.ndarray,
                                generated_data: np.ndarray,
                                save_name: str = 'statistical_summary'):
        """
        Plot statistical summary comparing real and generated data
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            save_name: Name for saved figure
        """
        real_data = self._prepare_data(real_data)
        generated_data = self._prepare_data(generated_data)
        
        _, _, num_channels = real_data.shape
        
        # Calculate statistics
        stats = {
            'Mean': (real_data.mean(axis=(0, 1)), generated_data.mean(axis=(0, 1))),
            'Std': (real_data.std(axis=(0, 1)), generated_data.std(axis=(0, 1))),
            'Min': (real_data.min(axis=(0, 1)), generated_data.min(axis=(0, 1))),
            'Max': (real_data.max(axis=(0, 1)), generated_data.max(axis=(0, 1)))
        }
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        feature_labels = self.feature_names if self.feature_names else [f'F{i}' for i in range(num_channels)]
        x = np.arange(num_channels)
        width = 0.35
        
        for idx, (stat_name, (real_stat, gen_stat)) in enumerate(stats.items()):
            ax = axes[idx]
            ax.bar(x - width/2, real_stat, width, label='Real', alpha=0.8)
            ax.bar(x + width/2, gen_stat, width, label='Generated', alpha=0.8)
            ax.set_xlabel('Features', fontsize=11)
            ax.set_ylabel(stat_name, fontsize=11)
            ax.set_title(f'{stat_name} Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Statistical Summary: Real vs Generated', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistical summary to {save_path}")
        plt.close()
        
        return fig
    
    def generate_full_report(self,
                            real_data: np.ndarray,
                            generated_data: np.ndarray,
                            num_samples: int = 5,
                            random_seed: int = 42,
                            report_name: str = 'full_report'):
        """
        Generate a comprehensive visualization report
        
        Args:
            real_data: Real time series data
            generated_data: Generated time series data
            num_samples: Number of samples for time series plots
            random_seed: Random seed for reproducibility
            report_name: Base name for report files
        """
        print(f"\nGenerating comprehensive visualization report...")
        print(f"Save directory: {self.save_dir}")
        
        # Time series comparison
        print("1. Creating time series comparison...")
        self.plot_random_samples(real_data, generated_data, num_samples, 
                                random_seed, f'{report_name}_random_samples')
        
        # Feature distributions
        print("2. Creating feature distributions...")
        self.plot_feature_distributions(real_data, generated_data, 
                                       save_name=f'{report_name}_distributions')
        
        # PCA embedding
        print("3. Creating PCA embedding...")
        self.plot_pca_embedding(real_data, generated_data, 
                               save_name=f'{report_name}_pca')
        
        # t-SNE embedding
        print("4. Creating t-SNE embedding...")
        self.plot_tsne_embedding(real_data, generated_data, random_seed=random_seed,
                                save_name=f'{report_name}_tsne')
        
        # Statistical summary
        print("5. Creating statistical summary...")
        self.plot_statistical_summary(real_data, generated_data, 
                                     save_name=f'{report_name}_stats')
        
        print(f"\nFull report generated in {self.save_dir}/")
        print(f"Report prefix: {report_name}")
    
    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare data to consistent shape (batch, seq_len, channels)
        
        Args:
            data: Input data in various shapes
            
        Returns:
            Data in shape (batch, seq_len, channels)
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Handle different input shapes
        if len(data.shape) == 4:
            # (batch, channels, 1, seq_len) -> (batch, seq_len, channels)
            data = data.squeeze(2).transpose(0, 2, 1)
        elif len(data.shape) == 3 and data.shape[1] < data.shape[2]:
            # (batch, channels, seq_len) -> (batch, seq_len, channels)
            data = data.transpose(0, 2, 1)
        # If already (batch, seq_len, channels), leave as is
        
        return data


# Example usage and testing
if __name__ == '__main__':
    print("Testing TimeSeriesVisualizer...")
    
    # Create dummy data
    batch_size, seq_len, channels = 100, 96, 7
    real_data = np.random.randn(batch_size, channels, 1, seq_len)
    generated_data = np.random.randn(batch_size, channels, 1, seq_len) * 1.1 + 0.1
    
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(
        save_dir='test_visualizations',
        feature_names=feature_names
    )
    
    # Generate full report
    visualizer.generate_full_report(
        real_data=real_data,
        generated_data=generated_data,
        num_samples=3,
        random_seed=42,
        report_name='test_report'
    )
    
    print("\nVisualizer test completed!")
