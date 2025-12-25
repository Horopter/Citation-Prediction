"""
PyTorch-based PCA utilities for GPU-friendly dimensionality reduction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TorchPCA:
    """
    PyTorch-based PCA that supports GPU and CPU fallback.
    Uses SVD for dimensionality reduction.
    """
    
    def __init__(self, n_components: int, device: Optional[torch.device] = None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep
            device: torch device (auto-detected if None)
        """
        self.n_components = n_components
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        
    def fit(self, X: np.ndarray) -> 'TorchPCA':
        """
        Fit PCA on data.
        
        Args:
            X: Input data (n_samples, n_features)
        """
        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        # Center the data
        self.mean_ = X_tensor.mean(dim=0, keepdim=True)
        X_centered = X_tensor - self.mean_
        
        # Compute SVD
        U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top n_components
        self.components_ = V[:self.n_components].cpu().numpy()
        self.explained_variance_ = (S[:self.n_components] ** 2 / (X.shape[0] - 1)).cpu().numpy()
        
        return self
    
    def partial_fit(self, X: np.ndarray) -> 'TorchPCA':
        """
        Partial fit for incremental PCA (uses full fit for simplicity).
        For true incremental PCA, would need to implement IPCA algorithm.
        """
        return self.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        # Ensure mean_ is 1D for broadcasting
        mean_tensor = torch.from_numpy(self.mean_.squeeze()).to(self.device)
        components_tensor = torch.from_numpy(self.components_).to(self.device)
        
        # Center and transform
        X_centered = X_tensor - mean_tensor
        X_transformed = torch.matmul(X_centered, components_tensor.T)
        
        return X_transformed.cpu().numpy()
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class IncrementalTorchPCA:
    """
    Incremental PCA using PyTorch for GPU support.
    Uses SVD-based approach for GPU acceleration.
    """
    
    def __init__(self, n_components: int, batch_size: int = 5000, 
                 device: Optional[torch.device] = None):
        """
        Initialize Incremental PCA.
        
        Args:
            n_components: Number of components to keep
            batch_size: Batch size (for compatibility, not used in fit)
            device: torch device (auto-detected if None)
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.components_ = None
        self.mean_ = None
        
    def partial_fit(self, X: np.ndarray) -> 'IncrementalTorchPCA':
        """
        Partial fit (for compatibility). Uses full fit internally.
        For true incremental PCA, would need IPCA algorithm implementation.
        """
        return self.fit(X)
    
    def fit(self, X: np.ndarray) -> 'IncrementalTorchPCA':
        """
        Fit PCA on dataset using PyTorch SVD (GPU-friendly).
        
        Args:
            X: Input data (n_samples, n_features)
        """
        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        # Center the data
        self.mean_ = X_tensor.mean(dim=0, keepdim=True)
        X_centered = X_tensor - self.mean_
        
        # Compute SVD (GPU-accelerated)
        U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top n_components
        self.components_ = V[:self.n_components].cpu().numpy()
        self.mean_ = self.mean_.cpu().numpy()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        # Ensure mean_ is 1D for broadcasting
        mean_tensor = torch.from_numpy(self.mean_.squeeze()).to(self.device)
        components_tensor = torch.from_numpy(self.components_).to(self.device)
        
        # Center and transform
        X_centered = X_tensor - mean_tensor
        X_transformed = torch.matmul(X_centered, components_tensor.T)
        
        return X_transformed.cpu().numpy()
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

