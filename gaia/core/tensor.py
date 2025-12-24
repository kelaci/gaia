"""
Tensor operations and utilities.

This module provides fundamental tensor operations used throughout GAIA,
including weight initialization, activation functions, and normalization.
"""

import numpy as np
from typing import Tuple, Optional, Union

def initialize_weights(shape: Tuple[int, ...], init_type: str = 'he') -> np.ndarray:
    """
    Initialize weights with specified initialization.

    Args:
        shape: Shape of the weight matrix
        init_type: Initialization type ('he', 'xavier', 'normal', 'uniform')

    Returns:
        Initialized weight matrix

    Raises:
        ValueError: If unknown initialization type is specified
    """
    if init_type == 'he':
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif init_type == 'xavier':
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    elif init_type == 'normal':
        return np.random.randn(*shape) * 0.01
    elif init_type == 'uniform':
        return np.random.uniform(-0.01, 0.01, shape)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")

def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    """
    Apply activation function.

    Args:
        x: Input tensor
        activation: Activation function name ('relu', 'sigmoid', 'tanh', 'linear')

    Returns:
        Activated tensor

    Raises:
        ValueError: If unknown activation function is specified
    """
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'linear':
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")

def apply_activation_derivative(x: np.ndarray, activation: str) -> np.ndarray:
    """
    Apply activation function derivative.

    Args:
        x: Input tensor (pre-activation or post-activation depending on function)
        activation: Activation function name

    Returns:
        Derivative tensor
    """
    if activation == 'relu':
        return (x > 0).astype(float)
    elif activation == 'sigmoid':
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif activation == 'tanh':
        return 1.0 - np.tanh(x)**2
    elif activation == 'linear':
        return np.ones_like(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")

def normalize_tensor(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize tensor along specified axis.

    Args:
        x: Input tensor
        axis: Axis to normalize along
        eps: Small constant to prevent division by zero

    Returns:
        Normalized tensor
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

def temporal_convolution(x: np.ndarray, kernel: np.ndarray, mode: str = 'same') -> np.ndarray:
    """
    Apply temporal convolution.

    Args:
        x: Input tensor (time, features)
        kernel: Convolution kernel (kernel_time,) or (kernel_time, features)
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Convolved tensor (time, features)
    """
    if len(x.shape) != 2:
        raise ValueError(f"Input x must be 2D (time, features), got {x.shape}")

    time_steps, features = x.shape
    
    if len(kernel.shape) == 1:
        # 1D kernel applied to each feature independently
        result = np.zeros_like(x)
        for f in range(features):
            result[:, f] = np.convolve(x[:, f], kernel, mode=mode)
        return result
    elif len(kernel.shape) == 2:
        # 2D kernel (kernel_time, features)
        if kernel.shape[1] != features:
            raise ValueError(f"Kernel features {kernel.shape[1]} must match input features {features}")
        
        result = np.zeros_like(x)
        for f in range(features):
            result[:, f] = np.convolve(x[:, f], kernel[:, f], mode=mode)
        return result
    else:
        raise NotImplementedError("Multi-dimensional temporal convolution not yet implemented")

def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix of input tensor.

    Args:
        x: Input tensor (batch/time, features)

    Returns:
        Correlation matrix (features, features)
    """
    if len(x.shape) != 2:
        raise ValueError(f"Input x must be 2D (time, features), got {x.shape}")
        
    # Standardize inputs
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-8
    x_standardized = (x - x_mean) / x_std
    
    # Compute correlation matrix: (X.T @ X) / (N - 1)
    n = x.shape[0]
    if n > 1:
        corr = np.dot(x_standardized.T, x_standardized) / (n - 1)
    else:
        corr = np.eye(x.shape[1])
        
    return corr