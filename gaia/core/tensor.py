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

def temporal_convolution(x: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """
    Apply temporal convolution.

    Args:
        x: Input tensor (time, features)
        kernel: Convolution kernel (time,)
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Convolved tensor

    TODO:
        - Implement efficient temporal convolution
        - Add support for multi-dimensional kernels
        - Optimize for different convolution modes
    """
    # Placeholder implementation - will be optimized
    if len(x.shape) == 2 and len(kernel.shape) == 1:
        # Simple 1D convolution along time axis
        result = np.zeros_like(x)
        kernel_len = len(kernel)

        for t in range(x.shape[0]):
            start = max(0, t - kernel_len // 2)
            end = min(x.shape[0], t + kernel_len // 2 + 1)
            window = x[start:end, :]
            result[t, :] = np.dot(window.T, kernel[start-t+kernel_len//2:end-t+kernel_len//2])

        return result
    else:
        raise NotImplementedError("Multi-dimensional temporal convolution not yet implemented")

def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix of input tensor.

    Args:
        x: Input tensor (time, features)

    Returns:
        Correlation matrix (features, features)

    TODO:
        - Implement efficient correlation computation
        - Add support for different correlation measures
    """
    # Simple covariance-based correlation
    if len(x.shape) == 2:
        cov = np.cov(x, rowvar=False)
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)
    else:
        raise ValueError("Input must be 2D tensor (time, features)")