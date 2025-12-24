"""
ReactiveLayer implementation.

Fast, reactive processing layer with fixed weights for initial feature extraction.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from gaia.core.base import Layer
from gaia.core.types import Tensor, ActivationFunction
from gaia.core.tensor import initialize_weights, apply_activation

class ReactiveLayer(Layer):
    """
    Fast, reactive processing layer with fixed weights.

    This layer provides rapid feedforward processing without plasticity,
    suitable for initial feature extraction in hierarchical processing.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        biases: Bias vector (output_size,)
        activation_fn: Activation function name
        init_type: Weight initialization type
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu', init_type: str = 'he'):
        """
        Initialize ReactiveLayer.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
            init_type: Weight initialization type ('he', 'xavier', 'normal', 'uniform')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.activation_fn = activation
        self.init_type = init_type

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases."""
        self.weights = initialize_weights((self.output_size, self.input_size), self.init_type)
        self.biases = np.zeros(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)

        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[1]}")

        # Linear transformation
        output = np.dot(x, self.weights.T) + self.biases

        # Apply activation
        return apply_activation(output, self.activation_fn)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer (batch_size, output_size)

        Returns:
            Gradient for previous layer (batch_size, input_size)

        TODO:
            - Implement proper gradient computation
            - Add support for different loss functions
            - Optimize for memory efficiency
        """
        # Placeholder implementation
        return np.dot(grad, self.weights)

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        ReactiveLayer has fixed weights - no update implemented.

        Args:
            lr: Learning rate (ignored for this layer type)
        """
        # ReactiveLayer has fixed weights - no update
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        # No internal state to reset
        pass

    def activation(self, x: Tensor) -> Tensor:
        """
        Apply activation function.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return apply_activation(x, self.activation_fn)

    def get_weights(self) -> Tensor:
        """
        Get current weights.

        Returns:
            Weight matrix
        """
        return self.weights.copy()

    def set_weights(self, weights: Tensor) -> None:
        """
        Set weights.

        Args:
            weights: New weight matrix

        Raises:
            ValueError: If weight shape doesn't match expected dimensions
        """
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        self.weights = weights.copy()

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Dictionary containing layer configuration
        """
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_fn,
            'init_type': self.init_type
        }

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"ReactiveLayer({self.input_size}→{self.output_size}, {self.activation_fn})"