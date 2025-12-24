"""
Core module providing base classes, type definitions, and utility functions.
"""

from .base import Module, Layer, PlasticComponent, HierarchicalLevel
from .types import *
from .tensor import *

__all__ = [
    'Module', 'Layer', 'PlasticComponent', 'HierarchicalLevel',
    'Tensor', 'Shape', 'PlasticityParams', 'LearningRate', 'TimeStep',
    'WeightMatrix', 'ActivationFunction', 'ConfigDict',
    'HierarchyConfig', 'PlasticityConfig', 'ESConfig',
    'initialize_weights', 'apply_activation', 'normalize_tensor',
    'temporal_convolution'
]