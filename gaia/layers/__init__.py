"""
Layer implementations for GAIA.

This module provides various layer types for hierarchical processing,
including reactive layers, Hebbian learning cores, and temporal layers.
"""

from .reactive import ReactiveLayer
from .hebbian import HebbianCore
from .temporal import TemporalLayer

__all__ = ['ReactiveLayer', 'HebbianCore', 'TemporalLayer']