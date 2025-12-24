"""
Plasticity rules implementation.

Various Hebbian learning rules for synaptic plasticity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class PlasticityRule(ABC):
    """
    Base class for plasticity rules.

    This abstract class defines the interface for all plasticity rules
    used in GAIA's learning mechanisms.
    """

    @abstractmethod
    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply plasticity rule to weights.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector

        Returns:
            Updated weight matrix
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """
        Get rule parameters.

        Returns:
            Dictionary of rule parameters
        """
        pass

    def __str__(self) -> str:
        """String representation of the rule."""
        return self.__class__.__name__

class HebbianRule(PlasticityRule):
    """
    Classic Hebbian learning rule: Δw = η * pre * post

    This rule implements the fundamental Hebbian learning principle:
    "neurons that fire together, wire together."
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize HebbianRule.

        Args:
            learning_rate: Learning rate parameter
        """
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector

        Returns:
            Updated weight matrix
        """
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """
        Get rule parameters.

        Returns:
            Dictionary of rule parameters
        """
        return {'learning_rate': self.learning_rate}

class OjasRule(PlasticityRule):
    """
    Oja's learning rule: Δw = η * post * (pre - post * w)

    This rule extends Hebbian learning with a weight decay term
    that prevents unbounded weight growth.
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize OjasRule.

        Args:
            learning_rate: Learning rate parameter
        """
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Oja's learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector

        Returns:
            Updated weight matrix
        """
        weight_update = self.learning_rate * np.outer(
            post_activity,
            pre_activity - np.dot(weights.T, post_activity)
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """
        Get rule parameters.

        Returns:
            Dictionary of rule parameters
        """
        return {'learning_rate': self.learning_rate}

class BCMRule(PlasticityRule):
    """
    Bienenstock-Cooper-Munro rule with sliding threshold.

    This rule implements a more sophisticated Hebbian learning rule
    with a dynamic threshold that adapts based on post-synaptic activity.
    """

    def __init__(self, learning_rate: float = 0.01, theta: float = 1.0):
        """
        Initialize BCMRule.

        Args:
            learning_rate: Learning rate parameter
            theta: Initial threshold parameter
        """
        self.learning_rate = learning_rate
        self.theta = theta

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply BCM learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector

        Returns:
            Updated weight matrix
        """
        # Update threshold based on post-synaptic activity
        self.theta = 0.9 * self.theta + 0.1 * np.mean(post_activity)

        weight_update = self.learning_rate * np.outer(
            post_activity * (post_activity - self.theta),
            pre_activity
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """
        Get rule parameters.

        Returns:
            Dictionary of rule parameters
        """
        return {'learning_rate': self.learning_rate, 'theta': self.theta}

class STDPRule(PlasticityRule):
    """
    Spike-Timing Dependent Plasticity rule.

    This rule implements a more biologically plausible learning rule
    that depends on the relative timing of pre- and post-synaptic spikes.

    TODO:
        - Implement full STDP rule
        - Add support for different timing windows
        - Consider different STDP variants
    """

    def __init__(self, learning_rate: float = 0.01, tau: float = 20.0):
        """
        Initialize STDPRule.

        Args:
            learning_rate: Learning rate parameter
            tau: Time constant for STDP window
        """
        self.learning_rate = learning_rate
        self.tau = tau

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply STDP learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector

        Returns:
            Updated weight matrix

        TODO:
            - Implement proper STDP computation
            - Add timing-dependent weight updates
            - Consider different STDP kernels
        """
        # Placeholder implementation
        # In practice, this would use spike timing information
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """
        Get rule parameters.

        Returns:
            Dictionary of rule parameters
        """
        return {'learning_rate': self.learning_rate, 'tau': self.tau}

def create_plasticity_rule(rule_name: str, **params) -> PlasticityRule:
    """
    Factory function for creating plasticity rules.

    Args:
        rule_name: Name of the plasticity rule
        **params: Parameters for the rule

    Returns:
        PlasticityRule instance

    Raises:
        ValueError: If unknown rule name is specified
    """
    rule_name = rule_name.lower()
    if rule_name == 'hebbian':
        return HebbianRule(**params)
    elif rule_name == 'oja':
        return OjasRule(**params)
    elif rule_name == 'bcm':
        return BCMRule(**params)
    elif rule_name == 'stdp':
        return STDPRule(**params)
    else:
        raise ValueError(f"Unknown plasticity rule: {rule_name}")