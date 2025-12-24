# Core Components

## 📦 Core Module Structure

The core module provides the foundation for all GAIA components, including base classes, type definitions, and utility functions.

### `core/base.py` - Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np

class Module(ABC):
    """Base class for all GAIA modules."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the module."""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation."""
        pass

    @abstractmethod
    def update(self, lr: float) -> None:
        """Update module parameters."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal state."""
        pass

class Layer(Module):
    """Base class for all layer implementations."""

    @abstractmethod
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get current weights."""
        pass

    @abstractmethod
    def set_weights(self, weights: np.ndarray) -> None:
        """Set weights."""
        pass

class PlasticComponent(ABC):
    """Base class for components with plasticity."""

    @abstractmethod
    def get_plasticity_params(self) -> Dict[str, float]:
        """Get plasticity parameters."""
        pass

    @abstractmethod
    def set_plasticity_params(self, params: Dict[str, float]) -> None:
        """Set plasticity parameters."""
        pass

class HierarchicalLevel(ABC):
    """Base class for hierarchical levels."""

    @abstractmethod
    def process_time_step(self, input_data: np.ndarray, t: int) -> np.ndarray:
        """Process a single time step."""
        pass

    @abstractmethod
    def get_representation(self) -> np.ndarray:
        """Get current representation."""
        pass
```

### `core/types.py` - Type Definitions

```python
import numpy as np
from typing import Dict, List, Tuple, Any

# Type aliases for better code clarity
Tensor = np.ndarray
Shape = Tuple[int, ...]
PlasticityParams = Dict[str, float]
LearningRate = float
TimeStep = int
WeightMatrix = np.ndarray
ActivationFunction = callable

# Configuration types
ConfigDict = Dict[str, Any]
HierarchyConfig = Dict[str, Any]
PlasticityConfig = Dict[str, float]
ESConfig = Dict[str, float]
```

### `core/tensor.py` - Tensor Operations

```python
import numpy as np
from typing import Tuple, Optional

def initialize_weights(shape: Tuple[int, ...], init_type: str = 'he') -> np.ndarray:
    """
    Initialize weights with specified initialization.

    Args:
        shape: Shape of the weight matrix
        init_type: Initialization type ('he', 'xavier', 'normal', 'uniform')

    Returns:
        Initialized weight matrix
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

def normalize_tensor(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize tensor along specified axis.

    Args:
        x: Input tensor
        axis: Axis to normalize along

    Returns:
        Normalized tensor
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)

def temporal_convolution(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply temporal convolution.

    Args:
        x: Input tensor (time, features)
        kernel: Convolution kernel (time,)

    Returns:
        Convolved tensor
    """
    # TODO: Implement efficient temporal convolution
    pass
```

## 📦 Layer Implementations

### `layers/reactive.py` - ReactiveLayer

```python
import numpy as np
from typing import Optional, Dict
from gaia.core.base import Layer
from gaia.core.types import Tensor, ActivationFunction

class ReactiveLayer(Layer):
    """
    Fast, reactive processing layer with fixed weights.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        biases: Bias vector (output_size,)
        activation_fn: Activation function
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu', init_type: str = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.activation_fn = activation
        self.init_type = init_type

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases."""
        from gaia.core.tensor import initialize_weights

        self.weights = initialize_weights((self.output_size, self.input_size), self.init_type)
        self.biases = np.zeros(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)
        """
        from gaia.core.tensor import apply_activation

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
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Args:
            lr: Learning rate
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
        from gaia.core.tensor import apply_activation
        return apply_activation(x, self.activation_fn)

    def get_weights(self) -> Tensor:
        """Get current weights."""
        return self.weights

    def set_weights(self, weights: Tensor) -> None:
        """Set weights."""
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        self.weights = weights
```

### `layers/hebbian.py` - HebbianCore

```python
import numpy as np
from typing import Dict, Optional
from gaia.core.base import PlasticComponent
from gaia.core.types import Tensor, PlasticityParams

class HebbianCore(PlasticComponent):
    """
    Hebbian learning implementation with multiple plasticity rules.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        pre_synaptic: Pre-synaptic activity trace
        post_synaptic: Post-synaptic activity trace
        plasticity_params: Plasticity parameters
    """

    def __init__(self, input_size: int, output_size: int,
                 plasticity_rule: str = 'hebbian',
                 params: Optional[Dict[str, float]] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.pre_synaptic = None
        self.post_synaptic = None
        self.plasticity_rule = plasticity_rule

        # Default plasticity parameters
        self.plasticity_params = params or {
            'learning_rate': 0.01,
            'decay_rate': 0.001,
            'ltp_coefficient': 1.0,
            'ltd_coefficient': 0.8,
            'homeostatic_strength': 0.1
        }

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and activity traces."""
        from gaia.core.tensor import initialize_weights

        self.weights = initialize_weights((self.output_size, self.input_size), 'he')
        self.pre_synaptic = np.zeros(self.input_size)
        self.post_synaptic = np.zeros(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Hebbian core.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)
        """
        # Update activity traces
        self.pre_synaptic = x.mean(axis=0)  # Average over batch
        output = np.dot(x, self.weights.T)
        self.post_synaptic = output.mean(axis=0)  # Average over batch

        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update weights using Hebbian learning.

        Args:
            lr: Learning rate (overrides plasticity_params if provided)
        """
        effective_lr = lr if lr is not None else self.plasticity_params['learning_rate']

        if self.plasticity_rule == 'hebbian':
            # Classic Hebbian: Δw = η * pre * post
            weight_update = effective_lr * np.outer(self.post_synaptic, self.pre_synaptic)
        elif self.plasticity_rule == 'oja':
            # Oja's rule: Δw = η * post * (pre - post * w)
            weight_update = effective_lr * np.outer(
                self.post_synaptic,
                self.pre_synaptic - np.dot(self.weights.T, self.post_synaptic)
            )
        elif self.plasticity_rule == 'bcm':
            # BCM rule with sliding threshold
            theta = np.mean(self.post_synaptic)
            weight_update = effective_lr * np.outer(
                self.post_synaptic * (self.post_synaptic - theta),
                self.pre_synaptic
            )
        else:
            raise ValueError(f"Unknown plasticity rule: {self.plasticity_rule}")

        # Apply weight update
        self.weights += weight_update

        # Apply weight decay
        self.weights *= (1.0 - self.plasticity_params['decay_rate'])

        # Homeostatic regulation
        self._homeostatic_regulation()

    def _homeostatic_regulation(self) -> None:
        """Apply homeostatic regulation to maintain stable activity."""
        # Simple weight normalization
        norm = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / (norm + 1e-8)

    def reset_state(self) -> None:
        """Reset internal state."""
        self.pre_synaptic = np.zeros(self.input_size)
        self.post_synaptic = np.zeros(self.output_size)

    def get_plasticity_params(self) -> PlasticityParams:
        """Get plasticity parameters."""
        return self.plasticity_params

    def set_plasticity_params(self, params: PlasticityParams) -> None:
        """Set plasticity parameters."""
        self.plasticity_params.update(params)
```

### `layers/temporal.py` - TemporalLayer

```python
import numpy as np
from typing import Optional, Dict
from gaia.core.base import Layer
from gaia.core.types import Tensor

class TemporalLayer(Layer):
    """
    Layer with temporal context processing.

    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        time_window: Number of time steps to maintain
        hidden_state: Current hidden state
        recurrent_weights: Recurrent weight matrix
    """

    def __init__(self, input_size: int, hidden_size: int,
                 time_window: int = 10, activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_window = time_window
        self.hidden_state = None
        self.activation = activation
        self.recurrent_weights = None

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        from gaia.core.tensor import initialize_weights

        # Input to hidden weights
        self.weights = initialize_weights((self.hidden_size, self.input_size), 'he')

        # Recurrent weights
        self.recurrent_weights = initialize_weights((self.hidden_size, self.hidden_size), 'he')

        # Hidden state initialization
        self.hidden_state = np.zeros(self.hidden_size)

    def forward(self, x: Tensor, t: Optional[int] = None) -> Tensor:
        """
        Forward pass with temporal processing.

        Args:
            x: Input tensor (batch_size, input_size)
            t: Optional time step

        Returns:
            Output tensor (batch_size, hidden_size)
        """
        from gaia.core.tensor import apply_activation

        # Linear transformation
        linear_output = np.dot(x, self.weights.T) + np.dot(self.hidden_state, self.recurrent_weights.T)

        # Apply activation
        output = apply_activation(linear_output, self.activation)

        # Update hidden state
        self.hidden_state = output.mean(axis=0)  # Average over batch

        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Args:
            lr: Learning rate
        """
        # TODO: Implement parameter update
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        self.hidden_state = np.zeros(self.hidden_size)

    def get_temporal_context(self) -> Tensor:
        """
        Get current temporal context.

        Returns:
            Temporal context tensor
        """
        return self.hidden_state.copy()
```

## 📦 Hierarchy Components

### `hierarchy/level.py` - HierarchicalLevel

```python
import numpy as np
from typing import Optional, List, Dict
from gaia.core.base import HierarchicalLevel
from gaia.core.types import Tensor

class HierarchicalLevel(HierarchicalLevel):
    """
    Implementation of a hierarchical level.

    Attributes:
        level_id: Level identifier
        input_size: Size of input features
        output_size: Size of output features
        temporal_resolution: Time compression factor
        parent_level: Reference to parent level
        child_levels: List of child levels
        processing_layers: List of processing layers
    """

    def __init__(self, level_id: int, input_size: int, output_size: int,
                 temporal_resolution: int = 1):
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_resolution = temporal_resolution
        self.parent_level = None
        self.child_levels: List['HierarchicalLevel'] = []
        self.processing_layers = []
        self.current_representation = None
        self.time_step = 0

    def add_layer(self, layer: Layer) -> None:
        """Add a processing layer to this level."""
        self.processing_layers.append(layer)

    def process_time_step(self, input_data: Tensor, t: int) -> Tensor:
        """
        Process a single time step.

        Args:
            input_data: Input data for this time step
            t: Global time step

        Returns:
            Processed output
        """
        # Only process every temporal_resolution time steps
        if t % self.temporal_resolution != 0:
            return self.current_representation

        # Process through all layers
        output = input_data
        for layer in self.processing_layers:
            output = layer.forward(output)

        # Store current representation
        self.current_representation = output
        self.time_step = t

        return output

    def get_representation(self) -> Tensor:
        """Get current representation."""
        return self.current_representation

    def communicate_with_parent(self) -> Optional[Tensor]:
        """
        Communicate with parent level.

        Returns:
            Data to send to parent, or None
        """
        if self.parent_level is None:
            return None

        # TODO: Implement communication protocol
        return self.current_representation

    def communicate_with_children(self, data: Tensor) -> None:
        """
        Communicate with child levels.

        Args:
            data: Data received from parent
        """
        # TODO: Implement communication protocol
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        for layer in self.processing_layers:
            layer.reset_state()
        self.current_representation = None
        self.time_step = 0
```

### `hierarchy/manager.py` - HierarchyManager

```python
from typing import List, Dict, Optional
import numpy as np
from gaia.core.types import Tensor
from gaia.hierarchy.level import HierarchicalLevel

class HierarchyManager:
    """
    Manages multiple hierarchical levels.

    Attributes:
        levels: List of hierarchical levels
        communication_schedule: Communication timing
    """

    def __init__(self):
        self.levels: List[HierarchicalLevel] = []
        self.communication_schedule = {}

    def add_level(self, level: HierarchicalLevel) -> None:
        """Add a level to the hierarchy."""
        self.levels.append(level)

        # Sort levels by level_id
        self.levels.sort(key=lambda x: x.level_id)

        # Update parent/child relationships
        self._update_hierarchy_relationships()

    def _update_hierarchy_relationships(self) -> None:
        """Update parent/child relationships between levels."""
        for i, level in enumerate(self.levels):
            # Set parent (level above)
            if i > 0:
                level.parent_level = self.levels[i-1]

            # Set children (levels below)
            level.child_levels = [self.levels[j] for j in range(i+1, len(self.levels))
                                if self.levels[j].level_id == level.level_id + 1]

    def process_hierarchy(self, input_data: Tensor, time_steps: int) -> Dict[int, List[Tensor]]:
        """
        Process input through the entire hierarchy.

        Args:
            input_data: Input data sequence (time_steps, input_size)
            time_steps: Number of time steps to process

        Returns:
            Dictionary of representations at each level
        """
        representations = {level.level_id: [] for level in self.levels}

        for t in range(time_steps):
            current_input = input_data[t]

            # Process through each level
            for level in self.levels:
                if level.level_id == 0:
                    # Input level
                    output = level.process_time_step(current_input, t)
                else:
                    # Higher levels
                    output = level.process_time_step(output, t)

                representations[level.level_id].append(output)

            # Hierarchical communication
            self._hierarchical_communication(t)

        return representations

    def _hierarchical_communication(self, t: int) -> None:
        """Handle communication between hierarchical levels."""
        # TODO: Implement communication protocol
        pass

    def get_all_representations(self) -> Dict[int, Tensor]:
        """
        Get current representations from all levels.

        Returns:
            Dictionary of current representations
        """
        return {level.level_id: level.get_representation() for level in self.levels}

    def reset_state(self) -> None:
        """Reset state of all levels."""
        for level in self.levels:
            level.reset_state()
```

## 📦 Plasticity Components

### `plasticity/controller.py` - PlasticityController

```python
import numpy as np
from typing import List, Dict, Optional
from gaia.core.base import PlasticComponent
from gaia.core.types import Tensor, PlasticityParams

class PlasticityController:
    """
    Controls plasticity parameters using Evolutionary Strategy.

    Attributes:
        target_modules: List of modules to control
        es_optimizer: Evolutionary Strategy optimizer
        plasticity_params: Current plasticity parameters
        adaptation_rate: Rate of parameter adaptation
        exploration_noise: Noise for exploration
    """

    def __init__(self, target_modules: List[PlasticComponent],
                 adaptation_rate: float = 0.01, exploration_noise: float = 0.1):
        self.target_modules = target_modules
        self.adaptation_rate = adaptation_rate
        self.exploration_noise = exploration_noise

        # Initialize ES optimizer
        from gaia.plasticity.es_optimizer import EvolutionaryStrategy
        self.es_optimizer = EvolutionaryStrategy()

        # Initialize plasticity parameters
        self.plasticity_params = self._initialize_params()

    def _initialize_params(self) -> Tensor:
        """Initialize plasticity parameters."""
        # Get parameter dimensions from target modules
        param_dims = sum(len(module.get_plasticity_params()) for module in self.target_modules)

        # Initialize with reasonable defaults
        initial_params = np.ones(param_dims) * 0.01

        return initial_params

    def adapt_plasticity(self, performance_metric: float) -> None:
        """
        Adapt plasticity parameters based on performance.

        Args:
            performance_metric: Current performance metric
        """
        # Sample perturbed parameters
        perturbed_params = self.es_optimizer.generate_population(self.plasticity_params)

        # Evaluate fitness of perturbed parameters
        fitness_scores = []
        for params in perturbed_params:
            # Apply parameters temporarily
            self._apply_params_temporarily(params)

            # Evaluate performance (simplified for now)
            fitness = self._evaluate_performance()
            fitness_scores.append(fitness)

        # Update parameters based on fitness
        self.es_optimizer.update_mean(self.plasticity_params, perturbed_params, fitness_scores)
        self.plasticity_params = self.es_optimizer.get_mean()

        # Apply updated parameters
        self._apply_params()

    def _apply_params_temporarily(self, params: Tensor) -> None:
        """Temporarily apply parameters for evaluation."""
        # TODO: Implement temporary parameter application
        pass

    def _apply_params(self) -> None:
        """Apply current parameters to target modules."""
        param_index = 0
        for module in self.target_modules:
            module_params = module.get_plasticity_params()
            num_params = len(module_params)

            # Update each parameter
            for i, param_name in enumerate(module_params.keys()):
                module_params[param_name] = self.plasticity_params[param_index]
                param_index += 1

            module.set_plasticity_params(module_params)

    def _evaluate_performance(self) -> float:
        """Evaluate current performance."""
        # TODO: Implement proper performance evaluation
        return np.random.random()  # Placeholder

    def get_current_params(self) -> Tensor:
        """Get current plasticity parameters."""
        return self.plasticity_params.copy()
```

### `plasticity/es_optimizer.py` - EvolutionaryStrategy

```python
import numpy as np
from typing import List, Tuple

class EvolutionaryStrategy:
    """
    Evolutionary Strategy optimizer for plasticity parameters.

    Attributes:
        population_size: Number of individuals in population
        sigma: Mutation strength
        learning_rate: Learning rate for mean update
        elite_fraction: Fraction of elites to select
    """

    def __init__(self, population_size: int = 50, sigma: float = 0.1,
                 learning_rate: float = 0.01, elite_fraction: float = 0.2):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction
        self.mean = None

    def generate_population(self, initial_mean: np.ndarray) -> List[np.ndarray]:
        """
        Generate population of perturbed parameters.

        Args:
            initial_mean: Initial parameter vector

        Returns:
            List of perturbed parameter vectors
        """
        self.mean = initial_mean.copy()
        population = []

        for _ in range(self.population_size):
            # Sample from multivariate normal distribution
            perturbation = np.random.randn(*self.mean.shape) * self.sigma
            perturbed_params = self.mean + perturbation
            population.append(perturbed_params)

        return population

    def update_mean(self, current_mean: np.ndarray,
                   population: List[np.ndarray],
                   fitness_scores: List[float]) -> None:
        """
        Update mean based on fitness scores.

        Args:
            current_mean: Current mean parameters
            population: List of parameter vectors
            fitness_scores: Corresponding fitness scores
        """
        # Select elites
        elite_indices = self._select_elites(fitness_scores)
        elites = [population[i] for i in elite_indices]

        # Update mean toward elites
        self.mean = current_mean + self.learning_rate * np.mean(elites - current_mean, axis=0)

    def _select_elites(self, fitness_scores: List[float]) -> List[int]:
        """Select elite individuals based on fitness."""
        # Get indices of top performers
        num_elites = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(fitness_scores)[-num_elites:]

        return list(elite_indices)

    def get_mean(self) -> np.ndarray:
        """Get current mean parameters."""
        return self.mean.copy()

    def adapt_sigma(self, fitness_improvement: float) -> None:
        """
        Adapt mutation strength based on fitness improvement.

        Args:
            fitness_improvement: Improvement in fitness
        """
        # Simple adaptation rule
        if fitness_improvement > 0:
            self.sigma *= 1.1  # Increase exploration
        else:
            self.sigma *= 0.9  # Decrease exploration
```

### `plasticity/rules.py` - Plasticity Rules

```python
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class PlasticityRule(ABC):
    """Base class for plasticity rules."""

    @abstractmethod
    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """Apply plasticity rule to weights."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        pass

class HebbianRule(PlasticityRule):
    """Classic Hebbian learning rule: Δw = η * pre * post"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate}

class OjasRule(PlasticityRule):
    """Oja's learning rule: Δw = η * post * (pre - post * w)"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Oja's learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        weight_update = self.learning_rate * np.outer(
            post_activity,
            pre_activity - np.dot(weights.T, post_activity)
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate}

class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro rule with sliding threshold."""

    def __init__(self, learning_rate: float = 0.01, theta: float = 1.0):
        self.learning_rate = learning_rate
        self.theta = theta

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply BCM learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        # Update threshold based on post-synaptic activity
        self.theta = 0.9 * self.theta + 0.1 * np.mean(post_activity)

        weight_update = self.learning_rate * np.outer(
            post_activity * (post_activity - self.theta),
            pre_activity
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate, 'theta': self.theta}
```

## 📦 Meta-Learning Components

### `meta_learning/optimizer.py` - MetaOptimizer

```python
import numpy as np
from typing import List, Dict, Callable
from gaia.core.types import Tensor

class MetaOptimizer:
    """
    Meta-optimizer for plasticity learning.

    Attributes:
        inner_loop: Plasticity controller for inner optimization
        outer_loop: Outer optimization algorithm
        task_distribution: Distribution of tasks
        learning_history: History of learning performance
    """

    def __init__(self, plasticity_controller, outer_optimizer='adam'):
        self.inner_loop = plasticity_controller
        self.outer_optimizer = outer_optimizer
        self.task_distribution = None
        self.learning_history = []

    def meta_train(self, num_episodes: int, tasks: List[Callable]) -> None:
        """
        Perform meta-training.

        Args:
            num_episodes: Number of training episodes
            tasks: List of task functions
        """
        for episode in range(num_episodes):
            # Sample a task
            task = np.random.choice(tasks)

            # Inner loop: adapt to task
            task_performance = self.inner_loop_adaptation(task)

            # Outer loop: update meta-parameters
            self.outer_update(task_performance)

            # Record learning history
            self.learning_history.append(task_performance)

    def inner_loop_adaptation(self, task: Callable) -> float:
        """
        Inner loop adaptation to a specific task.

        Args:
            task: Task function

        Returns:
            Task performance
        """
        # Reset plasticity controller
        self.inner_loop.reset_state()

        # Adapt to task
        performance = 0.0
        for step in range(10):  # Fixed number of adaptation steps
            task_data = task(step)
            performance += self.inner_loop.adapt_plasticity(task_data)

        return performance / 10  # Average performance

    def outer_update(self, performance: float) -> None:
        """
        Outer loop update of meta-parameters.

        Args:
            performance: Task performance
        """
        # TODO: Implement meta-parameter update
        pass

    def evaluate_meta_performance(self) -> Dict[str, float]:
        """
        Evaluate meta-learning performance.

        Returns:
            Dictionary of performance metrics
        """
        if not self.learning_history:
            return {'average_performance': 0.0, 'improvement': 0.0}

        metrics = {
            'average_performance': np.mean(self.learning_history),
            'improvement': self.learning_history[-1] - self.learning_history[0],
            'stability': np.std(self.learning_history[-10:]) if len(self.learning_history) >= 10 else 0.0
        }

        return metrics
```

### `meta_learning/metrics.py` - Performance Metrics

```python
import numpy as np
from typing import Dict, List
from gaia.core.base import Module

def measure_plasticity_efficiency(module: Module) -> float:
    """
    Measure plasticity efficiency of a module.

    Args:
        module: Module to evaluate

    Returns:
        Plasticity efficiency score (0-1)
    """
    # TODO: Implement proper efficiency measurement
    return np.random.random()

def measure_adaptation_speed(module: Module) -> float:
    """
    Measure adaptation speed of a module.

    Args:
        module: Module to evaluate

    Returns:
        Adaptation speed score (0-1)
    """
    # TODO: Implement proper speed measurement
    return np.random.random()

def measure_stability_plasticity_tradeoff(module: Module) -> float:
    """
    Measure stability-plasticity tradeoff.

    Args:
        module: Module to evaluate

    Returns:
        Tradeoff score (0-1, higher is better)
    """
    # TODO: Implement proper tradeoff measurement
    return np.random.random()

def evaluate_meta_learning_curve(learning_history: List[float]) -> Dict[str, float]:
    """
    Evaluate meta-learning curve.

    Args:
        learning_history: History of learning performance

    Returns:
        Dictionary of evaluation metrics
    """
    if len(learning_history) < 2:
        return {
            'convergence_rate': 0.0,
            'final_performance': 0.0,
            'improvement': 0.0,
            'stability': 0.0
        }

    metrics = {
        'convergence_rate': (learning_history[-1] - learning_history[0]) / len(learning_history),
        'final_performance': learning_history[-1],
        'improvement': learning_history[-1] - learning_history[0],
        'stability': np.std(learning_history[-10:]) if len(learning_history) >= 10 else 0.0
    }

    return metrics
```

## 📦 Configuration & Utilities

### `config/defaults.py` - Default Configurations

```python
# Default hierarchy configuration
DEFAULT_HIERARCHY_CONFIG = {
    "num_levels": 4,
    "temporal_compression": 2,
    "base_resolution": 1,
    "level_sizes": [64, 128, 256, 512]
}

# Default plasticity configuration
DEFAULT_PLASTICITY_CONFIG = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,
    "ltd_coefficient": 0.8,
    "decay_rate": 0.001,
    "homeostatic_strength": 0.1
}

# Default ES configuration
DEFAULT_ES_CONFIG = {
    "population_size": 50,
    "sigma": 0.1,
    "learning_rate": 0.01,
    "elite_fraction": 0.2
}

# Default layer configurations
DEFAULT_LAYER_CONFIGS = {
    "reactive": {
        "activation": "relu",
        "init_type": "he"
    },
    "hebbian": {
        "plasticity_rule": "hebbian",
        "params": DEFAULT_PLASTICITY_CONFIG
    },
    "temporal": {
        "activation": "tanh",
        "time_window": 10
    }
}
```

### `utils/logging.py` - Logging Utilities

```python
import logging
from typing import Optional, Dict
import numpy as np

def setup_logging(name: str = 'gaia', level: str = 'INFO',
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for GAIA.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_tensor_stats(logger: logging.Logger, tensor: np.ndarray,
                    name: str, level: str = 'DEBUG') -> None:
    """
    Log statistics about a tensor.

    Args:
        logger: Logger instance
        tensor: Tensor to log
        name: Tensor name
        level: Logging level
    """
    stats = {
        'shape': tensor.shape,
        'mean': np.mean(tensor),
        'std': np.std(tensor),
        'min': np.min(tensor),
        'max': np.max(tensor)
    }

    getattr(logger, level.lower())(f"Tensor {name} stats: {stats}")

def log_plasticity_update(logger: logging.Logger, params: Dict[str, float],
                         performance: float, step: int) -> None:
    """
    Log plasticity parameter update.

    Args:
        logger: Logger instance
        params: Plasticity parameters
        performance: Current performance
        step: Training step
    """
    logger.info(f"Step {step}: Performance = {performance:.4f}, Params = {params}")
```

### `utils/visualization.py` - Visualization Tools

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def plot_hierarchy_representations(representations: Dict[int, List[np.ndarray]],
                                 title: str = "Hierarchical Representations") -> None:
    """
    Plot representations from different hierarchical levels.

    Args:
        representations: Dictionary of representations by level
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    for level, reps in representations.items():
        # Average representation over time
        avg_rep = np.mean(reps, axis=0)

        plt.subplot(2, 2, level + 1)
        plt.imshow(avg_rep.reshape(1, -1), aspect='auto')
        plt.title(f"Level {level}")
        plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(learning_history: List[float],
                       title: str = "Learning Curve") -> None:
    """
    Plot learning curve.

    Args:
        learning_history: History of learning performance
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_history)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.grid(True)
    plt.show()

def plot_weight_matrix(weights: np.ndarray, title: str = "Weight Matrix") -> None:
    """
    Plot weight matrix.

    Args:
        weights: Weight matrix
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_plasticity_parameters(params_history: List[Dict[str, float]],
                             title: str = "Plasticity Parameters") -> None:
    """
    Plot plasticity parameter evolution.

    Args:
        params_history: History of plasticity parameters
        title: Plot title
    """
    if not params_history:
        return

    param_names = list(params_history[0].keys())
    num_params = len(param_names)

    plt.figure(figsize=(12, 6))
    for i, param_name in enumerate(param_names):
        values = [params[param_name] for params in params_history]
        plt.plot(values, label=param_name)

    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 📦 Examples

### `examples/basic_demo.py` - Basic Usage Example

```python
import numpy as np
from gaia.core.base import Module
from gaia.layers.reactive import ReactiveLayer
from gaia.layers.hebbian import HebbianCore
from gaia.layers.temporal import TemporalLayer
from gaia.hierarchy.level import HierarchicalLevel
from gaia.hierarchy.manager import HierarchyManager

def basic_demo():
    """Basic demonstration of GAIA components."""
    print("GAIA Basic Demo")
    print("=" * 50)

    # Create a simple hierarchy
    manager = HierarchyManager()

    # Level 0: Input level
    level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20, activation='relu'))
    manager.add_level(level0)

    # Level 1: Intermediate level
    level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40, plasticity_rule='hebbian'))
    manager.add_level(level1)

    # Level 2: High-level
    level2 = HierarchicalLevel(2, input_size=40, output_size=80, temporal_resolution=4)
    level2.add_layer(TemporalLayer(40, 80, time_window=5))
    manager.add_level(level2)

    # Generate some random input data
    time_steps = 20
    input_data = np.random.randn(time_steps, 10)

    # Process through hierarchy
    print("Processing input through hierarchy...")
    representations = manager.process_hierarchy(input_data, time_steps)

    # Display results
    print("\nRepresentations:")
    for level_id, reps in representations.items():
        print(f"Level {level_id}: {len(reps)} representations, shape {reps[0].shape}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    basic_demo()
```

## 📦 Main Package Structure

### `__init__.py` Files

```python
# gaia/__init__.py
from .core import *
from .layers import *
from .hierarchy import *
from .plasticity import *
from .meta_learning import *
from .utils import *
from .config import *

# gaia/core/__init__.py
from .base import Module, Layer, PlasticComponent, HierarchicalLevel
from .types import *
from .tensor import *

# gaia/layers/__init__.py
from .reactive import ReactiveLayer
from .hebbian import HebbianCore
from .temporal import TemporalLayer

# gaia/hierarchy/__init__.py
from .level import HierarchicalLevel
from .manager import HierarchyManager

# gaia/plasticity/__init__.py
from .controller import PlasticityController
from .es_optimizer import EvolutionaryStrategy
from .rules import *

# gaia/meta_learning/__init__.py
from .optimizer import MetaOptimizer
from .metrics import *

# gaia/utils/__init__.py
from .logging import *
from .visualization import *

# gaia/config/__init__.py
from .defaults import *
```

## 🎯 Implementation Notes

### Type Hints
- All functions and methods include comprehensive type hints
- Custom type aliases for better code clarity
- Type checking can be enabled with mypy

### Error Handling
- Input validation for all public methods
- Clear error messages for debugging
- Graceful fallbacks where appropriate

### Testing
- Each component has clear interfaces for unit testing
- Empty methods marked with TODO for implementation
- Example usage provided for each major component

### Performance
- Vectorized operations using numpy
- Efficient memory usage patterns
- Minimal computational overhead

## 🔮 Next Steps

The core components are now fully specified with proper typing, empty methods, and TODO comments. The next phase involves:

1. **Implementation**: Filling in the TODO methods
2. **Testing**: Creating unit tests for each component
3. **Integration**: Building complete examples
4. **Optimization**: Performance tuning

This skeleton provides a solid foundation for GAIA v4/v4.1 development!