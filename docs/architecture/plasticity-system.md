# Plasticity System

## 🎯 Plasticity Overview

The plasticity system is the heart of GAIA's learning capability, enabling adaptive behavior through meta-learning of plasticity parameters using Evolutionary Strategies.

## 🏗️ Plasticity Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            PlasticityController                             │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        EvolutionaryStrategy                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Population Generation                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Fitness Evaluation                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Parameter Update                                │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Plasticity Rules                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianRule                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     OjasRule                                        │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     BCMRule                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Target Modules                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianCore                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Plasticity Control Flow

1. **Performance Evaluation**: Measure current system performance
2. **Parameter Perturbation**: Generate population of perturbed parameters
3. **Fitness Assessment**: Evaluate each perturbation's impact
4. **Elite Selection**: Select top-performing parameter sets
5. **Parameter Update**: Update mean parameters toward elites
6. **Application**: Apply new parameters to target modules

## 📦 PlasticityController

### Core Functionality

```python
class PlasticityController:
    def __init__(self, target_modules, adaptation_rate=0.01, exploration_noise=0.1):
        self.target_modules = target_modules  # Modules to control
        self.es_optimizer = EvolutionaryStrategy()  # ES optimizer
        self.plasticity_params = self._initialize_params()  # Current parameters

    def adapt_plasticity(self, performance_metric):
        """Main adaptation loop."""
        # 1. Generate perturbed parameters
        # 2. Evaluate fitness
        # 3. Update parameters
        # 4. Apply to modules

    def _apply_params(self):
        """Apply parameters to target modules."""
```

### Key Features

1. **Meta-Learning**: Learns optimal plasticity parameters
2. **Evolutionary Optimization**: Uses ES for parameter search
3. **Dynamic Adaptation**: Continuously adapts to task requirements
4. **Multi-Module Control**: Manages multiple target modules

## 📦 EvolutionaryStrategy

### Core Algorithm

```python
class EvolutionaryStrategy:
    def __init__(self, population_size=50, sigma=0.1,
                 learning_rate=0.01, elite_fraction=0.2):
        self.population_size = population_size
        self.sigma = sigma  # Mutation strength
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction

    def generate_population(self, initial_mean):
        """Generate population through Gaussian perturbation."""
        population = []
        for _ in range(self.population_size):
            perturbation = np.random.randn(*initial_mean.shape) * self.sigma
            population.append(initial_mean + perturbation)
        return population

    def update_mean(self, current_mean, population, fitness_scores):
        """Update mean toward elite performers."""
        elites = self._select_elites(fitness_scores)
        self.mean = current_mean + self.learning_rate * np.mean(elites - current_mean, axis=0)
```

### Evolutionary Parameters

- **Population Size**: Number of parameter perturbations (50)
- **Sigma (σ)**: Mutation strength (0.1)
- **Learning Rate**: Mean update rate (0.01)
- **Elite Fraction**: Fraction of top performers to select (0.2)

## 📦 Plasticity Rules

### HebbianRule

```python
class HebbianRule(PlasticityRule):
    """Δw = η * pre * post"""

    def apply(self, weights, pre_activity, post_activity):
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update
```

### OjasRule

```python
class OjasRule(PlasticityRule):
    """Δw = η * post * (pre - post * w)"""

    def apply(self, weights, pre_activity, post_activity):
        weight_update = self.learning_rate * np.outer(
            post_activity,
            pre_activity - np.dot(weights.T, post_activity)
        )
        return weights + weight_update
```

### BCMRule

```python
class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro rule with sliding threshold."""

    def apply(self, weights, pre_activity, post_activity):
        self.theta = 0.9 * self.theta + 0.1 * np.mean(post_activity)
        weight_update = self.learning_rate * np.outer(
            post_activity * (post_activity - self.theta),
            pre_activity
        )
        return weights + weight_update
```

## 🔧 Configuration

### Default Plasticity Configuration

```python
DEFAULT_PLASTICITY_CONFIG = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,      # Long-Term Potentiation
    "ltd_coefficient": 0.8,      # Long-Term Depression
    "decay_rate": 0.001,         # Weight decay
    "homeostatic_strength": 0.1  # Homeostatic regulation
}

DEFAULT_ES_CONFIG = {
    "population_size": 50,
    "sigma": 0.1,
    "learning_rate": 0.01,
    "elite_fraction": 0.2
}
```

### Parameter Ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| learning_rate | [0.001, 0.1] | Base learning rate |
| ltp_coefficient | [0.5, 2.0] | LTP strength |
| ltd_coefficient | [0.1, 1.0] | LTD strength |
| decay_rate | [0.0001, 0.01] | Weight decay rate |
| homeostatic_strength | [0.01, 0.5] | Homeostatic regulation |

## 📊 Adaptation Process

### Parameter Evolution

```
Initial Parameters → Perturbation → Evaluation → Selection → Update → New Parameters
```

### Fitness Landscape

- **Performance Metrics**: Task accuracy, adaptation speed, stability
- **Multi-Objective Optimization**: Balance multiple performance aspects
- **Dynamic Fitness**: Adaptive fitness functions based on task requirements

## 🎯 Design Principles

### Meta-Learning
- Learn optimal plasticity parameters
- Adapt to different task distributions
- Continuous improvement over time

### Evolutionary Optimization
- Population-based search
- Parallel evaluation
- Robust to local optima

### Modular Control
- Control multiple modules simultaneously
- Module-specific parameter sets
- Dynamic module registration

## 🔮 Future Enhancements

### v4.1 Features
- **Advanced ES Variants**: CMA-ES, Natural Evolution Strategies
- **Multi-Objective Optimization**: Pareto front optimization
- **Adaptive Population Sizing**: Dynamic population adjustment

### v4.2 Features
- **Neuroevolution**: Direct neural architecture evolution
- **Plasticity Rule Discovery**: Automatic rule generation
- **Transfer Learning**: Cross-task plasticity adaptation

## 📋 Implementation Checklist

- [x] PlasticityController base class
- [x] EvolutionaryStrategy implementation
- [x] Basic plasticity rules (Hebbian, Oja, BCM)
- [x] Parameter application mechanism
- [ ] Advanced ES variants
- [ ] Multi-objective optimization
- [ ] Neuroevolution capabilities

## 🎯 Usage Example

```python
# Create target modules
hebbian_core = HebbianCore(input_size=20, output_size=40)
temporal_layer = TemporalLayer(input_size=40, output_size=80)

# Create plasticity controller
controller = PlasticityController(
    target_modules=[hebbian_core, temporal_layer],
    adaptation_rate=0.01,
    exploration_noise=0.1
)

# Adaptation loop
for episode in range(100):
    # Run task and get performance
    performance = run_task()

    # Adapt plasticity parameters
    controller.adapt_plasticity(performance)

    # Log progress
    print(f"Episode {episode}: Performance = {performance:.4f}")
```

## 📊 Performance Monitoring

### Key Metrics

1. **Adaptation Speed**: Time to reach target performance
2. **Stability**: Variance in performance over time
3. **Plasticity Efficiency**: Learning rate vs. forgetting rate
4. **Parameter Convergence**: Stability of plasticity parameters

### Visualization

```python
# Plot parameter evolution
params_history = controller.get_parameter_history()
plot_plasticity_parameters(params_history)

# Plot performance curve
plot_learning_curve(performance_history)
```

This plasticity system provides GAIA with powerful meta-learning capabilities for adaptive behavior!