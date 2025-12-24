# Quick Start Guide

## 🚀 Getting Started with GAIA

Welcome to GAIA! This guide will help you set up and run your first GAIA experiment.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB+ recommended
- **Storage**: 1GB+ free space

### Python Dependencies
```
numpy>=1.20.0
matplotlib>=3.4.0
scipy>=1.7.0
```

## 🛠️ Installation

### From Source

```bash
# Clone the repository
git clone git@github.com:kelaci/gaia.git
cd gaia

# Install dependencies
pip install -r requirements.txt

# Install GAIA in development mode
pip install -e .
```

### Using pip (coming soon)
```bash
pip install gaia-ai
```

## 🎯 Basic Usage

### Import GAIA Components

```python
import numpy as np
from gaia.layers.reactive import ReactiveLayer
from gaia.layers.hebbian import HebbianCore
from gaia.layers.temporal import TemporalLayer
from gaia.hierarchy.level import HierarchicalLevel
from gaia.hierarchy.manager import HierarchyManager
from gaia.plasticity.controller import PlasticityController
```

### Create a Simple Hierarchy

```python
# Initialize hierarchy manager
manager = HierarchyManager()

# Level 0: Input processing
level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
level0.add_layer(ReactiveLayer(10, 20, activation='relu'))
manager.add_level(level0)

# Level 1: Feature extraction
level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
level1.add_layer(HebbianCore(20, 40, plasticity_rule='hebbian'))
manager.add_level(level1)

# Level 2: Sequence processing
level2 = HierarchicalLevel(2, input_size=40, output_size=80, temporal_resolution=4)
level2.add_layer(TemporalLayer(40, 80, time_window=5))
manager.add_level(level2)
```

### Process Input Data

```python
# Generate random input data (100 time steps, 10 features)
time_steps = 100
input_data = np.random.randn(time_steps, 10)

# Process through hierarchy
print("Processing input through hierarchy...")
representations = manager.process_hierarchy(input_data, time_steps)

# Display results
print("\nHierarchical representations:")
for level_id, reps in representations.items():
    print(f"Level {level_id}: {len(reps)} representations, shape {reps[0].shape}")
```

## 🔧 Plasticity Control

### Create Plasticity Controller

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
```

### Adaptation Loop

```python
def simple_task(performance_history):
    """Simple task function for demonstration."""
    # Simulate task performance based on current parameters
    current_performance = 0.5 + 0.5 * np.random.random()
    if len(performance_history) > 0:
        current_performance = 0.8 * performance_history[-1] + 0.2 * current_performance
    return current_performance

# Run adaptation loop
performance_history = []
for episode in range(50):
    # Simulate task performance
    performance = simple_task(performance_history)
    performance_history.append(performance)

    # Adapt plasticity parameters
    controller.adapt_plasticity(performance)

    # Log progress
    if episode % 10 == 0:
        print(f"Episode {episode}: Performance = {performance:.4f}")

print("\nAdaptation completed!")
```

## 📊 Visualization

### Plot Learning Curve

```python
from gaia.utils.visualization import plot_learning_curve

plot_learning_curve(performance_history, title="Plasticity Adaptation")
```

### Plot Hierarchy Representations

```python
from gaia.utils.visualization import plot_hierarchy_representations

# Use a subset of representations for visualization
sample_reps = {level: reps[:10] for level, reps in representations.items()}
plot_hierarchy_representations(sample_reps)
```

## 🎓 Advanced Example: Meta-Learning

### Meta-Optimization Setup

```python
from gaia.meta_learning.optimizer import MetaOptimizer

# Create meta-optimizer
meta_optimizer = MetaOptimizer(controller)

# Define task distribution
def create_task(task_id):
    """Create a simple task function."""
    def task(step):
        # Task returns performance based on step and task_id
        base_performance = 0.5 + 0.1 * task_id
        noise = 0.1 * np.random.randn()
        return base_performance + noise
    return task

# Create multiple tasks
tasks = [create_task(i) for i in range(5)]
```

### Run Meta-Learning

```python
# Meta-training loop
num_episodes = 20
meta_optimizer.meta_train(num_episodes, tasks)

# Evaluate meta-learning performance
metrics = meta_optimizer.evaluate_meta_performance()
print("\nMeta-Learning Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

## 🔍 Debugging and Logging

### Setup Logging

```python
from gaia.utils.logging import setup_logging

# Setup comprehensive logging
logger = setup_logging(
    name='gaia_demo',
    level='INFO',
    log_file='gaia_demo.log'
)

logger.info("Starting GAIA demo")
```

### Log Tensor Statistics

```python
from gaia.utils.logging import log_tensor_stats

# Log statistics about input data
log_tensor_stats(logger, input_data, "input_data", level='INFO')
```

## 📚 Next Steps

### Explore the Architecture
- [Architecture Overview](../architecture/overview.md)
- [Core Components](../architecture/core-components.md)
- [Hierarchy System](../architecture/hierarchy.md)
- [Plasticity System](../architecture/plasticity-system.md)

### Try Advanced Features
- Experiment with different plasticity rules
- Create custom hierarchical configurations
- Implement your own tasks for meta-learning
- Extend GAIA with new layer types

### Contribute to GAIA
- [Development Roadmap](../development/roadmap.md)
- [Contributing Guidelines](../development/contributing.md)
- [Issue Tracker](https://github.com/kelaci/gaia/issues)

## 🎯 Troubleshooting

### Common Issues

**Import Errors**
- Ensure GAIA is installed (`pip install -e .`)
- Check Python path includes the gaia directory

**Performance Issues**
- Reduce input data size for testing
- Check for memory leaks with `tracemalloc`
- Profile code with `cProfile`

**Visualization Problems**
- Ensure matplotlib is installed (`pip install matplotlib`)
- Check for display issues in headless environments

### Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check the comprehensive docs
- **Community**: Join the Slack channel (link coming soon)

## 📋 Checklist for Your First GAIA Project

1. [ ] Install GAIA and dependencies
2. [ ] Run the basic hierarchy example
3. [ ] Experiment with plasticity control
4. [ ] Try the meta-learning example
5. [ ] Create your own custom hierarchy
6. [ ] Implement a simple task
7. [ ] Visualize your results
8. [ ] Share your findings with the community!

This quick start guide provides everything you need to begin exploring GAIA's hierarchical neural architecture and meta-learning capabilities!