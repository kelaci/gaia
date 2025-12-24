# PyTorch Integration Guide

## 🔗 Overview

GAIA exists in two complementary implementations:
- **NumPy v4.x**: Modular framework for research and prototyping
- **PyTorch v3.1**: Production-ready implementation with GPU support

This guide explains how they relate and when to use each.

---

## 1. Implementation Comparison

### 1.1 Feature Matrix

| Feature | NumPy v4.x | PyTorch v3.1 |
|---------|------------|--------------|
| **GPU Acceleration** | ❌ | ✅ |
| **Automatic Differentiation** | ❌ | ✅ |
| **BitNet Quantization** | ❌ | ✅ |
| **Dual-Timescale Traces** | ✅ | ✅ |
| **Active Inference** | Partial | ✅ Full |
| **Ensemble Uncertainty** | ❌ | ✅ |
| **Diagnostic Tracking** | Basic | ✅ Comprehensive |
| **Hierarchical Processing** | ✅ Full | Partial |
| **Meta-Learning Framework** | ✅ | ✅ |

### 1.2 Architecture Mapping

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         GAIA Architecture Mapping                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   NumPy v4.x                          PyTorch v3.1                       │
│   ──────────                          ────────────                       │
│                                                                          │
│   HebbianCore          ◄────────────► DiagnosticPlasticLinear           │
│   • pre/post traces                   • fast/slow traces                 │
│   • Hebbian/Oja/BCM                   • BitNet + Hebbian                 │
│                                                                          │
│   TemporalLayer        ◄────────────► EnhancedDeepPlasticMember.l2      │
│   • Recurrent weights                 • Layer with temporal context      │
│   • Time window                       • Hidden state                     │
│                                                                          │
│   PlasticityController ◄────────────► GaiaAgentEnhanced                 │
│   • ES optimizer                      • Active Inference                 │
│   • Parameter adaptation              • EFE-based selection              │
│                                                                          │
│   HierarchyManager     ◄────────────► (Not directly mapped)             │
│   • Multi-level processing            • Could use stacked agents         │
│                                                                          │
│   MetaOptimizer        ◄────────────► Meta-learning via outer loop      │
│   • Task distribution                 • Performance tracking             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. When to Use Each

### 2.1 Use NumPy v4.x When:

✅ **Prototyping** new plasticity rules  
✅ **Educational** purposes and understanding  
✅ **CPU-only** environments  
✅ **Hierarchical** multi-level processing is needed  
✅ **Debugging** mathematical details  

```python
# NumPy v4.x example
from gaia.layers.hebbian import HebbianCore
from gaia.hierarchy.manager import HierarchyManager

# Create hierarchy
manager = HierarchyManager()
level0 = HierarchicalLevel(0, 10, 20)
level0.add_layer(ReactiveLayer(10, 20))
manager.add_level(level0)

# Process
representations = manager.process_hierarchy(input_data, time_steps=100)
```

### 2.2 Use PyTorch v3.1 When:

✅ **GPU acceleration** needed  
✅ **Training** in production environments  
✅ **Active Inference** action selection  
✅ **Uncertainty estimation** via ensembles  
✅ **Integration** with other PyTorch models  

```python
# PyTorch v3.1 example
from gaia_protocol import GaiaAgentEnhanced, GaiaConfigEnhanced

# Create agent
cfg = GaiaConfigEnhanced(state_dim=4, action_dim=2)
agent = GaiaAgentEnhanced(cfg)

# Training loop
for step in range(1000):
    action = agent.select_action(state)
    next_state = env.step(action)
    agent.learn(state, action, next_state)
```

---

## 3. Porting Code

### 3.1 NumPy → PyTorch

**HebbianCore → DiagnosticPlasticLinear:**

```python
# NumPy v4.x
class HebbianCore:
    def forward(self, x):
        output = np.dot(x, self.weights.T)
        self.pre_synaptic = x.mean(axis=0)
        self.post_synaptic = output.mean(axis=0)
        return output
    
    def update(self, lr):
        weight_update = lr * np.outer(self.post_synaptic, self.pre_synaptic)
        self.weights += weight_update

# PyTorch v3.1 equivalent
class DiagnosticPlasticLinear:
    def forward(self, x):
        w_static = self.bitnet_quantize(self.weight)
        w_plastic = 0.1 * self.fast_trace + 0.05 * self.slow_trace
        y = F.linear(x, w_static + w_plastic)
        self.update_traces(x, y)  # Online update
        return y
```

**Key Differences:**
- PyTorch uses `F.linear()` instead of `np.dot()`
- Traces updated online during forward pass
- Quantization added for efficiency
- Dual timescales (fast/slow) instead of single trace

### 3.2 PyTorch → NumPy

**DiagnosticPlasticLinear → HebbianCore:**

```python
# PyTorch v3.1
def update_traces(self, x, y):
    with torch.no_grad():
        y_active = F.relu(y)
        delta = torch.matmul(y_active.t(), x) / x.shape[0]
        self.fast_trace.mul_(self.cfg.fast_trace_decay)
        self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)

# NumPy v4.x equivalent
def update(self, lr):
    # Compute Hebbian delta
    y_active = np.maximum(0, self.post_synaptic)
    delta = np.outer(y_active, self.pre_synaptic)
    
    # Update with decay (mimicking fast trace)
    self.weight_update = 0.95 * self.weight_update + 0.05 * delta
    self.weights += lr * self.weight_update
```

---

## 4. Hybrid Usage

### 4.1 Using Both Implementations

```python
# Research workflow
import numpy as np
import torch
from gaia.layers.hebbian import HebbianCore  # NumPy for prototyping
from gaia_protocol import DiagnosticPlasticLinear  # PyTorch for training

# 1. Prototype with NumPy
numpy_layer = HebbianCore(10, 20, plasticity_rule='bcm')
# ... experiment with different rules

# 2. Port to PyTorch for training
pytorch_layer = DiagnosticPlasticLinear(10, 20, cfg)

# 3. Transfer learned weights (if applicable)
pytorch_layer.weight.data = torch.from_numpy(numpy_layer.weights.T).float()
```

### 4.2 Shared Configuration

```python
# config.py - Shared configuration
from dataclasses import dataclass

@dataclass
class UnifiedGaiaConfig:
    """Configuration compatible with both implementations"""
    
    # Dimensions
    input_dim: int = 10
    hidden_dim: int = 64
    output_dim: int = 10
    
    # Plasticity
    learning_rate: float = 0.01
    fast_decay: float = 0.95
    slow_decay: float = 0.99
    homeostatic_target: float = 5.0
    
    def to_numpy_config(self):
        """Convert to NumPy v4.x format"""
        return {
            'learning_rate': self.learning_rate,
            'decay_rate': 1 - self.fast_decay,  # NumPy uses decay_rate
            'homeostatic_strength': 0.1,
        }
    
    def to_pytorch_config(self):
        """Convert to PyTorch v3.1 format"""
        from gaia_protocol import GaiaConfigEnhanced
        return GaiaConfigEnhanced(
            hidden_dim=self.hidden_dim,
            fast_trace_decay=self.fast_decay,
            slow_trace_decay=self.slow_decay,
            fast_trace_lr=self.learning_rate,
            homeostatic_target=self.homeostatic_target,
        )
```

---

## 5. Integration Patterns

### 5.1 NumPy Hierarchy + PyTorch Plasticity

```python
"""
Use NumPy for hierarchical structure,
PyTorch for plastic learning within levels
"""

import numpy as np
import torch
from gaia.hierarchy.manager import HierarchyManager
from gaia_protocol import DiagnosticPlasticLinear, GaiaConfigEnhanced

class HybridHierarchicalLevel:
    """Hierarchical level with PyTorch plastic core"""
    
    def __init__(self, level_id, input_size, output_size, cfg):
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        
        # PyTorch plastic layer
        self.plastic_layer = DiagnosticPlasticLinear(
            input_size, output_size, cfg
        ).to(device)
    
    def process_time_step(self, input_data, t):
        # Convert numpy to torch
        x_torch = torch.from_numpy(input_data).float().to(device)
        if x_torch.dim() == 1:
            x_torch = x_torch.unsqueeze(0)
        
        # Process through PyTorch layer
        output = self.plastic_layer(x_torch)
        
        # Convert back to numpy
        return output.detach().cpu().numpy()

# Usage
cfg = GaiaConfigEnhanced()
manager = HierarchyManager()

level0 = HybridHierarchicalLevel(0, 10, 20, cfg)
manager.add_level(level0)  # Would need adapter
```

### 5.2 PyTorch Agent with NumPy Analysis

```python
"""
Use PyTorch for training,
NumPy for offline analysis
"""

from gaia_protocol import GaiaAgentEnhanced
import numpy as np
from gaia.utils.visualization import plot_weight_matrix

# Train with PyTorch
agent = GaiaAgentEnhanced(cfg)
for step in range(1000):
    # ... training loop

# Analyze with NumPy tools
for i, model in enumerate(agent.wm.models):
    weights_np = model.l1.weight.detach().cpu().numpy()
    fast_trace_np = model.l1.fast_trace.cpu().numpy()
    
    # Use NumPy visualization
    plot_weight_matrix(weights_np, title=f"Model {i} Weights")
    plot_weight_matrix(fast_trace_np, title=f"Model {i} Fast Trace")
```

---

## 6. Migration Guide

### 6.1 From NumPy v4.x to PyTorch v3.1

**Step 1: Install PyTorch dependencies**
```bash
pip install torch torchvision
```

**Step 2: Create configuration**
```python
# Old NumPy config
numpy_config = {
    'learning_rate': 0.01,
    'decay_rate': 0.001,
}

# New PyTorch config
pytorch_config = GaiaConfigEnhanced(
    fast_trace_lr=0.01,
    fast_trace_decay=0.95,  # 1 - decay_rate * 50 (roughly)
)
```

**Step 3: Update layer creation**
```python
# Old
from gaia.layers.hebbian import HebbianCore
layer = HebbianCore(10, 20)

# New
from gaia_protocol import DiagnosticPlasticLinear
layer = DiagnosticPlasticLinear(10, 20, cfg)
```

**Step 4: Update forward pass**
```python
# Old
output = layer.forward(input_data)
layer.update(0.01)

# New (update happens in forward)
output = layer(torch.from_numpy(input_data).float())
```

### 6.2 From PyTorch v3.1 to NumPy v4.x

**Step 1: Extract weights**
```python
weights = layer.weight.detach().cpu().numpy()
fast_trace = layer.fast_trace.cpu().numpy()
```

**Step 2: Create NumPy layer**
```python
from gaia.layers.hebbian import HebbianCore
numpy_layer = HebbianCore(in_size, out_size)
numpy_layer.weights = weights.T  # Transpose for NumPy convention
```

**Step 3: Port plasticity logic**
```python
# PyTorch fast trace → NumPy activity history
numpy_layer.activity_history = [(fast_trace, fast_trace)]
```

---

## 7. Advanced Integration

### 7.1 Custom PyTorch Module with NumPy Analysis

```python
class AnalyzablePlasticLinear(DiagnosticPlasticLinear):
    """
    PyTorch layer with NumPy-compatible analysis methods
    """
    
    def compute_correlation_matrix(self) -> np.ndarray:
        """Compute weight correlation using NumPy"""
        weights = self.weight.detach().cpu().numpy()
        return np.corrcoef(weights)
    
    def compute_sparsity(self) -> float:
        """Compute weight sparsity"""
        weights = self.weight.detach().cpu().numpy()
        return np.mean(np.abs(weights) < 0.01)
    
    def export_for_analysis(self) -> dict:
        """Export all data for NumPy analysis"""
        return {
            'weights': self.weight.detach().cpu().numpy(),
            'fast_trace': self.fast_trace.cpu().numpy(),
            'slow_trace': self.slow_trace.cpu().numpy(),
            'trace_norm_history': self.trace_norm_history.cpu().numpy(),
            'update_magnitude_history': self.update_magnitude_history.cpu().numpy(),
        }
```

### 7.2 Unified Testing Framework

```python
"""
Test both implementations with same test cases
"""

import numpy as np
import torch
import pytest

class TestPlasticityUnified:
    """Unified tests for both implementations"""
    
    @pytest.fixture
    def numpy_layer(self):
        from gaia.layers.hebbian import HebbianCore
        return HebbianCore(10, 20)
    
    @pytest.fixture
    def pytorch_layer(self):
        from gaia_protocol import DiagnosticPlasticLinear, GaiaConfigEnhanced
        cfg = GaiaConfigEnhanced()
        return DiagnosticPlasticLinear(10, 20, cfg)
    
    def test_output_shape_numpy(self, numpy_layer):
        x = np.random.randn(5, 10)
        y = numpy_layer.forward(x)
        assert y.shape == (5, 20)
    
    def test_output_shape_pytorch(self, pytorch_layer):
        x = torch.randn(5, 10)
        y = pytorch_layer(x)
        assert y.shape == (5, 20)
    
    def test_trace_bounded_numpy(self, numpy_layer):
        for _ in range(100):
            x = np.random.randn(5, 10)
            numpy_layer.forward(x)
            numpy_layer.update(0.01)
        
        # NumPy uses homeostatic normalization in update()
        assert np.linalg.norm(numpy_layer.weights) < 100
    
    def test_trace_bounded_pytorch(self, pytorch_layer):
        for _ in range(100):
            x = torch.randn(5, 10)
            pytorch_layer(x)
        
        assert pytorch_layer.fast_trace.norm().item() <= 5.5  # Near homeostatic target
```

---

## 8. Best Practices

### 8.1 Development Workflow

```
1. PROTOTYPE (NumPy v4.x)
   └── Rapid iteration on algorithms
   └── Easy debugging and visualization
   └── Validate mathematical correctness

2. PORT (To PyTorch v3.1)
   └── Implement GPU-compatible version
   └── Add autodiff capabilities
   └── Integrate with training pipeline

3. TRAIN (PyTorch v3.1)
   └── Use GPU acceleration
   └── Leverage ensemble uncertainty
   └── Active Inference action selection

4. ANALYZE (Both)
   └── NumPy for visualization
   └── PyTorch for gradient analysis
   └── Compare implementations
```

### 8.2 Code Organization

```
gaia/
├── gaia/                    # NumPy v4.x
│   ├── layers/
│   ├── hierarchy/
│   ├── plasticity/
│   └── ...
├── gaia_torch/              # PyTorch v3.1
│   ├── layers/
│   │   └── plastic_linear.py
│   ├── models/
│   │   └── ensemble.py
│   ├── agents/
│   │   └── active_inference.py
│   └── ...
├── gaia_protocol.py         # Single-file PyTorch implementation
└── tests/
    ├── test_numpy.py
    ├── test_pytorch.py
    └── test_integration.py
```

---

*For theoretical background, see [Theoretical Foundations](../science/theoretical-foundations.md). For validation procedures, see [Validation Guide](validation.md).*