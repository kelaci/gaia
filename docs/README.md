# GAIA Documentation Hub

## 🧠 Generalized Adaptive Intelligent Architecture

Welcome to the GAIA documentation! GAIA is a research-grade implementation of hierarchical neural architectures with **Hebbian plasticity**, **Active Inference**, and **meta-learning** capabilities.

---

## 📚 Documentation Structure

```
docs/
├── 📖 README.md                         ← You are here
│
├── 🏗️ architecture/                     ← System design & components
│   ├── overview.md                      Core architecture overview
│   ├── core-components.md               Module descriptions
│   ├── hierarchy.md                     Hierarchical processing
│   ├── plasticity-system.md             Plasticity control
│   └── advanced-plasticity.md           ⭐ PyTorch v3.1 implementation
│
├── 🔬 science/                          ← Theoretical foundations
│   └── theoretical-foundations.md       ⭐ FEP, Hebbian learning, quantization
│
├── 📘 guides/                           ← How-to guides
│   ├── quickstart.md                    Getting started
│   ├── validation.md                    ⭐ Testing & diagnostics
│   └── pytorch-integration.md           ⭐ NumPy ↔ PyTorch integration
│
├── 🔮 research/                         ← Future directions
│   └── future-directions.md             ⭐ Meta-plasticity, neuromodulation
│
└── 🛠️ development/                      ← Development info
    └── roadmap.md                       Version roadmap
```

---

## 🚀 Quick Links

### Getting Started
- **[Quickstart Guide](guides/quickstart.md)** - Get up and running in 5 minutes
- **[Architecture Overview](architecture/overview.md)** - Understand the system design

### Understanding GAIA
- **[Theoretical Foundations](science/theoretical-foundations.md)** - The science behind GAIA
- **[Advanced Plasticity](architecture/advanced-plasticity.md)** - PyTorch v3.1 deep dive

### Working with GAIA
- **[Validation Guide](guides/validation.md)** - Testing and diagnostics
- **[PyTorch Integration](guides/pytorch-integration.md)** - Using both implementations

### Research & Development
- **[Future Directions](research/future-directions.md)** - Research extensions
- **[Development Roadmap](development/roadmap.md)** - Version planning

---

## 🎯 GAIA Overview

### Two Implementations

| Implementation | Framework | Focus | Status |
|----------------|-----------|-------|--------|
| **v4.x** | NumPy | Modular hierarchy, prototyping | ✅ Stable |
| **v3.1** | PyTorch | GPU training, Active Inference | ✅ Stable |

### Key Features

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAIA FEATURES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🧬 PLASTICITY                    🎯 ACTIVE INFERENCE            │
│  • Dual-timescale traces         • Free Energy Principle        │
│  • Hebbian/Oja/BCM rules         • EFE-based action selection   │
│  • Homeostatic regulation        • Epistemic exploration        │
│                                                                 │
│  📊 HIERARCHY                     🔢 QUANTIZATION                │
│  • Multi-level processing        • BitNet 1.58-bit weights      │
│  • Temporal abstraction          • Hybrid digital-analog        │
│  • Inter-level communication     • Deployment efficient         │
│                                                                 │
│  🔬 META-LEARNING                 📈 DIAGNOSTICS                 │
│  • ES-based optimization         • Trace norm tracking          │
│  • Task adaptation               • Stability validation         │
│  • Performance tracking          • Learning curves              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Reading Order

### For Researchers
1. [Theoretical Foundations](science/theoretical-foundations.md) - Understand the science
2. [Architecture Overview](architecture/overview.md) - See how it's implemented
3. [Future Directions](research/future-directions.md) - Explore extensions

### For Engineers
1. [Quickstart Guide](guides/quickstart.md) - Get running quickly
2. [PyTorch Integration](guides/pytorch-integration.md) - Choose your implementation
3. [Validation Guide](guides/validation.md) - Test your setup

### For Contributors
1. [Development Roadmap](development/roadmap.md) - See what's planned
2. [Core Components](architecture/core-components.md) - Understand the codebase
3. [Advanced Plasticity](architecture/advanced-plasticity.md) - Deep dive into implementation

---

## 🔬 Scientific Foundation

GAIA implements concepts from:

| Domain | Concept | GAIA Component |
|--------|---------|----------------|
| Neuroscience | Hebbian Learning | `HebbianCore`, `DiagnosticPlasticLinear` |
| Neuroscience | Memory Consolidation | Dual-timescale traces |
| Neuroscience | Homeostatic Plasticity | Trace normalization |
| Cognitive Science | Free Energy Principle | `approximate_efe()` |
| Cognitive Science | Active Inference | `select_action()` |
| Machine Learning | Ensemble Methods | `EnsembleWorldModel` |
| Machine Learning | Quantization | BitNet 1.58-bit |

---

## 💻 Code Examples

### NumPy v4.x
```python
from gaia.layers.hebbian import HebbianCore
from gaia.hierarchy.manager import HierarchyManager

# Create hierarchy
manager = HierarchyManager()
level = HierarchicalLevel(0, 64, 128)
level.add_layer(HebbianCore(64, 128, plasticity_rule='bcm'))
manager.add_level(level)

# Process sequence
representations = manager.process_hierarchy(data, time_steps=100)
```

### PyTorch v3.1
```python
from gaia_protocol import GaiaAgentEnhanced, GaiaConfigEnhanced

# Configure
cfg = GaiaConfigEnhanced(
    state_dim=4,
    action_dim=2,
    fast_trace_decay=0.95,
    homeostatic_target=5.0
)

# Create agent
agent = GaiaAgentEnhanced(cfg)

# Train
for step in range(1000):
    action = agent.select_action(state)
    next_state = env.step(action)
    agent.learn(state, action, next_state)
```

---

## 🏆 Design Philosophy

### Biological Plausibility
- Hebbian learning rules ("neurons that fire together...")
- Dual-timescale memory (hippocampus + neocortex analogy)
- Homeostatic regulation (synaptic scaling)

### Computational Efficiency
- BitNet quantization (10x memory reduction)
- Online learning (no separate training phase)
- Modular architecture (plug-and-play components)

### Research Focus
- Clear mathematical formulations
- Comprehensive diagnostics
- Extensible design for new research

---

## 📊 Validation Status

```
✅ NumPy v4.x Tests          - All passing
✅ PyTorch v3.1 Validation   - Stable
✅ Trace Stability           - Bounded
✅ Gradient Stability        - No explosions
✅ Memory Stability          - No leaks
```

Run validation:
```bash
# NumPy tests
python test_gaia.py

# PyTorch validation
python -c "from gaia_protocol import run_comprehensive_validation; run_comprehensive_validation()"
```

---

## 🤝 Contributing

See [Development Roadmap](development/roadmap.md) for:
- Current development priorities
- Contribution guidelines
- Coding standards

---

## 📚 References

Key papers informing GAIA's design:

1. **Friston, K. (2010)** - Free Energy Principle
2. **Hebb, D.O. (1949)** - Hebbian Learning
3. **McClelland et al. (1995)** - Complementary Learning Systems
4. **Ma et al. (2024)** - BitNet: 1-bit LLMs

---

## 📧 Contact

- **Repository**: [github.com/kelaci/gaia](https://github.com/kelaci/gaia)
- **Issues**: [GitHub Issues](https://github.com/kelaci/gaia/issues)

---

*GAIA v4.1.0 | Research-grade hierarchical neural architecture with plasticity*