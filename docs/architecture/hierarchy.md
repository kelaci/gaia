# Hierarchy System

## 🎯 Hierarchical Processing Overview

The hierarchy system is the core of GAIA's temporal abstraction capability, enabling processing at multiple time scales and levels of abstraction.

## 🏗️ Hierarchy Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            HierarchyManager                                │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 3 (High-level)                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer (8x)                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 2 (Intermediate)                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer (4x)                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 1 (Low-level)                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianCore (2x)                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 0 (Input)                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     ReactiveLayer (1x)                          │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Temporal Abstraction Mechanism

### Time Compression
- Each level processes input at a different temporal resolution
- Level 0: 1x (every time step)
- Level 1: 2x (every 2nd time step)
- Level 2: 4x (every 4th time step)
- Level 3: 8x (every 8th time step)

### Information Flow
```
Input → Level 0 → Level 1 → Level 2 → Level 3
  ↓       ↓        ↓         ↓
Raw    Features  Sequences  Concepts
```

## 📦 HierarchyManager

### Core Functionality

```python
class HierarchyManager:
    def __init__(self):
        self.levels = []  # List of HierarchicalLevel instances
        self.communication_schedule = {}

    def add_level(self, level: HierarchicalLevel):
        """Add a level to the hierarchy with automatic relationship management."""

    def process_hierarchy(self, input_data, time_steps):
        """Process input through all levels with temporal abstraction."""

    def _hierarchical_communication(self, t):
        """Handle bidirectional communication between levels."""
```

### Key Features

1. **Automatic Relationship Management**: Automatically sets parent/child relationships
2. **Temporal Processing**: Handles different temporal resolutions
3. **Communication Protocol**: Manages inter-level communication
4. **State Management**: Coordinates state across all levels

## 📦 HierarchicalLevel

### Core Functionality

```python
class HierarchicalLevel:
    def __init__(self, level_id, input_size, output_size, temporal_resolution=1):
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_resolution = temporal_resolution
        self.parent_level = None
        self.child_levels = []
        self.processing_layers = []

    def process_time_step(self, input_data, t):
        """Process input only at appropriate temporal resolution."""

    def communicate_with_parent(self):
        """Send representation to parent level."""

    def communicate_with_children(self, data):
        """Receive data from parent and distribute to children."""
```

### Processing Flow

1. **Temporal Filtering**: Only process every `temporal_resolution` time steps
2. **Layer Processing**: Sequential processing through all layers
3. **Representation Storage**: Maintain current representation
4. **Communication**: Bidirectional communication with adjacent levels

## 🔧 Configuration

### Default Hierarchy Configuration

```python
DEFAULT_HIERARCHY_CONFIG = {
    "num_levels": 4,
    "temporal_compression": 2,  # Each level compresses time by this factor
    "base_resolution": 1,       # Base temporal resolution
    "level_sizes": [64, 128, 256, 512]  # Feature sizes at each level
}
```

### Custom Configuration Example

```python
custom_config = {
    "num_levels": 3,
    "temporal_compression": 3,  # More aggressive compression
    "base_resolution": 1,
    "level_sizes": [128, 256, 512],
    "communication_interval": 5  # Communicate every 5 time steps
}
```

## 📊 Communication Protocols

### Bidirectional Communication

1. **Bottom-Up**: Raw data → Features → Sequences → Concepts
2. **Top-Down**: Contextual information → Expectations → Attention

### Communication Timing

- **Synchronous**: All levels communicate at each time step
- **Asynchronous**: Levels communicate based on temporal resolution
- **Scheduled**: Communication at specific intervals

## 🎯 Design Principles

### Modularity
- Each level operates independently
- Clear interfaces between levels
- Easy to add/remove levels

### Scalability
- Linear scaling with number of levels
- Efficient memory usage
- Parallel processing capabilities

### Flexibility
- Configurable temporal resolutions
- Customizable layer compositions
- Adaptable communication protocols

## 🔮 Future Enhancements

### v4.1 Features
- **Attention Mechanisms**: Selective focus at different levels
- **Dynamic Hierarchies**: Adaptive level creation/destruction
- **Cross-Level Learning**: Shared learning across levels

### v4.2 Features
- **Multi-Modal Hierarchies**: Separate hierarchies for different modalities
- **Hierarchy Optimization**: Automatic level configuration
- **Memory Systems**: Long-term memory integration

## 📋 Implementation Checklist

- [x] HierarchyManager base class
- [x] HierarchicalLevel implementation
- [x] Temporal abstraction mechanism
- [x] Basic communication protocol
- [ ] Advanced communication strategies
- [ ] Attention mechanisms
- [ ] Dynamic hierarchy management

## 🎯 Usage Example

```python
# Create hierarchy manager
manager = HierarchyManager()

# Add levels with different temporal resolutions
level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
level0.add_layer(ReactiveLayer(10, 20))
manager.add_level(level0)

level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
level1.add_layer(HebbianCore(20, 40))
manager.add_level(level1)

# Process input data
input_data = np.random.randn(100, 10)  # 100 time steps, 10 features
representations = manager.process_hierarchy(input_data, 100)

# Access results
level0_reps = representations[0]  # 100 representations
level1_reps = representations[1]  # 50 representations (every 2nd step)
```

This hierarchy system provides the foundation for GAIA's temporal abstraction and multi-scale processing capabilities!