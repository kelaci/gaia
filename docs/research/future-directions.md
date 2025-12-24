# Research Extensions & Future Directions

## 🔮 Overview

This document outlines promising research directions for extending GAIA's capabilities. These extensions are grounded in neuroscience literature and aim to enhance the system's learning efficiency, adaptability, and biological plausibility.

---

## 1. Meta-Plasticity: Learning to Learn

### 1.1 Concept

**Meta-plasticity** is the "plasticity of plasticity" — the ability to dynamically adjust learning rules themselves based on experience.

### 1.2 Current GAIA State

GAIA currently uses fixed trace decay rates:
```python
fast_trace_decay = 0.95  # Fixed
slow_trace_decay = 0.99  # Fixed
```

### 1.3 Proposed Extension

**Learnable Decay Rates:**

```python
class MetaPlasticLinear(DiagnosticPlasticLinear):
    """
    Extension with learnable plasticity parameters
    """
    
    def __init__(self, in_features: int, out_features: int, cfg: GaiaConfig):
        super().__init__(in_features, out_features, cfg)
        
        # Meta-parameters (learnable)
        self.log_fast_decay = nn.Parameter(torch.tensor(np.log(0.95)))
        self.log_slow_decay = nn.Parameter(torch.tensor(np.log(0.99)))
        self.log_fast_lr = nn.Parameter(torch.tensor(np.log(0.05)))
        
    @property
    def fast_decay(self):
        return torch.sigmoid(self.log_fast_decay)  # Ensures (0, 1)
    
    @property
    def slow_decay(self):
        return torch.sigmoid(self.log_slow_decay)
```

**Meta-Learning Objective:**

```python
def meta_loss(agent, task_batch):
    """
    MAML-style meta-learning for plasticity parameters
    
    Objective: Maximize adaptation speed across tasks
    """
    meta_grad = 0
    
    for task in task_batch:
        # Inner loop: adapt with current plasticity params
        adapted_agent = inner_adapt(agent, task, steps=10)
        
        # Outer loop: evaluate adaptation quality
        performance = evaluate(adapted_agent, task.test_set)
        meta_grad += compute_grad(performance, agent.meta_params)
    
    return meta_grad / len(task_batch)
```

### 1.4 Expected Benefits

| Benefit | Description |
|---------|-------------|
| Task-Adaptive | Decay rates adjust per-task |
| Sample Efficiency | Faster adaptation to new domains |
| Biological Plausibility | Mirrors BCM-style metaplasticity |

### 1.5 Research Questions

1. What is the optimal timescale for meta-parameter updates?
2. Should meta-parameters be global or per-layer?
3. How to prevent meta-parameter collapse?

---

## 2. Attention over Trace History

### 2.1 Concept

Instead of simple exponential decay, use **attention mechanisms** to selectively retrieve relevant past experiences from trace history.

### 2.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attention-Based Trace                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Trace History: [T₁, T₂, T₃, ..., Tₙ]                        │
│                      │                                          │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │    Query      │                                  │
│   Current ──▶│   Attention   │──▶ Weighted Trace               │
│   Context    │   (Q, K, V)   │                                  │
│              └───────────────┘                                  │
│                                                                 │
│   Output: ∑ᵢ αᵢ · Tᵢ  where αᵢ = softmax(Q · Kᵢ)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation Sketch

```python
class AttentiveTraceMemory(nn.Module):
    """
    Attention-based trace retrieval
    
    Instead of exponential decay, selectively attend to
    relevant historical traces based on current context.
    """
    
    def __init__(self, trace_dim: int, history_len: int = 100, n_heads: int = 4):
        super().__init__()
        self.history_len = history_len
        
        # Trace history buffer
        self.register_buffer("trace_history", 
                           torch.zeros(history_len, trace_dim))
        self.write_ptr = 0
        
        # Attention components
        self.query_proj = nn.Linear(trace_dim, trace_dim)
        self.key_proj = nn.Linear(trace_dim, trace_dim)
        self.value_proj = nn.Linear(trace_dim, trace_dim)
        self.n_heads = n_heads
    
    def write(self, trace: torch.Tensor):
        """Store new trace in history"""
        self.trace_history[self.write_ptr] = trace.detach()
        self.write_ptr = (self.write_ptr + 1) % self.history_len
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve weighted combination of historical traces"""
        Q = self.query_proj(query)
        K = self.key_proj(self.trace_history)
        V = self.value_proj(self.trace_history)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.T) / np.sqrt(Q.shape[-1])
        weights = F.softmax(scores, dim=-1)
        
        return torch.matmul(weights, V)
```

### 2.4 Biological Inspiration

This mirrors **hippocampal memory indexing**:
- Hippocampus stores episodic traces
- Retrieval is cue-dependent (attention-like)
- Consolidation is selective (not all traces equal)

### 2.5 Research Questions

1. What should the query be? (Current input? Prediction error?)
2. How to balance recency vs. relevance?
3. Computational cost vs. benefit tradeoff?

---

## 3. Multi-Scale Temporal Hierarchies

### 3.1 Concept

Extend from dual-timescale (fast/slow) to **multi-scale** temporal hierarchies with ultrafast, fast, medium, slow, and ultraslow traces.

### 3.2 Biological Motivation

Different brain regions operate at different timescales:

| Timescale | Duration | Brain Region | Function |
|-----------|----------|--------------|----------|
| Ultrafast | ~10ms | Sensory cortex | Immediate perception |
| Fast | ~100ms | PFC | Working memory |
| Medium | ~1s | Hippocampus | Episode encoding |
| Slow | ~minutes | Striatum | Skill learning |
| Ultraslow | ~hours/days | Neocortex | Semantic memory |

### 3.3 Proposed Extension

```python
class MultiScaleTraceLayer(nn.Module):
    """
    Multi-scale temporal trace system
    
    Implements 5 timescales: ultrafast → ultraslow
    """
    
    TIMESCALES = {
        'ultrafast': {'decay': 0.8, 'lr': 0.1, 'weight': 0.2},
        'fast':      {'decay': 0.95, 'lr': 0.05, 'weight': 0.15},
        'medium':    {'decay': 0.99, 'lr': 0.02, 'weight': 0.1},
        'slow':      {'decay': 0.999, 'lr': 0.005, 'weight': 0.05},
        'ultraslow': {'decay': 0.9999, 'lr': 0.001, 'weight': 0.02},
    }
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Static weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Multi-scale traces
        for name, params in self.TIMESCALES.items():
            self.register_buffer(f"{name}_trace", 
                               torch.zeros(out_features, in_features))
    
    def update_all_traces(self, hebbian_delta: torch.Tensor):
        """Update all timescale traces"""
        with torch.no_grad():
            for name, params in self.TIMESCALES.items():
                trace = getattr(self, f"{name}_trace")
                trace.mul_(params['decay'])
                trace.add_(hebbian_delta, alpha=params['lr'])
                
                # Homeostatic normalization per scale
                norm = trace.norm()
                if norm > 5.0 * params['weight']:
                    trace.mul_(5.0 * params['weight'] / (norm + 1e-6))
    
    def get_effective_modulation(self) -> torch.Tensor:
        """Combine all traces weighted by timescale importance"""
        modulation = torch.zeros_like(self.weight)
        for name, params in self.TIMESCALES.items():
            trace = getattr(self, f"{name}_trace")
            modulation += params['weight'] * trace
        return modulation
```

### 3.4 Hierarchy Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Scale Temporal Hierarchy                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Ultraslow (τ=0.9999) ─────────────────────────────────────── │
│   Slow (τ=0.999) ─────────────────────────────────             │
│   Medium (τ=0.99) ─────────────────────────                    │
│   Fast (τ=0.95) ─────────────────                              │
│   Ultrafast (τ=0.8) ─────────                                  │
│                                                                 │
│   Time →                                                        │
│   ════════════════════════════════════════════════════════════ │
│                                                                 │
│   Effective = Σᵢ wᵢ · Traceᵢ                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Research Questions

1. Optimal number of timescales?
2. How should weights be distributed across scales?
3. Should timescales be task-dependent?

---

## 4. Synaptic Tagging & Capture

### 4.1 Concept

**Synaptic Tagging and Capture (STC)** is a biological mechanism for selective memory consolidation:
- Strong activation creates a "tag" at synapses
- Tags capture plasticity-related proteins (PRPs)
- Only tagged synapses undergo long-term consolidation

### 4.2 Implementation Sketch

```python
class TaggingPlasticLinear(DiagnosticPlasticLinear):
    """
    Synaptic Tagging and Capture implementation
    
    Key mechanisms:
    1. Tags set by strong Hebbian updates
    2. PRPs (proteins) enable consolidation
    3. Only tagged synapses consolidate to slow trace
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Synaptic tags (binary mask)
        self.register_buffer("synaptic_tags", 
                           torch.zeros_like(self.fast_trace))
        
        # Tag decay (tags fade without capture)
        self.tag_decay = 0.95
        
        # Tag threshold
        self.tag_threshold = 0.1
        
        # PRP availability (global signal)
        self.prp_level = 0.0
    
    def update_traces(self, x: torch.Tensor, y: torch.Tensor):
        """Enhanced update with tagging mechanism"""
        with torch.no_grad():
            # Standard Hebbian update
            y_active = F.relu(y)
            delta = torch.matmul(y_active.t(), x) / x.shape[0]
            
            # === Tagging ===
            # Strong updates create tags
            strong_update_mask = delta.abs() > self.tag_threshold
            self.synaptic_tags[strong_update_mask] = 1.0
            
            # Decay existing tags
            self.synaptic_tags *= self.tag_decay
            
            # === Fast trace (always updated) ===
            self.fast_trace.mul_(self.cfg.fast_trace_decay)
            self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)
            
            # === Slow trace (only tagged synapses) ===
            if self.prp_level > 0.5:  # PRPs available
                tagged_fast = self.fast_trace * self.synaptic_tags
                self.slow_trace.mul_(self.cfg.slow_trace_decay)
                self.slow_trace.add_(tagged_fast, alpha=self.cfg.slow_trace_lr)
                
                # Clear captured tags
                self.synaptic_tags *= (1 - self.synaptic_tags * 0.5)
    
    def inject_prp(self, amount: float = 1.0):
        """Simulate PRP release (e.g., from reward signal)"""
        self.prp_level = min(1.0, self.prp_level + amount)
    
    def decay_prp(self):
        """PRPs decay over time"""
        self.prp_level *= 0.99
```

### 4.3 Biological Relevance

```
┌─────────────────────────────────────────────────────────────────┐
│              Synaptic Tagging & Capture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Weak Input ───▶ No Tag ───▶ Fast Trace Only (decays)         │
│                                                                 │
│   Strong Input ──▶ TAG SET ──┬─▶ Fast Trace                    │
│                              │                                  │
│                              └─▶ If PRPs available:            │
│                                  TAG CAPTURED → Slow Trace     │
│                                                                 │
│   PRPs released by:                                             │
│   • Reward signals                                              │
│   • Novelty detection                                           │
│   • Emotional arousal                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Research Questions

1. What triggers PRP release in artificial systems?
2. Optimal tag threshold?
3. Can tagging be learned end-to-end?

---

## 5. Neuromodulation

### 5.1 Concept

**Neuromodulation** refers to the global regulation of neural activity by diffuse neurotransmitter systems (dopamine, serotonin, norepinephrine, acetylcholine).

### 5.2 Current GAIA State

Fixed plasticity parameters:
```python
fast_trace_lr = 0.05  # Always the same
```

### 5.3 Proposed Extension

```python
class NeuromodulatedPlasticLinear(DiagnosticPlasticLinear):
    """
    Context-dependent plasticity via neuromodulation
    
    Neuromodulators:
    - Dopamine (DA): Reward prediction, motivation
    - Norepinephrine (NE): Arousal, attention
    - Acetylcholine (ACh): Learning, memory encoding
    - Serotonin (5-HT): Mood, behavioral flexibility
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Neuromodulator levels (normalized 0-1)
        self.modulators = {
            'dopamine': 0.5,
            'norepinephrine': 0.5,
            'acetylcholine': 0.5,
            'serotonin': 0.5,
        }
    
    def compute_effective_lr(self) -> float:
        """
        Compute effective learning rate based on neuromodulator state
        
        DA high → higher LR (reward-driven learning)
        ACh high → higher LR (attention/encoding)
        NE high → broader exploration
        5-HT high → more stable, less reactive
        """
        da = self.modulators['dopamine']
        ach = self.modulators['acetylcholine']
        ne = self.modulators['norepinephrine']
        sht = self.modulators['serotonin']
        
        # Effective learning rate
        base_lr = self.cfg.fast_trace_lr
        modulation = (da * 0.4 + ach * 0.4 + ne * 0.1 + (1 - sht) * 0.1)
        
        return base_lr * (0.5 + modulation)
    
    def compute_exploration_noise(self) -> float:
        """NE increases exploration"""
        return 0.1 * self.modulators['norepinephrine']
    
    def set_modulator(self, name: str, level: float):
        """Set neuromodulator level"""
        self.modulators[name] = np.clip(level, 0, 1)
    
    def process_reward(self, reward: float):
        """
        Update neuromodulators based on reward
        
        Positive reward → DA spike
        Negative reward → NE spike (arousal)
        Neutral → ACh boost (attention)
        """
        if reward > 0.5:
            self.modulators['dopamine'] = min(1.0, self.modulators['dopamine'] + 0.2)
        elif reward < -0.5:
            self.modulators['norepinephrine'] = min(1.0, self.modulators['norepinephrine'] + 0.2)
        else:
            self.modulators['acetylcholine'] = min(1.0, self.modulators['acetylcholine'] + 0.1)
        
        # Decay all modulators toward baseline
        for key in self.modulators:
            self.modulators[key] = 0.9 * self.modulators[key] + 0.1 * 0.5
```

### 5.4 Modulator Effects

| Modulator | High Level | Low Level |
|-----------|------------|-----------|
| **Dopamine** | Increased LR, reinforcement | Reduced motivation |
| **Norepinephrine** | Increased exploration | Focused exploitation |
| **Acetylcholine** | Enhanced encoding | Retrieval mode |
| **Serotonin** | Stability, patience | Reactivity, impulsivity |

### 5.5 Research Questions

1. How to learn neuromodulator dynamics?
2. Should different layers have different modulation?
3. Integration with Active Inference framework?

---

## 6. Additional Research Directions

### 6.1 Sparse Coding

```python
# Top-k sparsity in hidden activations
def sparse_activation(x, k=10):
    """Keep only top-k activations per sample"""
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(-1, topk_idx, topk_vals)
    return sparse_x
```

**Benefits**: Reduced interference, improved capacity

### 6.2 Predictive Coding

```python
# Prediction error drives learning
def predictive_coding_update(predicted, actual, trace):
    """Update based on prediction error, not raw activity"""
    error = actual - predicted
    return trace + lr * outer(error, input)
```

**Benefits**: More efficient information encoding

### 6.3 Sleep-Like Consolidation

```python
# Offline replay for consolidation
def sleep_consolidation(agent, steps=100):
    """Replay experiences without new input"""
    agent.plasticity_enabled = False  # No new encoding
    
    for _ in range(steps):
        # Replay from memory buffer
        state, action, next_state = sample_memory()
        
        # Strengthen relevant traces
        agent.consolidate_replay(state, action, next_state)
    
    agent.plasticity_enabled = True
```

**Benefits**: Memory consolidation, catastrophic forgetting prevention

### 6.4 Continual Learning

```python
# EWC-style parameter importance
def elastic_weight_consolidation(agent, task_data, lambda_ewc=0.4):
    """Protect important weights for previous tasks"""
    fisher = compute_fisher_information(agent, task_data)
    
    # During future learning:
    # loss += lambda_ewc * sum((param - old_param)^2 * fisher)
```

**Benefits**: Multi-task learning without forgetting

---

## 7. Implementation Priorities

### 7.1 Short-Term (1-3 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Multi-scale traces | Low | High |
| 2 | Basic neuromodulation | Medium | High |
| 3 | Sleep consolidation | Low | Medium |

### 7.2 Medium-Term (3-6 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Meta-plasticity | High | Very High |
| 2 | Synaptic tagging | Medium | High |
| 3 | Attention over traces | High | High |

### 7.3 Long-Term (6-12 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Full neuromodulation | Very High | Very High |
| 2 | Predictive coding | High | High |
| 3 | Continual learning | High | High |

---

## 8. Collaboration Opportunities

### 8.1 Neuroscience

- Memory consolidation mechanisms
- Neuromodulator dynamics
- Sleep and offline processing

### 8.2 Machine Learning

- Meta-learning algorithms
- Continual learning benchmarks
- Attention mechanisms

### 8.3 Neuromorphic Computing

- Hardware implementation
- Spiking neural networks
- Low-power deployment

---

## 9. References

1. Frey, U., & Morris, R. G. (1997). Synaptic tagging and long-term potentiation
2. Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of synaptic plasticity
3. Doya, K. (2002). Metalearning and neuromodulation
4. Vaswani, A., et al. (2017). Attention is all you need
5. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks
6. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex

---

*This document is a living research guide. For implementation details, see [Advanced Plasticity](../architecture/advanced-plasticity.md). For theoretical background, see [Theoretical Foundations](../science/theoretical-foundations.md).*