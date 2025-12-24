# Advanced Plasticity System

## 🧬 Overview

GAIA's advanced plasticity system implements **dual-timescale Hebbian learning** with **BitNet quantization** and **diagnostic tracking**. This document details the PyTorch implementation (v3.1) which extends the conceptual framework of the NumPy-based v4.x architecture.

---

## 1. DiagnosticPlasticLinear

### 1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DiagnosticPlasticLinear                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input x ──────────┬────────────────────────────────┐             │
│                     │                                 │             │
│                     ▼                                 ▼             │
│              ┌─────────────┐                   ┌───────────┐        │
│              │  W_static   │                   │  Hebbian  │        │
│              │ (learnable) │                   │  Update   │        │
│              └──────┬──────┘                   └─────┬─────┘        │
│                     │                                 │             │
│                     ▼                                 ▼             │
│              ┌─────────────┐                   ┌───────────┐        │
│              │   BitNet    │                   │ fast_trace │        │
│              │  Quantize   │                   │  (τ=0.95)  │        │
│              └──────┬──────┘                   └─────┬─────┘        │
│                     │                                 │             │
│                     │                                 ▼             │
│                     │                          ┌───────────┐        │
│                     │                          │ slow_trace │        │
│                     │                          │  (τ=0.99)  │        │
│                     │                          └─────┬─────┘        │
│                     │                                 │             │
│                     └────────────┬────────────────────┘             │
│                                  ▼                                  │
│                     ┌───────────────────────┐                       │
│                     │  W_eff = Q(W_static)  │                       │
│                     │  + 0.1*fast + 0.05*slow │                     │
│                     └───────────┬───────────┘                       │
│                                 │                                   │
│                                 ▼                                   │
│                     ┌───────────────────────┐                       │
│                     │   y = W_eff · x       │                       │
│                     └───────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementation

```python
class DiagnosticPlasticLinear(nn.Module):
    """
    Enhanced Plastic Linear layer with:
    - BitNet 1.58-bit quantization
    - Dual-timescale Hebbian traces
    - Homeostatic regulation
    - Comprehensive diagnostics
    """
    
    def __init__(self, in_features: int, out_features: int, cfg: GaiaConfig):
        super().__init__()
        self.cfg = cfg
        
        # Statikus súlyok (Static weights)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Plasztikus nyomok (Plastic traces) - registered as buffers
        self.register_buffer("fast_trace", torch.zeros(out_features, in_features))
        self.register_buffer("slow_trace", torch.zeros(out_features, in_features))
        
        # Diagnostic tracking
        self.plasticity_enabled = True
        self.register_buffer("trace_norm_history", torch.zeros(cfg.trace_history_len))
        self.register_buffer("update_magnitude_history", torch.zeros(cfg.trace_history_len))
        self.step_counter = 0
```

### 1.3 BitNet Quantization

```python
def bitnet_quantize(self, w: torch.Tensor) -> torch.Tensor:
    """
    1.58-bit quantization: weights → {-1, 0, +1}
    
    Process:
    1. Compute per-row scaling factor (mean absolute value)
    2. Normalize weights by scale
    3. Round to nearest integer and clamp to [-1, 1]
    4. Re-scale for effective weight magnitudes
    
    Memory: 32 bits → 1.58 bits per weight
    Compute: Multiplications → Additions (hardware efficient)
    """
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    w_normalized = w / scale
    w_quantized = w_normalized.round().clamp(-1, 1)
    return w_quantized * scale
```

### 1.4 Trace Update Mechanism

```python
def update_traces(self, x: torch.Tensor, y: torch.Tensor):
    """
    Hebbian trace update with homeostatic regulation
    
    Hungarian Comments Preserved:
    - Hebbian update: csak aktív neuronok (only active neurons)
    - Gyors nyom frissítése (Fast trace update)
    - Lassú nyom - konszolidáció (Slow trace - consolidation)
    """
    if not self.plasticity_enabled:
        return
    
    with torch.no_grad():
        # Hebbian update: csak aktív neuronok
        y_active = F.relu(y)
        delta = torch.matmul(y_active.t(), x) / x.shape[0]
        
        # Gyors nyom frissítése
        self.fast_trace.mul_(self.cfg.fast_trace_decay)      # τ_fast = 0.95
        self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)  # η_fast = 0.05
        
        # Homeostatic normalization
        fast_norm = self.fast_trace.norm()
        if fast_norm > self.cfg.homeostatic_target:  # target = 5.0
            self.fast_trace.mul_(self.cfg.homeostatic_target / (fast_norm + 1e-6))
        
        # Lassú nyom (konszolidáció)
        self.slow_trace.mul_(self.cfg.slow_trace_decay)      # τ_slow = 0.99
        self.slow_trace.add_(self.fast_trace, alpha=self.cfg.slow_trace_lr)  # η_slow = 0.01
```

### 1.5 Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: Quantize static weights (kvantált váz)
    w_static = self.bitnet_quantize(self.weight)
    
    # Step 2: Compute plastic modulation (plasztikus moduláció)
    w_plastic = 0.1 * self.fast_trace + 0.05 * self.slow_trace
    
    # Step 3: Combine for effective weight (effektív súly)
    w_effective = w_static + w_plastic
    
    # Step 4: Linear transformation
    y = F.linear(x, w_effective)
    
    # Step 5: Update traces (online learning)
    self.update_traces(x, y)
    
    return y
```

---

## 2. Configuration System

### 2.1 GaiaConfigEnhanced

```python
@dataclass
class GaiaConfigEnhanced:
    # === Core Dimensions ===
    state_dim: int = 1          # State space dimensionality
    action_dim: int = 1         # Action space dimensionality
    hidden_dim: int = 64        # Hidden layer size
    n_ensemble: int = 5         # Number of ensemble members
    
    # === Hebbian Plasticity ===
    fast_trace_decay: float = 0.95      # τ_fast: Fast trace decay
    fast_trace_lr: float = 0.05         # η_fast: Fast trace learning rate
    slow_trace_decay: float = 0.99      # τ_slow: Slow trace decay
    slow_trace_lr: float = 0.01         # η_slow: Slow trace learning rate
    homeostatic_target: float = 5.0     # H: Maximum trace norm
    
    # === Active Inference ===
    planning_samples: int = 30          # Number of action samples
    exploration_weight: float = 0.5     # β: Epistemic bonus weight
    temperature: float = 0.1            # τ: Action selection temperature
    
    # === Learning ===
    weight_scale: float = 5.0           # Policy update scaling
    wm_lr: float = 1e-3                 # World model learning rate
    policy_lr: float = 1e-3             # Policy learning rate
    
    # === Diagnostics ===
    track_metrics: bool = True          # Enable metric tracking
    trace_history_len: int = 1000       # History buffer size
```

### 2.2 Parameter Sensitivity

| Parameter | Range | Effect | Sensitivity |
|-----------|-------|--------|-------------|
| `fast_trace_decay` | 0.9-0.99 | Memory duration | High |
| `fast_trace_lr` | 0.01-0.1 | Adaptation speed | Medium |
| `homeostatic_target` | 1.0-10.0 | Stability vs. capacity | High |
| `exploration_weight` | 0.0-1.0 | Exploit vs. explore | Medium |
| `temperature` | 0.01-1.0 | Action selection sharpness | Low |

---

## 3. EnhancedDeepPlasticMember

### 3.1 Architecture

```python
class EnhancedDeepPlasticMember(nn.Module):
    """
    Deep network with plastic layers and layer normalization
    
    Structure:
        Input → PlasticLinear → LayerNorm → ReLU
              → PlasticLinear → LayerNorm → ReLU
              → Linear (output)
    """
    
    def __init__(self, cfg: GaiaConfigEnhanced):
        super().__init__()
        inp = cfg.state_dim + cfg.action_dim
        h = cfg.hidden_dim
        
        # Plastic layers (with traces)
        self.l1 = DiagnosticPlasticLinear(inp, h, cfg)
        self.l2 = DiagnosticPlasticLinear(h, h, cfg)
        
        # Output layer (non-plastic)
        self.l3 = nn.Linear(h, cfg.state_dim)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        h = self.ln1(F.relu(self.l1(x)))
        h = self.ln2(F.relu(self.l2(h)))
        return self.l3(h)
```

### 3.2 Design Rationale

**Why Layer Normalization?**
- Stabilizes activations during online learning
- Reduces internal covariate shift from trace updates
- Complements homeostatic regulation

**Why Non-Plastic Output Layer?**
- Predictions need to be stable for EFE computation
- Plasticity in hidden layers captures features
- Output layer learned via gradient descent

---

## 4. Ensemble World Model

### 4.1 Implementation

```python
class EnhancedEnsembleWorldModel(nn.Module):
    """
    Ensemble of plastic world models for uncertainty estimation
    
    Properties:
    - n_ensemble independent models
    - Each model has its own plastic traces
    - Mean prediction + disagreement-based uncertainty
    """
    
    def __init__(self, cfg: GaiaConfigEnhanced):
        super().__init__()
        self.models = nn.ModuleList(
            [EnhancedDeepPlasticMember(cfg) for _ in range(cfg.n_ensemble)]
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collect predictions from all ensemble members
        preds = torch.stack([m(state, action) for m in self.models])
        
        # Mean: expected next state
        # Std: epistemic uncertainty (model disagreement)
        return preds.mean(dim=0), preds.std(dim=0)
```

### 4.2 Uncertainty Interpretation

```
┌────────────────────────────────────────────────────────────────┐
│                  Ensemble Predictions                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Model 1: ─────•──────                                       │
│   Model 2: ────────•───                  High Uncertainty     │
│   Model 3: ──•─────────    ←─ Large spread                    │
│   Model 4: ───────────•                                       │
│   Model 5: ────•───────                                       │
│                                                                │
│   vs.                                                          │
│                                                                │
│   Model 1: ──────•─────                                       │
│   Model 2: ──────•─────                  Low Uncertainty      │
│   Model 3: ─────•──────    ←─ Small spread                    │
│   Model 4: ──────•─────                                       │
│   Model 5: ──────•─────                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Diagnostics System

### 5.1 Tracked Metrics

```python
def get_diagnostics(self) -> Dict[str, float]:
    """
    Comprehensive diagnostic metrics for research analysis
    """
    return {
        # Trace state
        "fast_trace_norm": self.fast_trace.norm().item(),
        "slow_trace_norm": self.slow_trace.norm().item(),
        
        # Historical statistics
        "mean_trace_norm": self.trace_norm_history[:valid_steps].mean().item(),
        "mean_update_mag": self.update_magnitude_history[:valid_steps].mean().item(),
        
        # Weight state
        "weight_static_norm": self.weight.norm().item(),
    }
```

### 5.2 Metric Interpretation

| Metric | Healthy Range | Warning Signs |
|--------|---------------|---------------|
| `fast_trace_norm` | 0.5 - 5.0 | > 5.5 (unstable), < 0.1 (not learning) |
| `slow_trace_norm` | 0.1 - 2.0 | > 3.0 (accumulating noise) |
| `mean_update_mag` | 0.01 - 0.5 | > 1.0 (volatile), < 0.001 (stagnant) |
| `weight_static_norm` | 0.1 - 5.0 | Growing unboundedly |

### 5.3 Validation Protocol

```python
def run_comprehensive_validation():
    """
    200-step validation protocol
    
    Checkpoints at steps: 50, 100, 150, 200
    Metrics: WM Loss, Epistemic Uncertainty, Trace Norms, State
    """
    agent = GaiaAgentEnhanced(cfg)
    state = torch.tensor([[0.0]], device=device)
    
    for step in range(200):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.01 * torch.randn_like(state)
        wm_loss, uncertainty = agent.learn(state, action, next_state)
        state = next_state
        
        if (step + 1) % 50 == 0:
            summary = agent.get_summary()
            # Log metrics...
    
    # Final stability check
    final_trace_norm = diag.get("l1_fast_trace_norm", 0)
    stable = 0.0 < final_trace_norm < cfg.homeostatic_target * 1.1
```

---

## 6. Integration with Active Inference

### 6.1 Action Selection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Active Inference Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐     ┌──────────────┐     ┌─────────────────┐   │
│   │   State   │────▶│ Action Model │────▶│ Sample Actions  │   │
│   └───────────┘     │   (μ, σ)     │     │ (30 samples)    │   │
│                     └──────────────┘     └────────┬────────┘   │
│                                                    │            │
│                                                    ▼            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Ensemble World Model                          │  │
│   │    (state, action) → (mean_next, std_next)              │  │
│   └────────────────────────────┬────────────────────────────┘  │
│                                │                               │
│                                ▼                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Expected Free Energy                          │  │
│   │    EFE = pragmatic - β * epistemic                      │  │
│   └────────────────────────────┬────────────────────────────┘  │
│                                │                               │
│                                ▼                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Softmax Selection                             │  │
│   │    probs = softmax(-EFE / τ)                            │  │
│   │    action = multinomial(probs)                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Learning Loop

```python
def learn(self, state, action, next_state):
    """
    Combined world model and policy learning
    
    World Model: Supervised learning on (s, a) → s'
    Policy: Reinforcement via performance improvement signal
    """
    # === World Model Update ===
    preds = torch.stack([m(state, action) for m in self.wm.models])
    target = next_state.unsqueeze(0).expand_as(preds)
    wm_loss = F.mse_loss(preds, target)
    
    self.wm_opt.zero_grad()
    wm_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)  # Stability
    self.wm_opt.step()
    
    # === Policy Update ===
    with torch.no_grad():
        before = F.mse_loss(state, self.preferred_state)
        after = F.mse_loss(next_state, self.preferred_state)
        improvement = (before - after).item()
    
    if improvement > 0:  # Only update if action was beneficial
        logp = self.am.log_prob(state, action)
        weight = torch.sigmoid(torch.tensor(improvement * self.cfg.weight_scale))
        loss = -(logp * weight)
        
        self.act_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.am.parameters(), 1.0)
        self.act_opt.step()
```

---

## 7. Performance Characteristics

### 7.1 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Forward pass | O(n² × h) | O(n × h) |
| Trace update | O(n² × h) | O(n × h) |
| BitNet quantize | O(n × h) | O(n × h) |
| Ensemble forward | O(E × n² × h) | O(E × n × h) |

Where: n = input dim, h = hidden dim, E = ensemble size

### 7.2 Memory Usage

```
Static weights:     4 bytes/weight (float32)
Quantized weights:  ~0.2 bytes/weight (1.58-bit)
Traces (fast+slow): 8 bytes/weight (2 × float32)
Diagnostics:        ~8KB per layer (1000 × 2 floats)
```

### 7.3 Stability Guarantees

✅ **Bounded traces** via homeostatic normalization  
✅ **Gradient clipping** prevents exploding gradients  
✅ **Layer normalization** stabilizes activations  
✅ **Sigmoid-weighted policy updates** prevent overcorrection  

---

## 8. Usage Examples

### 8.1 Basic Usage

```python
# Configuration
cfg = GaiaConfigEnhanced(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    n_ensemble=5
)

# Create agent
agent = GaiaAgentEnhanced(cfg)

# Training loop
for episode in range(1000):
    state = env.reset()
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, next_state)
        state = next_state
        if done:
            break
```

### 8.2 Accessing Diagnostics

```python
# Get ensemble diagnostics
diag = agent.wm.get_ensemble_diagnostics()
print(f"Fast trace norm: {diag['l1_fast_trace_norm']:.4f}")
print(f"Slow trace norm: {diag['l1_slow_trace_norm']:.4f}")

# Get learning summary
summary = agent.get_summary()
print(f"Mean WM loss: {summary['wm_loss_mean']:.4f}")
print(f"Mean uncertainty: {summary['epistemic_uncertainty_mean']:.4f}")
```

### 8.3 Disabling Plasticity

```python
# For inference-only mode
for model in agent.wm.models:
    model.l1.plasticity_enabled = False
    model.l2.plasticity_enabled = False
```

---

*For the theoretical foundations, see [Theoretical Foundations](../science/theoretical-foundations.md). For research extensions, see [Future Directions](../research/future-directions.md).*