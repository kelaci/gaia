# Validation & Diagnostics Guide

## 🔬 Overview

This guide covers comprehensive validation procedures for GAIA systems, including stability checks, diagnostic interpretation, and troubleshooting common issues.

---

## 1. Quick Validation

### 1.1 NumPy Implementation (v4.x)

```bash
# Run the test suite
python test_gaia.py

# Expected output:
# 🚀 Running GAIA v4/v4.1 Tests
# ==================================================
# 🧪 Testing Layers...
# ✅ ReactiveLayer test passed
# ✅ HebbianCore test passed
# ✅ TemporalLayer test passed
# ...
# 🎉 All tests passed successfully!
```

### 1.2 PyTorch Implementation (v3.1)

```python
from gaia_protocol import run_comprehensive_validation

# Run 200-step validation
agent = run_comprehensive_validation()

# Expected output:
# 🔬 GAIA PROTOCOL v3.1 — COMPREHENSIVE VALIDATION
# ==================================================
# Step 50:
#   WM Loss: 0.0234 ± 0.0056
#   Epistemic: 0.0123
#   Trace Norm: 2.3456
#   State: 0.1234
# ...
# ✅ STABLE
```

---

## 2. Diagnostic Metrics

### 2.1 Trace Norms

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `fast_trace_norm` | 0.5 - 5.0 | 5.0 - 7.0 | > 7.0 or < 0.1 |
| `slow_trace_norm` | 0.1 - 2.0 | 2.0 - 4.0 | > 4.0 |

**Interpretation:**
```
fast_trace_norm ≈ 0.0  → Not learning (plasticity disabled?)
fast_trace_norm > 7.0  → Homeostatic failure (unstable)
fast_trace_norm ≈ 5.0  → Homeostatic regulation active (normal)
```

### 2.2 Update Magnitudes

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `mean_update_mag` | 0.01 - 0.5 | 0.5 - 1.0 | > 1.0 or < 0.001 |

**Interpretation:**
```
mean_update_mag < 0.001 → Stagnant learning (lr too low?)
mean_update_mag > 1.0   → Volatile updates (lr too high?)
```

### 2.3 World Model Loss

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `wm_loss` | < 0.1 | 0.1 - 0.5 | > 0.5 |

**Interpretation:**
```
wm_loss increasing    → Catastrophic forgetting
wm_loss oscillating   → Learning rate too high
wm_loss plateau       → Local minimum (increase exploration)
```

### 2.4 Epistemic Uncertainty

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `epistemic_uncertainty` | 0.01 - 0.2 | 0.2 - 0.5 | > 0.5 |

**Interpretation:**
```
uncertainty high      → Ensemble disagreement (more data needed)
uncertainty → 0       → Overconfident (diversity collapse?)
uncertainty stable    → Good ensemble diversity
```

---

## 3. Stability Checks

### 3.1 Bounded Trace Test

```python
def test_trace_stability(agent, steps=1000):
    """
    Verify trace norms remain bounded
    """
    state = torch.zeros(1, cfg.state_dim, device=device)
    max_trace_norm = 0
    
    for step in range(steps):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.1 * torch.randn_like(state)
        agent.learn(state, action, next_state)
        state = next_state
        
        # Check trace norms
        diag = agent.wm.get_ensemble_diagnostics()
        trace_norm = diag.get('l1_fast_trace_norm', 0)
        max_trace_norm = max(max_trace_norm, trace_norm)
        
        if trace_norm > cfg.homeostatic_target * 1.5:
            print(f"❌ UNSTABLE at step {step}: trace_norm = {trace_norm:.4f}")
            return False
    
    print(f"✅ STABLE: max_trace_norm = {max_trace_norm:.4f}")
    return True
```

### 3.2 Gradient Explosion Test

```python
def test_gradient_stability(agent, steps=100):
    """
    Verify gradients remain bounded during training
    """
    state = torch.zeros(1, cfg.state_dim, device=device)
    
    for step in range(steps):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.1 * torch.randn_like(state)
        
        # Manual loss computation with gradient tracking
        preds = torch.stack([m(state, action) for m in agent.wm.models])
        target = next_state.unsqueeze(0).expand_as(preds)
        loss = F.mse_loss(preds, target)
        
        agent.wm_opt.zero_grad()
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0
        for p in agent.wm.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 100:
            print(f"❌ GRADIENT EXPLOSION at step {step}: norm = {total_grad_norm:.4f}")
            return False
        
        agent.wm_opt.step()
        state = next_state
    
    print(f"✅ GRADIENTS STABLE")
    return True
```

### 3.3 Memory Leak Test

```python
def test_memory_stability(agent, steps=1000):
    """
    Verify memory usage remains stable
    """
    import gc
    import psutil
    
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    state = torch.zeros(1, cfg.state_dim, device=device)
    
    for step in range(steps):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.1 * torch.randn_like(state)
        agent.learn(state, action, next_state)
        state = next_state
        
        if step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_growth = final_memory - initial_memory
    
    if memory_growth > 100:  # 100 MB threshold
        print(f"❌ MEMORY LEAK: grew by {memory_growth:.1f} MB")
        return False
    
    print(f"✅ MEMORY STABLE: grew by {memory_growth:.1f} MB")
    return True
```

---

## 4. Performance Benchmarks

### 4.1 Forward Pass Timing

```python
import time

def benchmark_forward_pass(agent, batch_size=32, iterations=1000):
    """
    Benchmark forward pass performance
    """
    state = torch.randn(batch_size, cfg.state_dim, device=device)
    action = torch.randn(batch_size, cfg.action_dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = agent.wm(state, action)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = agent.wm(state, action)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    throughput = (iterations * batch_size) / elapsed
    
    print(f"Forward Pass Benchmark:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    
    return throughput
```

### 4.2 Learning Step Timing

```python
def benchmark_learning_step(agent, iterations=100):
    """
    Benchmark complete learning step
    """
    state = torch.randn(1, cfg.state_dim, device=device)
    
    # Warmup
    for _ in range(10):
        action = agent.select_action(state)
        next_state = state * 0.9 + action
        agent.learn(state, action, next_state)
        state = next_state
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        action = agent.select_action(state)
        next_state = state * 0.9 + action
        agent.learn(state, action, next_state)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - start)
        state = next_state
    
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print(f"Learning Step Benchmark:")
    print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Max time: {max(times)*1000:.2f} ms")
    print(f"  Min time: {min(times)*1000:.2f} ms")
    
    return mean_time
```

---

## 5. Troubleshooting

### 5.1 Trace Norm Explosion

**Symptoms:**
- `fast_trace_norm > 10`
- NaN values in outputs
- Erratic behavior

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| `homeostatic_target` too high | Reduce to 3.0-5.0 |
| `fast_trace_lr` too high | Reduce to 0.01-0.03 |
| Numerical overflow | Add `torch.clamp` after operations |

**Quick Fix:**
```python
# Emergency trace reset
for model in agent.wm.models:
    model.l1.fast_trace.zero_()
    model.l1.slow_trace.zero_()
    model.l2.fast_trace.zero_()
    model.l2.slow_trace.zero_()
```

### 5.2 Not Learning

**Symptoms:**
- `fast_trace_norm ≈ 0`
- Constant outputs
- No improvement in loss

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| `plasticity_enabled = False` | Set to `True` |
| `fast_trace_lr = 0` | Set to 0.01-0.05 |
| Dead ReLU neurons | Use LeakyReLU or check initialization |

**Diagnostic:**
```python
# Check if updates are happening
agent.wm.models[0].l1.step_counter  # Should increase
agent.wm.models[0].l1.update_magnitude_history[-10:]  # Should be non-zero
```

### 5.3 Ensemble Collapse

**Symptoms:**
- `epistemic_uncertainty → 0`
- All ensemble members predict same values
- Poor exploration

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Same initialization | Use different seeds per member |
| Same traces | Each member has independent traces (should be automatic) |
| Overfitting | Increase `exploration_weight` |

**Fix:**
```python
# Re-initialize ensemble with diversity
for i, model in enumerate(agent.wm.models):
    torch.manual_seed(i * 1000)
    model.l1._reset_parameters()
    model.l2._reset_parameters()
```

### 5.4 Gradient Explosion

**Symptoms:**
- NaN in loss
- Very large weight updates
- Training crashes

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Learning rate too high | Reduce `wm_lr` and `policy_lr` |
| No gradient clipping | Add `clip_grad_norm_(params, 1.0)` |
| Bad initialization | Use smaller `weight * 0.02` |

---

## 6. Visualization

### 6.1 Trace Evolution

```python
def plot_trace_evolution(agent, steps=500):
    """
    Visualize trace norm evolution over time
    """
    state = torch.zeros(1, cfg.state_dim, device=device)
    fast_norms = []
    slow_norms = []
    
    for step in range(steps):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.01 * torch.randn_like(state)
        agent.learn(state, action, next_state)
        state = next_state
        
        diag = agent.wm.get_ensemble_diagnostics()
        fast_norms.append(diag.get('l1_fast_trace_norm', 0))
        slow_norms.append(diag.get('l1_slow_trace_norm', 0))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fast_norms, label='Fast Trace')
    plt.axhline(y=cfg.homeostatic_target, color='r', linestyle='--', 
                label=f'Homeostatic Target ({cfg.homeostatic_target})')
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.title('Fast Trace Norm')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(slow_norms, label='Slow Trace')
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.title('Slow Trace Norm')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

### 6.2 Learning Curves

```python
def plot_learning_curves(agent):
    """
    Visualize learning progress
    """
    metrics = agent.metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # WM Loss
    axes[0, 0].plot(metrics['wm_loss'])
    axes[0, 0].set_title('World Model Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('MSE Loss')
    
    # Epistemic Uncertainty
    axes[0, 1].plot(metrics['epistemic_uncertainty'])
    axes[0, 1].set_title('Epistemic Uncertainty')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Std')
    
    # Policy Improvement
    axes[1, 0].plot(metrics['policy_improvement'])
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Policy Improvement')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Δ Performance')
    
    # Trace Norms
    if 'trace_norms' in metrics and metrics['trace_norms']:
        axes[1, 1].plot(metrics['trace_norms'])
        axes[1, 1].axhline(y=cfg.homeostatic_target, color='r', linestyle='--')
        axes[1, 1].set_title('Trace Norms')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Norm')
    
    plt.tight_layout()
    plt.show()
```

---

## 7. Automated Validation Suite

### 7.1 Complete Validation Script

```python
def run_full_validation_suite(cfg=None):
    """
    Run complete validation suite
    """
    if cfg is None:
        cfg = GaiaConfigEnhanced()
    
    print("="*60)
    print("🔬 GAIA FULL VALIDATION SUITE")
    print("="*60)
    
    agent = GaiaAgentEnhanced(cfg)
    
    results = {}
    
    # Test 1: Basic Functionality
    print("\n📋 Test 1: Basic Functionality")
    try:
        state = torch.zeros(1, cfg.state_dim, device=device)
        action = agent.select_action(state)
        assert action.shape == (1, cfg.action_dim)
        print("  ✅ Action selection: PASS")
        results['basic_functionality'] = True
    except Exception as e:
        print(f"  ❌ Action selection: FAIL ({e})")
        results['basic_functionality'] = False
    
    # Test 2: Trace Stability
    print("\n📋 Test 2: Trace Stability (500 steps)")
    results['trace_stability'] = test_trace_stability(agent, steps=500)
    
    # Test 3: Gradient Stability
    print("\n📋 Test 3: Gradient Stability (100 steps)")
    agent = GaiaAgentEnhanced(cfg)  # Fresh agent
    results['gradient_stability'] = test_gradient_stability(agent, steps=100)
    
    # Test 4: Memory Stability
    print("\n📋 Test 4: Memory Stability (500 steps)")
    agent = GaiaAgentEnhanced(cfg)  # Fresh agent
    results['memory_stability'] = test_memory_stability(agent, steps=500)
    
    # Test 5: Performance Benchmark
    print("\n📋 Test 5: Performance Benchmark")
    agent = GaiaAgentEnhanced(cfg)
    throughput = benchmark_forward_pass(agent)
    results['throughput'] = throughput
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for k, v in results.items() if v is True or (isinstance(v, (int, float)) and v > 0))
    total = len(results)
    
    for test, result in results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
        else:
            status = f"📈 {result:.0f}"
        print(f"  {test}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return results

# Run validation
if __name__ == "__main__":
    results = run_full_validation_suite()
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/validate.yml
name: GAIA Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run NumPy tests
      run: python test_gaia.py
    
    - name: Run PyTorch validation
      run: python -c "from gaia_protocol import run_comprehensive_validation; run_comprehensive_validation()"
    
    - name: Run stability tests
      run: pytest tests/test_stability.py -v
```

---

*For implementation details, see [Advanced Plasticity](../architecture/advanced-plasticity.md). For theoretical background, see [Theoretical Foundations](../science/theoretical-foundations.md).*