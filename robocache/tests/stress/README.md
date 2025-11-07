# Stress Tests

Production-grade reliability and stability tests for RoboCache.

## Overview

These tests validate:
- **Long-running stability** (24h burn-in)
- **Memory leak detection** (GPU + CPU)
- **Concurrent inference** (multithreaded, multistream)
- **Back-pressure handling** (slow consumer)
- **Exception resilience** (graceful error handling)

## Quick Start

### 1-Hour Burn-In (Quick)
```bash
cd robocache
export STRESS_TEST_DURATION=3600  # 1 hour
pytest tests/stress/test_long_running.py::test_24h_burn_in -v -s
```

### 24-Hour Burn-In (Full)
```bash
cd robocache
export STRESS_TEST_DURATION=86400  # 24 hours
pytest tests/stress/test_long_running.py::test_24h_burn_in -v -s
```

### Concurrent Tests
```bash
cd robocache
pytest tests/stress/test_concurrent.py -v -s
```

### All Stress Tests
```bash
cd robocache
pytest tests/stress/ -m slow -v -s
```

## Test Descriptions

### `test_24h_burn_in`
- Runs continuous inference for 24 hours
- Checks GPU/CPU memory every 1000 iterations
- Fails if memory leak > 100MB (GPU) or 500MB (CPU)
- **Duration:** 24h (configurable)

### `test_repeated_allocation`
- 1000 cycles of large allocation → deallocation
- Tests OOM resilience
- **Duration:** ~5 minutes

### `test_backpressure_handling`
- Fast producer, slow consumer (100ms delay)
- Validates no memory accumulation
- **Duration:** ~10 seconds

### `test_multithreaded_inference`
- 4 threads × 100 iterations
- Concurrent inference from multiple threads
- **Duration:** ~30 seconds

### `test_multistream_inference`
- 4 CUDA streams × 100 iterations
- Concurrent inference on multiple streams
- **Duration:** ~30 seconds

### `test_exception_handling`
- 1000 iterations with 10% invalid inputs
- Tests graceful error handling
- **Duration:** ~10 seconds

## Expected Results

### H100 (24h burn-in)
```
[0.00h] Iter 1000: GPU=120.3MB (+0.1), CPU=350.2MB (+0.3)
[1.00h] Iter 60000: GPU=120.5MB (+0.3), CPU=350.8MB (+0.9)
[12.00h] Iter 720000: GPU=121.2MB (+1.0), CPU=352.1MB (+2.2)
[24.00h] Iter 1440000: GPU=122.0MB (+1.8), CPU=354.5MB (+4.6)

✓ Burn-in complete: 1440000 iterations, no memory leaks
```

### A100 (24h burn-in)
```
Similar performance, ~10-20% slower iteration rate
```

## CI Integration

Stress tests run nightly on self-hosted GPU runners:

```yaml
# .github/workflows/stress_tests.yml
schedule:
  - cron: '0 0 * * *'  # Daily at midnight
```

## Monitoring

### Memory Leak Detection
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Monitor CPU memory
watch -n 1 'ps aux | grep pytest'
```

### Performance Baseline
```bash
# Profile during stress test
nsys profile -o stress_profile python -m pytest tests/stress/test_long_running.py
```

## Troubleshooting

### OOM During Burn-In
```bash
# Reduce batch size
export ROBOCACHE_BATCH_SIZE=2
pytest tests/stress/test_long_running.py
```

### Slow Progress
```bash
# Check GPU utilization
nvidia-smi dmon -s u
```

### False Positive Memory Leak
```bash
# Increase leak threshold
# Edit test_long_running.py:
assert gpu_leak < 200  # Increase from 100MB
```

## References

- [NVIDIA DCGM](https://developer.nvidia.com/dcgm) - GPU health monitoring
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) - Memory profiling
- [pytest-timeout](https://pypi.org/project/pytest-timeout/) - Test timeouts

## Citation

```bibtex
@software{robocache2025,
  title={RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models},
  author={GOATnote Engineering},
  year={2025}
}
```

