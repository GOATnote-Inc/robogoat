# Golden Reference Outputs

This directory contains reference outputs for validating RoboCache performance and correctness.

## Files

### acceptance_thresholds.yaml
Pass/fail criteria for validation:
- Latency: < 20ms/step, ≥ 1.3× speedup vs baseline
- Accuracy: < 1e-3 L2 error (BF16 precision)
- Throughput: > 2000 episodes/sec on H100
- Convergence: 0.95 correlation with baseline loss

### baseline_metrics.json (To be generated)
PyTorch CPU reference metrics:
```bash
python train_robot_policy.py --mode baseline --save-metrics baseline_metrics.json
```

### robocache_metrics.json (To be generated)
RoboCache GPU metrics:
```bash
python train_robot_policy.py --mode robocache --save-metrics robocache_metrics.json
```

## Validation Protocol

```bash
# 1. Generate baseline
python train_robot_policy.py --mode baseline --steps 100 --save-metrics golden_outputs/baseline_metrics.json

# 2. Generate RoboCache output
python train_robot_policy.py --mode robocache --steps 100 --save-metrics golden_outputs/robocache_metrics.json

# 3. Validate against thresholds
python validate_against_golden.py \
  --baseline golden_outputs/baseline_metrics.json \
  --candidate golden_outputs/robocache_metrics.json \
  --thresholds golden_outputs/acceptance_thresholds.yaml
```

## Expected Output

```
✅ Latency Test: 14.04ms < 20ms threshold (PASS)
✅ Speedup Test: 1.85x ≥ 1.3x threshold (PASS)
✅ L2 Error Test: 3.2e-4 < 1e-3 threshold (PASS)
✅ Gradient Correlation: 0.997 ≥ 0.99 threshold (PASS)
✅ Convergence Test: Loss correlation 0.983 ≥ 0.95 (PASS)

ALL TESTS PASSED (5/5)
```

## Regenerating Golden Outputs

If you update kernels or change configurations, regenerate:

```bash
# Full regeneration
make golden-outputs

# Or manually:
python train_robot_policy.py --mode baseline --save-metrics golden_outputs/baseline_metrics.json --seed 42
python train_robot_policy.py --mode robocache --save-metrics golden_outputs/robocache_metrics.json --seed 42
```

## Hardware-Specific Outputs

Different GPUs have different performance envelopes:

- `baseline_metrics.json` - CPU reference (hardware-independent)
- `robocache_h100_metrics.json` - H100 GPU metrics
- `robocache_a100_metrics.json` - A100 GPU metrics

Thresholds account for architecture differences.

