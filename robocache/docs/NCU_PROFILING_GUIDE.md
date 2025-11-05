# NCU Profiling Guide for RoboCache

**Production-grade GPU profiling workflow for H100/A100 optimization**

Based on best practices from NVIDIA Nsight Compute and voxelization-kit-secure validation framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Basic Profiling](#basic-profiling)
4. [Key Metrics](#key-metrics)
5. [Interpretation Guide](#interpretation-guide)
6. [Acceptance Gates](#acceptance-gates)
7. [Common Issues](#common-issues)
8. [Advanced Profiling](#advanced-profiling)

---

## Overview

**Nsight Compute (NCU)** is NVIDIA's kernel profiler that provides detailed performance metrics for CUDA kernels. This guide establishes a systematic workflow for profiling RoboCache kernels with clear acceptance criteria.

### Why NCU Profiling Matters

- **Memory bottlenecks**: Identify if kernel is DRAM-bound (>70% HBM utilization)
- **Compute efficiency**: Check SM utilization and arithmetic intensity
- **Coalescing**: Validate memory access patterns
- **Optimization validation**: Ensure changes improve metrics, not just wall-clock time

---

## Setup

### Install Nsight Compute

```bash
# NCU is included in CUDA Toolkit 11.0+
# Verify installation
ncu --version

# If not found, install CUDA Toolkit:
# https://developer.nvidia.com/cuda-downloads

# On Brev H100 instances, NCU is pre-installed
which ncu  # Should show /usr/local/cuda/bin/ncu
```

### Security Note

**Never run NCU with `sudo` on production systems.** Use `--target-processes all` for user-space profiling.

---

## Basic Profiling

### Quick Profile (Single Kernel Launch)

```bash
# Profile one kernel iteration with full metrics
ncu --set full \
    --launch-skip 100 \
    --launch-count 1 \
    --target-processes all \
    ./build/benchmark_voxelization
```

**Flags explained:**
- `--set full`: Collect all metrics (comprehensive but slow)
- `--launch-skip 100`: Skip first 100 launches (warmup)
- `--launch-count 1`: Profile exactly 1 kernel launch
- `--target-processes all`: Profile without sudo

### Fast Profile (Key Metrics Only)

```bash
# Profile with minimal metrics (faster, recommended for CI)
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    --target-processes all \
    --launch-skip 100 --launch-count 1 \
    ./build/benchmark_voxelization
```

### Save Results to File

```bash
# Save NCU report for later analysis
ncu --set full \
    --launch-skip 100 --launch-count 1 \
    --target-processes all \
    -o ncu_report_phase3_occupancy \
    --force-overwrite \
    ./build/benchmark_voxelization

# View saved report
ncu-ui ncu_report_phase3_occupancy.ncu-rep
```

---

## Key Metrics

### Memory Metrics

| Metric | Description | H100 Peak | Good Range |
|--------|-------------|-----------|------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization | 3000 GB/s | 30-50% for memory-bound |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 cache load sectors | - | Compare before/after |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | L1 cache store sectors | - | Compare before/after |
| `lts__t_sectors_srcunit_tex_aperture_device.sum` | L2 cache sectors | - | Compare before/after |

### Compute Metrics

| Metric | Description | H100 Peak | Good Range |
|--------|-------------|-----------|------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM (compute) utilization | - | 50-90% for compute-bound |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | Branch uniformity | - | >90% (avoid divergence) |
| `smsp__average_warps_active` | Active warps per SM | - | >16 (high occupancy) |

### Occupancy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Warp occupancy | >50% |
| `launch__registers_per_thread` | Register usage per thread | <64 (H100) |
| `launch__shared_mem_per_block_static` | Static shared memory | <100KB (H100) |

---

## Interpretation Guide

### Memory-Bound Kernels (Trajectory Resampling, Voxelization)

**Characteristics:**
- High DRAM throughput (>40%)
- Low SM throughput (<30%)
- Memory-latency dominated

**Optimization Priorities:**
1. **Coalescing**: Reduce L1/L2 sector counts per transaction
2. **Caching**: Use shared memory for frequently accessed data
3. **Prefetching**: Overlap memory transfers with computation
4. **Data layout**: SOA vs AOS for better access patterns

**Example Analysis:**

```
dram__throughput.avg.pct_of_peak_sustained_elapsed = 72%
sm__throughput.avg.pct_of_peak_sustained_elapsed = 18%

→ MEMORY-BOUND: Focus on reducing DRAM accesses
```

### Compute-Bound Kernels (TSDF, Jacobian)

**Characteristics:**
- Low DRAM throughput (<20%)
- High SM throughput (>60%)
- Compute-limited

**Optimization Priorities:**
1. **Arithmetic intensity**: Increase FLOPs per memory access
2. **Instruction throughput**: Use FP16/BF16 tensor cores
3. **Parallelism**: Increase occupancy (more warps per SM)

**Example Analysis:**

```
dram__throughput.avg.pct_of_peak_sustained_elapsed = 15%
sm__throughput.avg.pct_of_peak_sustained_elapsed = 78%

→ COMPUTE-BOUND: Focus on increasing arithmetic intensity
```

### Balanced Kernels (Multimodal Fusion)

**Characteristics:**
- Moderate DRAM throughput (30-50%)
- Moderate SM throughput (40-60%)
- Mixed workload

**Optimization Priorities:**
1. Balance memory and compute
2. Kernel fusion to amortize memory accesses
3. Use persistent kernels to reduce launch overhead

---

## Acceptance Gates

**Use these criteria for CI/CD validation:**

### Correctness Gates (Mandatory)

```bash
# Run before profiling
./scripts/validate_correctness.sh

# Must pass: 0 mismatches between CPU and GPU
```

### Performance Gates (Recommended)

| Gate | Criteria | Action if Failed |
|------|----------|------------------|
| **Baseline comparison** | ±10% from saved baseline | Investigate regression |
| **DRAM efficiency** | <75% for memory-bound | Fix coalescing or caching |
| **Occupancy** | >40% warp occupancy | Reduce register/shared mem usage |
| **Branch uniformity** | >80% for control-heavy | Reduce divergence |

### Example Baseline JSON

```json
{
  "kernel": "voxelize_occupancy_kernel",
  "date": "2025-11-04",
  "gpu": "H100 PCIe",
  "config": "batch=32, points=100k, grid=128^3",
  "metrics": {
    "dram_throughput_pct": 45.2,
    "sm_throughput_pct": 22.1,
    "kernel_time_ms": 0.216,
    "occupancy_pct": 58.3
  }
}
```

### Validation Script

```bash
#!/bin/bash
# check_ncu_gates.sh

BASELINE="ncu_baseline_phase3.json"
CURRENT_DRAM=$(ncu --query-metric dram__throughput.avg.pct_of_peak_sustained_elapsed ...)

# Compare against baseline
python3 scripts/compare_ncu_results.py $BASELINE $CURRENT_DRAM
```

---

## Common Issues

### Issue 1: Low DRAM Utilization (<10%)

**Symptoms:**
```
dram__throughput = 3%
kernel_time = 0.5ms (unexpectedly slow)
```

**Causes:**
- Under-occupancy (not enough warps)
- Small problem size (not saturating GPU)
- Excessive register usage

**Solutions:**
```bash
# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ...

# Check register usage
ncu --metrics launch__registers_per_thread ...

# Fix: Reduce registers, increase blocks
```

### Issue 2: High DRAM Utilization but Slow (>70%)

**Symptoms:**
```
dram__throughput = 75%
kernel_time = 2.0ms (slower than expected)
```

**Causes:**
- Uncoalesced memory accesses
- Excessive L1/L2 cache misses
- Atomic contention

**Solutions:**
```bash
# Check L1 cache efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ...

# Fix: Align data, use shared memory, reduce atomics
```

### Issue 3: Branch Divergence

**Symptoms:**
```
smsp__sass_average_branch_targets_threads_uniform = 45%
```

**Causes:**
- Data-dependent branching
- Binary search in warps

**Solutions:**
- Predication instead of branching
- Warp-level primitives
- Sort data to reduce divergence

---

## Advanced Profiling

### Source Code Correlation

```bash
# Profile with source mapping
ncu --set full \
    --source yes \
    --launch-skip 100 --launch-count 1 \
    -o ncu_source_report \
    ./build/benchmark_voxelization

# View in UI to see hotspots
ncu-ui ncu_source_report.ncu-rep
```

### Memory Access Patterns

```bash
# Detailed memory analysis
ncu --section MemoryWorkloadAnalysis \
    --launch-skip 100 --launch-count 1 \
    ./build/benchmark_voxelization
```

### Compare Two Kernels

```bash
# Baseline
ncu -o baseline --set full --launch-count 1 ./baseline_version

# Optimized
ncu -o optimized --set full --launch-count 1 ./optimized_version

# Compare in UI
ncu-ui baseline.ncu-rep optimized.ncu-rep
```

### Automated Profiling Script

```bash
#!/bin/bash
# scripts/profile_all_phases.sh

for phase in 1 2 3 4; do
    echo "Profiling Phase $phase..."
    ncu --metrics \
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 100 --launch-count 1 \
        --target-processes all \
        -o "ncu_phase${phase}" --force-overwrite \
        ./build/benchmark_phase${phase}
done

echo "✅ All phases profiled. Results in ncu_phase*.ncu-rep"
```

---

## Profiling Checklist

Before submitting optimizations, ensure:

- [ ] CPU/GPU correctness validated (`./scripts/validate_correctness.sh`)
- [ ] NCU baseline captured (save to `docs/ncu_baselines/`)
- [ ] Key metrics collected (DRAM, SM, occupancy)
- [ ] Performance within ±10% of expected
- [ ] No regressions vs baseline
- [ ] Results documented in PR

---

## References

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [H100 Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Voxelization Kit Security](../voxelkit_validation.txt)

---

**Questions?** See `docs/FAQ.md` or reach out to the team.

**Last Updated:** 2025-11-04

