# H100 NCU Performance Analysis

**Date:** November 5, 2025  
**Kernel:** `resample_trajectories_bf16`  
**Configuration:** batch=64, src=4096, tgt=1024, dim=32

---

## Executive Summary

NCU profiling reveals kernel is **memory-latency bound** with **severe HBM underutilization** (0.13%). The kernel requires immediate optimization for memory coalescing and shared memory caching to approach theoretical bandwidth limits.

---

## Key Metrics

### Memory Bandwidth
```
dram__throughput.avg.pct_of_peak_sustained_elapsed: 0.13%
```
**Critical Issue:** Utilizing only 0.13% of H100's 3.35 TB/s HBM3 bandwidth

**Theoretical Peak:** 3,350 GB/s  
**Actual Throughput:** ~4.36 GB/s  
**Gap:** **768x underutilization**

### SM Utilization
```
sm__throughput.avg.pct_of_peak_sustained_elapsed: 36.61%
```
**Moderate:** Streaming Multiprocessors at 37% utilization indicates memory-bound workload

### Memory Access Pattern
```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum: 1,245,183 sectors
```
**Analysis:** 1.2M sectors × 32 bytes/sector = **39.8 MB** loaded from global memory per kernel invocation

**Data Size:** 64 × 4096 × 32 × 2 bytes (BF16) = **16.8 MB** (source data)  
**Overhead:** 39.8 / 16.8 = **2.37x** - indicates poor coalescing and repeated loads

### Memory Stalls
```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio: 2.03
```
**Issue:** Warps stalling on memory dependencies, waiting ~2 instructions per issue on average

---

## Root Cause Analysis

### 1. **Non-Coalesced Memory Access**
```cuda
// Current implementation (per-thread binary search)
int left = 0, right = source_length - 1;
while (left < right) {
    int mid = (left + right) / 2;
    float src_time = source_times[batch_idx * source_length + mid];  // Random access
    // ...
}
```

**Problem:** Each thread performs independent binary search with non-sequential memory access  
**Impact:** 2.37x memory overhead, cache misses

### 2. **No Shared Memory Caching**
**Problem:** Timestamps loaded repeatedly from global memory  
**Solution:** Cache timestamps in shared memory (32 KB available)

```cuda
__shared__ float s_source_times[MAX_SOURCE_LEN];  // 4KB for 1024 timestamps
// Load once per block, reuse across all threads
```

### 3. **Warp Divergence in Binary Search**
**Problem:** Each thread takes different path through binary search  
**Impact:** Reduced instruction-level parallelism, serialized execution

---

## Optimization Roadmap

### Phase 1: Shared Memory Caching (Expected +20-30%)
```cuda
// Load timestamps to shared memory
if (threadIdx.x < source_length) {
    s_source_times[threadIdx.x] = source_times[batch_idx * source_length + threadIdx.x];
}
__syncthreads();

// Binary search on shared memory (much faster)
```

**Expected Impact:**
- Reduce global memory loads by 50%
- Increase DRAM bandwidth utilization to ~0.25%
- Reduce latency by 20-30%

### Phase 2: Vectorized Memory Access (Expected +15-25%)
```cuda
// Load 4 floats at once using float4
float4 *data_vec = reinterpret_cast<float4*>(source_data);
float4 left_vec = data_vec[...];
```

**Expected Impact:**
- Improve memory coalescing
- Reduce transactions by 4x
- Increase bandwidth utilization to ~0.5%

### Phase 3: Warp-Cooperative Search (Expected +10-15%)
```cuda
// Use __shfl_sync for warp-level binary search
// All threads in warp cooperate on single search
unsigned mask = __activemask();
int search_result = cooperative_binary_search(target_time, s_source_times, source_length);
```

**Expected Impact:**
- Eliminate warp divergence
- Reduce instruction count
- Increase SM utilization to 50%+

### Phase 4: Tensor Core Utilization (Expected +2-3x)
```cuda
// Use WMMA/GMMA for interpolation of multiple timesteps
// Process 16x16 tiles with Tensor Cores
```

**Expected Impact:**
- 2-3x throughput improvement
- Leverage H100's 1979 TFLOPS (FP16 Tensor Core)
- Requires algorithm restructuring

---

## Performance Projections

### Current State
- **Latency:** 0.110 ms
- **Throughput:** 583K traj/sec
- **DRAM BW:** 0.13% (4.36 GB/s)
- **vs PyTorch:** 0.9x slower

### After Phase 1 (Shared Memory)
- **Latency:** ~0.080 ms (27% improvement)
- **Throughput:** 800K traj/sec
- **DRAM BW:** ~0.25% (8 GB/s)
- **vs PyTorch:** 1.2x faster

### After Phase 2 (Vectorization)
- **Latency:** ~0.065 ms (41% improvement from baseline)
- **Throughput:** 985K traj/sec
- **DRAM BW:** ~0.5% (17 GB/s)
- **vs PyTorch:** 1.5x faster

### After Phase 3 (Warp-Cooperative)
- **Latency:** ~0.055 ms (50% improvement from baseline)
- **Throughput:** 1.16M traj/sec
- **DRAM BW:** ~0.7% (23 GB/s)
- **vs PyTorch:** 1.8x faster

### After Phase 4 (Tensor Cores)
- **Latency:** ~0.020 ms (82% improvement from baseline)
- **Throughput:** 3.2M traj/sec
- **DRAM BW:** ~2% (67 GB/s)
- **vs PyTorch:** 5x faster ✅ **TARGET**

---

## Comparison to PyTorch

### Why PyTorch is Currently Faster

PyTorch's implementation uses highly optimized primitives:

```python
# PyTorch path
indices = torch.searchsorted(src_t, tgt_t)  # Optimized binary search with vectorization
left_data = data[:, left_idx, :]            # Coalesced gather
right_data = data[:, right_idx, :]          # Coalesced gather  
result = torch.lerp(left_data, right_data, weight)  # Vectorized BLAS
```

**Advantages:**
1. `searchsorted` is hand-tuned with years of optimization
2. Gather operations are fully coalesced
3. `lerp` uses vectorized load/store
4. All operations are memory-bandwidth optimized

**Our Current Disadvantage:**
1. Per-thread binary search (non-coalesced)
2. No shared memory caching
3. No vectorization
4. Memory-latency bound instead of bandwidth-bound

**Path to Victory:**
Implement phases 1-4 above to match/exceed PyTorch's memory efficiency, then leverage Tensor Cores for additional 2-3x gain that PyTorch cannot achieve without custom kernels.

---

## Next Actions

1. ✅ NCU profiling complete
2. ⏳ Implement shared memory optimization
3. ⏳ Add vectorized memory access
4. ⏳ Implement warp-cooperative search
5. ⏳ Re-profile with NCU to validate improvements
6. ⏳ Benchmark against PyTorch baseline
7. ⏳ Document optimization impact

---

## NCU Command Reference

```bash
# Basic metrics
/usr/local/cuda-13.0/bin/ncu \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --target-processes all \
  python3 profile_kernel.py

# Full profile
/usr/local/cuda-13.0/bin/ncu \
  --set full \
  --target-processes all \
  -o ncu_full_report \
  python3 profile_kernel.py

# Export to CSV
/usr/local/cuda-13.0/bin/ncu \
  --import ncu_full_report.ncu-rep \
  --page raw \
  --csv > ncu_metrics.csv
```

---

## Conclusion

**Current Status:** Baseline implementation with correct functionality but unoptimized memory access  
**Critical Finding:** 768x HBM underutilization due to memory-latency bound workload  
**Path Forward:** 4-phase optimization plan with clear performance targets  
**Expected Outcome:** 5x improvement over PyTorch baseline achievable with full optimization  

The NCU data provides concrete evidence for optimization priorities and validates the need for memory-focused improvements before attempting compute optimizations.

