# GPU Kernel Tuning Guide

**Version:** 1.0.0  
**Date:** 2025-11-06  
**Architectures:** SM80 (A100), SM90 (H100), SM100 (Blackwell)  
**Author:** Brandon Dent <b@thegoatnote.com>

---

## Introduction

This guide provides architecture-specific optimization techniques for CUDA kernels
in RoboCache, targeting NVIDIA Ampere (A100), Hopper (H100), and future Blackwell
(B100/B200) GPUs.

**Target Audience:** GPU engineers, performance specialists, ML engineers optimizing
data pipelines for robot foundation models.

**Prerequisites:**
- CUDA programming fundamentals
- Understanding of GPU memory hierarchy
- Familiarity with Nsight profiling tools

---

## Architecture Comparison

| Feature | Ampere (SM80) | Hopper (SM90) | Blackwell (SM100) |
|---------|---------------|---------------|-------------------|
| **Compute** |
| SMs | 108 (A100) | 132 (H100) | TBD (B100/B200) |
| CUDA Cores/SM | 64 FP32 | 128 FP32 | TBD |
| Tensor Cores | 3rd Gen | 4th Gen | 5th Gen (expected) |
| FP8 Support | No | Yes (Transformer Engine) | Yes (Enhanced) |
| **Memory** |
| L2 Cache | 40 MB | 50 MB | TBD (expected 60+ MB) |
| HBM | HBM2e (80GB) | HBM3 (80GB) | HBM3e (expected) |
| Bandwidth | 1.6 TB/s | 3.0 TB/s | TBD (expected 4+ TB/s) |
| **Features** |
| TMA | No | Yes | Yes (Enhanced) |
| Thread Block Clusters | No | Yes | Yes |
| Async Barriers | Limited | Full | Full |
| DPX Instructions | No | Yes | Yes |

---

## Memory Optimization

### 1. Coalesced Memory Access

**Pattern:** Threads in a warp access consecutive memory addresses.

**Good Example (Coalesced):**
```cuda
// Coalesced: stride-1 access
__global__ void coalesced_read(const float* data, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = data[idx];  // ✓ Sequential access
    }
}
```

**Bad Example (Uncoalesced):**
```cuda
// Uncoalesced: strided access
__global__ void uncoalesced_read(const float* data, float* output, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = data[idx * stride];  // ✗ Non-sequential
    }
}
```

**Nsight Metric:** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- Target: >80% sector utilization
- RoboCache trajectory kernel: 85%+

---

### 2. Shared Memory Optimization

**SM80/SM90:** 164 KB shared memory per SM  
**Bank conflicts:** Avoid when stride = 32 elements (for 4-byte types)

**Bank-Conflict-Free Pattern:**
```cuda
__shared__ float smem[256 + 8];  // Padding to avoid bank conflicts

__global__ void shared_memory_kernel(const float* input, float* output) {
    int tid = threadIdx.x;
    
    // Load to shared memory with padding
    smem[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    
    // Access without bank conflicts
    float value = smem[tid];  // No conflict
    output[blockIdx.x * blockDim.x + tid] = value * 2.0f;
}
```

**Nsight Metric:** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- Target: 0 conflicts
- RoboCache: 0 conflicts

---

### 3. L1 Cache Utilization

**Memory-Latency Bound Kernels:** Optimize for L1 cache residency

**Pattern:**
```cuda
// Maximize L1 hit rate with temporal locality
__global__ void l1_optimized(const float* data, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Reuse data[idx] multiple times
        float val = data[idx];
        float result = val * val + val * 2.0f + expf(val);
        output[idx] = result;
    }
}
```

**RoboCache Trajectory Kernel:**
- Binary search creates temporal locality
- L1 hit rate: >85% on H100
- DRAM bandwidth: 1.59% (memory-latency optimized)

**Nsight Metric:** `l1tex__t_sector_hit_rate.pct`
- Target: >70% for memory-latency bound
- Target: <30% for bandwidth-bound

---

### 4. Vectorized Loads (SM80+)

**128-bit loads** for coalesced float4/double2 access:

```cuda
// Vectorized load (128-bit)
__global__ void vectorized_load(const float4* data, float4* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 val = data[idx];  // Single 128-bit load
        output[idx] = make_float4(val.x * 2, val.y * 2, val.z * 2, val.w * 2);
    }
}
```

**RoboCache:** Uses vectorized BF16x2 loads for Tensor Core acceleration.

---

## Compute Optimization

### 5. Warp-Level Primitives

**Cooperative Groups (SM80+):**

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void warp_reduce(const float* data, float* output, int N) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < N) ? data[idx] : 0.0f;
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
    
    if (warp.thread_rank() == 0) {
        atomicAdd(output, val);
    }
}
```

**Benefits:**
- No shared memory required
- Implicit synchronization
- Lower latency than `__syncthreads()`

---

### 6. Occupancy Tuning

**SM80:** 2048 threads/SM max, 32 warps/SM  
**SM90:** 2048 threads/SM max, 48 warps/SM (Thread Block Clusters)

**Occupancy Calculator:**
```python
# Example: 128 threads/block, 48 KB shared memory
# SM80: 16 blocks/SM (2048/128), limited by shared memory (164KB/48KB = 3.4)
# Actual: 3 blocks/SM, occupancy = 37.5%

# Optimization: Reduce shared memory to 32 KB
# New: 5 blocks/SM, occupancy = 62.5% ✓
```

**RoboCache Targets:**
- Memory-latency bound: 50-75% occupancy
- Bandwidth-bound: 75-100% occupancy

**Nsight Metric:** `sm__warps_active.avg.pct_of_peak_sustained_active`

---

### 7. Fast Math (--use_fast_math)

**Trade-off:** 2-5× speedup vs. IEEE 754 compliance

**Enabled in RoboCache:**
```bash
nvcc --use_fast_math  # -ffast-math for host code
```

**Implications:**
- `expf(), logf(), sqrtf()` use lookup tables
- Denormals flush to zero
- Less precise rounding

**Validation:** Ensure correctness tests pass with tight tolerances.

---

## Architecture-Specific Features

### Hopper (SM90) Only

#### 8. Tensor Memory Accelerator (TMA)

**Feature:** Async global→shared DMA without thread participation

```cuda
// Traditional (SM80)
__shared__ float smem[256];
smem[threadIdx.x] = global_data[...];
__syncthreads();

// TMA (SM90)
__shared__ float smem[256];
tma_load_async(smem, global_data, ...);  // Hardware DMA
__sync_barrier_wait();
```

**Benefits:**
- Free up threads for computation
- Higher bandwidth (overlapped loads)
- Reduced register pressure

**RoboCache Status:** Planned for Q2 2026

---

#### 9. Thread Block Clusters

**Feature:** Fast inter-block communication via distributed shared memory

```cuda
__global__ void cluster_kernel() __cluster_dims__(2, 1, 1) {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();
    
    // Access neighbor block's shared memory
    __shared__ float smem[256];
    float neighbor_value = cluster.map_shared_rank(smem, neighbor_rank)[idx];
}
```

**Use Case:** Multi-block reductions, graph algorithms

---

#### 10. DPX Instructions

**Feature:** Dynamic programming acceleration (7× speedup for edit distance, etc.)

**RoboCache:** Not applicable (preprocessing-focused)

---

### Blackwell (SM100) Planned

#### 11. 5th-Gen Tensor Cores

**Expected:**
- Enhanced FP8 precision
- Larger MMA tile sizes
- Better sparse matrix support

**Preparation:** Abstract Tensor Core calls via CUTLASS templates

---

#### 12. WGMMA (Warp Group Matrix Multiply-Accumulate)

**Feature:** Warp-group level matrix operations (128 threads)

**RoboCache Plan:** Evaluate for large-scale voxel grid operations

---

## Profiling Workflows

### Nsight Compute

**Memory-Latency Bound:**
```bash
ncu --set full --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed, \
  dram__throughput.avg.pct_of_peak_sustained_elapsed, \
  l1tex__t_sector_hit_rate.pct \
  ./trajectory_benchmark
```

**Expected:**
- SM throughput: 80-95%
- DRAM throughput: <10%
- L1 hit rate: >70%

**Bandwidth-Bound:**
```bash
ncu --set full --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed, \
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct, \
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  ./voxelize_benchmark
```

**Expected:**
- DRAM throughput: 70-90%
- Coalescing: >80%
- L1 hit rate: <30%

---

### Nsight Systems

**End-to-End Timeline:**
```bash
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  -o timeline.nsys-rep \
  python scripts/train_demo.py
```

**Look for:**
- Kernel overlaps (pipelining)
- CPU/GPU bubbles
- Memory copy overhead

---

## Common Pitfalls

### ❌ Pitfall 1: Ignoring Warp Divergence

```cuda
// Bad: Divergent branches
if (idx % 2 == 0) {
    // Half the warp idles
    expensive_computation();
}

// Good: Branchless or separate kernels
int result = (idx % 2 == 0) ? expensive_computation() : 0;
```

**Nsight Metric:** `smsp__sass_average_branch_targets_threads_uniform.pct`
- Target: >95%

---

### ❌ Pitfall 2: Excessive Register Usage

**Symptom:** Low occupancy despite small shared memory

```cuda
// Bad: Too many registers (50+)
__global__ void register_heavy() {
    float a[20];  // 20 registers
    // ... complex computation ...
}

// Good: Spill to shared memory or recompute
__shared__ float smem[256];
```

**Nsight Metric:** `launch__registers_per_thread`
- Target: <40 for high occupancy

---

### ❌ Pitfall 3: Unaligned Memory Access

```cuda
// Bad: Misaligned by 1 byte
float* misaligned = (float*)((char*)aligned + 1);
float val = *misaligned;  // Serialized loads

// Good: Ensure 128-bit alignment
float* aligned = ...;  // cudaMalloc guarantees 256-byte alignment
```

---

## RoboCache Kernel Analysis

### Trajectory Resampling

**Profile:**
- Memory-latency bound (L1-optimized)
- Binary search: High temporal locality
- SM throughput: 85%+
- DRAM throughput: 1.59%
- L1 hit rate: 85%+

**Optimization:**
- Shared memory for timestamps (reduce L1 pressure)
- Warp-level scan for sorted data

---

### Multimodal Fusion

**Profile:**
- Compute-bound (3× interpolation)
- SM throughput: 90%+ (target)
- DRAM throughput: 15% (target)

**Optimization:**
- Vectorized loads for feature concatenation
- Thread-level parallelism across streams

---

### Voxelization

**Profile:**
- Bandwidth-bound (sparse writes)
- DRAM throughput: 70-90% (target)
- Atomic contention: Point-dependent

**Optimization:**
- Deterministic atomics (same hardware)
- Grid partitioning for reduced contention

---

## Checklist for New Kernels

1. **Memory Access Pattern**
   - [ ] Coalesced loads/stores (>80% efficiency)
   - [ ] Vectorized where possible (float4, bf16x2)
   - [ ] Minimize strided access

2. **Occupancy**
   - [ ] Target 50-75% for memory-latency bound
   - [ ] Target 75-100% for bandwidth-bound
   - [ ] Balance registers vs. shared memory

3. **Compute Efficiency**
   - [ ] Minimize warp divergence (>95% uniform)
   - [ ] Use warp-level primitives
   - [ ] Enable --use_fast_math (if acceptable)

4. **Profiling**
   - [ ] Nsight Compute metrics collected
   - [ ] Bottleneck identified (memory vs. compute)
   - [ ] Comparison to roofline model

5. **Validation**
   - [ ] Correctness tests pass (GPU vs. CPU)
   - [ ] Performance tests meet targets
   - [ ] Multiple input sizes tested

---

## References

- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Compute User Guide: https://docs.nvidia.com/nsight-compute/
- Hopper Architecture Whitepaper: https://resources.nvidia.com/en-us-tensor-core
- CUTLASS 4.3.0 Documentation: https://github.com/NVIDIA/cutlass

---

**Maintained By:** Brandon Dent <b@thegoatnote.com>  
**Last Updated:** 2025-11-06  
**Next Review:** Q1 2026 (Blackwell updates)

