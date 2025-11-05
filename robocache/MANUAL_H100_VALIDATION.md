# Manual H100 Validation Guide for Phase 2

**Expert validation mode - Critical review of all claims**

---

## Prerequisites

1. Brev authentication:
```bash
brev login --token <YOUR_TOKEN>
```

2. Access H100:
```bash
brev shell awesome-gpu-name
```

---

## Step 1: Prepare Workspace on H100

```bash
# Set environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Navigate to workspace
cd /workspace/robocache

# Verify GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Should show: NVIDIA H100 PCIe, 9.0

# Verify CUDA
nvcc --version | grep "release"
# Should show: release 13.0

# Check existing files
ls -la kernels/cutlass/
# Should have trajectory_resample.{h,cu} from Phase 1
```

---

## Step 2: Copy Phase 2 Files to H100

**From your local machine**, copy the Phase 2 files:

```bash
# Option A: Use git (if repo is tracked)
cd /workspace/robocache
git pull origin claude/robocache-trajectory-resampling-011CUmL9iZ88eGvKKKSz7LuQ

# Option B: Manual copy (if Option A doesn't work)
# On LOCAL machine:
cd /Users/kiteboard/robogoat
tar czf phase2_files.tar.gz \
    robocache/kernels/cutlass/multimodal_fusion.h \
    robocache/kernels/cutlass/multimodal_fusion.cu \
    robocache/benchmarks/benchmark_multimodal_fusion.cu

# Copy to H100 (replace IP with actual H100 IP from `brev ls`)
scp phase2_files.tar.gz shadeform@<H100_IP>:/workspace/

# On H100:
cd /workspace
tar xzf phase2_files.tar.gz
```

---

## Step 3: Build Phase 2 on H100

```bash
cd /workspace/robocache

# Create clean build
rm -rf build
mkdir build
cd build

# Configure CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DBUILD_TORCH_EXTENSION=OFF \
    -DROBOCACHE_BUNDLE_CUTLASS=ON

# Build (should take 2-3 minutes)
make -j$(nproc) benchmark_multimodal_fusion

# Verify binary created
ls -lh benchmark_multimodal_fusion
# Should show ~500KB executable
```

**Expected build output:**
```
[100%] Building CUDA object CMakeFiles/benchmark_multimodal_fusion.dir/benchmarks/benchmark_multimodal_fusion.cu.o
[100%] Linking CUDA executable benchmark_multimodal_fusion
[100%] Built target benchmark_multimodal_fusion
```

---

## Step 4: Run Benchmark (Critical Validation)

```bash
cd /workspace/robocache/build

# Run benchmark
./benchmark_multimodal_fusion 2>&1 | tee ../benchmark_results.txt
```

**Expected output format:**
```
╔══════════════════════════════════════════════════════════════════════╗
║          RoboCache Phase 2: Multimodal Sensor Fusion                ║
║                  H100 Performance Benchmark                          ║
╚══════════════════════════════════════════════════════════════════════╝

GPU: NVIDIA H100 PCIe
Compute capability: 9.0

════════════════════════════════════════════════════════════════════════
Configuration: Small (1-sec episodes, batch=32)
════════════════════════════════════════════════════════════════════════
...
Average latency:    0.XXX ms
Throughput:         XXXXX samples/sec
Bandwidth:          XXX.X GB/s
HBM3 efficiency:    XX.XX%
```

---

## Step 5: Critical Performance Analysis

### Target Metrics (Phase 2 Claims to Verify)

| Configuration | Max Latency | Min Bandwidth | Min Efficiency |
|---------------|-------------|---------------|----------------|
| Small (1-sec, batch=32) | 0.05 ms | 200 GB/s | 6% |
| Medium (5-sec, batch=128) | 0.15 ms | 250 GB/s | 8% |
| Large (10-sec, batch=256) | 0.30 ms | 280 GB/s | 9% |

### Validation Checklist

- [ ] **Latency**: Does it meet targets above?
- [ ] **Bandwidth**: Is it >= minimum?
- [ ] **Efficiency**: Is it >= minimum for memory-latency bound?
- [ ] **Speedup**: Is it 50-125x vs CPU (15ms baseline)?
- [ ] **No crashes**: Benchmark completes for all 3 configs

### Calculate Speedup

```bash
# From benchmark output, extract latency for medium config
# Example: 0.10 ms for batch=128

# CPU baseline: 15 ms per sample
# GPU: 0.10 ms for 128 samples = 0.10/128 = 0.00078 ms per sample
# Speedup = 15 / 0.00078 = 19,230x per sample
# OR: Throughput = 128 / 0.10ms * 1000 = 1.28M samples/sec
# vs CPU: 1/15ms * 1000 = 67 samples/sec
# Speedup = 1.28M / 67 = 19,100x

# This should be in range 50-125x claimed
```

---

## Step 6: NCU Profiling (Critical - Verify Optimization Claims)

```bash
cd /workspace/robocache/build

# Run NCU profiling (requires sudo)
sudo ncu \
    --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
    sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    --target-processes all \
    --kernel-name fused_multimodal_alignment_kernel \
    ./benchmark_multimodal_fusion \
    2>&1 | tee ../ncu_results.txt
```

### NCU Validation Checklist

From `ncu_results.txt`, verify:

- [ ] **DRAM Throughput**: Should be < 5% (shared memory working)
- [ ] **L1 Cache Hit Rate**: Should be > 50% (time arrays cached)
- [ ] **Memory Coalescing**: Check load efficiency
- [ ] **Compute Utilization**: Should be low (memory-latency bound)

**Critical questions:**
1. Is DRAM throughput < 5%? (If not, shared memory isn't working)
2. Is this memory-latency bound? (Low compute, high memory wait)
3. Does it match Phase 1 efficiency (~10%)? (Should be similar)

---

## Step 7: Correctness Validation

```bash
# Create simple correctness test
cat > /workspace/robocache/test_phase2_correctness.cu << 'EOF'
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include "multimodal_fusion.h"

int main() {
    // Small test: 1 batch, simple case
    int batch = 1, vision_len = 10, proprio_len = 10, force_len = 10, target_len = 5;
    int vision_dim = 4, proprio_dim = 2, force_dim = 2;
    
    // Allocate
    __nv_bfloat16 *d_vision, *d_proprio, *d_force, *d_output;
    float *d_vision_t, *d_proprio_t, *d_force_t, *d_target_t;
    
    cudaMalloc(&d_vision, batch * vision_len * vision_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vision_t, batch * vision_len * sizeof(float));
    cudaMalloc(&d_proprio, batch * proprio_len * proprio_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_proprio_t, batch * proprio_len * sizeof(float));
    cudaMalloc(&d_force, batch * force_len * force_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_force_t, batch * force_len * sizeof(float));
    cudaMalloc(&d_target_t, batch * target_len * sizeof(float));
    cudaMalloc(&d_output, batch * target_len * (vision_dim+proprio_dim+force_dim) * sizeof(__nv_bfloat16));
    
    // Initialize with simple values
    cudaMemset(d_vision, 0, batch * vision_len * vision_dim * sizeof(__nv_bfloat16));
    cudaMemset(d_proprio, 0, batch * proprio_len * proprio_dim * sizeof(__nv_bfloat16));
    cudaMemset(d_force, 0, batch * force_len * force_dim * sizeof(__nv_bfloat16));
    
    // Call kernel
    auto err = robocache::kernels::fused_multimodal_alignment<__nv_bfloat16>(
        d_vision, d_vision_t, vision_len, vision_dim,
        d_proprio, d_proprio_t, proprio_len, proprio_dim,
        d_force, d_force_t, force_len, force_dim,
        d_target_t, target_len,
        d_output, batch, 0
    );
    
    cudaDeviceSynchronize();
    
    if (err == cudaSuccess) {
        std::cout << "✓ Correctness test PASSED - kernel executed without errors\n";
        return 0;
    } else {
        std::cout << "❌ Correctness test FAILED: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
}
EOF

# Compile and run
nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90 \
    -I../kernels/cutlass -I../_deps/cutlass-src/include \
    test_phase2_correctness.cu ../kernels/cutlass/multimodal_fusion.cu ../kernels/cutlass/trajectory_resample.cu \
    -o test_correctness

./test_correctness
# Should print: ✓ Correctness test PASSED
```

---

## Step 8: Final Validation Report

### Create Summary

```bash
cd /workspace/robocache

cat > PHASE2_VALIDATION_RESULTS.md << 'EOF'
# Phase 2 Validation Results

**Date:** $(date)
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader)
**CUDA:** $(nvcc --version | grep release)

## Performance Summary

[Paste benchmark_results.txt key metrics here]

## NCU Analysis

[Paste key NCU metrics from ncu_results.txt]

## Verdict

- [ ] Latency targets met
- [ ] Bandwidth targets met
- [ ] Efficiency reasonable for workload
- [ ] NCU confirms optimizations working
- [ ] Correctness test passed

**Overall Status:** PASS / FAIL / NEEDS_WORK

**Issues found:**
1. [List any performance or correctness issues]

**Recommendations:**
1. [List any improvements needed]
EOF
```

---

## Troubleshooting

### Build Fails

```bash
# Check CUTLASS fetched correctly
ls -la build/_deps/cutlass-src/include/
# Should have cutlass/ directory with headers

# Check compiler flags
cd build
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1 benchmark_multimodal_fusion
```

### Runtime Crashes

```bash
# Run with cuda-memcheck
cuda-memcheck ./benchmark_multimodal_fusion

# Check for out-of-bounds
compute-sanitizer --tool memcheck ./benchmark_multimodal_fusion
```

### Poor Performance

```bash
# Profile with nsys
nsys profile --stats=true ./benchmark_multimodal_fusion

# Check GPU utilization
nvidia-smi dmon -s u -c 10  # While benchmark runs in another terminal
```

---

## Success Criteria

Phase 2 is **validated** if:

1. ✅ Builds without errors on H100 + CUDA 13.0
2. ✅ All 3 benchmark configs complete without crashes
3. ✅ Latency/bandwidth/efficiency within targets
4. ✅ NCU shows < 5% DRAM (shared memory working)
5. ✅ Speedup vs CPU is 50-125x range
6. ✅ Correctness test passes

**Current Status:** [To be filled after validation]

---

## Contact

If validation fails, provide:
1. Full build log
2. benchmark_results.txt
3. ncu_results.txt (if available)
4. Any error messages

This will help diagnose issues quickly.

