# RoboCache Multi-GPU Programming Guide

**Production-Grade Multi-Device CUDA**

Expert-level guide to safe and efficient multi-GPU programming in RoboCache. Demonstrates CUDA stream semantics, device management, and multi-GPU workload distribution.

---

## Table of Contents

1. [Overview](#overview)
2. [CUDAGuard](#cudaguard-raii-device-switcher)
3. [Stream Management](#stream-management)
4. [Multi-GPU Patterns](#multi-gpu-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Multi-GPU?

**Use cases:**
- **Scale throughput:** Process more data per second
- **Reduce latency:** Distribute batch across GPUs for parallel processing
- **Memory scaling:** Handle batches larger than single-GPU memory

**RoboCache multi-GPU features:**
- ✅ CUDAGuard for safe device switching
- ✅ Stream pool management
- ✅ Cross-device data transfer (P2P when available)
- ✅ Workload distribution utilities
- ✅ Thread-safe device management

---

## CUDAGuard: RAII Device Switcher

### Basic Usage

```cpp
#include "multi_gpu.cuh"
using namespace robocache::multigpu;

// Switch to device 1, automatically restore on scope exit
{
    CUDAGuard guard(1);
    // All CUDA operations now target device 1
    cudaMalloc(...);
    kernel<<<grid, block>>>(...);
}  // Automatically restores previous device
```

### Why CUDAGuard?

**Problem:** Manual device switching is error-prone:

```cpp
// BAD: Fragile, error-prone
int prev_device;
cudaGetDevice(&prev_device);
cudaSetDevice(target_device);

// ... do work ...

cudaSetDevice(prev_device);  // Easy to forget!
```

**Solution:** RAII-style automatic restoration:

```cpp
// GOOD: Safe, automatic
CUDAGuard guard(target_device);
// ... do work ...
// Automatically restores device on scope exit
```

### Advanced Usage

**Switch to tensor's device:**
```cpp
CUDAGuard guard(tensor);  // Switch to tensor.device()
```

**Conditional guard (no-op if device == -1):**
```cpp
CUDAGuard guard(maybe_device);  // No switch if -1
```

**Nested guards:**
```cpp
{
    CUDAGuard guard1(0);  // Switch to device 0
    // work on device 0
    
    {
        CUDAGuard guard2(1);  // Switch to device 1
        // work on device 1
    }  // Restore to device 0
    
    // Back on device 0
}  // Restore to original device
```

---

## Stream Management

### Why Streams?

**Streams enable:**
- Concurrent kernel execution
- Overlapping compute and memory transfer
- Better GPU utilization

**RoboCache provides:**
- Stream pool per device (avoid creation overhead)
- Automatic stream management
- Thread-safe stream access

### Using Stream Pools

```cpp
// Get stream pool for device 0 (creates 4 streams by default)
auto& streams = StreamPool::get_streams(0, 4);

// Launch kernels on different streams
for (int i = 0; i < 4; i++) {
    kernel<<<grid, block, 0, streams[i]>>>(data[i], ...);
}

// Synchronize all streams
StreamPool::synchronize_all(0);
```

### PyTorch Integration

```python
# Create custom stream
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result = robocache.voxelize(points, ...)

# Synchronize
stream.synchronize()
```

---

## Multi-GPU Patterns

### Pattern 1: Data Parallelism

**Goal:** Process different batches on different GPUs

```cpp
// Split batch across GPUs
auto splits = split_batch(total_batch_size, num_gpus);

std::vector<torch::Tensor> results;
for (auto [device_id, start_idx, count] : splits) {
    CUDAGuard guard(device_id);
    
    // Slice batch for this GPU
    auto points_slice = points.slice(0, start_idx, start_idx + count);
    
    // Process on this GPU
    auto result = voxelize_occupancy(points_slice, ...);
    results.push_back(result);
}

// Gather results to device 0
auto final_result = gather_from_devices(results, 0);
```

**Speedup:** Linear with number of GPUs (ideal: 4 GPUs = 4x throughput)

---

### Pattern 2: Pipeline Parallelism

**Goal:** Stage different operations on different GPUs

```cpp
// Stage 1: Preprocessing on GPU 0
CUDAGuard guard0(0);
auto preprocessed = preprocess(raw_data_gpu0);

// Transfer to GPU 1
auto data_gpu1 = copy_to_device_async(preprocessed, 1);

// Stage 2: Voxelization on GPU 1
CUDAGuard guard1(1);
auto voxelized = voxelize(data_gpu1, ...);

// Transfer to GPU 2
auto data_gpu2 = copy_to_device_async(voxelized, 2);

// Stage 3: Postprocessing on GPU 2
CUDAGuard guard2(2);
auto result = postprocess(data_gpu2);
```

**Speedup:** Overlap different stages (throughput can exceed single-GPU)

---

### Pattern 3: Model Parallelism

**Goal:** Split large models across GPUs (each GPU processes different layers)

```cpp
// Layer 1 on GPU 0
CUDAGuard guard0(0);
auto layer1_out = forward_layer1(input_gpu0);

// Transfer to GPU 1
auto layer1_out_gpu1 = copy_to_device_async(layer1_out, 1);

// Layer 2 on GPU 1
CUDAGuard guard1(1);
auto layer2_out = forward_layer2(layer1_out_gpu1);

// Continue...
```

**Use case:** Models too large for single GPU memory

---

## Performance Optimization

### 1. Enable Peer-to-Peer Access

**P2P allows direct GPU-to-GPU memory access** (bypasses CPU/PCIe):

```cpp
// Enable P2P between device 0 and 1
if (enable_peer_access(0, 1)) {
    // P2P enabled - fast transfers
} else {
    // P2P not available - use staging buffer
}
```

**Benchmark:**
```
Without P2P: 12 GB/s (PCIe 3.0 x16)
With P2P:    50 GB/s (NVLink 2.0)  ← 4.2x faster!
```

---

### 2. Overlap Compute and Transfer

```cpp
// Stream 0: Compute on current batch
kernel<<<grid, block, 0, stream0>>>(batch_data[i], ...);

// Stream 1: Transfer next batch (overlapped!)
cudaMemcpyAsync(batch_data[i+1], ..., cudaMemcpyHostToDevice, stream1);

// Synchronize stream 0 before using result
cudaStreamSynchronize(stream0);
```

**Speedup:** Hide memory transfer latency

---

### 3. Load Balancing

**Problem:** Unequal workload → GPU underutilization

```cpp
// BAD: Fixed split (GPU 0 may finish early)
split_batch(100, 4);  // [25, 25, 25, 25]
```

**Solution:** Dynamic load balancing

```cpp
// GOOD: Adjust split based on GPU performance
auto split = split_batch_weighted(100, gpu_weights);
// e.g., [35, 30, 20, 15] if GPU 0 is faster
```

---

### 4. Memory Pinning

**Pinned (page-locked) memory** enables faster H2D/D2H transfers:

```cpp
// Allocate pinned host memory
torch::Tensor host_data = torch::empty(
    size,
    torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)
);

// Transfer (faster than pageable memory)
auto gpu_data = host_data.to('cuda:0');
```

**Speedup:** ~2x faster H2D/D2H transfers

---

## Best Practices

### 1. Always Use CUDAGuard

```cpp
// BAD: Manual device management
cudaSetDevice(target_device);
// ... work ...
cudaSetDevice(prev_device);  // Easy to forget!

// GOOD: Automatic restoration
CUDAGuard guard(target_device);
// ... work ...
// Automatic restoration
```

---

### 2. Validate All Tensors Are on Same Device

```cpp
validate_same_device(
    {points, features, labels},
    {"points", "features", "labels"}
);
```

**Catches:** Cross-device bugs early (before kernel launch)

---

### 3. Synchronize Before Returning to User

```cpp
torch::Tensor my_kernel_wrapper(...) {
    // Launch kernels
    kernel<<<grid, block, 0, stream>>>(args);
    
    // CRITICAL: Synchronize before returning!
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return result;
}
```

**Why?** User expects result to be ready immediately.

---

### 4. Use Streams for Concurrency

```cpp
// BAD: Sequential execution
for (int i = 0; i < N; i++) {
    kernel<<<grid, block>>>(data[i]);
    cudaDeviceSynchronize();  // ← Serializes execution!
}

// GOOD: Concurrent execution
for (int i = 0; i < N; i++) {
    kernel<<<grid, block, 0, streams[i % 4]>>>(data[i]);
}
// Synchronize all at end
for (auto stream : streams) {
    cudaStreamSynchronize(stream);
}
```

**Speedup:** Up to 4x (if 4 independent kernels)

---

### 5. Batch P2P Transfers

```cpp
// BAD: Many small transfers
for (int i = 0; i < 1000; i++) {
    cudaMemcpy(dst + i, src + i, sizeof(float), cudaMemcpyDeviceToDevice);
}

// GOOD: Single large transfer
cudaMemcpy(dst, src, 1000 * sizeof(float), cudaMemcpyDeviceToDevice);
```

**Speedup:** ~100x (avoid per-transfer overhead)

---

## Troubleshooting

### Error: "Tensor on different device"

**Symptom:**
```
RuntimeError: Tensor 'features' is on different device. Expected: cuda:0, got: cuda:1.
```

**Fix:**
```python
# Move all tensors to same device
device = points.device
features = features.to(device)
labels = labels.to(device)
```

---

### Error: "Peer access already enabled"

**Symptom:**
```
CUDA Error: cudaErrorPeerAccessAlreadyEnabled
```

**Fix:** This is expected - P2P was already enabled. Clear error:
```cpp
cudaError_t err = cudaDeviceEnablePeerAccess(device, 0);
if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();  // Clear error
}
```

---

### Performance: Slower with Multi-GPU

**Possible causes:**
1. **Insufficient batch size:** Overhead dominates for tiny batches
   - **Fix:** Increase batch size (e.g., 32 → 128)

2. **Memory bottleneck:** Waiting for transfers
   - **Fix:** Use P2P, overlap compute/transfer

3. **Load imbalance:** Some GPUs idle
   - **Fix:** Dynamic load balancing

4. **PCIe bottleneck:** Limited inter-GPU bandwidth
   - **Fix:** Use NVLink if available, or minimize cross-GPU communication

---

### Memory: Out of Memory on Multi-GPU

**Symptom:**
```
RuntimeError: CUDA out of memory on device 1
```

**Fix:**
1. **Reduce batch per GPU:**
   ```python
   batch_per_gpu = total_batch // num_gpus
   # Reduce further if OOM
   batch_per_gpu = batch_per_gpu // 2
   ```

2. **Free cached memory:**
   ```python
   torch.cuda.empty_cache()
   ```

3. **Use gradient checkpointing** (if training)

---

## Advanced Topics

### Stream Priority

```cpp
// High-priority stream for latency-critical work
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

cudaStream_t high_priority_stream;
cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, priority_high);

// Use for critical kernels
kernel<<<grid, block, 0, high_priority_stream>>>(args);
```

---

### CUDA Events for Fine-Grained Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

---

### Multi-Process Service (MPS)

**For servers with many concurrent users:**

```bash
# Enable MPS (allows concurrent processes to share GPU)
nvidia-cuda-mps-control -d

# Run your application
python train.py

# Disable MPS
echo quit | nvidia-cuda-mps-control
```

**Benefit:** Better GPU utilization when multiple processes use GPU

---

## Conclusion

**RoboCache multi-GPU support demonstrates production-grade CUDA engineering:**

✅ **CUDAGuard** for safe device switching  
✅ **Stream pools** for efficient concurrency  
✅ **P2P access** for fast inter-GPU transfers  
✅ **Workload distribution** utilities  
✅ **Thread-safe** device management  

**For NVIDIA Interview:** Shows deep understanding of:
- CUDA stream semantics
- Multi-device programming patterns
- Performance optimization strategies
- Production error handling

**Next Steps:**
- Memory profiling and chunking
- Advanced Hopper features (TMA, WGMMA)
- Distributed training integration (NCCL)

---

**Key Takeaway:** Multi-GPU programming is about **safety**, **concurrency**, and **communication efficiency**. RoboCache provides the infrastructure to do it right.

