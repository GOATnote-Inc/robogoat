# CUDA Kernel Inventory

Canonical list of shipped CUDA kernels in RoboCache.

## Shipped Kernels

### 1. Trajectory Resampling
**File:** `csrc/cuda/resample_kernel.cu`  
**Function:** `trajectory_resample_cuda_kernel`  
**Launch:** `csrc/cpp/resample_ops.cpp::resample_trajectories_cuda()`

**Algorithm:**
- Binary search for interval finding
- Linear interpolation between timestamps
- Supports BF16/FP32

**Performance:**
- H100: 0.018ms for (4, 50, 64)
- A100: 0.057ms for (4, 50, 64)

### 2. Multimodal Fusion
**File:** `csrc/cuda/multimodal_kernel.cu`  
**Function:** `multimodal_fusion_fp32_kernel`, `multimodal_fusion_bf16_kernel`  
**Launch:** `csrc/cuda/multimodal_kernel.cu::launch_multimodal_fusion()`

**Algorithm:**
- Fuses 3 sensor streams to common timeline
- Per-stream binary search + linear interpolation
- Single-pass kernel

**Performance:**
- H100: 0.018ms for 3-stream fusion
- A100: 0.057ms for 3-stream fusion

### 3. Point Cloud Voxelization
**File:** `csrc/cuda/voxelize_kernel.cu`  
**Function:** `voxelize_count_kernel`, `voxelize_occupancy_kernel`, etc.  
**Launch:** `csrc/cuda/voxelize_kernel.cu::launch_voxelize()`

**Modes:**
- `count`: Atomic increment per voxel
- `occupancy`: Binary occupancy grid
- `mean`: Feature averaging
- `max`: Feature max pooling

**Performance:**
- H100 (count): 34.5 B pts/s
- A100 (count): 15.6 B pts/s

## Archived Experiments

These implementations are NOT shipped, archived for reference only:

- `.archive/kernel_experiments/multimodal_fusion.cu` (early CUTLASS prototype)
- `.archive/kernel_experiments/multimodal_fusion_fixed.cu` (fixed version, superseded)
- `kernels/cutlass/trajectory_resample_*.cu` (various optimization attempts)

## Build Configuration

**CMake:** `cpp/CMakeLists.txt`  
**Target Architectures:**
- `sm_80` (A100)
- `sm_90` (H100)

**Compilation Flags:**
```cmake
-O3 --use_fast_math -lineinfo --expt-relaxed-constexpr -std=c++17
```

## Verification

To verify kernel implementations:
```bash
cd robocache
python -c "import robocache; print(robocache._cuda_available)"  # Should print True

# Run correctness tests
pytest tests/test_cuda_correctness.py -v

# Run with Compute Sanitizer
compute-sanitizer --tool memcheck python -m pytest tests/test_cuda_correctness.py
```

---

**Last Updated:** November 7, 2025  
**Maintainer:** GOATnote Engineering

