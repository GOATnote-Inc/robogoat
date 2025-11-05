# Security Policy

## Supported Versions

We take security seriously and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities to: **security@robogoat.ai**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information (as much as you can provide):

* Type of issue (e.g. buffer overflow, use-after-free, memory corruption, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Security Best Practices for Users

### Input Validation

RoboCache performs extensive input validation to prevent common vulnerabilities:

**Tensor Validation:**
```python
# All tensors are validated for:
# - Device (must be CUDA)
# - Dtype (must match expected type)
# - Shape (must match expected dimensions)
# - Contiguity (must be contiguous in memory)

# Example of safe usage:
points = torch.randn(batch, num_points, 3, device='cuda').contiguous()
result = robocache.voxelize(points, grid_size, voxel_size, origin)
```

**Memory Safety:**
```python
# RoboCache checks available memory before allocation
# and provides warnings if operation may OOM

# Safe usage with memory monitoring:
free, total = torch.cuda.mem_get_info()
if free < required_memory * 1.2:  # 20% safety margin
    # Use chunking to avoid OOM
    result = process_in_chunks(data, chunk_size)
```

### Buffer Overflow Protection

All CUDA kernels include bounds checking:

```cuda
// Example: Safe voxel access with bounds check
__device__ bool point_to_voxel_idx(
    float px, float py, float pz,
    const float* origin,
    float voxel_size,
    int depth, int height, int width,
    int& vx, int& vy, int& vz
) {
    // Convert to voxel coordinates
    vx = __float2int_rd((px - origin[0]) / voxel_size);
    vy = __float2int_rd((py - origin[1]) / voxel_size);
    vz = __float2int_rd((pz - origin[2]) / voxel_size);
    
    // CRITICAL: Bounds check before array access
    return (vx >= 0 && vx < depth &&
            vy >= 0 && vy < height &&
            vz >= 0 && vz < width);
}
```

### Multi-GPU Safety

When using multiple GPUs, RoboCache provides thread-safe device management:

```python
# Safe multi-GPU usage with CUDAGuard
from robocache.utils import CUDAGuard

# Automatic device restoration (RAII-style)
with CUDAGuard(device_id):
    result = process_on_device(data)
# Device automatically restored after context exit
```

### Integer Overflow Protection

Large tensor operations are checked for integer overflow:

```cpp
// Safe size calculation with overflow check
TORCH_CHECK(
    batch_size <= INT_MAX / (depth * height * width),
    "Total grid size would overflow int32. "
    "Reduce batch_size or grid resolution."
);

size_t total_size = static_cast<size_t>(batch_size) * 
                    static_cast<size_t>(depth) * 
                    static_cast<size_t>(height) * 
                    static_cast<size_t>(width);
```

## Known Security Considerations

### 1. Out-of-Memory (OOM) Conditions

**Risk:** Large batches or high-resolution grids can exhaust GPU memory.

**Mitigation:**
- Pre-allocation memory checks with `check_memory_available()`
- Automatic chunking with `auto_chunk()`
- Detailed error messages with memory usage information

**Example:**
```python
# Automatic OOM prevention
from robocache.memory import suggest_chunking

if robocache.will_oom(batch_size, num_points, grid_dims):
    config = suggest_chunking(batch_size, num_points, grid_dims)
    print(f"Using chunking: {config.num_chunks} chunks")
```

### 2. Numerical Stability

**Risk:** Fast-math optimizations can cause IEEE 754 non-compliance.

**Mitigation:**
- Fast-math disabled by default (`-fmad=false` in CMakeLists.txt)
- CPU/GPU numerical parity validated
- Explicit floor/round operations for determinism

**Compile flags:**
```cmake
# CRITICAL: Disabled for numerical safety
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")  # DISABLED
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false")  # Disable FMA
```

### 3. Atomic Operation Safety

**Risk:** Non-deterministic atomic operations can cause race conditions.

**Mitigation:**
- Use `atomicAdd` for deterministic accumulation (not `atomicExch`)
- Two-pass algorithms for correctness (accumulate, then reduce)
- CPU reference validation for all atomic operations

**Safe atomic usage:**
```cuda
// SAFE: Deterministic accumulation
atomicAdd(&voxel_grid[voxel_idx], 1.0f);

// UNSAFE: Non-deterministic last-write-wins
// atomicExch(&voxel_grid[voxel_idx], 1.0f);  // DON'T USE
```

### 4. Multi-Threading Safety

**Risk:** Concurrent access to GPU resources can cause crashes.

**Mitigation:**
- Thread-safe `StreamPool` with mutex protection
- RAII-style `CUDAGuard` for device switching
- Validated on multi-threaded data loading

**Thread-safe usage:**
```cpp
// Thread-safe stream pool
auto& streams = StreamPool::get_streams(device_id);  // Mutex protected
```

### 5. Dependency Security

**Risk:** Vulnerable dependencies can introduce security issues.

**Mitigation:**
- Minimal dependencies (PyTorch, CUDA)
- Pinned versions in requirements.txt
- Regular security audits

**Dependencies:**
```
torch>=2.0.0,<3.0.0  # Pinned major version
numpy>=1.21.0,<2.0.0
```

## Security Scanning

Our CI pipeline includes:

- ✅ Static analysis with clang-tidy
- ✅ Memory sanitizer in debug builds (AddressSanitizer, MemorySanitizer)
- ✅ Bounds checking in all kernels
- ✅ Input validation tests

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

## Security Contacts

* Security Team: security@robogoat.ai
* Project Lead: Available via GitHub issues for non-security questions

## Attribution

We appreciate responsible disclosure and will acknowledge security researchers who report vulnerabilities to us (unless they prefer to remain anonymous).

## Updates

This security policy may be updated from time to time. Please check back regularly.

**Last Updated:** November 5, 2025

