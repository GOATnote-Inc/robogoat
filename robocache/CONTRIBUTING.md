# Contributing to RoboCache

Thank you for your interest in contributing to RoboCache! This document provides guidelines for contributing to this expert-level GPU-accelerated robotics library.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Performance Requirements](#performance-requirements)
6. [Testing Standards](#testing-standards)
7. [Documentation Requirements](#documentation-requirements)
8. [Pull Request Process](#pull-request-process)
9. [Security](#security)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- NVIDIA GPU with Compute Capability 9.0+ (Hopper/H100 recommended)
- CUDA 12.0+
- CMake 3.18+
- Python 3.8+
- PyTorch 2.0+

### Clone the Repository

```bash
git clone https://github.com/GOATnote-Inc/robogoat.git
cd robogoat/robocache
```

### Build from Source

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make -j$(nproc)
```

### Run Tests

```bash
# C++ tests
./build/test_voxelization

# Python tests
pytest tests/ -v
```

## Development Setup

### IDE Configuration

**VS Code (Recommended):**
```json
{
    "C_Cpp.default.compilerPath": "/usr/local/cuda/bin/nvcc",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/kernels",
        "/usr/local/cuda/include"
    ],
    "C_Cpp.default.cStandard": "c11",
    "C_Cpp.default.cppStandard": "c++17"
}
```

### Linting and Formatting

```bash
# C++ formatting (clang-format)
find kernels/ -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# Python formatting (black)
black tests/ benchmarks/

# Python linting (flake8)
flake8 tests/ benchmarks/ --max-line-length=100
```

## Contribution Guidelines

### What We're Looking For

We welcome contributions that:

- ✅ **Add new GPU-accelerated operations** for robotics data processing
- ✅ **Improve performance** of existing kernels (with benchmarks)
- ✅ **Enhance error handling** and user experience
- ✅ **Add comprehensive tests** and documentation
- ✅ **Fix bugs** with test cases demonstrating the fix

### What We're NOT Looking For

- ❌ CPU-only implementations (this is a GPU library)
- ❌ Unverified performance claims (no "should be faster")
- ❌ Breaking API changes without migration path
- ❌ Code without tests or documentation

### Code Quality Standards

All contributions must meet these standards:

1. **Correctness First**
   - CPU reference implementation for validation
   - Zero tolerance for CPU/GPU mismatches
   - Comprehensive edge case testing

2. **Performance Second**
   - Benchmarks on real hardware (H100 preferred)
   - Comparison to baseline (PyTorch native, etc.)
   - NCU profiling for bottleneck analysis

3. **Production Quality**
   - Comprehensive error handling
   - Input validation (shapes, dtypes, devices)
   - Memory safety (bounds checking, OOM prevention)
   - Thread-safe for multi-GPU

4. **Documentation Always**
   - API documentation with examples
   - Performance characteristics
   - Known limitations
   - Troubleshooting guide

## Performance Requirements

### Benchmarking Standards

All performance optimizations must include:

1. **Hardware Specifications**
```python
# Include in benchmark output
Hardware: NVIDIA H100 PCIe 80GB
CUDA: 12.0
Driver: 535.104.05
PyTorch: 2.1.0
```

2. **Statistical Analysis**
```python
# Run multiple iterations for statistical significance
num_warmup = 10
num_iterations = 100

times = []
for i in range(num_warmup + num_iterations):
    start = time.time()
    result = operation(input)
    torch.cuda.synchronize()
    if i >= num_warmup:
        times.append(time.time() - start)

mean_time = np.mean(times)
std_time = np.std(times)
print(f"Time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
```

3. **Comparison to Baseline**
```python
# Always compare to existing implementation
baseline_time = benchmark_baseline(input)
optimized_time = benchmark_optimized(input)
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x")
```

4. **NCU Profiling** (for kernel optimizations)
```bash
# Capture key metrics
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./benchmark_kernel

# Include analysis in PR description
```

### Performance Acceptance Criteria

- **Speedup:** Must be at least 1.1x faster (10% improvement)
- **Regression:** No more than 5% slowdown in any existing benchmark
- **Memory:** No more than 10% increase in peak memory usage
- **Validation:** Zero accuracy degradation (bit-exact or within tolerance)

## Testing Standards

### Test Coverage Requirements

All new features must include:

1. **Unit Tests**
```python
def test_voxelization_small():
    """Test voxelization with small input"""
    points = torch.randn(2, 100, 3, device='cuda')
    grid_size = torch.tensor([32, 32, 32], dtype=torch.int32, device='cuda')
    origin = torch.zeros(3, device='cuda')
    
    result = robocache.voxelize(points, grid_size, 0.1, origin)
    
    assert result.shape == (2, 32, 32, 32)
    assert result.device.type == 'cuda'
    assert not torch.isnan(result).any()
```

2. **Edge Case Tests**
```python
def test_voxelization_empty():
    """Test voxelization with empty point cloud"""
    points = torch.randn(2, 0, 3, device='cuda')  # Empty
    # ... should handle gracefully
```

3. **CPU Reference Validation**
```python
def test_voxelization_correctness():
    """Validate GPU result matches CPU reference"""
    points = generate_test_points()
    
    gpu_result = voxelize_gpu(points)
    cpu_result = voxelize_cpu(points.cpu())
    
    # Zero tolerance for mismatches
    assert torch.allclose(gpu_result.cpu(), cpu_result, atol=1e-6)
```

4. **Error Handling Tests**
```python
def test_voxelization_wrong_device():
    """Test error when input on wrong device"""
    points = torch.randn(2, 100, 3)  # CPU tensor
    
    with pytest.raises(RuntimeError, match="must be on CUDA"):
        robocache.voxelize(points, ...)
```

### Test Naming Convention

```python
def test_{operation}_{scenario}():
    """Test {operation} with {scenario}"""
    # Descriptive docstring
    # Test implementation
```

## Documentation Requirements

### API Documentation

All public functions must include:

```python
def voxelize_occupancy(
    points: torch.Tensor,
    grid_size: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor
) -> torch.Tensor:
    """
    Convert point cloud to occupancy voxel grid.
    
    Args:
        points: Point cloud [batch, num_points, 3] (float32, CUDA)
        grid_size: Grid dimensions [3] (int32, CUDA) - [depth, height, width]
        voxel_size: Size of each voxel in meters (positive float)
        origin: Grid origin [3] (float32, CUDA) - [x, y, z]
    
    Returns:
        Occupancy grid [batch, depth, height, width] (float32, CUDA)
        Values are 0.0 (empty) or 1.0 (occupied)
    
    Raises:
        RuntimeError: If inputs are invalid or operation fails
        
    Example:
        >>> points = torch.randn(4, 1024, 3, device='cuda')
        >>> grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
        >>> origin = torch.tensor([0.0, 0.0, 0.0], device='cuda')
        >>> voxels = robocache.voxelize_occupancy(points, grid_size, 0.1, origin)
        >>> print(voxels.shape)
        torch.Size([4, 64, 64, 64])
    
    Performance:
        - H100: ~0.017 ms for small grids (64³)
        - Speedup: 581x vs CPU reference
        - Bandwidth: 666 GB/s (19.9% of H100 peak)
    
    Notes:
        - Uses deterministic atomic operations (atomicAdd)
        - Two-pass algorithm for CPU/GPU parity
        - Memory required: batch * D * H * W * 4 bytes
    
    See Also:
        - voxelize_density: For point counts per voxel
        - voxelize_tsdf: For TSDF reconstruction
    """
```

### Performance Documentation

Create a markdown file for each major feature:

```markdown
# Voxelization Performance

## Benchmarks (H100)

| Config | Points | Grid | Latency | Throughput |
|--------|--------|------|---------|------------|
| Small  | 4K     | 64³  | 0.017ms | 460K/sec   |
| Medium | 16K    | 128³ | 0.558ms | 57K/sec    |
| Large  | 65K    | 256³ | 7.489ms | 8.5K/sec   |

## NCU Analysis

- Memory-bound: 0.2 FLOP/byte
- DRAM bandwidth: 666 GB/s (19.9% of peak)
- SM utilization: 85-90%
- Bottleneck: Atomic contention

## Optimization Opportunities

- [x] Deterministic atomics (v0.2.0)
- [ ] BF16 support (5-30% faster)
- [ ] Multi-GPU batching (4x scaling)
```

## Pull Request Process

### Before Submitting

1. **Run all tests**
```bash
pytest tests/ -v
./build/test_voxelization
```

2. **Format code**
```bash
clang-format -i kernels/**/*.{cu,cuh}
black tests/ benchmarks/
```

3. **Run benchmarks**
```bash
./benchmarks/run_all.sh
```

4. **Update documentation**
- Add API docs for new functions
- Update performance tables
- Add examples

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Performance improvement
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update

## Performance Impact
- Baseline: X.XX ms
- Optimized: Y.YY ms
- Speedup: Z.Zx

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] CPU reference validation passed
- [ ] Benchmarks on real hardware

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings

## Hardware Tested
- GPU: NVIDIA H100 PCIe
- CUDA: 12.0
- Driver: 535.104.05
```

### Review Process

1. **Automated Checks** (CI must pass)
   - Build succeeds
   - All tests pass
   - Linting passes
   - No new warnings

2. **Code Review** (at least one approval)
   - Correctness verified
   - Performance validated
   - Documentation complete
   - Tests comprehensive

3. **Merge** (squash and merge preferred)
   - Clear commit message
   - Reference issue number
   - Update changelog

## Security

See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities.

**Never commit:**
- Credentials or API keys
- Personal information
- Unvalidated user input handling
- Unsafe memory operations

## Questions?

- 💬 GitHub Discussions: For questions and general discussion
- 🐛 GitHub Issues: For bug reports and feature requests
- 📧 Email: contrib@robogoat.ai for private inquiries

## Recognition

Contributors are recognized in:
- README.md acknowledgments
- Release notes
- Git commit history

Thank you for contributing to RoboCache! 🚀

**Last Updated:** November 5, 2025

