# Contributing to RoboCache

Thank you for your interest in contributing to RoboCache! We welcome contributions from the community to help make GPU-accelerated robot learning data processing accessible to everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **CUDA Toolkit 13.x+** installed
- **CUTLASS 4.3.0** installed (see [build instructions](docs/build_instructions.md))
- **PyTorch 2.0+** with CUDA support
- **CMake 3.18+**
- **Python 3.8+**
- **Git** for version control

### Building from Source

Follow our comprehensive [build instructions](docs/build_instructions.md):

```bash
# Clone the repository
git clone https://github.com/yourusername/robocache.git
cd robocache

# Install CUTLASS 4.3.0
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout v4.3.0
sudo cp -r include/cutlass /usr/local/include/

# Build RoboCache
cd ../robocache
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package in development mode
cd ..
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import robocache; robocache.print_installation_info()"
```

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `pre-commit` for git hooks

### Set Up Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run formatting and linting before each commit.

## How to Contribute

We welcome various types of contributions:

### 1. Bug Reports

Found a bug? Please [open an issue](https://github.com/yourusername/robocache/issues/new?template=bug_report.md) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (GPU, CUDA version, PyTorch version)
- Error messages or stack traces

### 2. Feature Requests

Have an idea for a new kernel or feature? [Open a feature request](https://github.com/yourusername/robocache/issues/new?template=feature_request.md) describing:
- The problem you're trying to solve
- Your proposed solution
- Why this would benefit the robot learning community
- Any alternative solutions you've considered

### 3. Code Contributions

Contributing code? Great! Please:

1. **Check existing issues**: Look for open issues labeled `good first issue` or `help wanted`
2. **Discuss first**: For major changes, open an issue to discuss your approach
3. **Fork and branch**: Create a feature branch from `main`
4. **Follow style guidelines**: See [Code Style Guidelines](#code-style-guidelines)
5. **Add tests**: Include unit tests for new functionality
6. **Update docs**: Document new features and API changes
7. **Submit PR**: Open a pull request with a clear description

### 4. Documentation Improvements

Documentation is crucial! You can help by:
- Fixing typos or clarifying existing docs
- Adding examples and tutorials
- Improving API documentation
- Creating blog posts or video tutorials

### 5. Performance Optimizations

Optimizing GPU kernels? Please:
- Include benchmarks showing improvement
- Document the optimization technique used
- Consider different GPU architectures (H100, A100, RTX 4090)
- Profile with NSight Compute/Systems and include results

## Pull Request Process

### 1. Fork and Create Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/robocache.git
cd robocache
git checkout -b feature/my-awesome-feature
```

### 2. Make Your Changes

- Write clean, readable code following our style guidelines
- Add or update tests as needed
- Update documentation (README, docstrings, examples)
- Ensure all tests pass locally

### 3. Test Your Changes

```bash
# Run Python tests
pytest tests/ -v

# Run C++ benchmarks
cd build
./benchmark_trajectory_resample

# Run examples
cd ../examples
python basic_usage.py
```

### 4. Format and Lint

```bash
# Format Python code
black python/ tests/ examples/

# Check for linting issues
flake8 python/ tests/ examples/

# Type checking
mypy python/robocache
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add point cloud voxelization kernel

- Implement CUTLASS kernel for 3D voxel grid generation
- Add PyTorch bindings with dtype dispatch
- Include benchmark showing 50x speedup vs CPU
- Add example usage in examples/voxelization.py

Closes #42"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description
- Reference related issues/PRs
- Explain the "why" not just the "what"

### 6. Push and Open Pull Request

```bash
git push origin feature/my-awesome-feature
```

Then open a pull request on GitHub with:

- **Title**: Clear, descriptive summary
- **Description**:
  - What changes were made
  - Why these changes are needed
  - How to test the changes
  - Related issues (e.g., "Closes #42")
  - Benchmarks (for performance changes)
  - Breaking changes (if any)

### 7. Code Review Process

- **Automated checks**: CI/CD will run tests and linting
- **Maintainer review**: A maintainer will review your code
- **Address feedback**: Make requested changes, push updates
- **Approval**: Once approved, your PR will be merged!

**Review Timeline:**
- Initial response: Within 3 business days
- Small PRs (<100 lines): Usually merged within 1 week
- Large PRs (>500 lines): May take 2-3 weeks

## Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use Black formatter (line length: 100)
# Use type hints for function signatures
def resample_trajectories(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    Brief description in one line.

    Detailed description explaining the function's purpose,
    algorithm, and any important considerations.

    Args:
        source_data: Description with shape [batch, len, dim]
        source_times: Description with shape [batch, len]
        target_times: Description with shape [batch, len]

    Returns:
        Resampled trajectories [batch, target_len, dim]

    Raises:
        RuntimeError: When CUDA extension is not available
        ValueError: When tensor shapes are incompatible

    Example:
        >>> data = torch.randn(64, 100, 32, device='cuda')
        >>> resampled = resample_trajectories(data, src_t, tgt_t)
    """
    pass
```

**Key Points:**
- Use `black` for automatic formatting
- Type hints for all public functions
- Google-style docstrings with Args/Returns/Raises
- Line length: 100 characters
- Imports: stdlib, third-party, local (separated by blank lines)

### C++/CUDA Code Style

```cpp
// Namespace: robocache::kernels
// File naming: snake_case.cu, snake_case.h
// Functions: snake_case
// Constants: UPPER_CASE
// Classes: PascalCase (if needed)

namespace robocache {
namespace kernels {

// Device pointers: d_* prefix
// Host pointers: h_* prefix
// Shared memory: s_* prefix

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

/**
 * Comprehensive function documentation.
 *
 * Detailed explanation of algorithm, memory layout,
 * and performance characteristics.
 *
 * @param source_data Device pointer to input data
 * @param batch_size Number of trajectories
 * @return cudaSuccess on success, error code otherwise
 */
cudaError_t resample_trajectories_bf16(
    const void* source_data,
    int batch_size,
    cudaStream_t stream = 0
);

}  // namespace kernels
}  // namespace robocache
```

**Key Points:**
- Indentation: 4 spaces (no tabs)
- Braces: Opening brace on same line
- Pointer alignment: `int* ptr` (asterisk with type)
- Comments: Doxygen-style for public APIs
- Error handling: Always check `cudaError_t` return values

### CMake Style

```cmake
# Use lowercase for commands
# Use UPPER_CASE for variables
# Indent nested blocks with 2 spaces

set(ROBOCACHE_VERSION_MAJOR 0)
set(ROBOCACHE_VERSION_MINOR 1)

if(BUILD_TORCH_EXTENSION)
  add_library(robocache_cuda SHARED
    kernels/cutlass/trajectory_resample.cu
    kernels/cutlass/trajectory_resample_torch.cu
  )
endif()
```

## Testing Guidelines

### Python Unit Tests

Write tests using `pytest`:

```python
# tests/test_resample.py
import pytest
import torch
import robocache

def test_resample_basic():
    """Test basic trajectory resampling functionality."""
    batch, src_len, tgt_len, dim = 4, 10, 5, 8

    data = torch.randn(batch, src_len, dim, dtype=torch.float32, device='cuda')
    src_times = torch.linspace(0, 1, src_len, device='cuda').expand(batch, -1)
    tgt_times = torch.linspace(0, 1, tgt_len, device='cuda').expand(batch, -1)

    result = robocache.resample_trajectories(data, src_times, tgt_times)

    assert result.shape == (batch, tgt_len, dim)
    assert result.dtype == torch.float32
    assert result.device == torch.device('cuda:0')

def test_resample_bfloat16():
    """Test BF16 dtype support."""
    data = torch.randn(2, 10, 4, dtype=torch.bfloat16, device='cuda')
    # ... test implementation

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_resample_dtypes(dtype):
    """Test all supported dtypes."""
    # ... parameterized test
```

**Test Requirements:**
- Every public function must have tests
- Test edge cases (empty tensors, single element, large batches)
- Test error conditions (wrong device, invalid shapes)
- Use fixtures for common setup
- Mark GPU tests with `@pytest.mark.gpu`

### C++ Benchmarks

Benchmarks live in `benchmarks/`:

```cpp
// Include warmup iterations
for (int i = 0; i < 10; i++) {
    kernel<<<grid, block>>>();
}
cudaDeviceSynchronize();

// Benchmark with proper timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (int i = 0; i < num_iterations; i++) {
    kernel<<<grid, block>>>();
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Documentation Guidelines

### Markdown Files

- Use clear, descriptive headings
- Include code examples for every feature
- Add table of contents for long documents
- Use tables for comparing options
- Include performance numbers with context

### Python Docstrings

- Use Google-style docstrings
- Include type information in Args section
- Provide realistic examples
- Document performance characteristics
- Note any limitations or caveats

### C++ Documentation

- Use Doxygen-style comments
- Document all public APIs
- Include memory layout diagrams
- Note thread safety and synchronization
- Document performance characteristics

## Community

### Getting Help

- **Documentation**: Start with [README.md](README.md) and [docs/](docs/)
- **GitHub Discussions**: Ask questions, share ideas
- **GitHub Issues**: Report bugs, request features
- **Email**: B@thegoatnote.com

### Staying Connected

- **GitHub**: Watch the repository for updates
- **Releases**: Follow release notes for new versions
- **LinkedIn**: Connect with maintainer [Brandon Dent](https://www.linkedin.com/in/brandon-dent-84aba2130)

## Recognition

Contributors will be:
- Listed in release notes
- Acknowledged in README
- Credited in academic papers (for significant contributions)

## Questions?

Don't hesitate to ask! Open an issue, start a discussion, or reach out directly. We're here to help you contribute successfully.

---

**Thank you for contributing to RoboCache!** Your contributions help advance the robot learning community. 🤖🚀
