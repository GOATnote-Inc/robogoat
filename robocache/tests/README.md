# RoboCache Test Suite

This directory contains the comprehensive test suite for RoboCache.

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Only GPU tests
pytest tests/ -m gpu -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Only benchmarks
pytest tests/ -m benchmark -v

# Run specific test file
pytest tests/test_resample.py -v

# Run specific test class
pytest tests/test_resample.py::TestResampleBasic -v

# Run specific test function
pytest tests/test_resample.py::TestResampleBasic::test_resample_basic -v
```

### Run with Coverage

```bash
pytest tests/ --cov=robocache --cov-report=html -v
```

Then open `htmlcov/index.html` in your browser.

## Test Organization

### Test Files

- **`test_installation.py`**: Tests for installation, imports, and environment setup
- **`test_resample.py`**: Comprehensive tests for trajectory resampling functionality
- **`conftest.py`**: Pytest configuration and shared fixtures

### Test Categories

Tests are marked with the following markers:

- **`@pytest.mark.gpu`**: Requires a CUDA-capable GPU
- **`@pytest.mark.slow`**: Takes >10 seconds to run
- **`@pytest.mark.benchmark`**: Performance benchmark tests
- **`@pytest.mark.integration`**: Integration tests with external systems

### Test Classes

#### `TestResampleBasic`
Basic functionality tests covering:
- Small, medium, and large batch sizes
- Output shape and dtype verification
- Device placement

#### `TestResampleDtypes`
Data type support tests:
- FP32, FP16, BF16 support
- Dtype preservation

#### `TestResampleEdgeCases`
Edge cases and boundary conditions:
- Single trajectory (batch=1)
- Same-length resampling
- Upsampling and downsampling
- Small and large action dimensions

#### `TestResampleErrors`
Error handling and validation:
- CPU tensor error detection
- Shape mismatch errors
- Wrong dimension errors

#### `TestResampleCorrectness`
Numerical correctness verification:
- Linear interpolation accuracy
- Constant trajectory preservation

#### `TestResamplePerformance`
Performance benchmarks:
- Throughput measurement
- Latency analysis
- Performance regression detection

## Requirements

### Basic Testing

```bash
pip install pytest
```

### Full Testing Suite

```bash
pip install pytest pytest-cov pytest-timeout pytest-benchmark
```

Or install development dependencies:

```bash
pip install -e ".[dev]"
```

## Writing New Tests

### Test Structure

```python
import pytest
import torch

@pytest.mark.gpu
class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self, gpu_device):
        """Test basic functionality of my feature."""
        # Arrange
        input_data = torch.randn(4, 10, 8, device=gpu_device)

        # Act
        result = my_function(input_data)

        # Assert
        assert result.shape == (4, 10, 8)
        assert result.device == gpu_device
```

### Using Fixtures

```python
def test_with_fixtures(self, small_batch, sample_trajectory_data):
    """Use fixtures for common test data."""
    source_data, source_times, target_times = sample_trajectory_data(**small_batch)
    # Test implementation
```

### Parameterized Tests

```python
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_different_batch_sizes(self, batch_size, gpu_device):
    """Test with different batch sizes."""
    # Test implementation
```

## Continuous Integration

These tests are automatically run by GitHub Actions on:
- Every pull request
- Every push to main branch
- Nightly builds

## Performance Benchmarks

Benchmark tests measure performance and detect regressions:

```bash
# Run benchmarks
pytest tests/ -m benchmark -v

# Save benchmark baseline
pytest tests/ -m benchmark --benchmark-save=baseline

# Compare against baseline
pytest tests/ -m benchmark --benchmark-compare=baseline
```

## GPU Testing

GPU tests require:
- CUDA-capable GPU (H100, A100, RTX 4090, etc.)
- CUDA Toolkit 13.x+
- PyTorch with CUDA support
- RoboCache CUDA extension built

To skip GPU tests:

```bash
pytest tests/ -m "not gpu" -v
```

## Troubleshooting

### Tests Not Found

```bash
# Ensure you're in the robocache directory
cd robocache
pytest tests/ -v
```

### Import Errors

```bash
# Install robocache in development mode
pip install -e .
```

### CUDA Extension Not Available

```bash
# Build the CUDA extension
cd robocache
mkdir build && cd build
cmake .. && make -j
```

### Slow Test Performance

```bash
# Skip slow tests
pytest tests/ -m "not slow" -v

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

## Contributing Tests

When contributing new features, please:

1. Add corresponding tests
2. Ensure all existing tests pass
3. Add test coverage for edge cases
4. Include performance benchmarks for GPU kernels
5. Document test purpose and expected behavior

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
