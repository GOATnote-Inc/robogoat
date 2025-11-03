"""
Tests for trajectory resampling functionality.
"""

import pytest
import torch
import numpy as np


# Skip all tests if CUDA extension not available
try:
    import robocache

    CUDA_AVAILABLE = robocache.check_installation()["cuda_extension_available"]
except ImportError:
    CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension not available")


@pytest.mark.gpu
class TestResampleBasic:
    """Basic functionality tests for trajectory resampling."""

    def test_resample_basic(self, small_batch, sample_trajectory_data):
        """Test basic trajectory resampling functionality."""
        import robocache

        source_data, source_times, target_times = sample_trajectory_data(**small_batch)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        # Check output shape
        expected_shape = (
            small_batch["batch_size"],
            small_batch["target_len"],
            small_batch["action_dim"],
        )
        assert result.shape == expected_shape

        # Check dtype preserved
        assert result.dtype == source_data.dtype

        # Check device preserved
        assert result.device == source_data.device

    def test_resample_medium_batch(self, medium_batch, sample_trajectory_data):
        """Test with medium-sized batches."""
        import robocache

        source_data, source_times, target_times = sample_trajectory_data(**medium_batch)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        expected_shape = (
            medium_batch["batch_size"],
            medium_batch["target_len"],
            medium_batch["action_dim"],
        )
        assert result.shape == expected_shape

    @pytest.mark.slow
    def test_resample_large_batch(self, large_batch, sample_trajectory_data):
        """Test with large batches."""
        import robocache

        source_data, source_times, target_times = sample_trajectory_data(**large_batch)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        expected_shape = (
            large_batch["batch_size"],
            large_batch["target_len"],
            large_batch["action_dim"],
        )
        assert result.shape == expected_shape


@pytest.mark.gpu
class TestResampleDtypes:
    """Test different data types."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16]
    )
    def test_dtype_support(self, small_batch, sample_trajectory_data, dtype):
        """Test all supported dtypes."""
        import robocache

        source_data, source_times, target_times = sample_trajectory_data(
            **small_batch, dtype=dtype
        )

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.dtype == dtype
        assert result.shape == (
            small_batch["batch_size"],
            small_batch["target_len"],
            small_batch["action_dim"],
        )

    def test_bfloat16_recommended(self, medium_batch, sample_trajectory_data):
        """Test BF16 dtype (recommended for H100)."""
        import robocache

        source_data, source_times, target_times = sample_trajectory_data(
            **medium_batch, dtype=torch.bfloat16
        )

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.dtype == torch.bfloat16


@pytest.mark.gpu
class TestResampleEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_trajectory(self, gpu_device):
        """Test with batch size of 1."""
        import robocache

        source_data = torch.randn(1, 10, 8, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).unsqueeze(0)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).unsqueeze(0)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == (1, 5, 8)

    def test_same_length_resampling(self, gpu_device):
        """Test resampling to the same length."""
        import robocache

        length = 20
        source_data = torch.randn(4, length, 8, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, length, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, length, device=gpu_device).expand(4, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == source_data.shape

    def test_upsampling(self, gpu_device):
        """Test upsampling (target_len > source_len)."""
        import robocache

        source_data = torch.randn(4, 10, 8, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 50, device=gpu_device).expand(4, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == (4, 50, 8)

    def test_downsampling(self, gpu_device):
        """Test downsampling (target_len < source_len)."""
        import robocache

        source_data = torch.randn(4, 100, 8, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 100, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 20, device=gpu_device).expand(4, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == (4, 20, 8)

    def test_small_action_dim(self, gpu_device):
        """Test with small action dimension."""
        import robocache

        source_data = torch.randn(4, 10, 2, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).expand(4, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == (4, 5, 2)

    def test_large_action_dim(self, gpu_device):
        """Test with large action dimension."""
        import robocache

        source_data = torch.randn(4, 10, 128, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).expand(4, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        assert result.shape == (4, 5, 128)


@pytest.mark.gpu
class TestResampleErrors:
    """Test error handling and validation."""

    def test_cpu_tensor_error(self):
        """Test that CPU tensors raise an error."""
        import robocache

        source_data = torch.randn(4, 10, 8, dtype=torch.float32)  # CPU tensor
        source_times = torch.linspace(0, 1, 10).expand(4, -1)
        target_times = torch.linspace(0, 1, 5).expand(4, -1)

        with pytest.raises(RuntimeError, match="CUDA"):
            robocache.resample_trajectories(source_data, source_times, target_times)

    def test_shape_mismatch_error(self, gpu_device):
        """Test that shape mismatches raise errors."""
        import robocache

        source_data = torch.randn(4, 10, 8, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).expand(2, -1)  # Wrong batch

        with pytest.raises((RuntimeError, ValueError)):
            robocache.resample_trajectories(source_data, source_times, target_times)

    def test_wrong_dimension_error(self, gpu_device):
        """Test that wrong tensor dimensions raise errors."""
        import robocache

        # 2D tensor instead of 3D
        source_data = torch.randn(4, 10, dtype=torch.float32, device=gpu_device)
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(4, -1)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).expand(4, -1)

        with pytest.raises((RuntimeError, ValueError)):
            robocache.resample_trajectories(source_data, source_times, target_times)


@pytest.mark.gpu
class TestResampleCorrectness:
    """Test numerical correctness of resampling."""

    def test_linear_interpolation_correctness(self, gpu_device):
        """Test that linear interpolation is correct."""
        import robocache

        # Create simple linear trajectory
        batch_size = 2
        source_len = 11  # 0.0, 0.1, 0.2, ..., 1.0
        action_dim = 1

        # y = x (linear function)
        source_times = torch.linspace(0, 1, source_len, device=gpu_device).expand(
            batch_size, -1
        )
        source_data = source_times.unsqueeze(-1).expand(batch_size, source_len, action_dim)

        # Sample at midpoints
        target_times = torch.tensor(
            [[0.05, 0.15, 0.25], [0.05, 0.15, 0.25]], dtype=torch.float32, device=gpu_device
        )

        result = robocache.resample_trajectories(
            source_data.float(), source_times, target_times
        )

        # For y=x, interpolated values should equal the time points
        expected = target_times.unsqueeze(-1)

        # Allow small numerical error
        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-4)

    def test_constant_trajectory(self, gpu_device):
        """Test resampling of constant trajectory."""
        import robocache

        batch_size = 4
        constant_value = 3.14

        source_data = torch.full(
            (batch_size, 10, 8), constant_value, dtype=torch.float32, device=gpu_device
        )
        source_times = torch.linspace(0, 1, 10, device=gpu_device).expand(batch_size, -1)
        target_times = torch.linspace(0, 1, 5, device=gpu_device).expand(batch_size, -1)

        result = robocache.resample_trajectories(source_data, source_times, target_times)

        # Constant trajectory should remain constant
        expected = torch.full((batch_size, 5, 8), constant_value, device=gpu_device)
        assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.benchmark
class TestResamplePerformance:
    """Performance benchmarks for trajectory resampling."""

    @pytest.mark.slow
    def test_benchmark_medium_batch(self, medium_batch, sample_trajectory_data):
        """Benchmark with medium batch size."""
        import robocache
        import time

        source_data, source_times, target_times = sample_trajectory_data(
            **medium_batch, dtype=torch.bfloat16
        )

        # Warmup
        for _ in range(10):
            result = robocache.resample_trajectories(
                source_data, source_times, target_times
            )
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            result = robocache.resample_trajectories(
                source_data, source_times, target_times
            )
        torch.cuda.synchronize()
        end = time.time()

        avg_time_ms = (end - start) / num_iterations * 1000
        throughput = medium_batch["batch_size"] / (avg_time_ms / 1000)

        print(f"\nPerformance (BF16, batch={medium_batch['batch_size']}):")
        print(f"  Avg time: {avg_time_ms:.3f} ms")
        print(f"  Throughput: {throughput:.1f} trajectories/sec")

        # Sanity check - should be reasonably fast
        assert avg_time_ms < 10.0, "Performance regression detected"
