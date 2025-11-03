"""
Pytest configuration and fixtures for RoboCache tests.
"""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_device():
    """Get the default CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def small_batch():
    """Fixture for small batch testing."""
    return {
        "batch_size": 4,
        "source_len": 10,
        "target_len": 5,
        "action_dim": 8,
    }


@pytest.fixture
def medium_batch():
    """Fixture for medium batch testing."""
    return {
        "batch_size": 64,
        "source_len": 100,
        "target_len": 50,
        "action_dim": 32,
    }


@pytest.fixture
def large_batch():
    """Fixture for large batch testing."""
    return {
        "batch_size": 256,
        "source_len": 200,
        "target_len": 100,
        "action_dim": 64,
    }


@pytest.fixture
def sample_trajectory_data(gpu_device):
    """Generate sample trajectory data for testing."""

    def _generate(batch_size, source_len, target_len, action_dim, dtype=torch.float32):
        """Generate realistic trajectory data."""
        # Source data with realistic robot trajectories
        source_data = torch.randn(
            batch_size, source_len, action_dim, dtype=dtype, device=gpu_device
        )

        # Source times (monotonically increasing)
        source_times = torch.linspace(
            0, 1, source_len, dtype=torch.float32, device=gpu_device
        ).expand(batch_size, -1)

        # Target times
        target_times = torch.linspace(
            0, 1, target_len, dtype=torch.float32, device=gpu_device
        ).expand(batch_size, -1)

        return source_data, source_times, target_times

    return _generate


@pytest.fixture(autouse=True)
def reset_cuda():
    """Reset CUDA state between tests."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
