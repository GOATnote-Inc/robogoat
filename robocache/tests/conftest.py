"""
Pytest configuration and fixtures for RoboCache tests.
"""
import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def cuda_extension():
    """
    Verify CUDA extension is loaded and functional.
    
    This fixture FAILS the entire test suite if CUDA kernels are not available
    when running CUDA tests. This prevents silent fallback to PyTorch.
    """
    import robocache
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this system")
    
    if not robocache._cuda_available:
        pytest.fail(
            "CUDA extension not loaded. Build failed or extension import error.\n"
            "Run: cd robocache && python setup.py develop"
        )
    
    # Smoke test: verify kernel actually executes
    try:
        src = torch.randn(2, 10, 8, dtype=torch.float32, device='cuda')
        src_times = torch.linspace(0, 1, 10, device='cuda').unsqueeze(0).expand(2, -1)
        tgt_times = torch.linspace(0, 1, 5, device='cuda').unsqueeze(0).expand(2, -1)
        
        result = robocache.resample_trajectories(src, src_times, tgt_times, backend="cuda")
        assert result.shape == (2, 5, 8), "CUDA kernel returned wrong shape"
        assert result.is_cuda, "CUDA kernel returned CPU tensor"
        
    except Exception as e:
        pytest.fail(f"CUDA kernel execution failed: {e}")
    
    return True


@pytest.fixture
def require_cuda(cuda_extension):
    """Mark test as requiring working CUDA extension"""
    return cuda_extension


@pytest.fixture
def gpu_device():
    """Return CUDA device if available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset CUDA memory between tests"""
    if torch.cuda.is_available():
        yield
        torch.cuda.empty_cache()
    else:
        yield

