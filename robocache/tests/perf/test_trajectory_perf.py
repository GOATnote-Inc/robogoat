"""
Performance tests for trajectory resampling with regression gates.
"""

import pytest
import torch

from tests.perf.perf_guard import time_op, perf_guard


@pytest.fixture(scope="module")
def robocache_module():
    """Import robocache once per module."""
    try:
        import robocache
        return robocache
    except ImportError:
        pytest.skip("robocache not installed")


@pytest.fixture
def device():
    """Get CUDA device if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.mark.perf
class TestTrajectoryResamplePerf:
    """Performance tests for trajectory resampling."""
    
    def test_small_batch_perf(self, robocache_module, device):
        """Test small batch performance (8×250×128)."""
        batch_size, source_len, target_len, dim = 8, 250, 128, 128
        
        # Generate data
        source_data = torch.randn(batch_size, source_len, dim, device=device, dtype=torch.bfloat16)
        source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Benchmark
        fn = lambda: robocache_module.resample_trajectories(source_data, source_times, target_times)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 0.05ms P50, < 0.10ms P99
        perf_guard.require_lt_ms(
            "trajectory_resample_small",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=0.05,
            p99_max=0.10
        )
        
        perf_guard.record("trajectory_resample_small", stats)
    
    def test_medium_batch_perf(self, robocache_module, device):
        """Test medium batch performance (32×500×256)."""
        batch_size, source_len, target_len, dim = 32, 500, 256, 256
        
        source_data = torch.randn(batch_size, source_len, dim, device=device, dtype=torch.bfloat16)
        source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fn = lambda: robocache_module.resample_trajectories(source_data, source_times, target_times)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 0.20ms P50, < 0.40ms P99
        perf_guard.require_lt_ms(
            "trajectory_resample_medium",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=0.20,
            p99_max=0.40
        )
        
        perf_guard.record("trajectory_resample_medium", stats)
    
    def test_large_batch_perf(self, robocache_module, device):
        """Test large batch performance (64×1000×512)."""
        batch_size, source_len, target_len, dim = 64, 1000, 512, 512
        
        source_data = torch.randn(batch_size, source_len, dim, device=device, dtype=torch.bfloat16)
        source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fn = lambda: robocache_module.resample_trajectories(source_data, source_times, target_times)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 1.0ms P50, < 2.0ms P99
        perf_guard.require_lt_ms(
            "trajectory_resample_large",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=1.0,
            p99_max=2.0
        )
        
        perf_guard.record("trajectory_resample_large", stats)


@pytest.fixture(scope="session", autouse=True)
def save_perf_results(request):
    """Save performance results at end of session."""
    yield
    # After all tests
    from pathlib import Path
    output_dir = Path("bench/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_guard.save_results(output_dir / "pytest_perf_baseline.json")
    perf_guard.print_summary()

