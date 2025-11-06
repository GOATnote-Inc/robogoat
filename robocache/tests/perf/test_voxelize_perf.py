"""
Performance tests for point cloud voxelization with regression gates.
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
class TestVoxelizePerf:
    """Performance tests for point cloud voxelization."""
    
    def test_small_cloud_perf(self, robocache_module, device):
        """Test voxelization of small point cloud (10K points, 32³ grid)."""
        torch.manual_seed(42)
        n_points = 10000
        points = torch.rand(n_points, 3, device=device) * 3.0
        
        grid_size = (32, 32, 32)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        # Benchmark
        fn = lambda: robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 0.50ms P50, < 1.0ms P99
        perf_guard.require_lt_ms(
            "voxelize_small_10k_32cubed",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=0.50,
            p99_max=1.0
        )
        
        perf_guard.record("voxelize_small_10k_32cubed", stats)
    
    def test_medium_cloud_perf(self, robocache_module, device):
        """Test voxelization of medium point cloud (100K points, 64³ grid)."""
        torch.manual_seed(123)
        n_points = 100000
        points = torch.rand(n_points, 3, device=device) * 6.0
        
        grid_size = (64, 64, 64)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        fn = lambda: robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 5.0ms P50, < 10.0ms P99
        perf_guard.require_lt_ms(
            "voxelize_medium_100k_64cubed",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=5.0,
            p99_max=10.0
        )
        
        perf_guard.record("voxelize_medium_100k_64cubed", stats)
    
    def test_large_cloud_perf(self, robocache_module, device):
        """Test voxelization of large point cloud (1M points, 128³ grid)."""
        torch.manual_seed(456)
        n_points = 1000000
        points = torch.rand(n_points, 3, device=device) * 12.0
        
        grid_size = (128, 128, 128)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        fn = lambda: robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        stats = time_op(fn, warmup=5, iters=50, sync_fn=torch.cuda.synchronize)  # Fewer iters for large data
        
        # Performance gate: < 50ms P50, < 100ms P99
        perf_guard.require_lt_ms(
            "voxelize_large_1m_128cubed",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=50.0,
            p99_max=100.0
        )
        
        perf_guard.record("voxelize_large_1m_128cubed", stats)
    
    def test_high_resolution_grid_perf(self, robocache_module, device):
        """Test voxelization with high-resolution grid (256³)."""
        torch.manual_seed(789)
        n_points = 500000
        points = torch.rand(n_points, 3, device=device) * 25.0
        
        grid_size = (256, 256, 256)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        fn = lambda: robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        stats = time_op(fn, warmup=3, iters=20, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 100ms P50, < 200ms P99 (large grid is expensive)
        perf_guard.require_lt_ms(
            "voxelize_high_res_256cubed",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=100.0,
            p99_max=200.0
        )
        
        perf_guard.record("voxelize_high_res_256cubed", stats)
    
    def test_throughput_points_per_second(self, robocache_module, device):
        """Test voxelization throughput (points/sec)."""
        torch.manual_seed(999)
        n_points = 100000
        points = torch.rand(n_points, 3, device=device) * 6.0
        
        grid_size = (64, 64, 64)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        fn = lambda: robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Calculate throughput
        time_per_call_sec = stats.p50 / 1000.0  # Convert ms to seconds
        points_per_sec = n_points / time_per_call_sec
        
        print(f"\nVoxelization throughput: {points_per_sec/1e6:.2f}M points/sec")
        
        # Gate: Should process > 10M points/sec
        assert points_per_sec > 10e6, f"Throughput too low: {points_per_sec/1e6:.2f}M pts/sec"
        
        perf_guard.record("voxelize_throughput", stats)

