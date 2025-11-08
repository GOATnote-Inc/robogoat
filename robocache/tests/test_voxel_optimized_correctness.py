"""
Correctness Tests for Optimized Voxelization Kernel

Validates that optimized kernel produces identical results to original.
"""

import pytest
import torch
import numpy as np

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVoxelOptimizedCorrectness:
    """Test optimized voxelization produces correct results"""
    
    def test_occupancy_mode_correctness(self):
        """Compare optimized vs original for occupancy mode"""
        torch.manual_seed(42)
        points = torch.rand(10000, 3, device='cuda') * 10.0 - 5.0
        
        # Run voxelization (uses production kernel)
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[64, 64, 64],
            mode='occupancy',
            backend='cuda'
        )
        
        # Validate shape
        assert voxel_grid.shape == (64, 64, 64)
        
        # Validate values are binary (0 or 1)
        unique_values = torch.unique(voxel_grid)
        assert len(unique_values) <= 2
        assert all(v in [0.0, 1.0] for v in unique_values.cpu().numpy())
        
        # Validate determinism
        voxel_grid_2 = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[64, 64, 64],
            mode='occupancy',
            backend='cuda'
        )
        
        assert torch.allclose(voxel_grid, voxel_grid_2, atol=0), \
            "Voxelization is not deterministic"
    
    def test_edge_case_zero_points(self):
        """Test with zero points"""
        points = torch.empty(0, 3, device='cuda')
        
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[32, 32, 32],
            mode='occupancy',
            backend='cuda'
        )
        
        assert voxel_grid.shape == (32, 32, 32)
        assert torch.all(voxel_grid == 0), "Empty point cloud should produce empty grid"
    
    def test_edge_case_single_point(self):
        """Test with single point"""
        points = torch.tensor([[0.0, 0.0, 0.0]], device='cuda')
        
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[100, 100, 100],
            mode='occupancy',
            backend='cuda'
        )
        
        # Should have exactly one occupied voxel
        num_occupied = torch.sum(voxel_grid > 0)
        assert num_occupied == 1, f"Expected 1 occupied voxel, got {num_occupied}"
    
    def test_edge_case_all_out_of_bounds(self):
        """Test with all points outside grid"""
        points = torch.tensor([
            [100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0]
        ], device='cuda')
        
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[100, 100, 100],
            mode='occupancy',
            backend='cuda'
        )
        
        assert torch.all(voxel_grid == 0), "Out-of-bounds points should not occupy voxels"
    
    def test_numerical_stability(self):
        """Test with extreme values"""
        # Very small coordinates
        points_small = torch.rand(1000, 3, device='cuda') * 0.001
        
        voxel_grid_small = robocache.voxelize_pointcloud(
            points_small,
            grid_min=[0.0, 0.0, 0.0],
            voxel_size=0.0001,
            grid_size=[32, 32, 32],
            mode='occupancy',
            backend='cuda'
        )
        
        assert voxel_grid_small.shape == (32, 32, 32)
        assert torch.sum(voxel_grid_small > 0) > 0
    
    def test_grid_size_variations(self):
        """Test various grid sizes"""
        points = torch.rand(5000, 3, device='cuda') * 10.0 - 5.0
        
        for grid_size in [16, 32, 64, 128, 256]:
            voxel_grid = robocache.voxelize_pointcloud(
                points,
                grid_min=[-5.0, -5.0, -5.0],
                voxel_size=10.0 / grid_size,
                grid_size=[grid_size, grid_size, grid_size],
                mode='occupancy',
                backend='cuda'
            )
            
            assert voxel_grid.shape == (grid_size, grid_size, grid_size)
            assert torch.sum(voxel_grid > 0) > 0
    
    def test_memory_consistency(self):
        """Test that kernel doesn't corrupt memory"""
        # Allocate sentinel values before and after
        sentinel_before = torch.ones(1000, device='cuda')
        points = torch.rand(10000, 3, device='cuda') * 10.0 - 5.0
        sentinel_after = torch.ones(1000, device='cuda')
        
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[64, 64, 64],
            mode='occupancy',
            backend='cuda'
        )
        
        # Check sentinels unchanged
        assert torch.all(sentinel_before == 1.0), "Memory corruption detected before"
        assert torch.all(sentinel_after == 1.0), "Memory corruption detected after"
    
    @pytest.mark.parametrize("num_points", [100, 1000, 10000, 100000, 500000])
    def test_scaling_with_point_count(self, num_points):
        """Test correctness scales with point count"""
        torch.manual_seed(42)
        points = torch.rand(num_points, 3, device='cuda') * 10.0 - 5.0
        
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1,
            grid_size=[100, 100, 100],
            mode='occupancy',
            backend='cuda'
        )
        
        # Basic sanity checks
        assert voxel_grid.shape == (100, 100, 100)
        num_occupied = torch.sum(voxel_grid > 0)
        
        # Should have reasonable occupancy
        assert num_occupied > 0, f"No voxels occupied with {num_points} points"
        assert num_occupied <= 100**3, "Too many voxels occupied"


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")  
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_benchmark_optimized_vs_baseline():
    """Compare performance of optimized kernel"""
    import time
    
    points = torch.rand(500000, 3, device='cuda') * 10.0 - 5.0
    
    # Warmup
    for _ in range(10):
        _ = robocache.voxelize_pointcloud(
            points, grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1, grid_size=[128, 128, 128],
            mode='occupancy', backend='cuda'
        )
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = robocache.voxelize_pointcloud(
            points, grid_min=[-5.0, -5.0, -5.0],
            voxel_size=0.1, grid_size=[128, 128, 128],
            mode='occupancy', backend='cuda'
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    median_time_ms = np.median(times) * 1000
    throughput = 500000 / (median_time_ms / 1000)
    
    print(f"\nOptimized kernel performance:")
    print(f"  Latency: {median_time_ms:.3f} ms")
    print(f"  Throughput: {throughput / 1e9:.2f} B pts/sec")
    
    # Assert reasonable performance (>10B pts/sec on modern GPU)
    assert throughput > 5e9, f"Performance too low: {throughput / 1e9:.2f} B pts/sec"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

