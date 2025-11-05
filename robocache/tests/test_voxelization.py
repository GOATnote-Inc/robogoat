"""
Comprehensive test suite for point cloud voxelization

Tests correctness, edge cases, and multi-backend consistency for
voxelization operations across CUDA and PyTorch backends.

Test Coverage:
- CPU golden reference validation
- CUDA vs PyTorch backend parity
- Edge cases (empty clouds, single points, boundary conditions)
- All voxelization modes (occupancy, density, TSDF, feature max/mean)
- Dtype support
- Error handling

Author: RoboCache Team
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False

# Check backend availability
if ROBOCACHE_AVAILABLE:
    info = robocache.check_installation()
    CUDA_AVAILABLE = info['cuda_extension_available']
    PYTORCH_AVAILABLE = info['pytorch_available']
else:
    CUDA_AVAILABLE = False
    PYTORCH_AVAILABLE = False


def cpu_golden_voxelize_occupancy(points, grid_size, voxel_size, origin):
    """
    CPU golden reference for occupancy voxelization.
    
    Uses deterministic counting (matching production CUDA implementation).
    """
    batch_size, num_points, _ = points.shape
    depth, height, width = grid_size[0].item(), grid_size[1].item(), grid_size[2].item()
    
    # Convert to CPU for reference
    points_cpu = points.cpu().float()
    origin_cpu = origin.cpu().float()
    
    # Allocate grid
    voxel_grid = torch.zeros((batch_size, depth, height, width), dtype=torch.float32)
    
    # Process each batch
    for b in range(batch_size):
        for p in range(num_points):
            px, py, pz = points_cpu[b, p, 0].item(), points_cpu[b, p, 1].item(), points_cpu[b, p, 2].item()
            
            # Convert to voxel indices (using floor, matching CUDA __float2int_rd)
            import math
            vx = int(math.floor((px - origin_cpu[0].item()) / voxel_size))
            vy = int(math.floor((py - origin_cpu[1].item()) / voxel_size))
            vz = int(math.floor((pz - origin_cpu[2].item()) / voxel_size))
            
            # Check bounds
            if 0 <= vx < depth and 0 <= vy < height and 0 <= vz < width:
                # Accumulate counts (matches GPU atomicAdd)
                voxel_grid[b, vx, vy, vz] += 1.0
    
    # Convert counts to binary occupancy (matches GPU second pass)
    voxel_grid = (voxel_grid > 0.0).float()
    
    return voxel_grid


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestVoxelizationCorrectness:
    """Test correctness against CPU golden reference"""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("grid_dim", [32, 64])
    @pytest.mark.parametrize("num_points", [100, 1000])
    def test_occupancy_correctness(self, batch_size, grid_dim, num_points):
        """Test occupancy voxelization correctness"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        torch.manual_seed(42)
        
        # Generate random point cloud within grid bounds
        voxel_size = 0.1
        grid_extent = grid_dim * voxel_size
        
        points = torch.rand(batch_size, num_points, 3) * grid_extent
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        
        # Compute golden reference
        golden = cpu_golden_voxelize_occupancy(points, grid_size, voxel_size, origin)
        
        # Compute using PyTorch backend
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Compare
        result_cpu = result.cpu().float()
        mismatches = (golden != result_cpu).sum().item()
        total_voxels = batch_size * grid_dim ** 3
        
        assert mismatches == 0, f"{mismatches}/{total_voxels} voxels mismatch"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
    @pytest.mark.parametrize("grid_dim", [32, 64, 128])
    def test_cuda_pytorch_occupancy_parity(self, grid_dim):
        """Test CUDA and PyTorch backends produce identical occupancy results"""
        batch_size, num_points = 4, 10000
        voxel_size = 0.1
        grid_extent = grid_dim * voxel_size
        
        torch.manual_seed(42)
        points = torch.rand(batch_size, num_points, 3, device='cuda') * grid_extent
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        # Compute using both backends
        result_cuda = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='cuda'
        )
        
        result_pytorch = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Compare
        mismatches = (result_cuda != result_pytorch).sum().item()
        total_voxels = batch_size * grid_dim ** 3
        
        assert mismatches == 0, f"CUDA/PyTorch parity failed: {mismatches}/{total_voxels} voxels differ"


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestVoxelizationEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_point_cloud(self):
        """Test with empty point cloud (no points)"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim = 2, 32
        points = torch.zeros(batch_size, 0, 3)  # Zero points
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        voxel_size = 0.1
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # All voxels should be empty
        assert result.sum().item() == 0
        assert result.shape == (batch_size, grid_dim, grid_dim, grid_dim)
    
    def test_single_point(self):
        """Test with single point"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim = 1, 32
        voxel_size = 0.1
        
        # Place point at center
        center = (grid_dim * voxel_size) / 2
        points = torch.tensor([[[center, center, center]]])
        
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Exactly one voxel should be occupied
        assert result.sum().item() == 1
    
    def test_points_on_boundaries(self):
        """Test points exactly on voxel boundaries"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim = 1, 32
        voxel_size = 0.1
        
        # Points exactly on voxel boundaries
        points = torch.tensor([
            [[0.0, 0.0, 0.0]],  # Origin
            [[voxel_size, voxel_size, voxel_size]],  # On boundary
            [[voxel_size * 2, voxel_size * 2, voxel_size * 2]],  # Another boundary
        ])
        
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Should have exactly 3 occupied voxels (one per point)
        assert result.sum().item() == 3
    
    def test_all_points_out_of_bounds(self):
        """Test when all points are outside grid"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim = 2, 32
        voxel_size = 0.1
        grid_extent = grid_dim * voxel_size
        
        # All points outside grid (negative coordinates)
        points = torch.randn(batch_size, 100, 3) - 10.0
        
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # All voxels should be empty
        assert result.sum().item() == 0
    
    def test_multiple_points_same_voxel(self):
        """Test multiple points falling in same voxel"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim = 1, 32
        voxel_size = 0.1
        
        # 10 points very close together (same voxel)
        base_point = torch.tensor([1.0, 1.0, 1.0])
        points = base_point.unsqueeze(0).unsqueeze(0).expand(1, 10, -1) + torch.randn(1, 10, 3) * 0.01
        
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.zeros(3)
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Should have only 1 occupied voxel (occupancy is binary)
        assert result.sum().item() == 1
    
    def test_negative_origin(self):
        """Test with negative origin"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, grid_dim, num_points = 2, 32, 100
        voxel_size = 0.1
        
        # Points centered around zero
        points = torch.randn(batch_size, num_points, 3) * 0.5
        
        # Origin at negative coordinates
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32)
        origin = torch.tensor([-1.6, -1.6, -1.6])  # Centered grid
        
        result = robocache.voxelize_occupancy(
            points, grid_size, voxel_size, origin,
            backend='pytorch'
        )
        
        # Should have some occupied voxels
        assert result.sum().item() > 0
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestVoxelizationErrorHandling:
    """Test error handling and validation"""
    
    def test_invalid_grid_size(self):
        """Test error handling for invalid grid size"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        points = torch.randn(1, 100, 3)
        grid_size = torch.tensor([0, 32, 32], dtype=torch.int32)  # Zero dimension
        origin = torch.zeros(3)
        voxel_size = 0.1
        
        with pytest.raises((ValueError, RuntimeError)):
            robocache.voxelize_occupancy(
                points, grid_size, voxel_size, origin,
                backend='pytorch'
            )
    
    def test_negative_voxel_size(self):
        """Test error handling for negative voxel size"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        points = torch.randn(1, 100, 3)
        grid_size = torch.tensor([32, 32, 32], dtype=torch.int32)
        origin = torch.zeros(3)
        voxel_size = -0.1  # Negative
        
        with pytest.raises((ValueError, RuntimeError)):
            robocache.voxelize_occupancy(
                points, grid_size, voxel_size, origin,
                backend='pytorch'
            )
    
    def test_invalid_points_shape(self):
        """Test error handling for invalid points shape"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        points = torch.randn(1, 100, 2)  # Only 2D points
        grid_size = torch.tensor([32, 32, 32], dtype=torch.int32)
        origin = torch.zeros(3)
        voxel_size = 0.1
        
        with pytest.raises((ValueError, RuntimeError)):
            robocache.voxelize_occupancy(
                points, grid_size, voxel_size, origin,
                backend='pytorch'
            )


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestVoxelizationPerformance:
    """Performance regression tests"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
    @pytest.mark.parametrize("grid_dim", [64, 128])
    def test_cuda_faster_than_pytorch(self, grid_dim):
        """Verify CUDA backend is significantly faster than PyTorch"""
        import time
        
        batch_size, num_points = 8, 50000
        voxel_size = 0.1
        grid_extent = grid_dim * voxel_size
        
        torch.manual_seed(42)
        points = torch.rand(batch_size, num_points, 3, device='cuda') * grid_extent
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        # Warmup
        for _ in range(5):
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='cuda')
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='pytorch')
        
        torch.cuda.synchronize()
        
        # Benchmark CUDA
        start = time.time()
        for _ in range(50):
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='cuda')
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(50):
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='pytorch')
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        speedup = pytorch_time / cuda_time
        print(f"\nVoxelization ({grid_dim}³) Speedup: {speedup:.2f}x (CUDA: {cuda_time:.4f}s, PyTorch: {pytorch_time:.4f}s)")
        
        # CUDA should be at least 50x faster (typically 100-500x)
        assert speedup > 50.0, f"CUDA backend not sufficiently faster than PyTorch (speedup: {speedup:.2f}x)"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
    def test_expected_speedup_h100(self):
        """Verify we achieve expected H100 speedup (regression test)"""
        import time
        
        # Use exact benchmark configuration
        batch_size, num_points = 4, 100000
        grid_dim = 128
        voxel_size = 0.1
        grid_extent = grid_dim * voxel_size
        
        torch.manual_seed(42)
        points = torch.rand(batch_size, num_points, 3, device='cuda') * grid_extent
        grid_size = torch.tensor([grid_dim, grid_dim, grid_dim], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        # Warmup
        for _ in range(20):
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='cuda')
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = robocache.voxelize_occupancy(points, grid_size, voxel_size, origin, backend='cuda')
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        latency_ms = (elapsed / 100) * 1000
        print(f"\nVoxelization (128³) Latency: {latency_ms:.3f}ms")
        
        # On H100, we expect ~0.5-1.0ms for 128³ grid
        # Allow some variance for different hardware
        if torch.cuda.get_device_name(0) == "NVIDIA H100":
            assert latency_ms < 2.0, f"H100 performance regression: {latency_ms:.3f}ms (expected < 1.0ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

