"""
Correctness tests for point cloud voxelization.

Tests conversion of 3D point clouds to structured voxel grids.
"""

import pytest
import torch


@pytest.fixture
def robocache_module():
    """Import robocache once per module."""
    try:
        import robocache
        return robocache
    except ImportError:
        pytest.skip("robocache not installed")


@pytest.fixture
def device():
    """Get CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestVoxelizeCorrectness:
    """Correctness tests for point cloud voxelization."""
    
    def test_single_point(self, robocache_module, device):
        """Test voxelization of a single point."""
        # Point at origin
        points = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=torch.float32)
        grid_size = (10, 10, 10)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # Should activate voxel at (5, 5, 5)
        assert grid.shape == grid_size
        assert grid[5, 5, 5] == 1.0
        assert grid.sum() == 1.0, "Only one voxel should be occupied"
    
    def test_multiple_points_same_voxel(self, robocache_module, device):
        """Test multiple points in the same voxel."""
        # Multiple points in same voxel
        points = torch.tensor([
            [0.51, 0.51, 0.51],
            [0.52, 0.52, 0.52],
            [0.53, 0.53, 0.53],
        ], device=device, dtype=torch.float32)
        
        grid_size = (10, 10, 10)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # All points map to voxel (5, 5, 5)
        assert grid[5, 5, 5] == 1.0
        assert grid.sum() == 1.0, "Should still be one voxel (occupancy)"
    
    def test_grid_boundaries(self, robocache_module, device):
        """Test points at grid boundaries."""
        grid_size = (5, 5, 5)
        voxel_size = 1.0
        origin = torch.zeros(3, device=device)
        
        # Points at corners
        points = torch.tensor([
            [0.0, 0.0, 0.0],  # Min corner → (0, 0, 0)
            [4.9, 4.9, 4.9],  # Max corner → (4, 4, 4)
        ], device=device, dtype=torch.float32)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        assert grid[0, 0, 0] == 1.0
        assert grid[4, 4, 4] == 1.0
        assert grid.sum() == 2.0
    
    def test_out_of_bounds_points(self, robocache_module, device):
        """Test points outside grid bounds are ignored."""
        grid_size = (10, 10, 10)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        # Points: some in bounds, some out
        points = torch.tensor([
            [0.5, 0.5, 0.5],  # In bounds
            [-1.0, 0.5, 0.5],  # Out (negative x)
            [2.0, 0.5, 0.5],   # Out (x > grid)
            [0.5, -1.0, 0.5],  # Out (negative y)
            [0.5, 2.0, 0.5],   # Out (y > grid)
            [0.5, 0.5, -1.0],  # Out (negative z)
            [0.5, 0.5, 2.0],   # Out (z > grid)
        ], device=device, dtype=torch.float32)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # Only the first point should be counted
        assert grid.sum() == 1.0, "Out-of-bounds points should be ignored"
        assert grid[5, 5, 5] == 1.0
    
    def test_custom_origin(self, robocache_module, device):
        """Test voxelization with non-zero origin."""
        grid_size = (10, 10, 10)
        voxel_size = 0.1
        origin = torch.tensor([1.0, 1.0, 1.0], device=device)
        
        # Point at (1.5, 1.5, 1.5) → offset (0.5, 0.5, 0.5) → voxel (5, 5, 5)
        points = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        assert grid[5, 5, 5] == 1.0
        assert grid.sum() == 1.0
    
    def test_large_point_cloud(self, robocache_module, device):
        """Test voxelization of a large point cloud."""
        # Generate 10K random points in a cube
        torch.manual_seed(42)
        n_points = 10000
        points = torch.rand(n_points, 3, device=device) * 5.0  # Points in [0, 5]^3
        
        grid_size = (64, 64, 64)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # Sanity checks
        assert grid.shape == grid_size
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0
        assert grid.sum() > 0, "Some voxels should be occupied"
        assert grid.sum() <= n_points, "Can't have more occupied voxels than points"
        
        # Most voxels should be occupied (dense random cloud)
        occupied_ratio = grid.sum() / n_points
        assert occupied_ratio > 0.3, f"Expected >30% occupancy, got {occupied_ratio*100:.1f}%"
    
    @pytest.mark.parametrize("grid_size", [(32, 32, 32), (64, 64, 64), (128, 128, 128)])
    @pytest.mark.parametrize("voxel_size", [0.05, 0.1, 0.2])
    def test_parametric_configs(self, robocache_module, device, grid_size, voxel_size):
        """Test voxelization across various grid sizes and voxel sizes."""
        torch.manual_seed(123)
        
        # Generate points within grid bounds
        max_extent = grid_size[0] * voxel_size
        points = torch.rand(1000, 3, device=device) * max_extent * 0.9  # Stay within bounds
        
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        assert grid.shape == grid_size
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0
        assert not torch.isnan(grid).any()
        assert not torch.isinf(grid).any()
    
    def test_deterministic_voxelization(self, robocache_module, device):
        """Test that voxelization is deterministic (same input → same output)."""
        torch.manual_seed(456)
        points = torch.rand(500, 3, device=device) * 2.0
        
        grid_size = (32, 32, 32)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        # Run twice
        grid1 = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        grid2 = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # Should be identical
        assert torch.equal(grid1, grid2), "Voxelization should be deterministic"
    
    def test_sparse_point_cloud(self, robocache_module, device):
        """Test voxelization of a very sparse point cloud."""
        # Only 5 points in a large grid
        points = torch.tensor([
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [1.9, 1.9, 1.9],
        ], device=device, dtype=torch.float32)
        
        grid_size = (128, 128, 128)
        voxel_size = 0.05
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # Should have exactly 5 occupied voxels
        assert grid.sum() == 5.0, "Sparse cloud should have 5 occupied voxels"
        
        # Grid should be mostly empty
        occupancy_ratio = grid.sum() / (128 * 128 * 128)
        assert occupancy_ratio < 0.001, "Grid should be very sparse"
    
    def test_dense_point_cloud(self, robocache_module, device):
        """Test voxelization of a dense point cloud (multiple points per voxel)."""
        torch.manual_seed(789)
        
        # Generate 50K points in a small region (many collisions expected)
        n_points = 50000
        points = torch.rand(n_points, 3, device=device) * 1.0  # All in [0, 1]^3
        
        grid_size = (32, 32, 32)  # Coarse grid
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        occupied_voxels = grid.sum().item()
        
        # Many points should map to same voxels
        assert occupied_voxels < n_points, "Collisions should occur"
        
        # But most voxels in the region should be filled
        # 10^3 cube = 10 voxels per side, so ~1000 voxels in region
        assert occupied_voxels > 500, f"Expected >500 occupied voxels, got {occupied_voxels}"
    
    def test_axis_aligned_line(self, robocache_module, device):
        """Test voxelization of points along an axis (1D pattern)."""
        # Points along X axis
        n_points = 50
        points = torch.zeros(n_points, 3, device=device)
        points[:, 0] = torch.linspace(0, 4.9, n_points)
        points[:, 1] = 0.5
        points[:, 2] = 0.5
        
        grid_size = (50, 10, 10)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # All points should map to y=5, z=5
        # Check that occupied voxels are along the line
        occupied_mask = grid > 0.5
        
        # Sum over X axis: should have occupancy
        occupancy_yz = occupied_mask.sum(dim=0)
        
        # Most occupancy should be at (5, 5) in YZ plane
        assert occupancy_yz[5, 5] > 0, "Line should pass through (5, 5) in YZ"
        
        # Total occupied voxels should be close to n_points (one per point)
        assert grid.sum() >= n_points * 0.8, "Most points should occupy unique voxels"
    
    def test_dtype_consistency(self, robocache_module, device):
        """Test that output dtype is FP32."""
        points = torch.randn(100, 3, device=device, dtype=torch.float32)
        
        grid_size = (32, 32, 32)
        voxel_size = 0.1
        origin = torch.zeros(3, device=device)
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        assert grid.dtype == torch.float32, f"Expected FP32, got {grid.dtype}"
    
    def test_negative_coordinates(self, robocache_module, device):
        """Test handling of negative coordinates with appropriate origin."""
        # Points in negative space
        points = torch.tensor([
            [-0.5, -0.5, -0.5],
            [-0.1, -0.1, -0.1],
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
        ], device=device, dtype=torch.float32)
        
        # Origin at -1.0 to include negative points
        origin = torch.tensor([-1.0, -1.0, -1.0], device=device)
        grid_size = (20, 20, 20)
        voxel_size = 0.1
        
        grid = robocache_module.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        
        # All 4 points should be within bounds and occupy voxels
        assert grid.sum() == 4.0, "All points should be voxelized"

