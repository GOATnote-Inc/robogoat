"""
Comprehensive tests for RoboCache operations
Tests trajectory resampling, multimodal fusion, and voxelization
with both CUDA and PyTorch backends
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import robocache


class TestTrajectoryResampling:
    """Test trajectory resampling with various configurations"""
    
    def test_basic_resampling_cpu(self):
        """Test basic trajectory resampling on CPU"""
        B, S, T, D = 2, 10, 20, 8
        src = torch.randn(B, S, D, dtype=torch.float32)
        src_t = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1)
        tgt_t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
        
        result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
        
        assert result.shape == (B, T, D)
        assert result.dtype == torch.float32
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_basic_resampling_cuda(self):
        """Test basic trajectory resampling on CUDA"""
        B, S, T, D = 2, 10, 20, 8
        src = torch.randn(B, S, D, dtype=torch.bfloat16, device='cuda')
        src_t = torch.linspace(0, 1, S, device='cuda').unsqueeze(0).expand(B, -1)
        tgt_t = torch.linspace(0, 1, T, device='cuda').unsqueeze(0).expand(B, -1)
        
        result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
        
        assert result.shape == (B, T, D)
        assert result.device.type == 'cuda'
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_edge_case_single_source(self):
        """Test with single source timestep"""
        B, S, T, D = 1, 1, 5, 4
        src = torch.randn(B, S, D)
        src_t = torch.tensor([[0.0]])
        tgt_t = torch.linspace(0, 1, T).unsqueeze(0)
        
        result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
        
        # All outputs should match the single input
        for t in range(T):
            assert torch.allclose(result[0, t], src[0, 0], atol=1e-5)
    
    def test_edge_case_target_outside_range(self):
        """Test with target times outside source range"""
        B, S, T, D = 1, 5, 10, 4
        src = torch.randn(B, S, D)
        src_t = torch.linspace(0.2, 0.8, S).unsqueeze(0)
        tgt_t = torch.linspace(0, 1, T).unsqueeze(0)  # Outside [0.2, 0.8]
        
        result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
        
        # Targets before 0.2 should match first source
        assert torch.allclose(result[0, 0], src[0, 0], atol=1e-5)
        # Targets after 0.8 should match last source
        assert torch.allclose(result[0, -1], src[0, -1], atol=1e-5)


class TestMultimodalFusion:
    """Test multimodal sensor fusion"""
    
    def test_basic_fusion_cpu(self):
        """Test multimodal fusion on CPU"""
        B = 2
        vision = torch.randn(B, 10, 64, dtype=torch.float32)
        vision_t = torch.linspace(0, 1, 10).unsqueeze(0).expand(B, -1)
        proprio = torch.randn(B, 20, 16, dtype=torch.float32)
        proprio_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
        force = torch.randn(B, 15, 8, dtype=torch.float32)
        force_t = torch.linspace(0, 1, 15).unsqueeze(0).expand(B, -1)
        target_t = torch.linspace(0, 1, 12).unsqueeze(0).expand(B, -1)
        
        result = robocache.fuse_multimodal(
            vision, vision_t,
            proprio, proprio_t,
            force, force_t,
            target_t,
            backend='pytorch'
        )
        
        assert result.shape == (B, 12, 64 + 16 + 8)
        assert result.dtype == torch.float32
        assert not torch.isnan(result).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_basic_fusion_cuda(self):
        """Test multimodal fusion on CUDA"""
        B = 2
        vision = torch.randn(B, 10, 64, dtype=torch.bfloat16, device='cuda')
        vision_t = torch.linspace(0, 1, 10, device='cuda').unsqueeze(0).expand(B, -1)
        proprio = torch.randn(B, 20, 16, dtype=torch.bfloat16, device='cuda')
        proprio_t = torch.linspace(0, 1, 20, device='cuda').unsqueeze(0).expand(B, -1)
        force = torch.randn(B, 15, 8, dtype=torch.bfloat16, device='cuda')
        force_t = torch.linspace(0, 1, 15, device='cuda').unsqueeze(0).expand(B, -1)
        target_t = torch.linspace(0, 1, 12, device='cuda').unsqueeze(0).expand(B, -1)
        
        result = robocache.fuse_multimodal(
            vision, vision_t,
            proprio, proprio_t,
            force, force_t,
            target_t,
            backend='pytorch'
        )
        
        assert result.shape == (B, 12, 64 + 16 + 8)
        assert result.device.type == 'cuda'
        assert not torch.isnan(result).any()
    
    def test_dimension_concatenation(self):
        """Test that dimensions are correctly concatenated"""
        B = 1
        vision = torch.ones(B, 5, 10)
        vision_t = torch.linspace(0, 1, 5).unsqueeze(0)
        proprio = torch.ones(B, 5, 20) * 2
        proprio_t = torch.linspace(0, 1, 5).unsqueeze(0)
        force = torch.ones(B, 5, 30) * 3
        force_t = torch.linspace(0, 1, 5).unsqueeze(0)
        target_t = torch.linspace(0, 1, 5).unsqueeze(0)
        
        result = robocache.fuse_multimodal(
            vision, vision_t,
            proprio, proprio_t,
            force, force_t,
            target_t,
            backend='pytorch'
        )
        
        # Check concatenation order
        assert torch.allclose(result[0, 0, :10], torch.ones(10), atol=0.1)
        assert torch.allclose(result[0, 0, 10:30], torch.ones(20) * 2, atol=0.1)
        assert torch.allclose(result[0, 0, 30:], torch.ones(30) * 3, atol=0.1)


class TestVoxelization:
    """Test point cloud voxelization"""
    
    def test_basic_voxelization_cpu(self):
        """Test basic voxelization on CPU"""
        N = 100
        points = torch.randn(N, 3) * 10
        grid_size = (16, 16, 16)
        voxel_size = 1.0
        origin = torch.tensor([-8.0, -8.0, -8.0])
        
        result = robocache.voxelize_point_cloud(
            points, grid_size, voxel_size, origin, backend='pytorch'
        )
        
        assert result.shape == grid_size
        assert result.dtype == torch.float32
        # Check binary occupancy
        assert ((result == 0) | (result == 1)).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_basic_voxelization_cuda(self):
        """Test basic voxelization on CUDA"""
        N = 100
        points = torch.randn(N, 3, device='cuda') * 10
        grid_size = (16, 16, 16)
        voxel_size = 1.0
        origin = torch.tensor([-8.0, -8.0, -8.0], device='cuda')
        
        result = robocache.voxelize_point_cloud(
            points, grid_size, voxel_size, origin, backend='pytorch'
        )
        
        assert result.shape == grid_size
        assert result.device.type == 'cuda'
        assert ((result == 0) | (result == 1)).all()
    
    def test_voxelization_bounds(self):
        """Test that points outside bounds are ignored"""
        # All points outside grid
        N = 100
        points = torch.randn(N, 3) * 100  # Very far from origin
        grid_size = (8, 8, 8)
        voxel_size = 1.0
        origin = torch.tensor([0.0, 0.0, 0.0])
        
        result = robocache.voxelize_point_cloud(
            points, grid_size, voxel_size, origin, backend='pytorch'
        )
        
        # Most should be empty
        occupancy_rate = result.sum() / result.numel()
        assert occupancy_rate < 0.5  # Less than 50% occupied
    
    def test_voxelization_single_voxel(self):
        """Test multiple points in same voxel"""
        # 10 points in same location
        points = torch.zeros(10, 3)
        grid_size = (4, 4, 4)
        voxel_size = 1.0
        origin = torch.tensor([-2.0, -2.0, -2.0])
        
        result = robocache.voxelize_point_cloud(
            points, grid_size, voxel_size, origin, backend='pytorch'
        )
        
        # Should have exactly 1 occupied voxel
        assert result.sum() == 1


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_sequential_operations(self):
        """Test all 3 operations in sequence"""
        B = 2
        
        # 1. Trajectory resampling
        traj = torch.randn(B, 10, 16)
        traj_t = torch.linspace(0, 1, 10).unsqueeze(0).expand(B, -1)
        target_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
        
        resampled = robocache.resample_trajectories(traj, traj_t, target_t, backend='pytorch')
        assert resampled.shape == (B, 20, 16)
        
        # 2. Multimodal fusion
        vision = torch.randn(B, 15, 32)
        vision_t = torch.linspace(0, 1, 15).unsqueeze(0).expand(B, -1)
        proprio = torch.randn(B, 25, 8)
        proprio_t = torch.linspace(0, 1, 25).unsqueeze(0).expand(B, -1)
        force = torch.randn(B, 20, 4)
        force_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
        
        fused = robocache.fuse_multimodal(
            vision, vision_t, proprio, proprio_t, force, force_t, target_t, backend='pytorch'
        )
        assert fused.shape == (B, 20, 32 + 8 + 4)
        
        # 3. Voxelization
        points = torch.randn(50, 3) * 5
        grid = robocache.voxelize_point_cloud(
            points, (8, 8, 8), 1.0, torch.tensor([-4.0, -4.0, -4.0]), backend='pytorch'
        )
        assert grid.shape == (8, 8, 8)
        
        print("✅ All 3 operations executed successfully in sequence")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

