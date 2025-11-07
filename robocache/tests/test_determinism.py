"""Determinism tests - verify reproducible outputs under fixed seeds"""

import pytest
import torch
import numpy as np


@pytest.fixture
def seed():
    """Fixed seed for reproducibility"""
    return 42


def set_seed(seed):
    """Set all random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multimodal_fusion_deterministic(seed):
    """Multimodal fusion produces identical results with same seed"""
    import robocache
    
    batch = 4
    
    # Run 1
    set_seed(seed)
    vision1 = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
    vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
    proprio1 = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
    imu1 = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
    imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)
    
    out1 = robocache.fuse_multimodal(
        vision1, vision_times, proprio1, proprio_times, imu1, imu_times, target
    )
    
    # Run 2 with same seed
    set_seed(seed)
    vision2 = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
    proprio2 = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
    imu2 = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
    
    out2 = robocache.fuse_multimodal(
        vision2, vision_times, proprio2, proprio_times, imu2, imu_times, target
    )
    
    # Should be identical
    assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic multimodal fusion"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_voxelization_deterministic(seed):
    """Voxelization produces identical results with same seed"""
    import robocache
    
    # Run 1
    set_seed(seed)
    points1 = torch.rand(100000, 3, device='cuda') * 4.0 - 2.0
    grid1 = robocache.voxelize_pointcloud(
        points1, grid_min=[-2, -2, -2], voxel_size=0.0625, 
        grid_size=[128, 128, 128], mode='count'
    )
    
    # Run 2 with same seed
    set_seed(seed)
    points2 = torch.rand(100000, 3, device='cuda') * 4.0 - 2.0
    grid2 = robocache.voxelize_pointcloud(
        points2, grid_min=[-2, -2, -2], voxel_size=0.0625,
        grid_size=[128, 128, 128], mode='count'
    )
    
    # Should be identical
    assert torch.equal(grid1, grid2), "Non-deterministic voxelization"

