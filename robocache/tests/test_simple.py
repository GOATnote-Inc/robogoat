"""
Simple validation tests for RoboCache operations
Runs without pytest dependency
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import robocache

def test_trajectory_cpu():
    """Test trajectory resampling on CPU"""
    B, S, T, D = 2, 10, 20, 8
    src = torch.randn(B, S, D)
    src_t = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1)
    tgt_t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    
    result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
    
    assert result.shape == (B, T, D), f"Shape mismatch: {result.shape}"
    assert not torch.isnan(result).any(), "NaN detected"
    print("✅ Trajectory resampling (CPU)")

def test_multimodal_cpu():
    """Test multimodal fusion on CPU"""
    B = 2
    vision = torch.randn(B, 10, 64)
    vision_t = torch.linspace(0, 1, 10).unsqueeze(0).expand(B, -1)
    proprio = torch.randn(B, 20, 16)
    proprio_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
    force = torch.randn(B, 15, 8)
    force_t = torch.linspace(0, 1, 15).unsqueeze(0).expand(B, -1)
    target_t = torch.linspace(0, 1, 12).unsqueeze(0).expand(B, -1)
    
    result = robocache.fuse_multimodal(
        vision, vision_t, proprio, proprio_t, force, force_t, target_t, backend='pytorch'
    )
    
    assert result.shape == (B, 12, 64 + 16 + 8), f"Shape mismatch: {result.shape}"
    assert not torch.isnan(result).any(), "NaN detected"
    print("✅ Multimodal fusion (CPU)")

def test_voxelization_cpu():
    """Test voxelization on CPU"""
    N = 100
    points = torch.randn(N, 3) * 10
    grid_size = (16, 16, 16)
    voxel_size = 1.0
    origin = torch.tensor([-8.0, -8.0, -8.0])
    
    result = robocache.voxelize_point_cloud(
        points, grid_size, voxel_size, origin, backend='pytorch'
    )
    
    assert result.shape == grid_size, f"Shape mismatch: {result.shape}"
    assert ((result == 0) | (result == 1)).all(), "Non-binary values"
    print("✅ Voxelization (CPU)")

def test_integration():
    """Test all 3 operations in sequence"""
    B = 2
    
    # Trajectory
    traj = torch.randn(B, 10, 16)
    traj_t = torch.linspace(0, 1, 10).unsqueeze(0).expand(B, -1)
    target_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
    resampled = robocache.resample_trajectories(traj, traj_t, target_t, backend='pytorch')
    
    # Multimodal
    vision = torch.randn(B, 15, 32)
    vision_t = torch.linspace(0, 1, 15).unsqueeze(0).expand(B, -1)
    proprio = torch.randn(B, 25, 8)
    proprio_t = torch.linspace(0, 1, 25).unsqueeze(0).expand(B, -1)
    force = torch.randn(B, 20, 4)
    force_t = torch.linspace(0, 1, 20).unsqueeze(0).expand(B, -1)
    fused = robocache.fuse_multimodal(
        vision, vision_t, proprio, proprio_t, force, force_t, target_t, backend='pytorch'
    )
    
    # Voxelization
    points = torch.randn(50, 3) * 5
    grid = robocache.voxelize_point_cloud(
        points, (8, 8, 8), 1.0, torch.tensor([-4.0, -4.0, -4.0]), backend='pytorch'
    )
    
    print("✅ Integration test (all 3 operations)")

if __name__ == '__main__':
    print("=" * 60)
    print("RoboCache Test Suite")
    print("=" * 60)
    
    try:
        test_trajectory_cpu()
        test_multimodal_cpu()
        test_voxelization_cpu()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

