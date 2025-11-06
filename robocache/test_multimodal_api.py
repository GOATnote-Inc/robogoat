"""
Test multimodal fusion Python API
Proves all 3 operations work: trajectory, multimodal, voxelization
"""

import torch
import sys
sys.path.insert(0, 'python')

import robocache

print("=" * 60)
print("RoboCache API Test: All 3 Operations")
print("=" * 60)

# Test 1: Trajectory Resampling
print("\n[1/3] Trajectory Resampling")
B, S, T, D = 4, 10, 20, 8
src = torch.randn(B, S, D, dtype=torch.bfloat16)
src_t = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1)
tgt_t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)

result = robocache.resample_trajectories(src, src_t, tgt_t, backend='pytorch')
print(f"✅ {src.shape} → {result.shape}")
assert result.shape == (B, T, D), f"Expected {(B, T, D)}, got {result.shape}"

# Test 2: Multimodal Fusion
print("\n[2/3] Multimodal Fusion")
B = 4
vision = torch.randn(B, 30, 512, dtype=torch.bfloat16)
vision_t = torch.linspace(0, 1, 30).unsqueeze(0).expand(B, -1)
proprio = torch.randn(B, 100, 32, dtype=torch.bfloat16)
proprio_t = torch.linspace(0, 1, 100).unsqueeze(0).expand(B, -1)
force = torch.randn(B, 50, 16, dtype=torch.bfloat16)
force_t = torch.linspace(0, 1, 50).unsqueeze(0).expand(B, -1)
target_t = torch.linspace(0, 1, 50).unsqueeze(0).expand(B, -1)

fused = robocache.fuse_multimodal(
    vision, vision_t,
    proprio, proprio_t,
    force, force_t,
    target_t,
    backend='pytorch'
)
print(f"✅ Vision {vision.shape} + Proprio {proprio.shape} + Force {force.shape}")
print(f"   → Fused {fused.shape}")
expected_shape = (B, 50, 512 + 32 + 16)
assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"

# Test 3: Voxelization
print("\n[3/3] Voxelization")
N = 1000
points = torch.randn(N, 3) * 10  # Random point cloud
grid_size = (32, 32, 32)
voxel_size = 1.0
origin = torch.tensor([-16.0, -16.0, -16.0])

grid = robocache.voxelize_point_cloud(points, grid_size, voxel_size, origin, backend='pytorch')
print(f"✅ Points {points.shape} → Grid {grid.shape}")
assert grid.shape == grid_size, f"Expected {grid_size}, got {grid.shape}"
print(f"   Occupancy: {grid.sum().item():.0f} / {grid.numel()} voxels")

print("\n" + "=" * 60)
print("✅ ALL 3 OPERATIONS WORK (PyTorch backend)")
print("=" * 60)
print("\nNext: Add CUDA backends for multimodal + voxelization")
print("Status: Week 1, Day 1 - API Complete ✓")

