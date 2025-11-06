# Copyright (c) 2025 GOATnote Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Correctness tests for point cloud voxelization.
Compares GPU implementation against CPU reference.
"""

import pytest
import torch
import numpy as np


def cpu_reference_voxelize(points, grid_min, voxel_size, grid_size, mode='occupancy'):
    """CPU reference implementation"""
    N = points.shape[0]
    grid_x, grid_y, grid_z = grid_size
    
    # Initialize grid
    if mode in ['count', 'occupancy']:
        grid = torch.zeros(grid_x, grid_y, grid_z, dtype=torch.int32)
    else:
        grid = torch.zeros(grid_x, grid_y, grid_z, dtype=torch.float32)
    
    # Voxelize each point
    for i in range(N):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        
        # Convert to voxel coordinates
        vx = int((px - grid_min[0]) / voxel_size)
        vy = int((py - grid_min[1]) / voxel_size)
        vz = int((pz - grid_min[2]) / voxel_size)
        
        # Boundary check
        if vx < 0 or vx >= grid_x or vy < 0 or vy >= grid_y or vz < 0 or vz >= grid_z:
            continue
        
        if mode == 'count':
            grid[vx, vy, vz] += 1
        elif mode == 'occupancy':
            grid[vx, vy, vz] = 1
    
    return grid


@pytest.mark.cuda
@pytest.mark.parametrize("num_points", [100, 1000, 10000])
@pytest.mark.parametrize("grid_size", [[32, 32, 32], [64, 64, 64], [128, 128, 128]])
@pytest.mark.parametrize("mode", ['count', 'occupancy'])
def test_voxelize_correctness(num_points, grid_size, mode):
    """Test voxelization correctness against CPU reference"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Generate random points in [-10, 10]³
    torch.manual_seed(42)
    points = torch.rand(num_points, 3, device='cuda') * 20 - 10
    
    grid_min = [-10.0, -10.0, -10.0]
    voxel_size = 20.0 / grid_size[0]
    
    # GPU result
    try:
        gpu_result = robocache.voxelize_pointcloud(
            points,
            grid_min=grid_min,
            voxel_size=voxel_size,
            grid_size=grid_size,
            mode=mode
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise
    
    # CPU reference
    cpu_result = cpu_reference_voxelize(
        points.cpu(),
        grid_min,
        voxel_size,
        grid_size,
        mode
    )
    
    # Compare
    torch.testing.assert_close(
        gpu_result.cpu(),
        cpu_result,
        rtol=0,
        atol=0,
        msg=f"Mismatch for num_points={num_points}, grid_size={grid_size}, mode={mode}"
    )


@pytest.mark.cuda
def test_voxelize_shape():
    """Test output shape is correct"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    points = torch.rand(1000, 3, device='cuda') * 20 - 10
    grid_size = [64, 64, 64]
    
    try:
        result = robocache.voxelize_pointcloud(
            points,
            grid_min=[-10, -10, -10],
            voxel_size=0.3125,
            grid_size=grid_size,
            mode='occupancy'
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise
    
    assert result.shape == tuple(grid_size), f"Expected {tuple(grid_size)}, got {result.shape}"
    assert result.dtype == torch.int32


@pytest.mark.cuda
def test_voxelize_boundary_cases():
    """Test edge cases: out-of-bounds points, empty grid"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test 1: All points out of bounds
    points = torch.tensor([
        [100, 100, 100],
        [-100, -100, -100],
        [50, 50, 50]
    ], device='cuda', dtype=torch.float32)
    
    try:
        result = robocache.voxelize_pointcloud(
            points,
            grid_min=[-10, -10, -10],
            voxel_size=0.15625,
            grid_size=[128, 128, 128],
            mode='count'
        )
        assert result.sum() == 0, "Out-of-bounds points should not contribute"
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise
    
    # Test 2: All points in same voxel
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02]
    ], device='cuda', dtype=torch.float32)
    
    try:
        result = robocache.voxelize_pointcloud(
            points,
            grid_min=[-10, -10, -10],
            voxel_size=0.5,
            grid_size=[64, 64, 64],
            mode='count'
        )
        # All 3 points should be in the same voxel
        assert result.max() == 3, f"Expected 3 points in one voxel, got {result.max()}"
        assert (result > 0).sum() == 1, f"Expected 1 occupied voxel, got {(result > 0).sum()}"
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise


@pytest.mark.cuda
def test_voxelize_determinism():
    """Test that voxelization is deterministic across runs"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    torch.manual_seed(42)
    points = torch.rand(10000, 3, device='cuda') * 20 - 10
    
    grid_min = [-10, -10, -10]
    voxel_size = 0.15625
    grid_size = [128, 128, 128]
    
    # Run 5 times
    results = []
    for i in range(5):
        try:
            result = robocache.voxelize_pointcloud(
                points,
                grid_min=grid_min,
                voxel_size=voxel_size,
                grid_size=grid_size,
                mode='count'
            )
            results.append(result.cpu())
        except RuntimeError as e:
            if "not available" in str(e):
                pytest.skip("Voxelization kernel not compiled")
            raise
    
    # All results should be identical
    for i in range(1, len(results)):
        torch.testing.assert_close(
            results[i],
            results[0],
            rtol=0,
            atol=0,
            msg=f"Run {i} differs from run 0 (non-deterministic)"
        )


@pytest.mark.cuda
def test_voxelize_large_grid():
    """Test voxelization with large grids (128³ and 256³)"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    points = torch.rand(100000, 3, device='cuda') * 40 - 20
    
    # 128³ grid
    try:
        result_128 = robocache.voxelize_pointcloud(
            points,
            grid_min=[-20, -20, -20],
            voxel_size=40.0 / 128,
            grid_size=[128, 128, 128],
            mode='occupancy'
        )
        assert result_128.shape == (128, 128, 128)
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise
    
    # 256³ grid (stress test)
    try:
        result_256 = robocache.voxelize_pointcloud(
            points,
            grid_min=[-20, -20, -20],
            voxel_size=40.0 / 256,
            grid_size=[256, 256, 256],
            mode='occupancy'
        )
        assert result_256.shape == (256, 256, 256)
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise


@pytest.mark.cuda
@pytest.mark.parametrize("device_id", [0])
def test_voxelize_device_placement(device_id):
    """Test that kernel runs on specified device"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    if torch.cuda.device_count() <= device_id:
        pytest.skip(f"GPU {device_id} not available")
    
    device = f'cuda:{device_id}'
    points = torch.rand(1000, 3, device=device) * 20 - 10
    
    try:
        result = robocache.voxelize_pointcloud(
            points,
            grid_min=[-10, -10, -10],
            voxel_size=0.15625,
            grid_size=[128, 128, 128],
            mode='occupancy'
        )
        assert result.device.type == 'cuda'
        assert result.device.index == device_id
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Voxelization kernel not compiled")
        raise
