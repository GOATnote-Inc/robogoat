#!/usr/bin/env python3
"""
Profile point cloud voxelization for Nsight Systems/Compute.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx


def main():
    """Profile voxelization with NVTX ranges."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    # Import robocache
    try:
        import robocache
    except ImportError:
        print("ERROR: robocache not installed. Run: pip install -e .")
        sys.exit(1)
    
    device = "cuda"
    n_points = 100000
    grid_size = (128, 128, 128)
    voxel_size = 0.1
    
    print(f"Profiling voxelization:")
    print(f"  Points: {n_points}")
    print(f"  Grid: {grid_size}")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Device: {device}")
    
    # Generate data
    with nvtx.range("data_generation"):
        torch.manual_seed(42)
        points = torch.rand(n_points, 3, device=device) * 12.0  # Points in [0, 12]^3
        origin = torch.zeros(3, device=device)
    
    # Warmup
    with nvtx.range("warmup"):
        for _ in range(10):
            _ = robocache.voxelize_point_cloud(points, grid_size, voxel_size, origin)
        torch.cuda.synchronize()
    
    # Start profiling
    profiler.start()
    
    # Profile iterations
    with nvtx.range("profile_iterations"):
        for i in range(100):
            with nvtx.range(f"iteration_{i}"):
                result = robocache.voxelize_point_cloud(points, grid_size, voxel_size, origin)
                torch.cuda.synchronize()
    
    # Stop profiling
    profiler.stop()
    
    print(f"✅ Profiling complete. Grid shape: {result.shape}, Occupancy: {result.sum().item():.0f} voxels")


if __name__ == "__main__":
    main()

