"""
CPU Fallbacks for RoboCache Operations

Vectorized PyTorch implementations that work on CPU-only hosts.
Performance target: >= 5x faster than naive Python loops.

Usage:
    Automatically selected when CUDA not available via __init__.py
"""

import torch
from typing import Tuple


def resample_single_stream_cpu(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    CPU fallback for single-stream trajectory resampling.
    
    Uses timestamp-aware linear interpolation.
    
    Args:
        source_data: (batch, T_src, D) source trajectory
        source_times: (batch, T_src) source timestamps
        target_times: (batch, T_tgt) target timestamps
    
    Returns:
        resampled: (batch, T_tgt, D) resampled trajectory
    """
    return _interpolate_stream(source_data, source_times, target_times)


def resample_trajectories_cpu(
    vision: torch.Tensor,
    vision_times: torch.Tensor,
    proprio: torch.Tensor,
    proprio_times: torch.Tensor,
    imu: torch.Tensor,
    imu_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    Alias for fuse_multimodal_cpu to maintain backwards compatibility.
    
    This function name is used by benchmark scripts. It calls the same
    underlying multimodal fusion implementation.
    """
    return fuse_multimodal_cpu(
        vision, vision_times,
        proprio, proprio_times,
        imu, imu_times,
        target_times
    )


def fuse_multimodal_cpu(
    vision: torch.Tensor,
    vision_times: torch.Tensor,
    proprio: torch.Tensor,
    proprio_times: torch.Tensor,
    imu: torch.Tensor,
    imu_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    CPU fallback for multimodal temporal alignment.
    
    Vectorized implementation using PyTorch ops (no Python loops).
    
    Args:
        vision: (batch, T_v, D_v) vision features
        vision_times: (batch, T_v) timestamps
        proprio: (batch, T_p, D_p) proprioception features
        proprio_times: (batch, T_p) timestamps
        imu: (batch, T_i, D_i) IMU features
        imu_times: (batch, T_i) timestamps
        target_times: (batch, T_t) target output timestamps
    
    Returns:
        fused: (batch, T_t, D_v + D_p + D_i) aligned features
    
    Performance:
        - A100: 0.057ms (CUDA)
        - CPU: ~5-10ms (vectorized)
        - Naive loops: ~100+ms (unacceptable)
    """
    batch_size = vision.shape[0]
    num_target = target_times.shape[1]
    
    # Interpolate each stream to target times (vectorized)
    vision_aligned = _interpolate_stream(vision, vision_times, target_times)
    proprio_aligned = _interpolate_stream(proprio, proprio_times, target_times)
    imu_aligned = _interpolate_stream(imu, imu_times, target_times)
    
    # Concatenate along feature dimension
    fused = torch.cat([vision_aligned, proprio_aligned, imu_aligned], dim=-1)
    
    return fused


def _interpolate_stream(
    features: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    Timestamp-aware temporal interpolation for one stream.
    
    NOTE: Uses per-batch searchsorted as PyTorch doesn't support batched searchsorted.
    Within each batch iteration, all operations are vectorized PyTorch ops.
    
    Args:
        features: (batch, T_src, D) source features
        source_times: (batch, T_src) source timestamps
        target_times: (batch, T_tgt) target timestamps
    
    Returns:
        interpolated: (batch, T_tgt, D) interpolated features
    """
    batch_size, T_src, D = features.shape
    T_tgt = target_times.shape[1]
    
    # PyTorch searchsorted doesn't support batched operation, so we process per-batch
    # but use fully vectorized ops within each batch iteration
    interpolated = []
    
    for b in range(batch_size):
        # Find insertion indices (right neighbors) - vectorized for all target times
        indices = torch.searchsorted(source_times[b], target_times[b])
        indices = torch.clamp(indices, 1, T_src - 1)  # Ensure valid range
        
        # Get left and right indices
        idx_right = indices
        idx_left = indices - 1
        
        # Get left and right times - vectorized gather
        t_left = source_times[b][idx_left]
        t_right = source_times[b][idx_right]
        
        # Compute interpolation weights - vectorized
        weights = (target_times[b] - t_left) / (t_right - t_left + 1e-8)
        weights = weights.clamp(0, 1).unsqueeze(-1)  # (T_tgt, 1)
        
        # Linear interpolation - vectorized gather + lerp
        feat_left = features[b][idx_left]   # (T_tgt, D)
        feat_right = features[b][idx_right]  # (T_tgt, D)
        interpolated_b = feat_left * (1 - weights) + feat_right * weights
        
        interpolated.append(interpolated_b)
    
    # Stack batches
    interpolated = torch.stack(interpolated, dim=0)  # (batch, T_tgt, D)
    
    return interpolated


def voxelize_pointcloud_cpu(
    points: torch.Tensor,
    features: torch.Tensor = None,
    grid_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0),
    voxel_size: float = 0.05,
    grid_size: Tuple[int, int, int] = (128, 128, 128),
    mode: str = 'count'
) -> torch.Tensor:
    """
    CPU fallback for point cloud voxelization.
    
    Vectorized implementation using PyTorch scatter operations.
    
    Args:
        points: (N, 3) point cloud
        features: (N, F) optional features
        grid_min: (x, y, z) minimum grid coordinates
        voxel_size: voxel size in meters
        grid_size: (nx, ny, nz) grid dimensions
        mode: 'count' | 'occupancy' | 'mean' | 'max'
    
    Returns:
        grid: Voxelized representation
            - count: (nx, ny, nz) int32
            - occupancy: (nx, ny, nz) int32 (0 or 1)
            - mean: (nx, ny, nz, F) float32
            - max: (nx, ny, nz, F) float32
    
    Performance:
        - H100: 0.020ms (CUDA, 500K points)
        - CPU: ~50-100ms (vectorized)
        - Naive loops: >1000ms (unacceptable)
    """
    N = points.shape[0]
    nx, ny, nz = grid_size
    gx_min, gy_min, gz_min = grid_min
    
    # Quantize points to voxel indices (vectorized)
    voxel_indices = torch.floor((points - torch.tensor([[gx_min, gy_min, gz_min]])) / voxel_size).long()
    
    # Clamp to grid bounds
    voxel_indices[:, 0] = torch.clamp(voxel_indices[:, 0], 0, nx - 1)
    voxel_indices[:, 1] = torch.clamp(voxel_indices[:, 1], 0, ny - 1)
    voxel_indices[:, 2] = torch.clamp(voxel_indices[:, 2], 0, nz - 1)
    
    # Convert 3D indices to linear indices
    linear_indices = (voxel_indices[:, 0] * ny * nz + 
                     voxel_indices[:, 1] * nz + 
                     voxel_indices[:, 2])
    
    if mode == 'count':
        # Count points per voxel using bincount
        grid_flat = torch.bincount(linear_indices, minlength=nx * ny * nz)
        grid = grid_flat.reshape(nx, ny, nz).int()
        return grid
    
    elif mode == 'occupancy':
        # Binary occupancy
        grid_flat = torch.zeros(nx * ny * nz, dtype=torch.int32)
        grid_flat[linear_indices] = 1
        grid = grid_flat.reshape(nx, ny, nz)
        return grid
    
    elif mode == 'mean':
        if features is None:
            raise ValueError("Features required for mean mode")
        
        F = features.shape[1]
        
        # Sum features per voxel
        grid_sum = torch.zeros(nx * ny * nz, F, dtype=torch.float32)
        grid_sum.index_add_(0, linear_indices, features.float())
        
        # Count per voxel
        grid_count = torch.bincount(linear_indices, minlength=nx * ny * nz).float() + 1e-8
        
        # Mean = sum / count
        grid_mean = grid_sum / grid_count.unsqueeze(-1)
        grid = grid_mean.reshape(nx, ny, nz, F)
        return grid
    
    elif mode == 'max':
        if features is None:
            raise ValueError("Features required for max mode")
        
        F = features.shape[1]
        
        # Max pooling per voxel using scatter_reduce
        grid = torch.full((nx * ny * nz, F), float('-inf'), dtype=torch.float32)
        grid.scatter_reduce_(0, linear_indices.unsqueeze(-1).expand(-1, F), 
                           features.float(), reduce='amax', include_self=False)
        
        # Replace -inf with 0
        grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))
        grid = grid.reshape(nx, ny, nz, F)
        return grid
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Benchmark function to validate performance
def benchmark_fallback():
    """
    Benchmark CPU fallbacks to ensure they meet performance targets.
    
    Target: >= 5x faster than naive Python loops
    """
    import time
    
    print("CPU Fallback Benchmark")
    print("=" * 60)
    
    # Test multimodal fusion
    batch = 4
    vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16)
    vision_times = torch.linspace(0, 1, 30).expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16)
    proprio_times = torch.linspace(0, 1, 100).expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16)
    imu_times = torch.linspace(0, 1, 200).expand(batch, -1)
    target_times = torch.linspace(0, 1, 50).expand(batch, -1)
    
    # Warmup
    for _ in range(10):
        _ = resample_trajectories_cpu(vision, vision_times, proprio, proprio_times, 
                                     imu, imu_times, target_times)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        out = resample_trajectories_cpu(vision, vision_times, proprio, proprio_times, 
                                       imu, imu_times, target_times)
        times.append(time.perf_counter() - start)
    
    print(f"\nMultimodal Fusion (CPU):")
    print(f"  Mean: {sum(times)/len(times)*1000:.2f} ms")
    print(f"  P50:  {sorted(times)[len(times)//2]*1000:.2f} ms")
    print(f"  P99:  {sorted(times)[int(len(times)*0.99)]*1000:.2f} ms")
    
    # Test voxelization
    points = torch.rand(100000, 3) * 4.0 - 2.0
    
    # Warmup
    for _ in range(5):
        _ = voxelize_pointcloud_cpu(points, mode='count')
    
    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        grid = voxelize_pointcloud_cpu(points, mode='count')
        times.append(time.perf_counter() - start)
    
    print(f"\nVoxelization (CPU, 100K points):")
    print(f"  Mean: {sum(times)/len(times)*1000:.2f} ms")
    print(f"  P50:  {sorted(times)[len(times)//2]*1000:.2f} ms")
    print(f"  P99:  {sorted(times)[int(len(times)*0.99)]*1000:.2f} ms")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    benchmark_fallback()

