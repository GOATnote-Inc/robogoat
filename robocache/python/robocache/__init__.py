"""
RoboCache: GPU-Accelerated Data Engine for Robot Learning
Optimized for NVIDIA H100 (Hopper, SM90)
"""

__version__ = "0.1.0"

import torch
from typing import Optional, Literal

# Lazy import CUDA extension to avoid import-time JIT
_cuda_module = None

def _get_cuda_ext():
    """Lazy load CUDA extension"""
    global _cuda_module
    if _cuda_module is None:
        try:
            from . import _cuda_ext
            _cuda_module = _cuda_ext.get_cuda_module()
        except Exception as e:
            print(f"Warning: CUDA extension unavailable: {e}")
            _cuda_module = False
    return _cuda_module if _cuda_module else None

def resample_trajectories(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    backend: Optional[Literal["cuda", "pytorch"]] = None
) -> torch.Tensor:
    """
    Resample robot trajectories to uniform sampling rate.
    
    Args:
        source_data: [B, S, D] source trajectory (BF16 or FP32)
        source_times: [B, S] source timestamps (FP32)
        target_times: [B, T] target timestamps (FP32)
        backend: "cuda" (H100 optimized), "pytorch" (fallback), or None (auto)
        
    Returns:
        [B, T, D] resampled trajectory (same dtype as input)
    """
    # Validate inputs
    assert source_data.dim() == 3, f"source_data must be 3D, got {source_data.shape}"
    assert source_times.dim() == 2, f"source_times must be 2D, got {source_times.shape}"
    assert target_times.dim() == 2, f"target_times must be 2D, got {target_times.shape}"
    
    B, S, D = source_data.shape
    assert source_times.shape == (B, S), f"source_times shape mismatch"
    assert target_times.shape[0] == B, f"target_times batch mismatch"
    
    # Backend selection
    if backend is None:
        backend = "cuda" if source_data.is_cuda and _get_cuda_ext() else "pytorch"
    
    if backend == "cuda":
        cuda_ext = _get_cuda_ext()
        if cuda_ext is None:
            print("Warning: CUDA backend requested but unavailable, falling back to PyTorch")
            backend = "pytorch"
        else:
            # CUDA path
            src = source_data.contiguous()
            st = source_times.contiguous()
            tt = target_times.contiguous()
            
            # Convert to BF16 if needed
            if src.dtype == torch.float32:
                src = src.to(torch.bfloat16)
                out = cuda_ext.resample_trajectories_optimized(src, st, tt)
                return out.to(torch.float32)
            else:
                return cuda_ext.resample_trajectories_optimized(src, st, tt)
    
    # PyTorch fallback
    return _resample_pytorch(source_data, source_times, target_times)

def _resample_pytorch(source_data, source_times, target_times):
    """PyTorch fallback implementation"""
    B, S, D = source_data.shape
    T = target_times.shape[1]
    
    output = torch.zeros(B, T, D, dtype=source_data.dtype, device=source_data.device)
    
    for b in range(B):
        for t in range(T):
            tgt = target_times[b, t]
            
            # Find interval
            if tgt <= source_times[b, 0]:
                output[b, t] = source_data[b, 0]
            elif tgt >= source_times[b, -1]:
                output[b, t] = source_data[b, -1]
            else:
                # Binary search
                left, right = 0, S - 1
                while left < right - 1:
                    mid = (left + right) // 2
                    if source_times[b, mid] <= tgt:
                        left = mid
                    else:
                        right = mid
                
                # Linear interpolation
                t_left = source_times[b, left]
                t_right = source_times[b, right]
                alpha = (tgt - t_left) / (t_right - t_left + 1e-8)
                
                output[b, t] = (1 - alpha) * source_data[b, left] + alpha * source_data[b, right]
    
    return output

def voxelize_point_cloud(
    points: torch.Tensor,
    grid_size: tuple,
    voxel_size: float,
    origin: Optional[torch.Tensor] = None,
    backend: Optional[Literal["cuda", "pytorch"]] = None
) -> torch.Tensor:
    """
    Voxelize point cloud to occupancy grid.
    
    Args:
        points: [N, 3] point cloud (FP32)
        grid_size: (X, Y, Z) grid dimensions
        voxel_size: Size of each voxel (meters)
        origin: [3] grid origin (defaults to zeros)
        backend: "cuda" or "pytorch" (auto-select if None)
        
    Returns:
        [X, Y, Z] occupancy grid (FP32, 0 or 1)
    """
    if origin is None:
        origin = torch.zeros(3, device=points.device)
    
    # PyTorch fallback (CUDA not implemented in public API yet)
    X, Y, Z = grid_size
    grid = torch.zeros(X, Y, Z, device=points.device)
    
    for i in range(points.shape[0]):
        p = points[i]
        x = int((p[0] - origin[0]) / voxel_size)
        y = int((p[1] - origin[1]) / voxel_size)
        z = int((p[2] - origin[2]) / voxel_size)
        
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            grid[x, y, z] = 1.0
    
    return grid

def fuse_multimodal(
    vision_data: torch.Tensor,
    vision_times: torch.Tensor,
    proprio_data: torch.Tensor,
    proprio_times: torch.Tensor,
    force_data: torch.Tensor,
    force_times: torch.Tensor,
    target_times: torch.Tensor,
    backend: Optional[Literal["cuda", "pytorch"]] = None
) -> torch.Tensor:
    """
    Align multimodal sensor data to common timestamps.
    
    Args:
        vision_data: [B, Sv, Dv] vision features
        vision_times: [B, Sv] vision timestamps
        proprio_data: [B, Sp, Dp] proprioception
        proprio_times: [B, Sp] proprio timestamps
        force_data: [B, Sf, Df] force/torque
        force_times: [B, Sf] force timestamps
        target_times: [B, T] target timestamps
        backend: "cuda" or "pytorch"
        
    Returns:
        [B, T, Dv+Dp+Df] fused features
    """
    # Resample each modality independently
    v_aligned = resample_trajectories(vision_data, vision_times, target_times, backend)
    p_aligned = resample_trajectories(proprio_data, proprio_times, target_times, backend)
    f_aligned = resample_trajectories(force_data, force_times, target_times, backend)
    
    # Concatenate
    return torch.cat([v_aligned, p_aligned, f_aligned], dim=2)

__all__ = [
    "resample_trajectories",
    "voxelize_point_cloud",
    "fuse_multimodal",
]
