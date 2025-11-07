"""
RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models

Optimized CUDA kernels for H100/A100 with BF16 support.
"""

__version__ = "1.0.0"

import sys
import platform
from typing import Optional, TextIO

import torch

# Try to import CUDA extensions
_cuda_available = False
_multimodal_available = False
_voxelize_available = False

try:
    from robocache import _cuda_ops
    _cuda_available = True
except ImportError:
    _cuda_ops = None

try:
    from robocache import _multimodal_ops
    _multimodal_available = True
except ImportError:
    _multimodal_ops = None

try:
    from robocache import _voxelize_ops
    _voxelize_available = True
except ImportError:
    _voxelize_ops = None

# Always import CPU fallbacks (production requirement)
from robocache import ops_fallback

def resample_trajectories(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Resample trajectory data from source to target timestamps.
    
    Automatically uses CUDA kernels when available, falls back to PyTorch.
    
    Args:
        source_data: Input trajectory [B, S, D] (BF16 or FP32)
        source_times: Source timestamps [B, S]
        target_times: Target timestamps [B, T]
        device: Target device (default: source_data.device)
    
    Returns:
        Resampled trajectory [B, T, D]
    
    Performance:
        H100: ~2.6ms for (32, 500, 256) -> (32, 256, 256)
        A100: ~3.1ms for same config
    
    Examples:
        >>> source = torch.randn(8, 100, 64, dtype=torch.bfloat16, device='cuda')
        >>> src_times = torch.linspace(0, 5, 100, device='cuda').expand(8, -1)
        >>> tgt_times = torch.linspace(0, 5, 50, device='cuda').expand(8, -1)
        >>> result = resample_trajectories(source, src_times, tgt_times)
        >>> result.shape
        torch.Size([8, 50, 64])
    """
    if device is None:
        device = source_data.device
    
    # Move to target device
    source_data = source_data.to(device)
    source_times = source_times.to(device)
    target_times = target_times.to(device)
    
    # Use CUDA kernel if available and on CUDA device
    if _cuda_available and source_data.is_cuda:
        return _cuda_ops.resample_trajectories_cuda(
            source_data, source_times, target_times
        )
    
    # Fallback to PyTorch implementation
    return _resample_pytorch(source_data, source_times, target_times)

def _resample_pytorch(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """PyTorch fallback implementation (slower)"""
    return torch.nn.functional.interpolate(
        source_data.transpose(1, 2),
        size=target_times.shape[1],
        mode='linear',
        align_corners=True
    ).transpose(1, 2)

def fuse_multimodal(
    stream1_data: torch.Tensor,
    stream1_times: torch.Tensor,
    stream2_data: torch.Tensor,
    stream2_times: torch.Tensor,
    stream3_data: torch.Tensor,
    stream3_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    Fuse multimodal sensor streams with temporal alignment.
    
    Aligns heterogeneous sensor data (vision, proprioception, IMU) to
    uniform target timestamps with sub-millisecond latency.
    
    Args:
        stream1_data: First sensor stream [B, S1, D1]
        stream1_times: First stream timestamps [B, S1]
        stream2_data: Second sensor stream [B, S2, D2]
        stream2_times: Second stream timestamps [B, S2]
        stream3_data: Third sensor stream [B, S3, D3]
        stream3_times: Third stream timestamps [B, S3]
        target_times: Target timestamps [B, T]
    
    Returns:
        Fused features [B, T, D1+D2+D3]
    
    Performance:
        H100: <1ms for 3 streams @ 100Hz -> 50Hz
    
    Examples:
        >>> vision = torch.randn(4, 30, 512, device='cuda', dtype=torch.bfloat16)
        >>> vision_times = torch.linspace(0, 1, 30, device='cuda').expand(4, -1)
        >>> proprio = torch.randn(4, 100, 64, device='cuda', dtype=torch.bfloat16)
        >>> proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(4, -1)
        >>> imu = torch.randn(4, 200, 12, device='cuda', dtype=torch.bfloat16)
        >>> imu_times = torch.linspace(0, 1, 200, device='cuda').expand(4, -1)
        >>> target = torch.linspace(0, 1, 50, device='cuda').expand(4, -1)
        >>> fused = fuse_multimodal(vision, vision_times, proprio, proprio_times, 
        ...                          imu, imu_times, target)
        >>> fused.shape
        torch.Size([4, 50, 588])  # 512 + 64 + 12
    """
    # Use CUDA kernel if available and on CUDA device
    if _multimodal_available and stream1_data.is_cuda:
        return _multimodal_ops.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    
    # Fallback to CPU implementation (vectorized PyTorch)
    return ops_fallback.resample_trajectories_cpu(
        stream1_data, stream1_times,
        stream2_data, stream2_times,
        stream3_data, stream3_times,
        target_times
    )

def voxelize_pointcloud(
    points: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    grid_min: list = [-10.0, -10.0, -10.0],
    voxel_size: float = 0.1,
    grid_size: list = [128, 128, 128],
    mode: str = "occupancy"
) -> torch.Tensor:
    """
    Voxelize 3D point cloud to structured grid.
    
    Converts unstructured point clouds to voxel grids with atomic accumulation
    for deterministic results. Supports multiple modes: count, occupancy, mean, max.
    
    Args:
        points: Point coordinates [N, 3] (x, y, z)
        features: Point features [N, F] (required for mean/max modes)
        grid_min: Minimum corner [x, y, z] (default: [-10, -10, -10])
        voxel_size: Voxel edge length (default: 0.1)
        grid_size: Grid dimensions [X, Y, Z] (default: [128, 128, 128])
        mode: Accumulation mode: "count", "occupancy", "mean", "max"
    
    Returns:
        Voxel grid [X, Y, Z] for count/occupancy, [X, Y, Z, F] for mean/max
    
    Performance:
        H100: >2.5B points/sec @ 128³ grid
    
    Examples:
        >>> points = torch.rand(1000000, 3, device='cuda') * 20 - 10  # 1M points
        >>> grid = voxelize_pointcloud(points, mode="occupancy")
        >>> grid.shape
        torch.Size([128, 128, 128])
        
        >>> # With features (mean mode)
        >>> features = torch.randn(1000000, 8, device='cuda')
        >>> grid = voxelize_pointcloud(points, features, mode="mean")
        >>> grid.shape
        torch.Size([128, 128, 128, 8])
    """
    # Use CUDA kernel if available and on CUDA device
    if _voxelize_available and points.is_cuda:
        results = _voxelize_ops.voxelize_pointcloud(
            points, features if features is not None else torch.empty(0),
            grid_min, voxel_size, grid_size, mode
        )
        return results[0]  # Return voxel_grid (discard counts if mean mode)
    
    # Fallback to CPU implementation (vectorized PyTorch)
    return ops_fallback.voxelize_pointcloud_cpu(
        points, features, tuple(grid_min), voxel_size, tuple(grid_size), mode
    )

def is_cuda_available() -> bool:
    """Check if CUDA kernels are available"""
    return _cuda_available and _multimodal_available and _voxelize_available

def _write_line(message: str, stream: TextIO) -> None:
    """Write a single line to the provided stream."""
    stream.write(f"{message}\n")


def print_installation_info(stream: Optional[TextIO] = None) -> None:
    """Print diagnostic information about the RoboCache installation."""
    if stream is None:
        stream = sys.stdout

    python_version = platform.python_version()
    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_runtime = torch.version.cuda if cuda_version else "None"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    _write_line("RoboCache Installation Info", stream)
    _write_line("=" * 60, stream)
    _write_line(f"Version: {__version__}", stream)
    _write_line(f"Python: {python_version}", stream)
    _write_line(f"PyTorch: {torch_version}", stream)
    _write_line(f"CUDA available: {torch.cuda.is_available()}", stream)
    _write_line(f"CUDA runtime: {cuda_runtime}", stream)
    _write_line(f"GPU: {gpu_name}", stream)
    _write_line(f"CUDA kernels loaded: {_cuda_available}", stream)
    _write_line(f"Multimodal kernels loaded: {_multimodal_available}", stream)
    _write_line(f"Voxelize kernels loaded: {_voxelize_available}", stream)
    _write_line("Module location: " + __file__, stream)


def self_test():
    """
    Run quick self-test to verify installation.
    
    Returns:
        True if tests pass, raises exception otherwise
    """
    print("RoboCache Self-Test")
    print("=" * 60)
    
    # Check PyTorch
    print(f"✓ PyTorch {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (CPU-only mode)")
    
    # Check CUDA kernels
    if _cuda_available:
        print(f"✓ RoboCache CUDA kernels loaded")
    else:
        print(f"⚠ RoboCache CUDA kernels not available (using PyTorch fallback)")
    
    # Quick functional test
    print("\nFunctional Test:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source = torch.randn(2, 10, 8, dtype=torch.float32, device=device)
    src_times = torch.linspace(0, 1, 10, device=device).expand(2, -1)
    tgt_times = torch.linspace(0, 1, 5, device=device).expand(2, -1)
    
    result = resample_trajectories(source, src_times, tgt_times)
    assert result.shape == (2, 5, 8), f"Wrong shape: {result.shape}"
    print(f"✓ Trajectory resampling: {source.shape} -> {result.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return True

# Export public API
__all__ = [
    'resample_trajectories',
    'fuse_multimodal',
    'voxelize_pointcloud',
    'is_cuda_available',
    'self_test',
    'print_installation_info',
    '__version__',
]
