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
    device: Optional[str] = None,
    backend: Optional[str] = None
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
    
    # Force specific backend if requested
    if backend == "cuda":
        if not _cuda_available:
            raise RuntimeError(
                "CUDA backend requested but RoboCache CUDA kernels not available. "
                "Ensure CUDA extension was compiled successfully."
            )
        if not source_data.is_cuda:
            raise RuntimeError("CUDA backend requested but tensors are on CPU")
        return _cuda_ops.resample_trajectories_cuda(
            source_data, source_times, target_times
        )
    elif backend == "pytorch":
        return _resample_pytorch(source_data, source_times, target_times)
    
    # Auto-select: Use CUDA kernel if available and on CUDA device
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
    """
    PyTorch fallback implementation with timestamp-aware interpolation.
    
    Uses actual timestamp values for correct linear interpolation,
    not uniform spacing assumptions.
    """
    from robocache.ops_fallback import resample_single_stream_cpu
    return resample_single_stream_cpu(source_data, source_times, target_times)

def fuse_multimodal(
    stream1_data: torch.Tensor,
    stream1_times: torch.Tensor,
    stream2_data: torch.Tensor,
    stream2_times: torch.Tensor,
    stream3_data: torch.Tensor,
    stream3_times: torch.Tensor,
    target_times: torch.Tensor,
    backend: Optional[str] = None
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
        backend: Force backend selection: 'cuda', 'pytorch', or None (auto)
    
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
    # Force specific backend if requested
    if backend == "cuda":
        if not _multimodal_available:
            raise RuntimeError(
                "CUDA backend requested but RoboCache multimodal CUDA kernels not available. "
                "Ensure CUDA extension was compiled successfully."
            )
        if not stream1_data.is_cuda:
            raise RuntimeError("CUDA backend requested but tensors are on CPU")
        return _multimodal_ops.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    elif backend == "pytorch":
        return ops_fallback.fuse_multimodal_cpu(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    elif backend is not None:
        raise ValueError(f"Unknown backend: {backend}. Expected 'cuda', 'pytorch', or None")
    
    # Auto-select: Use CUDA kernel if available and on CUDA device
    if _multimodal_available and stream1_data.is_cuda:
        return _multimodal_ops.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    
    # Fallback to CPU implementation (vectorized PyTorch)
    return ops_fallback.fuse_multimodal_cpu(
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
    mode: str = "occupancy",
    backend: Optional[str] = None
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
        backend: Force backend selection: 'cuda', 'pytorch', or None (auto)
    
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
    # Force specific backend if requested
    if backend == "cuda":
        if not _voxelize_available:
            raise RuntimeError(
                "CUDA backend requested but RoboCache voxelization CUDA kernels not available. "
                "Ensure CUDA extension was compiled successfully."
            )
        if not points.is_cuda:
            raise RuntimeError("CUDA backend requested but tensors are on CPU")
        results = _voxelize_ops.voxelize_pointcloud(
            points, features if features is not None else torch.empty(0),
            grid_min, voxel_size, grid_size, mode
        )
        return results[0]
    elif backend == "pytorch":
        return ops_fallback.voxelize_pointcloud_cpu(
            points, features, tuple(grid_min), voxel_size, tuple(grid_size), mode
        )
    elif backend is not None:
        raise ValueError(f"Unknown backend: {backend}. Expected 'cuda', 'pytorch', or None")
    
    # Auto-select: Use CUDA kernel if available and on CUDA device
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

def check_installation() -> dict:
    """
    Check installation status and backend availability.
    
    Returns:
        dict with keys:
            - 'cuda_extension_available': bool - Trajectory resampling CUDA kernels
            - 'multimodal_extension_available': bool - Multimodal fusion CUDA kernels
            - 'voxelize_extension_available': bool - Voxelization CUDA kernels
            - 'pytorch_available': bool - PyTorch backend availability
            - 'cuda_device_available': bool - CUDA GPU device present
            - 'gpu_name': str | None - GPU device name if available
    
    Example:
        >>> info = robocache.check_installation()
        >>> if info['cuda_device_available'] and info['cuda_extension_available']:
        ...     print(f"CUDA acceleration available on {info['gpu_name']}")
    """
    return {
        'cuda_extension_available': _cuda_available,
        'multimodal_extension_available': _multimodal_available,
        'voxelize_extension_available': _voxelize_available,
        'pytorch_available': True,  # Always true if robocache loads
        'cuda_device_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

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
    Comprehensive self-test for RoboCache installation.
    
    Tests all operations, dtypes, and backends (CPU + CUDA if available).
    
    Returns:
        True if tests pass, raises exception otherwise
    """
    print("RoboCache Comprehensive Self-Test")
    print("=" * 60)
    
    # Check PyTorch
    print(f"✓ PyTorch {torch.__version__}")
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (CPU-only mode)")
    
    # Check extensions
    info = check_installation()
    if info['cuda_extension_available']:
        print(f"✓ RoboCache CUDA kernels loaded")
    else:
        print(f"⚠ RoboCache CUDA kernels not available")
    
    print("\nFunctional Tests:")
    
    # Test 1: Trajectory Resampling
    print("\n[1/6] Trajectory Resampling...")
    for dtype in [torch.float32, torch.bfloat16 if device == 'cuda' else torch.float32]:
        source = torch.randn(2, 10, 8, dtype=dtype, device=device)
        src_times = torch.linspace(0, 1, 10, device=device).expand(2, -1)
        tgt_times = torch.linspace(0, 1, 5, device=device).expand(2, -1)
        
        result = resample_trajectories(source, src_times, tgt_times)
        assert result.shape == (2, 5, 8), f"Wrong shape: {result.shape}"
        assert result.dtype == dtype, f"Wrong dtype: {result.dtype}"
        assert not torch.isnan(result.float()).any(), "Result contains NaN"
        print(f"  ✓ {dtype} on {device}: {source.shape} -> {result.shape}")
    
    # Test 2: Timestamp-Aware Interpolation
    print("\n[2/6] Timestamp Correctness...")
    source_data = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
    source_times = torch.tensor([[0.0, 0.5, 1.0]], device=device)
    target_times = torch.tensor([[0.25, 0.75]], device=device)
    result = resample_trajectories(source_data, source_times, target_times, backend='pytorch')
    expected = torch.tensor([[[1.5], [2.5]]], device=device)
    assert torch.allclose(result, expected, atol=1e-4), f"Timestamp interpolation incorrect"
    print(f"  ✓ Timestamp-aware interpolation verified")
    
    # Test 3: Voxelization
    print("\n[3/6] Point Cloud Voxelization...")
    points = torch.rand(1000, 3, device=device) * 10.0 - 5.0
    grid = voxelize_pointcloud(
        points,
        grid_min=[-5.0, -5.0, -5.0],
        voxel_size=0.1,
        grid_size=[100, 100, 100],
        mode='occupancy'
    )
    assert grid.shape == (100, 100, 100), f"Wrong grid shape: {grid.shape}"
    assert grid.sum() > 0, "Empty grid (should have occupied voxels)"
    print(f"  ✓ Voxelization: {points.shape[0]} points -> {grid.sum().item():.0f} occupied voxels")
    
    # Test 4: Multimodal Fusion
    print("\n[4/6] Multimodal Sensor Fusion...")
    vision = torch.randn(2, 10, 64, device=device)
    vision_times = torch.linspace(0, 1, 10, device=device).expand(2, -1)
    proprio = torch.randn(2, 20, 32, device=device)
    proprio_times = torch.linspace(0, 1, 20, device=device).expand(2, -1)
    imu = torch.randn(2, 40, 12, device=device)
    imu_times = torch.linspace(0, 1, 40, device=device).expand(2, -1)
    target_times = torch.linspace(0, 1, 15, device=device).expand(2, -1)
    
    fused = fuse_multimodal(
        vision, vision_times,
        proprio, proprio_times,
        imu, imu_times,
        target_times
    )
    expected_shape = (2, 15, 64+32+12)
    assert fused.shape == expected_shape, f"Wrong shape: {fused.shape} vs {expected_shape}"
    assert not torch.isnan(fused).any(), "Fused result contains NaN"
    print(f"  ✓ Multimodal fusion: 3 streams -> {fused.shape}")
    
    # Test 5: Error Handling
    print("\n[5/6] Error Handling...")
    try:
        _ = resample_trajectories(source, src_times, tgt_times, backend='invalid')
        assert False, "Should have raised error for invalid backend"
    except (ValueError, RuntimeError):
        print(f"  ✓ Invalid backend correctly rejected")
    
    # Test 6: CUDA vs CPU Parity (if CUDA available)
    if torch.cuda.is_available() and info['cuda_extension_available']:
        print("\n[6/6] CUDA/CPU Parity...")
        data_cpu = torch.randn(2, 10, 8)
        times_src = torch.linspace(0, 1, 10).expand(2, -1)
        times_tgt = torch.linspace(0, 1, 5).expand(2, -1)
        
        result_cpu = resample_trajectories(data_cpu, times_src, times_tgt, backend='pytorch')
        result_cuda = resample_trajectories(
            data_cpu.cuda(), times_src.cuda(), times_tgt.cuda(), backend='cuda'
        )
        
        max_diff = (result_cpu - result_cuda.cpu()).abs().max().item()
        assert max_diff < 1e-3, f"CUDA/CPU results differ: {max_diff}"
        print(f"  ✓ CUDA/CPU results match (max diff: {max_diff:.6f})")
    else:
        print("\n[6/6] CUDA/CPU Parity... (skipped, CUDA not available)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return True

# Compatibility wrapper for tests
def voxelize_occupancy(points, grid_size, voxel_size, origin, backend=None):
    """
    Compatibility wrapper for voxelize_pointcloud with occupancy mode.
    
    This function exists for backward compatibility with test code.
    New code should use voxelize_pointcloud(..., mode='occupancy').
    """
    return voxelize_pointcloud(
        points=points,
        features=None,
        grid_min=origin.tolist() if hasattr(origin, 'tolist') else origin,
        voxel_size=voxel_size,
        grid_size=grid_size.tolist() if hasattr(grid_size, 'tolist') else grid_size,
        mode='occupancy',
        backend=backend
    )

# Export public API
__all__ = [
    'resample_trajectories',
    'fuse_multimodal',
    'voxelize_pointcloud',
    'voxelize_occupancy',  # Compatibility wrapper
    'is_cuda_available',
    'check_installation',
    'self_test',
    'print_installation_info',
    '__version__',
]
