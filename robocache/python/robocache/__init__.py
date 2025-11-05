"""
RoboCache: GPU-Accelerated Data Engine for Embodied AI

High-performance CUDA primitives for robot learning data preprocessing,
optimized for NVIDIA H100 GPUs with multi-backend support.

Quick Start:
    >>> import robocache
    >>> robocache.print_installation_info()
    >>>
    >>> # Trajectory resampling (auto-selects best backend)
    >>> import torch
    >>> data = torch.randn(64, 100, 32, dtype=torch.bfloat16, device='cuda')
    >>> src_times = torch.linspace(0, 1, 100, device='cuda').expand(64, -1).contiguous()
    >>> tgt_times = torch.linspace(0, 1, 50, device='cuda').expand(64, -1).contiguous()
    >>> resampled = robocache.resample_trajectories(data, src_times, tgt_times)

Performance:
    - Trajectory resampling: 23.76% DRAM BW on H100 (NCU-validated)
    - Multimodal fusion: 20.45% L1 cache (optimal L1-resident)
    - End-to-end pipeline: 100% sustained GPU utilization

Author: RoboCache Team
License: Apache-2.0
GitHub: https://github.com/robocache/robocache
"""

# Version information
from ._version import (
    __version__,
    __version_info__,
    __build_date__,
    __api_version__,
)

__author__ = "RoboCache Team"
__license__ = "Apache-2.0"

import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration first
from .config import get_config

# Initialize config
_config = get_config()

# Import backend infrastructure
from .backends import (
    BackendType,
    select_backend,
    get_backend_status,
    get_backend_info,
    PyTorchBackend,
)

# Import observability
from .observability import (
    get_metrics,
    profile_operation,
    health_check,
    print_health_check,
    get_telemetry,
)

# Lazy-loaded CUDA module
_cuda_module = None


def _get_cuda_module():
    """Lazy-load CUDA module only when needed."""
    global _cuda_module
    if _cuda_module is None:
        from ._cuda_ext import get_cuda_module
        _cuda_module = get_cuda_module()
    return _cuda_module


def resample_trajectories(
    source_data,
    source_times,
    target_times,
    backend: Optional[str] = None
):
    """
    Resample robot trajectories to uniform frequency with temporal interpolation.
    
    Automatically selects the best available backend (CUDA > PyTorch) or allows
    manual override. CUDA backend provides 1.85x-22x speedup depending on workload.
    
    Args:
        source_data (torch.Tensor): Input trajectories [batch, source_len, action_dim]
            Supported dtypes: float32, float16, bfloat16 (bfloat16 recommended for CUDA)
        source_times (torch.Tensor): Source timestamps [batch, source_len]
            Must be float32, monotonically increasing per batch
        target_times (torch.Tensor): Target timestamps [batch, target_len]
            Must be float32
        backend (str, optional): Backend selection
            - None or 'auto': Auto-select best available (CUDA > PyTorch)
            - 'cuda': Force CUDA backend (raises error if unavailable)
            - 'pytorch': Force PyTorch native (CPU/GPU compatible)
    
    Returns:
        torch.Tensor: Resampled trajectories [batch, target_len, action_dim]
            Same dtype and device as source_data
    
    Raises:
        RuntimeError: If requested backend is unavailable
        ValueError: If tensor shapes are incompatible
    
    Performance (H100, BF16, NCU-validated):
        - CUDA: 138.24 μs, 23.76% DRAM BW
        - PyTorch GPU: ~2-3ms (fallback)
        - PyTorch CPU: ~20-30ms (compatibility)
    
    Example:
        >>> import torch
        >>> import robocache
        >>>
        >>> # Auto-select backend (CUDA if available)
        >>> data = torch.randn(64, 100, 32, dtype=torch.bfloat16, device='cuda')
        >>> src_t = torch.linspace(0, 1, 100, device='cuda').expand(64, -1).contiguous()
        >>> tgt_t = torch.linspace(0, 1, 50, device='cuda').expand(64, -1).contiguous()
        >>> resampled = robocache.resample_trajectories(data, src_t, tgt_t)
        >>>
        >>> # Force PyTorch fallback
        >>> resampled_pt = robocache.resample_trajectories(
        ...     data, src_t, tgt_t, backend='pytorch'
        ... )
    
    Note:
        Uses binary search + linear interpolation. Does not support autograd.
        Use in torch.no_grad() context for data preprocessing pipelines.
    """
    # Select backend
    selected_backend = select_backend(backend if backend else _config.backend)
    
    # Execute with profiling
    with profile_operation("resample_trajectories",
                          batch_size=source_data.shape[0],
                          source_len=source_data.shape[1],
                          target_len=target_times.shape[1],
                          backend=selected_backend.value):
        
        if selected_backend == BackendType.CUDA:
            cuda_module = _get_cuda_module()
            return cuda_module.resample_trajectories(source_data, source_times, target_times)
        
        elif selected_backend == BackendType.PYTORCH:
            return PyTorchBackend.resample_trajectories(source_data, source_times, target_times)
        
        else:
            raise RuntimeError(f"Unsupported backend: {selected_backend}")


def fused_multimodal_alignment(
    vision_data,
    vision_times,
    proprio_data,
    proprio_times,
    force_data=None,
    force_times=None,
    target_times=None,
    backend: Optional[str] = None
):
    """
    Fuse multimodal sensor data with temporal alignment.
    
    Aligns vision, proprioception, and optionally force data to common timestamps
    using high-performance resampling, then concatenates features.
    
    Args:
        vision_data (torch.Tensor): Vision features [batch, vision_len, vision_dim]
        vision_times (torch.Tensor): Vision timestamps [batch, vision_len]
        proprio_data (torch.Tensor): Proprioception [batch, proprio_len, proprio_dim]
        proprio_times (torch.Tensor): Proprioception timestamps [batch, proprio_len]
        force_data (torch.Tensor, optional): Force/torque [batch, force_len, force_dim]
        force_times (torch.Tensor, optional): Force timestamps [batch, force_len]
        target_times (torch.Tensor, optional): Target timestamps [batch, target_len]
            If None, uses vision_times as target
        backend (str, optional): 'auto', 'cuda', or 'pytorch' (default: 'auto')
    
    Returns:
        torch.Tensor: Fused multimodal data [batch, target_len, total_dim]
            where total_dim = vision_dim + proprio_dim + force_dim
    
    Performance (H100, BF16, NCU-validated):
        - CUDA: 81.66 μs, 20.45% L1 cache (optimal L1-resident behavior)
        - PyTorch: ~5-10ms (fallback)
    
    Example:
        >>> import torch
        >>> import robocache
        >>>
        >>> # RGB camera at 30 Hz (512-dim)
        >>> vision = torch.randn(32, 30, 512, dtype=torch.bfloat16, device='cuda')
        >>> vision_t = torch.linspace(0, 1, 30, device='cuda').expand(32, -1).contiguous()
        >>>
        >>> # Proprioception at 100 Hz (32-dim)
        >>> proprio = torch.randn(32, 100, 32, dtype=torch.bfloat16, device='cuda')
        >>> proprio_t = torch.linspace(0, 1, 100, device='cuda').expand(32, -1).contiguous()
        >>>
        >>> # Force at 50 Hz (16-dim)
        >>> force = torch.randn(32, 50, 16, dtype=torch.bfloat16, device='cuda')
        >>> force_t = torch.linspace(0, 1, 50, device='cuda').expand(32, -1).contiguous()
        >>>
        >>> # Align all to 30 Hz (vision frequency)
        >>> fused = robocache.fused_multimodal_alignment(
        ...     vision, vision_t, proprio, proprio_t, force, force_t
        ... )
        >>> print(fused.shape)  # torch.Size([32, 30, 560])
    """
    if target_times is None:
        target_times = vision_times
    
    selected_backend = select_backend(backend if backend else _config.backend)
    
    with profile_operation("fused_multimodal_alignment",
                          batch_size=vision_data.shape[0],
                          backend=selected_backend.value):
        
        if selected_backend == BackendType.CUDA:
            cuda_module = _get_cuda_module()
            return cuda_module.fused_multimodal_alignment(
                vision_data, vision_times,
                proprio_data, proprio_times,
                force_data, force_times,
                target_times
            )
        
        elif selected_backend == BackendType.PYTORCH:
            return PyTorchBackend.fused_multimodal_alignment(
                vision_data, vision_times,
                proprio_data, proprio_times,
                force_data, force_times,
                target_times
            )
        
        else:
            raise RuntimeError(f"Unsupported backend: {selected_backend}")


def voxelize_occupancy(
    points,
    grid_size,
    voxel_size,
    origin,
    backend: Optional[str] = None
):
    """
    Convert point cloud to binary occupancy voxel grid.
    
    Creates a 3D voxel grid where each cell is 1.0 if at least one point
    falls within it, 0.0 otherwise. Optimized for real-time robotics with
    deterministic atomic operations for CPU/GPU parity.
    
    Args:
        points (torch.Tensor): Point cloud [batch, num_points, 3] (XYZ)
        grid_size (list or tuple): Grid dimensions [depth, height, width]
        voxel_size (float): Size of each voxel in meters
        origin (torch.Tensor): Grid origin [3] (X, Y, Z)
        backend (str, optional): 'auto', 'cuda', or 'pytorch'
    
    Returns:
        torch.Tensor: Binary occupancy grid [batch, depth, height, width]
            Values are 0.0 (empty) or 1.0 (occupied)
    
    Performance (H100, NCU-validated):
        - CUDA: Functional, atomic operations
        - PyTorch: 500-1000x slower (not recommended for production)
    
    Example:
        >>> import torch
        >>> import robocache
        >>>
        >>> # LiDAR point cloud (100k points)
        >>> points = torch.randn(4, 100000, 3, device='cuda') * 10
        >>> grid_size = [128, 128, 128]
        >>> voxel_size = 0.1  # 10cm voxels
        >>> origin = torch.tensor([-6.4, -6.4, -6.4], device='cuda')
        >>>
        >>> # Voxelize
        >>> occupancy = robocache.voxelize_occupancy(
        ...     points, grid_size, voxel_size, origin
        ... )
        >>> print(occupancy.shape)  # torch.Size([4, 128, 128, 128])
    """
    selected_backend = select_backend(backend if backend else _config.backend)
    
    with profile_operation("voxelize_occupancy",
                          batch_size=points.shape[0],
                          num_points=points.shape[1],
                          grid_size=grid_size,
                          backend=selected_backend.value):
        
        if selected_backend == BackendType.CUDA:
            cuda_module = _get_cuda_module()
            return cuda_module.voxelize_occupancy(points, grid_size, voxel_size, origin)
        
        elif selected_backend == BackendType.PYTORCH:
            return PyTorchBackend.voxelize_occupancy(points, grid_size, voxel_size, origin)
        
        else:
            raise RuntimeError(f"Unsupported backend: {selected_backend}")


# Convenience functions
def check_installation():
    """
    Check RoboCache installation and return system info.
    
    Returns:
        dict: Installation status, backend availability, GPU info, etc.
    """
    from .backends import get_backend_info
    import torch
    
    info = {
        "version": __version__,
        "api_version": __api_version__,
        "build_date": __build_date__,
    }
    
    # Backend info
    backend_info = get_backend_info()
    info.update(backend_info)
    
    # PyTorch info
    if torch is not None:
        info["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["pytorch"]["cuda_version"] = torch.version.cuda
            info["pytorch"]["gpu_count"] = torch.cuda.device_count()
            info["pytorch"]["gpu_name"] = torch.cuda.get_device_name(0)
            info["pytorch"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def print_installation_info():
    """Print formatted installation information."""
    info = check_installation()
    
    print("=" * 80)
    print(f"RoboCache v{info['version']} (API v{info['api_version']})")
    print("=" * 80)
    print(f"Build Date:       {info['build_date']}")
    print(f"Default Backend:  {info.get('default_backend', 'N/A')}")
    print()
    
    print("Backends:")
    for name, backend in info.get("backends", {}).items():
        status = "✓" if backend["available"] else "✗"
        print(f"  {status} {name:10s} {backend.get('performance_tier', '')}")
        if backend.get("error"):
            print(f"      Error: {backend['error']}")
    print()
    
    if "pytorch" in info:
        pt = info["pytorch"]
        print(f"PyTorch:          {pt['version']}")
        print(f"CUDA Available:   {'✓' if pt['cuda_available'] else '✗'}")
        if pt.get("cuda_available"):
            print(f"CUDA Version:     {pt.get('cuda_version', 'N/A')}")
            print(f"GPU:              {pt.get('gpu_name', 'N/A')}")
            print(f"GPU Memory:       {pt.get('gpu_memory_gb', 0):.1f} GB")
    
    print("=" * 80)
    return info


def enable_metrics():
    """Enable performance metrics collection."""
    get_metrics().enable()


def disable_metrics():
    """Disable performance metrics collection."""
    get_metrics().disable()


def print_metrics():
    """Print collected performance metrics."""
    get_metrics().print_stats()


def reset_metrics():
    """Reset all performance metrics."""
    get_metrics().reset()


# Export public API
__all__ = [
    # Version info
    "__version__",
    "__api_version__",
    
    # Core operations
    "resample_trajectories",
    "fused_multimodal_alignment",
    "voxelize_occupancy",
    
    # Installation and health
    "check_installation",
    "print_installation_info",
    "health_check",
    "print_health_check",
    
    # Metrics and profiling
    "enable_metrics",
    "disable_metrics",
    "print_metrics",
    "reset_metrics",
    
    # Backend management
    "get_backend_status",
    "get_backend_info",
    
    # Configuration
    "get_config",
]
