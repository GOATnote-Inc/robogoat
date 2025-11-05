"""
Backend selection and management for RoboCache.

Supports multiple backends with automatic selection and fallback:
- CUDA: Optimized CUTLASS kernels (fastest, H100/A100)
- PyTorch: Pure PyTorch fallback (compatibility)
- Triton: Auto-tuned kernels (future)
"""

import enum
import os
import warnings
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BackendType(enum.Enum):
    """Available backend types."""
    CUDA = "cuda"
    PYTORCH = "pytorch"
    TRITON = "triton"  # Future
    AUTO = "auto"


class BackendStatus:
    """Track backend availability and reasons for unavailability."""
    
    def __init__(self):
        self.cuda_available = False
        self.cuda_error = None
        self.pytorch_available = False
        self.pytorch_error = None
        self.triton_available = False
        self.triton_error = None
        
        self._check_backends()
    
    def _check_backends(self):
        """Check which backends are available."""
        # Check PyTorch
        try:
            import torch
            self.pytorch_available = True
            logger.info(f"PyTorch backend available: {torch.__version__}")
        except ImportError as e:
            self.pytorch_error = f"PyTorch not installed: {e}"
            logger.warning(self.pytorch_error)
        
        # Check CUDA extension
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available in PyTorch")
            
            # Try to import/build CUDA extension
            from . import _cuda_ext
            self.cuda_available = True
            logger.info("CUDA backend available")
        except Exception as e:
            self.cuda_error = f"CUDA extension unavailable: {e}"
            logger.info(self.cuda_error)
        
        # Check Triton (future)
        try:
            import triton
            self.triton_available = True
            logger.info(f"Triton backend available: {triton.__version__}")
        except ImportError:
            self.triton_error = "Triton not installed"
    
    def get_default_backend(self) -> BackendType:
        """Get the best available backend."""
        if self.cuda_available:
            return BackendType.CUDA
        elif self.pytorch_available:
            return BackendType.PYTORCH
        else:
            raise RuntimeError(
                "No backends available. Install PyTorch: pip install torch"
            )
    
    def print_status(self):
        """Print backend status for debugging."""
        print("="*60)
        print("RoboCache Backend Status")
        print("="*60)
        print(f"CUDA:    {'✓ Available' if self.cuda_available else f'✗ {self.cuda_error}'}")
        print(f"PyTorch: {'✓ Available' if self.pytorch_available else f'✗ {self.pytorch_error}'}")
        print(f"Triton:  {'✓ Available' if self.triton_available else f'✗ {self.triton_error}'}")
        print(f"Default: {self.get_default_backend().value}")
        print("="*60)


# Global backend status
_backend_status = None


def get_backend_status() -> BackendStatus:
    """Get or create global backend status."""
    global _backend_status
    if _backend_status is None:
        _backend_status = BackendStatus()
    return _backend_status


def select_backend(requested: Optional[str] = None) -> BackendType:
    """
    Select the backend to use.
    
    Args:
        requested: Requested backend ('auto', 'cuda', 'pytorch', 'triton', or None)
    
    Returns:
        BackendType to use
    
    Raises:
        RuntimeError: If requested backend is unavailable
    """
    status = get_backend_status()
    
    # Handle None or 'auto'
    if requested is None or requested == 'auto':
        return status.get_default_backend()
    
    # Validate requested backend
    try:
        backend = BackendType(requested.lower())
    except ValueError:
        valid = [b.value for b in BackendType]
        raise ValueError(
            f"Invalid backend '{requested}'. Valid options: {valid}"
        )
    
    # Check if requested backend is available
    if backend == BackendType.CUDA:
        if not status.cuda_available:
            raise RuntimeError(
                f"CUDA backend requested but unavailable: {status.cuda_error}\n"
                f"Use backend='pytorch' for fallback."
            )
        return backend
    
    elif backend == BackendType.PYTORCH:
        if not status.pytorch_available:
            raise RuntimeError(
                f"PyTorch backend unavailable: {status.pytorch_error}"
            )
        return backend
    
    elif backend == BackendType.TRITON:
        if not status.triton_available:
            raise RuntimeError(
                f"Triton backend requested but unavailable: {status.triton_error}"
            )
        return backend
    
    elif backend == BackendType.AUTO:
        return status.get_default_backend()
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


class PyTorchBackend:
    """
    Pure PyTorch fallback implementations.
    
    These provide CPU/GPU compatible implementations without CUDA extensions.
    Performance is significantly slower than CUDA but ensures compatibility.
    """
    
    @staticmethod
    def resample_trajectories(source_data, source_times, target_times):
        """
        PyTorch implementation of trajectory resampling.
        
        Uses torch.searchsorted for binary search and linear interpolation.
        ~10-20x slower than CUDA but works on CPU/GPU.
        """
        import torch
        
        # Ensure contiguous
        source_data = source_data.contiguous()
        source_times = source_times.contiguous()
        target_times = target_times.contiguous()
        
        B, S, D = source_data.shape
        T = target_times.shape[1]
        
        # Allocate output
        output = torch.empty(B, T, D, dtype=source_data.dtype, device=source_data.device)
        
        for b in range(B):
            src_t = source_times[b]
            tgt_t = target_times[b]
            src_d = source_data[b]
            
            # Binary search for each target time
            indices = torch.searchsorted(src_t, tgt_t) - 1
            indices = torch.clamp(indices, 0, S - 2)
            
            # Get neighboring times and data
            t_left = src_t[indices]
            t_right = src_t[indices + 1]
            d_left = src_d[indices]
            d_right = src_d[indices + 1]
            
            # Linear interpolation weight
            alpha = ((tgt_t - t_left) / (t_right - t_left + 1e-8)).unsqueeze(-1)
            
            # Interpolate
            output[b] = d_left + alpha * (d_right - d_left)
        
        return output
    
    @staticmethod
    def fused_multimodal_alignment(vision_data, vision_times, proprio_data, proprio_times,
                                   force_data, force_times, target_times):
        """PyTorch implementation of multimodal fusion."""
        # Resample each modality to target times
        vision_resampled = PyTorchBackend.resample_trajectories(
            vision_data, vision_times, target_times
        )
        proprio_resampled = PyTorchBackend.resample_trajectories(
            proprio_data, proprio_times, target_times
        )
        
        if force_data is not None:
            force_resampled = PyTorchBackend.resample_trajectories(
                force_data, force_times, target_times
            )
            return torch.cat([vision_resampled, proprio_resampled, force_resampled], dim=-1)
        else:
            return torch.cat([vision_resampled, proprio_resampled], dim=-1)
    
    @staticmethod
    def voxelize_occupancy(points, grid_size, voxel_size, origin):
        """
        PyTorch implementation of occupancy voxelization.
        
        WARNING: Extremely slow (~500-1000x slower than CUDA).
        Only use for correctness testing or when CUDA is unavailable.
        """
        import torch
        
        B, N, _ = points.shape
        D, H, W = grid_size
        
        # Allocate output grid
        grid = torch.zeros(B, D, H, W, dtype=torch.float32, device=points.device)
        
        for b in range(B):
            pts = points[b]  # [N, 3]
            
            # Convert points to voxel indices
            voxel_coords = ((pts - origin) / voxel_size).long()
            
            # Filter out-of-bounds
            mask = (
                (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < D) &
                (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < H) &
                (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < W)
            )
            valid_coords = voxel_coords[mask]
            
            # Set occupied voxels (this is slow but works)
            for coord in valid_coords:
                grid[b, coord[0], coord[1], coord[2]] = 1.0
        
        return grid


def get_backend_info() -> Dict[str, Any]:
    """
    Get detailed backend information for debugging/monitoring.
    
    Returns:
        Dict with backend availability, performance characteristics, etc.
    """
    status = get_backend_status()
    
    info = {
        "backends": {
            "cuda": {
                "available": status.cuda_available,
                "error": status.cuda_error,
                "performance_tier": "optimal" if status.cuda_available else None,
            },
            "pytorch": {
                "available": status.pytorch_available,
                "error": status.pytorch_error,
                "performance_tier": "fallback" if status.pytorch_available else None,
            },
            "triton": {
                "available": status.triton_available,
                "error": status.triton_error,
                "performance_tier": "experimental" if status.triton_available else None,
            },
        },
        "default_backend": status.get_default_backend().value if (
            status.cuda_available or status.pytorch_available
        ) else None,
    }
    
    # Add environment info
    info["environment"] = {
        "ROBOCACHE_BACKEND": os.environ.get("ROBOCACHE_BACKEND"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    
    return info


__all__ = [
    "BackendType",
    "BackendStatus",
    "select_backend",
    "get_backend_status",
    "get_backend_info",
    "PyTorchBackend",
]

