"""Public Python API for the open-source RoboCache reference package."""

from __future__ import annotations

import os
import sys
import warnings
from enum import Enum
from functools import lru_cache
from typing import Dict, Literal, Optional

import torch

from ._version import __api_version__, __version__, __version_info__
from .backends.pytorch_backend import PyTorchBackend


def _env_flag(*names: str, default: bool = False) -> bool:
    """Return True if any environment variable in *names evaluates to truthy."""

    truthy = {"1", "true", "yes", "on"}
    seen = False
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        seen = True
        if value.strip().lower() in truthy:
            return True
    return default if not seen else False


CUDA_BACKEND_ENABLED = _env_flag(
    "ROBOCACHE_ENABLE_CUDA_BACKEND",
    "ROBOCACHE_BUILD_WITH_CUDA",
    default=False,
)


class BackendType(Enum):
    CUDA = "cuda"
    PYTORCH = "pytorch"


@lru_cache(maxsize=1)
def _cuda_extension_status() -> tuple[bool, Optional[str]]:
    """Return whether the CUDA extension can be imported."""

    if not CUDA_BACKEND_ENABLED:
        return False, "CUDA backend disabled for this build"

    if not torch.cuda.is_available():
        return False, "CUDA runtime unavailable"

    try:
        from . import _cuda_ext

        _cuda_ext.get_cuda_module()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _select_backend(requested: Optional[str]) -> BackendType:
    """Select a backend using local availability checks."""

    normalized = requested.lower() if isinstance(requested, str) else None

    if normalized in {None, "auto"}:
        if CUDA_BACKEND_ENABLED:
            cuda_ok, _ = _cuda_extension_status()
            if cuda_ok:
                return BackendType.CUDA
        return BackendType.PYTORCH

    if normalized == "cuda":
        if not CUDA_BACKEND_ENABLED:
            raise RuntimeError(
                "CUDA backend disabled for this build. Reinstall RoboCache with "
                "ROBOCACHE_ENABLE_CUDA_BACKEND=1 set during installation, or use "
                "backend='pytorch'."
            )

        cuda_ok, _ = _cuda_extension_status()
        if not cuda_ok:
            _, error = _cuda_extension_status()
            raise RuntimeError(
                f"CUDA backend unavailable: {error}. Use backend='pytorch' instead."
            )
        return BackendType.CUDA

    if normalized == "pytorch":
        return BackendType.PYTORCH

    raise ValueError(f"Unsupported backend '{requested}'. Expected 'cuda' or 'pytorch'.")

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
    
    try:
        backend_choice = _select_backend(backend)
    except RuntimeError as exc:
        if isinstance(backend, str) and backend.lower() == "cuda":
            raise NotImplementedError(
                "CUDA backend is disabled or unavailable in this build. "
                "Export ROBOCACHE_ENABLE_CUDA_BACKEND=1 during installation and "
                "provide a CUDA 13+ toolchain to enable it."
            ) from exc
        raise

    if backend_choice == BackendType.CUDA:
        from . import _cuda_ext

        module = _cuda_ext.get_cuda_module()

        if not (source_data.is_cuda and source_times.is_cuda and target_times.is_cuda):
            raise RuntimeError(
                "CUDA backend requires all tensors on CUDA devices. "
                "Use backend='pytorch' for CPU tensors."
            )

        if source_times.dtype != torch.float32 or target_times.dtype != torch.float32:
            raise TypeError("CUDA backend expects FP32 timestamps")

        original_dtype = source_data.dtype
        if original_dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                "CUDA backend supports float32 or bfloat16 source_data tensors"
            )

        # CUDA kernels are implemented in BF16. Allow FP32 inputs by down-casting and
        # returning to the original dtype for convenience in tests and examples.
        if original_dtype == torch.float32:
            source_data_cast = source_data.to(torch.bfloat16)
            convert_back = True
        else:
            source_data_cast = source_data
            convert_back = False

        result = module.resample_trajectories(
            source_data_cast.contiguous(),
            source_times.contiguous(),
            target_times.contiguous(),
        )

        if convert_back:
            result = result.to(original_dtype)

        return result

    if source_data.is_cuda:
        warnings.warn(
            "Falling back to the PyTorch reference implementation on CUDA tensors.",
            RuntimeWarning,
        )

    return PyTorchBackend.resample_trajectories(
        source_data, source_times, target_times
    )

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
    
    warnings.warn(
        "voxelize_point_cloud is deprecated in favour of voxelize_occupancy and "
        "currently executes the PyTorch reference implementation only.",
        DeprecationWarning,
    )

    return voxelize_occupancy(
        points=points,
        grid_size=grid_size,
        voxel_size=voxel_size,
        origin=origin,
        backend=backend,
    )


def voxelize_occupancy(
    points: torch.Tensor,
    grid_size: tuple,
    voxel_size: float,
    origin: Optional[torch.Tensor] = None,
    backend: Optional[Literal["cuda", "pytorch"]] = None,
) -> torch.Tensor:
    """Voxelize a point cloud into a binary occupancy grid using PyTorch."""

    if backend == "cuda":
        raise NotImplementedError(
            "CUDA voxelization kernels are unavailable in the reference build. "
            "Pass backend='pytorch' to use the compatibility implementation."
        )

    if origin is None:
        origin = torch.zeros(3, device=points.device)

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
    if backend == "cuda":
        raise NotImplementedError(
            "CUDA multimodal fusion kernels are unavailable in the reference build. "
            "Pass backend='pytorch' to use the compatibility implementation."
        )

    v_aligned = resample_trajectories(
        vision_data, vision_times, target_times, backend
    )
    p_aligned = resample_trajectories(
        proprio_data, proprio_times, target_times, backend
    )
    f_aligned = resample_trajectories(
        force_data, force_times, target_times, backend
    )

    # Concatenate
    return torch.cat([v_aligned, p_aligned, f_aligned], dim=2)


def check_installation() -> Dict[str, Optional[str]]:
    """Return backend availability information used by tests and docs."""

    info: Dict[str, Optional[str]] = {
        "pytorch_available": False,
        "pytorch_version": None,
        "cuda_extension_available": False,
        "cuda_extension_error": None,
        "cuda_backend_enabled": CUDA_BACKEND_ENABLED,
        "default_backend": None,
        "triton_available": False,
        "triton_error": None,
    }

    try:
        info["pytorch_available"] = True
        info["pytorch_version"] = torch.__version__
        info["cuda_runtime_available"] = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - defensive
        info["pytorch_error"] = str(exc)
        info["cuda_runtime_available"] = False
        return info

    cuda_ok, cuda_error = _cuda_extension_status()
    info["cuda_extension_available"] = cuda_ok
    info["cuda_extension_error"] = cuda_error

    if cuda_ok:
        info["default_backend"] = BackendType.CUDA.value
    elif info["pytorch_available"]:
        info["default_backend"] = BackendType.PYTORCH.value
    else:
        info["default_backend_error"] = "PyTorch not installed"

    return info


def print_installation_info(stream=None) -> None:
    """Pretty-print installation diagnostics for README examples."""

    info = check_installation()
    out = stream or sys.stdout

    print("=" * 60, file=out)
    print("RoboCache Installation Diagnostics", file=out)
    print("=" * 60, file=out)

    print(f"PyTorch: {'✓ ' + info['pytorch_version'] if info['pytorch_available'] else '✗'}", file=out)

    backend_state = "✓ Enabled" if info["cuda_backend_enabled"] else "✗ Disabled for this build"
    print(f"CUDA Backend Flag: {backend_state}", file=out)

    cuda_status = "✓ Available" if info["cuda_extension_available"] else f"✗ {info['cuda_extension_error']}"
    print(f"CUDA Extension: {cuda_status}", file=out)

    if "cuda_runtime_available" in info:
        runtime_state = "✓" if info["cuda_runtime_available"] else "✗"
        print(f"CUDA Runtime: {runtime_state}", file=out)

    if info["default_backend"]:
        print(f"Default Backend: {info['default_backend']}", file=out)
    elif "default_backend_error" in info:
        print(f"Default Backend: ✗ {info['default_backend_error']}", file=out)

    if info["triton_available"]:
        print("Triton: ✓ Available", file=out)
    elif info["triton_error"]:
        print(f"Triton: ✗ {info['triton_error']}", file=out)

    print("=" * 60, file=out)


__all__ = [
    "__version__",
    "__version_info__",
    "__api_version__",
    "resample_trajectories",
    "voxelize_point_cloud",
    "voxelize_occupancy",
    "fuse_multimodal",
]
