"""
CUDA extension loading for RoboCache.

This module handles lazy loading of CUDA extensions with proper error handling.
Import this module only when CUDA backend is actually requested (not at package import time).
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global CUDA extension cache
_cuda_module = None
_cuda_load_attempted = False
_cuda_load_error = None


def get_kernel_dir() -> Path:
    """Get the kernels directory path."""
    # Assuming package structure: robocache/python/robocache/_cuda_ext.py
    # Kernels are at: robocache/kernels/cutlass/
    package_dir = Path(__file__).parent
    kernel_dir = package_dir.parent.parent / "kernels" / "cutlass"
    
    if not kernel_dir.exists():
        raise FileNotFoundError(
            f"Kernels directory not found: {kernel_dir}\n"
            f"Expected structure: robocache/kernels/cutlass/"
        )
    
    return kernel_dir


def load_cuda_extension(verbose: bool = False, force_reload: bool = False):
    """
    Load the CUDA extension with JIT compilation.
    
    Args:
        verbose: Print compilation output
        force_reload: Force recompilation even if cached
    
    Returns:
        The loaded CUDA module
    
    Raises:
        RuntimeError: If CUDA extension fails to load
    """
    global _cuda_module, _cuda_load_attempted, _cuda_load_error
    
    # Return cached module if already loaded
    if _cuda_module is not None and not force_reload:
        return _cuda_module
    
    # Return cached error if already attempted
    if _cuda_load_attempted and not force_reload:
        if _cuda_load_error is not None:
            raise RuntimeError(f"CUDA extension previously failed to load: {_cuda_load_error}")
        return _cuda_module
    
    _cuda_load_attempted = True
    
    try:
        import torch
        from torch.utils.cpp_extension import load
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in PyTorch")
        
        kernel_dir = get_kernel_dir()
        
        # Check if kernel files exist
        required_files = [
            "trajectory_resample_optimized_v2.cu",
            "multimodal_fusion.cu",
            "point_cloud_voxelization.cu",
            "robocache_bindings_all.cu",
        ]
        
        missing_files = []
        for fname in required_files:
            if not (kernel_dir / fname).exists():
                missing_files.append(fname)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing kernel files in {kernel_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )
        
        logger.info(f"Loading CUDA extension from {kernel_dir}")
        
        # Build with JIT compilation
        _cuda_module = load(
            name='robocache_cuda',
            sources=[
                str(kernel_dir / "trajectory_resample_optimized_v2.cu"),
                str(kernel_dir / "multimodal_fusion.cu"),
                str(kernel_dir / "point_cloud_voxelization.cu"),
                str(kernel_dir / "robocache_bindings_all.cu"),
            ],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-lineinfo',
                '--expt-relaxed-constexpr',
                '-std=c++17',
                '-gencode=arch=compute_90,code=sm_90',  # H100 (Hopper)
                '-gencode=arch=compute_80,code=sm_80',  # A100 (Ampere)
            ],
            verbose=verbose,
        )
        
        logger.info("CUDA extension loaded successfully")
        return _cuda_module
    
    except Exception as e:
        _cuda_load_error = str(e)
        logger.error(f"Failed to load CUDA extension: {e}")
        raise RuntimeError(f"Failed to load CUDA extension: {e}") from e


def get_cuda_module():
    """
    Get the CUDA module, loading it if necessary.
    
    Returns:
        The CUDA module
    
    Raises:
        RuntimeError: If CUDA extension is not available
    """
    if _cuda_module is None:
        return load_cuda_extension(verbose=False)
    return _cuda_module


def is_cuda_available() -> bool:
    """
    Check if CUDA extension is available without raising errors.
    
    Returns:
        True if CUDA extension can be loaded, False otherwise
    """
    try:
        get_cuda_module()
        return True
    except Exception:
        return False


def get_cuda_info() -> dict:
    """
    Get information about CUDA extension status.
    
    Returns:
        Dict with CUDA extension status, error messages, etc.
    """
    info = {
        "loaded": _cuda_module is not None,
        "load_attempted": _cuda_load_attempted,
        "load_error": _cuda_load_error,
    }
    
    if _cuda_module is not None:
        try:
            kernel_dir = get_kernel_dir()
            info["kernel_dir"] = str(kernel_dir)
            info["available_functions"] = [
                name for name in dir(_cuda_module)
                if not name.startswith('_')
            ]
        except Exception as e:
            info["info_error"] = str(e)
    
    return info


__all__ = [
    "load_cuda_extension",
    "get_cuda_module",
    "is_cuda_available",
    "get_cuda_info",
]

