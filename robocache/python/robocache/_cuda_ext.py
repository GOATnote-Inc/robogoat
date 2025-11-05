"""
Lazy CUDA extension loader for RoboCache
Prevents JIT compilation at import time
"""

import os
import torch

_cached_module = None

def get_cuda_module():
    """Load CUDA extension with JIT compilation"""
    global _cached_module
    
    if _cached_module is not None:
        return _cached_module
    
    # Check if CUDA available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Get kernel directory
    kernel_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'kernels', 'cutlass')
    kernel_dir = os.path.abspath(kernel_dir)
    
    if not os.path.exists(kernel_dir):
        raise RuntimeError(f"Kernel directory not found: {kernel_dir}")
    
    # Define sources
    sources = [
        os.path.join(kernel_dir, 'trajectory_resample_optimized_v2.cu'),
    ]
    
    # Check sources exist
    for src in sources:
        if not os.path.exists(src):
            raise RuntimeError(f"Source file not found: {src}")
    
    # JIT compile
    try:
        from torch.utils.cpp_extension import load
        
        _cached_module = load(
            name='robocache_cuda',
            sources=sources,
            extra_cuda_cflags=[
                '-O3',
                '-std=c++17',
                '--use_fast_math',
                '-lineinfo',
                f'-I{kernel_dir}',
            ],
            verbose=False,
        )
        
        return _cached_module
        
    except Exception as e:
        raise RuntimeError(f"Failed to compile CUDA extension: {e}")

def is_cuda_available():
    """Check if CUDA extension can be loaded"""
    try:
        get_cuda_module()
        return True
    except:
        return False
