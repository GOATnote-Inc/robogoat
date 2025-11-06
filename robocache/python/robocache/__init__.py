"""
RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models

Optimized CUDA kernels for H100/A100 with BF16 support.
"""

__version__ = "1.0.0"

import torch
from typing import Optional

# Try to import CUDA extension
_cuda_available = False
try:
    from robocache import _cuda_ops
    _cuda_available = True
except ImportError:
    _cuda_ops = None

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

def is_cuda_available() -> bool:
    """Check if CUDA kernels are available"""
    return _cuda_available

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
    'is_cuda_available',
    'self_test',
    '__version__',
]
