#!/usr/bin/env python3
"""
Profile trajectory resampling operation for Nsight Systems/Compute.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx


def main():
    """Profile trajectory resampling with NVTX ranges."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    # Import robocache
    try:
        import robocache
    except ImportError:
        print("ERROR: robocache not installed. Run: pip install -e .")
        sys.exit(1)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 32
    source_len = 500
    target_len = 256
    dim = 256
    
    print(f"Profiling trajectory resampling:")
    print(f"  Batch: {batch_size}, Source: {source_len}, Target: {target_len}, Dim: {dim}")
    print(f"  Device: {device}, Dtype: {dtype}")
    
    # Generate data
    with nvtx.range("data_generation"):
        source_data = torch.randn(batch_size, source_len, dim, device=device, dtype=dtype)
        source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Warmup
    with nvtx.range("warmup"):
        for _ in range(10):
            _ = robocache.resample_trajectories(source_data, source_times, target_times)
        torch.cuda.synchronize()
    
    # Start profiling
    profiler.start()
    
    # Profile iterations
    with nvtx.range("profile_iterations"):
        for i in range(100):
            with nvtx.range(f"iteration_{i}"):
                result = robocache.resample_trajectories(source_data, source_times, target_times)
                torch.cuda.synchronize()
    
    # Stop profiling
    profiler.stop()
    
    print(f"✅ Profiling complete. Result shape: {result.shape}")


if __name__ == "__main__":
    main()

