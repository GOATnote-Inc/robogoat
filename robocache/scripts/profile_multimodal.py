#!/usr/bin/env python3
"""
Profile multimodal fusion operation for Nsight Systems/Compute.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx


def main():
    """Profile multimodal fusion with NVTX ranges."""
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
    
    print(f"Profiling multimodal fusion:")
    print(f"  Batch: {batch_size}")
    print(f"  Vision: 30 Hz, 256-dim")
    print(f"  Proprio: 100 Hz, 16-dim")
    print(f"  Force: 333 Hz, 6-dim")
    print(f"  Target: 50 Hz")
    print(f"  Device: {device}, Dtype: {dtype}")
    
    # Generate data
    with nvtx.range("data_generation"):
        vision_data = torch.randn(batch_size, 30, 256, device=device, dtype=dtype)
        vision_times = torch.linspace(0, 1, 30, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, 100, 16, device=device, dtype=dtype)
        proprio_times = torch.linspace(0, 1, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, 333, 6, device=device, dtype=dtype)
        force_times = torch.linspace(0, 1, 333, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 1, 50, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Warmup
    with nvtx.range("warmup"):
        for _ in range(10):
            _ = robocache.fuse_multimodal(
                vision_data, vision_times,
                proprio_data, proprio_times,
                force_data, force_times,
                target_times
            )
        torch.cuda.synchronize()
    
    # Start profiling
    profiler.start()
    
    # Profile iterations
    with nvtx.range("profile_iterations"):
        for i in range(100):
            with nvtx.range(f"iteration_{i}"):
                result = robocache.fuse_multimodal(
                    vision_data, vision_times,
                    proprio_data, proprio_times,
                    force_data, force_times,
                    target_times
                )
                torch.cuda.synchronize()
    
    # Stop profiling
    profiler.stop()
    
    print(f"✅ Profiling complete. Result shape: {result.shape}")


if __name__ == "__main__":
    main()

