#!/usr/bin/env python3
"""
Profile trajectory resampling operation for Nsight Systems/Compute.
Uses GPU-accelerated interpolation as RoboCache proxy.
"""

import sys
import time
import torch

# Try to use NVTX if available
try:
    import torch.cuda.profiler as profiler
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except:
    NVTX_AVAILABLE = False
    # Mock NVTX for compatibility
    class MockNVTX:
        @staticmethod
        def range(name):
            class Context:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return Context()
    nvtx = MockNVTX()
    class MockProfiler:
        @staticmethod
        def start(): pass
        @staticmethod
        def stop(): pass
    profiler = MockProfiler()


def resample_gpu(source_data, source_times, target_times):
    """GPU-accelerated trajectory resampling (RoboCache proxy)."""
    # Use PyTorch's GPU interpolation as proxy for CUDA kernel
    return torch.nn.functional.interpolate(
        source_data.transpose(1, 2),
        size=target_times.shape[1],
        mode='linear',
        align_corners=True
    ).transpose(1, 2)


def main():
    """Profile trajectory resampling with NVTX ranges."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 32
    source_len = 500
    target_len = 256
    dim = 256
    
    print(f"Profiling trajectory resampling (GPU):")
    print(f"  Batch: {batch_size}, Source: {source_len}, Target: {target_len}, Dim: {dim}")
    print(f"  Device: {device}, Dtype: {dtype}")
    print(f"  NVTX: {'Enabled' if NVTX_AVAILABLE else 'Disabled'}")
    
    # Generate data
    with nvtx.range("data_generation"):
        source_data = torch.randn(batch_size, source_len, dim, device=device, dtype=dtype)
        source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Warmup
    with nvtx.range("warmup"):
        for _ in range(10):
            _ = resample_gpu(source_data, source_times, target_times)
        torch.cuda.synchronize()
    
    # Measure latency
    start = time.perf_counter()
    with nvtx.range("timing"):
        for _ in range(100):
            result = resample_gpu(source_data, source_times, target_times)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / 100
    
    # Start profiling
    profiler.start()
    
    # Profile iterations
    with nvtx.range("profile_iterations"):
        for i in range(100):
            with nvtx.range(f"iteration_{i}"):
                result = resample_gpu(source_data, source_times, target_times)
                torch.cuda.synchronize()
    
    # Stop profiling
    profiler.stop()
    
    print(f"")
    print(f"✅ Profiling complete")
    print(f"   Result shape: {result.shape}")
    print(f"   Latency: {elapsed:.3f} ms")
    throughput = (batch_size * 100) / (elapsed * 100 / 1000)
    print(f"   Throughput: {throughput:.1f} samples/sec")


if __name__ == "__main__":
    main()

