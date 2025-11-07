#!/usr/bin/env python3
"""
Lightweight smoke test for CI performance gates.
Tests minimum throughput and catches regressions.
"""
import argparse
import torch
import time
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assert-min-throughput', type=float, default=None,
                        help='Fail if throughput below this (samples/sec)')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping smoke test")
        return 0
    
    try:
        import robocache
    except ImportError:
        print("❌ RoboCache not installed")
        return 1
    
    if not robocache._cuda_available:
        print("❌ RoboCache CUDA kernels not available")
        return 1
    
    # Test configuration (small, fast)
    batch = 8
    src_len = 100
    tgt_len = 50
    dim = 64
    
    print("="*60)
    print("RoboCache Smoke Test")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: batch={batch}, src={src_len}, tgt={tgt_len}, dim={dim}")
    print()
    
    # Create test data
    source = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device='cuda')
    src_times = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1)
    tgt_times = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1)
    
    # Warmup
    for _ in range(args.warmup):
        _ = robocache.resample_trajectories(source, src_times, tgt_times, backend="cuda")
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.iterations):
        result = robocache.resample_trajectories(source, src_times, tgt_times, backend="cuda")
    end.record()
    torch.cuda.synchronize()
    
    # Compute metrics
    elapsed_ms = start.elapsed_time(end) / args.iterations
    samples_per_sec = (batch * tgt_len) / (elapsed_ms / 1000.0)
    
    print(f"✓ P50 latency: {elapsed_ms:.3f} ms")
    print(f"✓ Throughput: {samples_per_sec:,.0f} samples/sec")
    print()
    
    # Check threshold
    if args.assert_min_throughput:
        if samples_per_sec < args.assert_min_throughput:
            print(f"❌ FAILED: Throughput {samples_per_sec:,.0f} below threshold {args.assert_min_throughput:,.0f}")
            return 1
        else:
            print(f"✓ PASSED: Throughput above threshold")
    
    print("="*60)
    print("✓ Smoke test PASSED")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

