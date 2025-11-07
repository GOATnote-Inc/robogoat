"""
RoboCache Benchmark Harness

Production-grade performance measurement with warmup, steady-state, and statistical analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Callable, Any

import numpy as np
import torch


class BenchmarkHarness:
    """Performance benchmark harness with CUDA event timing"""
    
    def __init__(self, warmup: int = 50, iterations: int = 200, device: str = 'cuda'):
        self.warmup = warmup
        self.iterations = iterations
        self.device = device
        
    def benchmark(self, fn: Callable, *args, **kwargs) -> Dict[str, float]:
        """
        Benchmark a function with CUDA event timing.
        
        Returns:
            dict: {p50_ms, p99_ms, mean_ms, std_ms, min_ms, max_ms}
        """
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark with CUDA events
        times = []
        
        if self.device == 'cuda':
            for _ in range(self.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                fn(*args, **kwargs)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        else:
            for _ in range(self.iterations):
                start = time.perf_counter()
                fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'p50_ms': float(np.percentile(times, 50)),
            'p99_ms': float(np.percentile(times, 99)),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
        }


def benchmark_multimodal_fusion(device: str = 'cuda'):
    """Benchmark multimodal fusion operation"""
    import robocache
    
    batch = 4
    vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device=device)
    vision_times = torch.linspace(0, 1, 30, device=device).expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device=device)
    proprio_times = torch.linspace(0, 1, 100, device=device).expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device=device)
    imu_times = torch.linspace(0, 1, 200, device=device).expand(batch, -1)
    target = torch.linspace(0, 1, 50, device=device).expand(batch, -1)
    
    harness = BenchmarkHarness(device=device)
    
    def fn():
        return robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target
        )
    
    results = harness.benchmark(fn)
    results['name'] = 'multimodal_fusion'
    results['device'] = device
    return results


def benchmark_voxelization(device: str = 'cuda', mode: str = 'occupancy'):
    """Benchmark voxelization operation"""
    import robocache
    
    num_points = 500000
    points = torch.rand(num_points, 3, device=device) * 4.0 - 2.0
    
    harness = BenchmarkHarness(device=device, iterations=100)
    
    def fn():
        return robocache.voxelize_pointcloud(
            points,
            grid_min=[-2.0, -2.0, -2.0],
            voxel_size=0.0625,
            grid_size=[128, 128, 128],
            mode=mode
        )
    
    results = harness.benchmark(fn)
    results['name'] = f'voxelization_{mode}'
    results['device'] = device
    results['num_points'] = num_points
    results['throughput_billion_pts_per_sec'] = num_points / (results['mean_ms'] / 1000) / 1e9
    return results


def run_full_suite(device: str = 'cuda', output_dir: str = 'bench_results'):
    """Run complete benchmark suite"""
    import robocache
    
    print(f"RoboCache Benchmark Suite")
    print(f"========================")
    print(f"Version: {robocache.__version__}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    results = []
    
    # Multimodal fusion
    print("[1/3] Multimodal fusion...")
    r = benchmark_multimodal_fusion(device)
    print(f"  P50: {r['p50_ms']:.3f}ms, P99: {r['p99_ms']:.3f}ms")
    results.append(r)
    
    # Voxelization modes
    for mode in ['count', 'occupancy']:
        print(f"[2-3/3] Voxelization ({mode})...")
        r = benchmark_voxelization(device, mode)
        print(f"  P50: {r['p50_ms']:.3f}ms, Throughput: {r['throughput_billion_pts_per_sec']:.2f}B pts/s")
        results.append(r)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Git SHA (if available)
    try:
        import subprocess
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        sha = 'unknown'
    
    output_file = output_path / f'benchmark_{device}_{sha}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'sha': sha,
            'device': device,
            'robocache_version': robocache.__version__,
            'results': results
        }, f, indent=2)
    
    print()
    print(f"✓ Results saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', default='bench_results')
    args = parser.parse_args()
    
    run_full_suite(args.device, args.output_dir)

