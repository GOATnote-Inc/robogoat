#!/usr/bin/env python3
"""
Reproducible benchmark harness for RoboCache kernels.
Emits machine-readable JSON with full configuration metadata.
"""
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

import torch


def get_system_info() -> Dict[str, Any]:
    """Collect system configuration metadata."""
    info = {
        "timestamp": datetime.utcnow().isoformat(),
        "hostname": os.uname().nodename,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "device_count": torch.cuda.device_count(),
            "driver_version": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True
            ).strip(),
            "gpu_clocks": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem", "--format=csv,noheader"],
                text=True
            ).strip(),
            "power_limit": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"],
                text=True
            ).strip(),
        })
    
    info["pytorch_version"] = torch.__version__
    info["python_version"] = subprocess.check_output(
        ["python3", "--version"], text=True
    ).strip()
    
    return info


def benchmark_kernel(
    kernel_func,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a kernel with statistical rigor."""
    
    # Warmup
    for _ in range(warmup):
        result = kernel_func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = kernel_func(*args, **kwargs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    
    latencies_ms = [l * 1000 for l in latencies]
    
    return {
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": (sum((x - sum(latencies_ms)/len(latencies_ms))**2 for x in latencies_ms) / len(latencies_ms)) ** 0.5,
        "iterations": iterations,
        "warmup": warmup,
    }


def benchmark_trajectory_resampling():
    """Benchmark trajectory resampling kernel."""
    from torch.utils.cpp_extension import load
    
    print("Building optimized trajectory resampling kernel...")
    robocache = load(
        name='robocache_cuda_optimized',
        sources=[
            'kernels/cutlass/trajectory_resample_optimized_v2.cu',
            'kernels/cutlass/trajectory_resample_optimized_v2_torch.cu',
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++17'],
        verbose=False
    )
    
    results = []
    
    # Test multiple configurations
    configs = [
        {"batch": 32, "src_len": 2048, "tgt_len": 512, "dim": 16},
        {"batch": 64, "src_len": 4096, "tgt_len": 1024, "dim": 32},
        {"batch": 128, "src_len": 8192, "tgt_len": 2048, "dim": 64},
    ]
    
    for config in configs:
        print(f"Testing config: {config}")
        
        batch = config["batch"]
        src_len = config["src_len"]
        tgt_len = config["tgt_len"]
        dim = config["dim"]
        
        # Create test data
        data = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device='cuda')
        src_t = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
        tgt_t = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
        
        # Benchmark optimized kernel
        opt_stats = benchmark_kernel(robocache.resample_trajectories, data, src_t, tgt_t)
        
        # Benchmark PyTorch baseline
        def pytorch_baseline(data, src_t, tgt_t):
            indices_l = torch.searchsorted(src_t, tgt_t).clamp(0, src_len-2)
            indices_r = (indices_l + 1).clamp(0, src_len-1)
            t_l = torch.gather(src_t, 1, indices_l)
            t_r = torch.gather(src_t, 1, indices_r)
            w = ((tgt_t - t_l) / (t_r - t_l + 1e-8)).clamp(0, 1).unsqueeze(-1)
            left_data = torch.gather(data, 1, indices_l.unsqueeze(-1).expand(-1, -1, dim))
            right_data = torch.gather(data, 1, indices_r.unsqueeze(-1).expand(-1, -1, dim))
            return left_data + w * (right_data - left_data)
        
        baseline_stats = benchmark_kernel(pytorch_baseline, data, src_t, tgt_t)
        
        # Calculate throughput and speedup
        throughput_opt = batch / (opt_stats["mean_ms"] / 1000)
        throughput_baseline = batch / (baseline_stats["mean_ms"] / 1000)
        speedup = baseline_stats["mean_ms"] / opt_stats["mean_ms"]
        
        results.append({
            "kernel": "trajectory_resampling",
            "config": config,
            "optimized": opt_stats,
            "baseline": baseline_stats,
            "speedup": speedup,
            "throughput_opt_per_sec": throughput_opt,
            "throughput_baseline_per_sec": throughput_baseline,
        })
    
    return results


def main():
    """Run benchmark suite and emit JSON."""
    print("=" * 80)
    print("RoboCache Benchmark Harness")
    print("=" * 80)
    
    # Collect system info
    system_info = get_system_info()
    print(f"\n📊 System Configuration:")
    print(f"  Device: {system_info.get('device_name', 'N/A')}")
    print(f"  CUDA: {system_info.get('cuda_version', 'N/A')}")
    print(f"  Driver: {system_info.get('driver_version', 'N/A')}")
    print(f"  PyTorch: {system_info['pytorch_version']}")
    
    # Run benchmarks
    print(f"\n🔥 Running trajectory resampling benchmarks...")
    trajectory_results = benchmark_trajectory_resampling()
    
    # Compile results
    output = {
        "system": system_info,
        "benchmarks": {
            "trajectory_resampling": trajectory_results,
        }
    }
    
    # Write JSON
    os.makedirs("benchmarks/results", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmarks/results/benchmark_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results written to: {output_file}")
    
    # Print summary
    print(f"\n📈 Performance Summary:")
    for result in trajectory_results:
        config = result["config"]
        speedup = result["speedup"]
        opt_lat = result["optimized"]["mean_ms"]
        base_lat = result["baseline"]["mean_ms"]
        print(f"\n  Config: batch={config['batch']}, src={config['src_len']}, tgt={config['tgt_len']}, dim={config['dim']}")
        print(f"    Optimized: {opt_lat:.3f} ms")
        print(f"    Baseline:  {base_lat:.3f} ms")
        print(f"    Speedup:   {speedup:.2f}x")
    
    return 0


if __name__ == "__main__":
    exit(main())

