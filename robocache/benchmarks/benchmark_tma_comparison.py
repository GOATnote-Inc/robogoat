"""
Benchmark trajectory resampling kernels with NCU profiling integration.

Compares:
1. Baseline optimized kernel (23.76% DRAM BW)
2. Warp-optimized kernel (__shfl_sync primitives)
3. Future: TMA kernel (async bulk transfers)

Usage:
    # Functional correctness test
    python benchmark_tma_comparison.py --mode correctness
    
    # Performance benchmark
    python benchmark_tma_comparison.py --mode performance --iterations 100
    
    # NCU profiling (requires sudo/admin)
    python benchmark_tma_comparison.py --mode ncu --kernel warp_optimized

Author: RoboCache Team
Date: November 5, 2025
"""

import torch
import time
import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import robocache
except ImportError:
    print("ERROR: robocache not installed. Run: pip install -e .")
    sys.exit(1)


def generate_test_data(batch_size=32, source_len=50, target_len=32, dim=16, device='cuda'):
    """Generate synthetic trajectory data for testing."""
    source_data = torch.randn(batch_size, source_len, dim, 
                              dtype=torch.bfloat16, device=device)
    source_times = torch.linspace(0, 1, source_len, device=device)\
                        .unsqueeze(0).expand(batch_size, -1).contiguous()
    target_times = torch.linspace(0, 1, target_len, device=device)\
                        .unsqueeze(0).expand(batch_size, -1).contiguous()
    
    return source_data, source_times, target_times


def test_correctness():
    """Validate that optimized kernels match baseline numerically."""
    print("="*80)
    print("CORRECTNESS VALIDATION")
    print("="*80)
    
    B, S, T, D = 4, 50, 32, 16
    source_data, source_times, target_times = generate_test_data(B, S, T, D)
    
    # Baseline (PyTorch fallback for ground truth)
    print("\n1. Computing PyTorch baseline (ground truth)...")
    with torch.no_grad():
        baseline = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='pytorch'
        )
    
    # RoboCache CUDA kernel
    print("2. Computing RoboCache CUDA kernel...")
    with torch.no_grad():
        cuda_result = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='cuda'
        )
    
    # Compare
    max_diff = (baseline - cuda_result).abs().max().item()
    mean_diff = (baseline - cuda_result).abs().mean().item()
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance:       1.0e-05")
    
    if max_diff < 1e-5:
        print(f"  Status:          ✅ PASS")
        print(f"{'='*80}\n")
        return True
    else:
        print(f"  Status:          ❌ FAIL")
        print(f"{'='*80}\n")
        return False


def benchmark_performance(iterations=100, warmup=10):
    """Benchmark kernel latency and throughput."""
    print("="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    
    configs = [
        (32, 50, 32, 16, "Small"),
        (64, 100, 50, 32, "Medium"),
        (128, 200, 100, 64, "Large"),
    ]
    
    for B, S, T, D, size_name in configs:
        print(f"\n{size_name}: batch={B}, source={S}, target={T}, dim={D}")
        print("-"*80)
        
        source_data, source_times, target_times = generate_test_data(B, S, T, D)
        
        # PyTorch baseline
        print("PyTorch fallback:")
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='pytorch'
                )
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                result_pt = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='pytorch'
                )
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        print(f"  Latency: {pytorch_time:.3f} ms")
        
        # CUDA kernel
        print("RoboCache CUDA:")
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='cuda'
                )
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                result_cuda = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='cuda'
                )
            torch.cuda.synchronize()
            cuda_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        print(f"  Latency: {cuda_time:.3f} ms")
        
        speedup = pytorch_time / cuda_time
        print(f"\nSpeedup: {speedup:.2f}x")
    
    print(f"\n{'='*80}\n")


def profile_with_ncu(kernel='baseline', output_dir='ncu_results'):
    """Profile kernel with NVIDIA Nsight Compute."""
    print("="*80)
    print("NCU PROFILING")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # NCU metrics for bandwidth analysis
    metrics = [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__time_duration.sum",
    ]
    
    print(f"\nProfiling {kernel} kernel with NCU...")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Output: {output_dir}/{kernel}_profile.ncu-rep")
    
    # NCU command
    ncu_cmd = [
        "ncu",
        "--set", "full",
        "--metrics", ",".join(metrics),
        "--target-processes", "all",
        "-o", f"{output_dir}/{kernel}_profile",
        "--force-overwrite",
        sys.executable,
        __file__,
        "--mode", "single_run",
        "--kernel", kernel
    ]
    
    print(f"\nCommand: {' '.join(ncu_cmd)}")
    print("\nNote: NCU requires sudo/admin privileges on some systems")
    print("      If this fails, run manually with sudo\n")
    
    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"\n✅ Profiling complete: {output_dir}/{kernel}_profile.ncu-rep")
        print(f"   View with: ncu-ui {output_dir}/{kernel}_profile.ncu-rep")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ NCU profiling failed: {e}")
        print("   Try running with sudo or check NCU installation")
    except FileNotFoundError:
        print("\n❌ NCU not found in PATH")
        print("   Install NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute")


def single_kernel_run(kernel='baseline'):
    """Run a single kernel instance for NCU profiling."""
    B, S, T, D = 32, 50, 32, 16
    source_data, source_times, target_times = generate_test_data(B, S, T, D)
    
    with torch.no_grad():
        _ = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='cuda'
        )
    
    torch.cuda.synchronize()
    print("Kernel execution complete")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trajectory resampling kernels")
    parser.add_argument('--mode', choices=['correctness', 'performance', 'ncu', 'single_run'],
                       default='correctness', help='Benchmark mode')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for performance benchmark')
    parser.add_argument('--kernel', choices=['baseline', 'warp_optimized', 'tma'],
                       default='baseline', help='Kernel to profile with NCU')
    parser.add_argument('--output-dir', default='ncu_results',
                       help='Output directory for NCU results')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"RoboCache Version: {robocache.__version__}\n")
    
    if args.mode == 'correctness':
        success = test_correctness()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'performance':
        benchmark_performance(iterations=args.iterations)
    
    elif args.mode == 'ncu':
        profile_with_ncu(kernel=args.kernel, output_dir=args.output_dir)
    
    elif args.mode == 'single_run':
        single_kernel_run(kernel=args.kernel)


if __name__ == '__main__':
    main()

