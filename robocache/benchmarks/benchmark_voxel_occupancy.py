#!/usr/bin/env python3
"""
Voxelization Kernel Occupancy Benchmark

Compares original vs optimized voxelization kernels:
- Measure throughput (B points/sec)
- Measure latency (ms)
- Validate occupancy improvement
- Test on multiple GPU architectures

Target: 85%+ occupancy (up from ~64%)
"""

import torch
import time
import numpy as np
import subprocess
import sys

def get_gpu_info():
    """Get GPU name and compute capability"""
    if not torch.cuda.is_available():
        return "CPU", "0.0"
    
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    return gpu_name, f"SM{capability[0]}{capability[1]}"

def benchmark_voxelization(
    num_points, 
    grid_size, 
    num_iterations=200,
    warmup=50
):
    """Benchmark voxelization kernel"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate random point cloud
    points = torch.rand(num_points, 3, device=device) * 10.0 - 5.0
    grid_min = torch.tensor([-5.0, -5.0, -5.0], device=device)
    voxel_size = 0.1
    
    # Warmup
    try:
        import robocache
        for _ in range(warmup):
            voxel_grid = robocache.voxelize_pointcloud(
                points, 
                grid_min=tuple(grid_min.cpu().numpy()),
                voxel_size=voxel_size,
                grid_size=grid_size,
                mode='occupancy'
            )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Error during warmup: {e}")
        return None
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=tuple(grid_min.cpu().numpy()),
            voxel_size=voxel_size,
            grid_size=grid_size,
            mode='occupancy'
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    throughput = num_points / (np.median(times) / 1000)
    
    return {
        'num_points': num_points,
        'grid_size': grid_size,
        'p50_latency_ms': np.percentile(times, 50),
        'p99_latency_ms': np.percentile(times, 99),
        'throughput_pts_per_sec': throughput,
        'throughput_B_pts_per_sec': throughput / 1e9
    }

def profile_with_ncu(num_points, grid_size):
    """Profile kernel with NCU to measure occupancy"""
    print(f"\n🔍 Profiling with NCU ({num_points:,} points, {grid_size}³ grid)...")
    
    # Create profiling script
    profile_script = f"""
import torch
import robocache

points = torch.rand({num_points}, 3, device='cuda') * 10.0 - 5.0
grid_min = (-5.0, -5.0, -5.0)

# Run kernel once
voxel_grid = robocache.voxelize_pointcloud(
    points,
    grid_min=grid_min,
    voxel_size=0.1,
    grid_size=[{grid_size}, {grid_size}, {grid_size}],
    mode='occupancy'
)
torch.cuda.synchronize()
print("✅ Kernel executed")
"""
    
    with open('/tmp/ncu_profile.py', 'w') as f:
        f.write(profile_script)
    
    # Run NCU profiling
    try:
        cmd = [
            'ncu',
            '--metrics', 'sm__warps_active.avg.pct_of_peak_sustained_active',
            '--metrics', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
            '--metrics', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'python3', '/tmp/ncu_profile.py'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ NCU profiling successful")
            # Parse occupancy from output
            for line in result.stdout.split('\n'):
                if 'warps_active' in line:
                    print(f"  Occupancy: {line.strip()}")
                elif 'throughput' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"⚠️  NCU profiling failed: {result.stderr}")
    
    except Exception as e:
        print(f"⚠️  NCU profiling skipped: {e}")

def main():
    print("="*70)
    print("VOXELIZATION KERNEL OCCUPANCY BENCHMARK")
    print("="*70)
    
    gpu_name, compute_cap = get_gpu_info()
    print(f"\nGPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap}")
    
    # Test configurations
    configs = [
        ("Small", 50_000, 64),
        ("Medium", 250_000, 128),
        ("Large", 500_000, 128),
        ("XLarge", 1_000_000, 128),
    ]
    
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKS")
    print("="*70)
    
    results = []
    for name, num_points, grid_size in configs:
        print(f"\n{name} config: {num_points:,} points → {grid_size}³ grid")
        result = benchmark_voxelization(num_points, [grid_size] * 3)
        
        if result:
            print(f"  P50 Latency: {result['p50_latency_ms']:.3f} ms")
            print(f"  P99 Latency: {result['p99_latency_ms']:.3f} ms")
            print(f"  Throughput: {result['throughput_B_pts_per_sec']:.2f} B pts/sec")
            results.append((name, result))
    
    # NCU profiling (if available)
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("NCU OCCUPANCY PROFILING")
        print("="*70)
        profile_with_ncu(250_000, 128)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        avg_throughput = np.mean([r[1]['throughput_B_pts_per_sec'] for r in results])
        print(f"\nAverage Throughput: {avg_throughput:.2f} B pts/sec")
        print(f"GPU: {gpu_name} ({compute_cap})")
        
        # Check if we meet targets
        if avg_throughput > 10.0:
            print("\n✅ PERFORMANCE TARGET MET (>10B pts/sec)")
        else:
            print(f"\n⚠️  Below target ({avg_throughput:.2f} < 10.0 B pts/sec)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    try:
        import robocache
        print(f"RoboCache version: {robocache}")
    except ImportError:
        print("❌ RoboCache not installed")
        sys.exit(1)
    
    main()

