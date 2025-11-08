#!/usr/bin/env python3
"""
Multi-GPU DDP Scaling Benchmark

Validates scaling efficiency of multi-GPU preprocessing:
- Tests 1, 2, 4, 8 GPUs
- Measures throughput and latency
- Calculates scaling efficiency
- Validates NCCL communication overhead

Expected results (H100 cluster with NVLink):
  1 GPU:  2.5 GB/sec baseline
  2 GPUs: 4.8 GB/sec (96% efficiency)
  4 GPUs: 9.2 GB/sec (92% efficiency)
  8 GPUs: 17.1 GB/sec (85% efficiency)
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np
from typing import List, Dict

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False
    print("⚠️  RoboCache not available")


def benchmark_single_gpu(
    rank: int,
    world_size: int,
    num_points: int,
    grid_size: int,
    num_iterations: int,
    results_queue
):
    """Worker function for each GPU"""
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    
    if world_size > 1:
        # Initialize DDP
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:12356',
            world_size=world_size,
            rank=rank
        )
    
    # Generate test data
    points_per_gpu = num_points // world_size
    points = torch.rand(points_per_gpu, 3, device=device) * 10.0 - 5.0
    
    # Warmup
    for _ in range(10):
        if ROBOCACHE_AVAILABLE:
            voxel_grid = robocache.voxelize_pointcloud(
                points,
                grid_min=[-5.0, -5.0, -5.0],
                voxel_size=0.1,
                grid_size=[grid_size, grid_size, grid_size],
                mode='occupancy',
                backend='cuda'
            )
        else:
            voxel_grid = torch.zeros((grid_size, grid_size, grid_size), device=device)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        # Voxelization
        if ROBOCACHE_AVAILABLE:
            voxel_grid = robocache.voxelize_pointcloud(
                points,
                grid_min=[-5.0, -5.0, -5.0],
                voxel_size=0.1,
                grid_size=[grid_size, grid_size, grid_size],
                mode='occupancy',
                backend='cuda'
            )
        else:
            voxel_grid = torch.zeros((grid_size, grid_size, grid_size), device=device)
        
        # All-gather (if multi-GPU)
        if world_size > 1:
            gathered = [torch.zeros_like(voxel_grid) for _ in range(world_size)]
            dist.all_gather(gathered, voxel_grid)
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    # Report results
    if rank == 0:
        times = np.array(times) * 1000  # ms
        avg_latency = np.mean(times)
        p50_latency = np.percentile(times, 50)
        p99_latency = np.percentile(times, 99)
        throughput_pts_sec = num_points / (avg_latency / 1000)
        throughput_gb_sec = (num_points * 3 * 4) / (avg_latency / 1000) / 1e9  # float32
        
        results_queue.put({
            'num_gpus': world_size,
            'num_points': num_points,
            'grid_size': grid_size,
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p99_latency_ms': p99_latency,
            'throughput_pts_sec': throughput_pts_sec,
            'throughput_gb_sec': throughput_gb_sec
        })


def run_benchmark(num_gpus: int, num_points: int, grid_size: int, num_iterations: int):
    """Run benchmark with specified number of GPUs"""
    import multiprocessing as std_mp
    results_queue = std_mp.Queue()
    
    if num_gpus == 1:
        # Single GPU (no DDP)
        benchmark_single_gpu(0, 1, num_points, grid_size, num_iterations, results_queue)
    else:
        # Multi-GPU with DDP
        mp.spawn(
            benchmark_single_gpu,
            args=(num_gpus, num_points, grid_size, num_iterations, results_queue),
            nprocs=num_gpus,
            join=True
        )
    
    # Get results
    if not results_queue.empty():
        return results_queue.get()
    return None


def main():
    print("="*70)
    print("MULTI-GPU DDP SCALING BENCHMARK")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    num_available_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_available_gpus}")
    for i in range(num_available_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if not ROBOCACHE_AVAILABLE:
        print("\n⚠️  RoboCache not available, using dummy benchmark")
    
    # Test configuration
    num_points = 500_000
    grid_size = 128
    num_iterations = 100
    
    print(f"\nTest Configuration:")
    print(f"  Points per batch: {num_points:,}")
    print(f"  Grid size: {grid_size}³")
    print(f"  Iterations: {num_iterations}")
    
    # Run benchmarks for 1, 2, 4, 8 GPUs
    gpu_configs = [1, 2, 4, 8]
    gpu_configs = [n for n in gpu_configs if n <= num_available_gpus]
    
    print("\n" + "="*70)
    print("SCALING BENCHMARKS")
    print("="*70)
    
    results = []
    baseline_throughput = None
    
    for num_gpus in gpu_configs:
        print(f"\n🔄 Testing with {num_gpus} GPU(s)...")
        result = run_benchmark(num_gpus, num_points, grid_size, num_iterations)
        
        if result:
            results.append(result)
            
            if baseline_throughput is None:
                baseline_throughput = result['throughput_gb_sec']
            
            scaling_efficiency = (result['throughput_gb_sec'] / baseline_throughput / num_gpus) * 100
            
            print(f"  ✅ Completed")
            print(f"     Latency (P50): {result['p50_latency_ms']:.2f} ms")
            print(f"     Throughput: {result['throughput_gb_sec']:.2f} GB/sec")
            print(f"     Scaling Efficiency: {scaling_efficiency:.1f}%")
    
    # Summary table
    print("\n" + "="*70)
    print("SCALING SUMMARY")
    print("="*70)
    
    if results:
        print("\n| GPUs | Latency (P50) | Throughput | Scaling Efficiency |")
        print("|------|---------------|------------|-------------------|")
        
        for result in results:
            num_gpus = result['num_gpus']
            latency = result['p50_latency_ms']
            throughput = result['throughput_gb_sec']
            efficiency = (throughput / baseline_throughput / num_gpus) * 100
            
            print(f"| {num_gpus:4d} | {latency:11.2f} ms | {throughput:8.2f} GB/s | {efficiency:16.1f}% |")
        
        # Final assessment
        print("\n" + "="*70)
        print("ASSESSMENT")
        print("="*70)
        
        if len(results) >= 2:
            two_gpu_efficiency = (results[1]['throughput_gb_sec'] / baseline_throughput / 2) * 100
            
            if two_gpu_efficiency > 90:
                print("\n✅ EXCELLENT: >90% scaling efficiency with 2 GPUs")
            elif two_gpu_efficiency > 80:
                print("\n✅ GOOD: >80% scaling efficiency with 2 GPUs")
            elif two_gpu_efficiency > 70:
                print("\n⚠️  ACCEPTABLE: >70% scaling efficiency with 2 GPUs")
            else:
                print(f"\n❌ POOR: {two_gpu_efficiency:.1f}% scaling efficiency")
                print("   Possible causes:")
                print("   - No NVLink (high inter-GPU communication latency)")
                print("   - Batch size too small (communication overhead dominates)")
                print("   - CPU bottleneck in data loading")
        
        # Check for NVLink
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '--status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'Active' in result.stdout:
                print("\n✅ NVLink detected and active")
            else:
                print("\n⚠️  NVLink not detected - using PCIe (slower)")
        except:
            print("\n⚠️  Could not check NVLink status")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

