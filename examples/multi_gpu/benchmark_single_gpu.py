#!/usr/bin/env python3
"""
Single GPU Baseline Benchmarks

Measures baseline performance on each GPU independently before multi-GPU testing.

Usage:
    # H100 baseline
    python benchmark_single_gpu.py --gpu 0 --arch h100 --output h100_single.json
    
    # A100 baseline
    python benchmark_single_gpu.py --gpu 1 --arch a100 --output a100_single.json
"""

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import torch

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False
    print("ERROR: RoboCache not installed")
    exit(1)


class BenchmarkSuite:
    """Comprehensive single-GPU benchmarks"""
    
    def __init__(self, gpu_id: int, arch: str):
        self.gpu_id = gpu_id
        self.arch = arch
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Warm up
        torch.cuda.set_device(self.device)
        _ = torch.randn(1000, 1000, device=self.device) @ torch.randn(1000, 1000, device=self.device)
        torch.cuda.synchronize(self.device)
        
        print(f"\n{'='*70}")
        print(f"Single GPU Benchmark - {self.arch.upper()}")
        print(f"GPU ID: {gpu_id}")
        print(f"Device: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(gpu_id)}")
        print(f"{'='*70}\n")
    
    def benchmark_multimodal_fusion(self, num_iters: int = 200) -> Dict:
        """Benchmark multimodal fusion"""
        print(f"[1/3] Multimodal Fusion Benchmark ({num_iters} iterations)...")
        
        # Create test data
        batch_size = 4
        vision = torch.randn(batch_size, 30, 512, dtype=torch.bfloat16, device=self.device)
        vision_times = torch.linspace(0, 1, 30, device=self.device).expand(batch_size, -1)
        
        proprio = torch.randn(batch_size, 100, 64, dtype=torch.bfloat16, device=self.device)
        proprio_times = torch.linspace(0, 1, 100, device=self.device).expand(batch_size, -1)
        
        imu = torch.randn(batch_size, 200, 12, dtype=torch.bfloat16, device=self.device)
        imu_times = torch.linspace(0, 1, 200, device=self.device).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 1, 50, device=self.device).expand(batch_size, -1)
        
        # Warm-up
        for _ in range(20):
            _ = robocache.fuse_multimodal(
                vision, vision_times,
                proprio, proprio_times,
                imu, imu_times,
                target_times
            )
        torch.cuda.synchronize(self.device)
        
        # Benchmark
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            out = robocache.fuse_multimodal(
                vision, vision_times,
                proprio, proprio_times,
                imu, imu_times,
                target_times
            )
            torch.cuda.synchronize(self.device)
            times.append(time.perf_counter() - start)
        
        return {
            'p50_latency_ms': float(np.percentile(times, 50) * 1000),
            'p99_latency_ms': float(np.percentile(times, 99) * 1000),
            'mean_latency_ms': float(np.mean(times) * 1000),
            'std_latency_ms': float(np.std(times) * 1000),
            'throughput_batches_per_sec': 1.0 / np.mean(times),
            'output_shape': list(out.shape),
        }
    
    def benchmark_voxelization(self, num_iters: int = 100) -> Dict:
        """Benchmark point cloud voxelization"""
        print(f"[2/3] Voxelization Benchmark ({num_iters} iterations)...")
        
        # Create test data
        num_points = 500000
        points = torch.rand(num_points, 3, device=self.device, dtype=torch.float32) * 4.0 - 2.0
        
        grid_min = [-2.0, -2.0, -2.0]
        voxel_size = 0.0625  # 128^3 grid over 4m span
        grid_size = [128, 128, 128]
        
        modes = ['count', 'occupancy', 'mean', 'max']
        results = {}
        
        for mode in modes:
            print(f"  Mode: {mode}...")
            
            # For mean/max, need features
            features = torch.randn(num_points, 8, device=self.device, dtype=torch.float32) if mode in ['mean', 'max'] else None
            
            # Warm-up
            for _ in range(10):
                _ = robocache.voxelize_pointcloud(
                    points,
                    features=features,
                    grid_min=grid_min,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                    mode=mode
                )
            torch.cuda.synchronize(self.device)
            
            # Benchmark
            times = []
            for _ in range(num_iters):
                start = time.perf_counter()
                out = robocache.voxelize_pointcloud(
                    points,
                    features=features,
                    grid_min=grid_min,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                    mode=mode
                )
                torch.cuda.synchronize(self.device)
                times.append(time.perf_counter() - start)
            
            throughput = num_points / np.mean(times) / 1e9  # Billion points/sec
            
            results[mode] = {
                'p50_latency_ms': float(np.percentile(times, 50) * 1000),
                'p99_latency_ms': float(np.percentile(times, 99) * 1000),
                'mean_latency_ms': float(np.mean(times) * 1000),
                'throughput_billion_pts_per_sec': float(throughput),
                'output_shape': list(out.shape),
            }
        
        return results
    
    def benchmark_memory_bandwidth(self) -> Dict:
        """Measure memory bandwidth"""
        print(f"[3/3] Memory Bandwidth Benchmark...")
        
        # Large tensor copy to measure bandwidth
        size = 100_000_000  # 100M floats = 400MB
        src = torch.randn(size, device=self.device, dtype=torch.float32)
        
        # Warm-up
        for _ in range(10):
            dst = src.clone()
        torch.cuda.synchronize(self.device)
        
        # Benchmark copy
        times = []
        for _ in range(50):
            start = time.perf_counter()
            dst = src.clone()
            torch.cuda.synchronize(self.device)
            times.append(time.perf_counter() - start)
        
        bytes_transferred = size * 4 * 2  # float32 * read + write
        bandwidth_gb_s = bytes_transferred / np.mean(times) / 1e9
        
        return {
            'bandwidth_gb_per_sec': float(bandwidth_gb_s),
            'latency_ms': float(np.mean(times) * 1000),
        }
    
    def run_all(self) -> Dict:
        """Run all benchmarks"""
        results = {
            'gpu_id': self.gpu_id,
            'architecture': self.arch,
            'device_name': torch.cuda.get_device_name(self.gpu_id),
            'compute_capability': f"{torch.cuda.get_device_capability(self.gpu_id)[0]}.{torch.cuda.get_device_capability(self.gpu_id)[1]}",
            'memory_total_gb': torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9,
        }
        
        results['multimodal_fusion'] = self.benchmark_multimodal_fusion()
        results['voxelization'] = self.benchmark_voxelization()
        results['memory_bandwidth'] = self.benchmark_memory_bandwidth()
        
        return results


def print_summary(results: Dict):
    """Print benchmark summary"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY - {results['architecture'].upper()}")
    print(f"{'='*70}\n")
    
    print(f"GPU: {results['device_name']}")
    print(f"Compute Capability: {results['compute_capability']}")
    print(f"Memory: {results['memory_total_gb']:.1f} GB")
    
    print(f"\n📊 Multimodal Fusion:")
    mf = results['multimodal_fusion']
    print(f"  P50 Latency: {mf['p50_latency_ms']:.3f} ms")
    print(f"  P99 Latency: {mf['p99_latency_ms']:.3f} ms")
    print(f"  Throughput: {mf['throughput_batches_per_sec']:.1f} batches/sec")
    
    print(f"\n📊 Voxelization:")
    for mode, data in results['voxelization'].items():
        print(f"  {mode.upper():>10}: {data['p50_latency_ms']:.3f} ms | {data['throughput_billion_pts_per_sec']:.2f} B pts/sec")
    
    print(f"\n📊 Memory Bandwidth:")
    bw = results['memory_bandwidth']
    print(f"  Measured: {bw['bandwidth_gb_per_sec']:.1f} GB/s")
    
    # Theoretical peak bandwidth
    if 'H100' in results['device_name']:
        theoretical = 3350
    elif 'A100' in results['device_name']:
        theoretical = 1935
    else:
        theoretical = None
    
    if theoretical:
        efficiency = bw['bandwidth_gb_per_sec'] / theoretical * 100
        print(f"  Theoretical: {theoretical} GB/s")
        print(f"  Efficiency: {efficiency:.1f}%")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Single GPU baseline benchmarks')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU ID to benchmark')
    parser.add_argument('--arch', type=str, required=True, choices=['h100', 'a100'],
                        help='GPU architecture (for labeling)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Validate GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    if args.gpu >= torch.cuda.device_count():
        print(f"ERROR: GPU {args.gpu} not found (only {torch.cuda.device_count()} GPUs available)")
        return 1
    
    # Run benchmarks
    suite = BenchmarkSuite(args.gpu, args.arch)
    results = suite.run_all()
    
    # Print summary
    print_summary(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())

