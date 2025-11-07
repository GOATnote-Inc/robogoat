#!/usr/bin/env python3
"""
Multi-GPU Scaling Benchmarks

Tests RoboCache performance scaling across multiple GPUs with PyTorch DDP.

Usage:
    # 2x H100 with NVLink
    torchrun --nproc_per_node=2 benchmark_multi_gpu.py --arch h100 --output h100_2gpu.json
    
    # 2x A100 with NVLink
    torchrun --nproc_per_node=2 benchmark_multi_gpu.py --arch a100 --output a100_2gpu.json
    
    # 4 GPUs
    torchrun --nproc_per_node=4 benchmark_multi_gpu.py --arch mixed --output 4gpu.json
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False
    print("ERROR: RoboCache not installed")
    exit(1)


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size


def check_nvlink(local_rank: int, world_size: int) -> bool:
    """Check if GPUs are connected via NVLink"""
    if world_size == 1:
        return False
    
    # Check NVLINK topology
    # This is a simplified check - real implementation would parse nvidia-smi topo -m
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', 'topo', '-m'],
            capture_output=True, text=True
        )
        # Look for NV* entries (NVLink connections)
        has_nvlink = 'NV' in result.stdout and 'GPU' in result.stdout
        return has_nvlink
    except:
        return False


class MultiGPUBenchmark:
    """Multi-GPU performance benchmarks"""
    
    def __init__(self, local_rank: int, world_size: int, arch: str):
        self.local_rank = local_rank
        self.world_size = world_size
        self.arch = arch
        self.device = torch.device(f'cuda:{local_rank}')
        
        if local_rank == 0:
            print(f"\n{'='*70}")
            print(f"Multi-GPU Benchmark - {world_size} GPUs")
            print(f"Architecture: {arch.upper()}")
            print(f"{'='*70}\n")
            
            # Check for NVLink
            has_nvlink = check_nvlink(local_rank, world_size)
            print(f"NVLink: {'✓ Detected' if has_nvlink else '✗ Not detected (using PCIe)'}")
            
            for rank in range(world_size):
                device_name = torch.cuda.get_device_name(rank)
                print(f"GPU {rank}: {device_name}")
            print()
    
    def benchmark_multimodal_fusion_ddp(self, num_iters: int = 200) -> Dict:
        """Benchmark multimodal fusion with DDP"""
        if self.local_rank == 0:
            print(f"[1/3] Multimodal Fusion DDP ({num_iters} iterations)...")
        
        # Each GPU processes its own batch
        batch_size = 4  # Per GPU
        vision = torch.randn(batch_size, 30, 512, dtype=torch.bfloat16, device=self.device)
        vision_times = torch.linspace(0, 1, 30, device=self.device).expand(batch_size, -1)
        
        proprio = torch.randn(batch_size, 100, 64, dtype=torch.bfloat16, device=self.device)
        proprio_times = torch.linspace(0, 1, 100, device=self.device).expand(batch_size, -1)
        
        imu = torch.randn(batch_size, 200, 12, dtype=torch.bfloat16, device=self.device)
        imu_times = torch.linspace(0, 1, 200, device=self.device).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 1, 50, device=self.device).expand(batch_size, -1)
        
        # Warm-up
        for _ in range(20):
            out = robocache.fuse_multimodal(
                vision, vision_times,
                proprio, proprio_times,
                imu, imu_times,
                target_times
            )
            # Simulate DDP all-reduce
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize(self.device)
        dist.barrier()
        
        # Benchmark
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            
            # Local computation
            out = robocache.fuse_multimodal(
                vision, vision_times,
                proprio, proprio_times,
                imu, imu_times,
                target_times
            )
            
            # DDP communication
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            
            torch.cuda.synchronize(self.device)
            times.append(time.perf_counter() - start)
        
        dist.barrier()
        
        # Gather results to rank 0
        times_tensor = torch.tensor(times, device=self.device)
        gathered = [torch.zeros_like(times_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, times_tensor)
        
        if self.local_rank == 0:
            # Use slowest GPU time (bottleneck)
            all_times = [t.cpu().numpy() for t in gathered]
            max_times = np.max(all_times, axis=0)  # Slowest GPU per iteration
            
            return {
                'p50_latency_ms': float(np.percentile(max_times, 50) * 1000),
                'p99_latency_ms': float(np.percentile(max_times, 99) * 1000),
                'mean_latency_ms': float(np.mean(max_times) * 1000),
                'std_latency_ms': float(np.std(max_times) * 1000),
                'throughput_batches_per_sec': self.world_size * batch_size / np.mean(max_times),
            }
        else:
            return {}
    
    def benchmark_voxelization_ddp(self, num_iters: int = 100) -> Dict:
        """Benchmark voxelization with DDP"""
        if self.local_rank == 0:
            print(f"[2/3] Voxelization DDP ({num_iters} iterations, occupancy mode)...")
        
        # Each GPU processes different point cloud
        num_points = 500000
        points = torch.rand(num_points, 3, device=self.device, dtype=torch.float32) * 4.0 - 2.0
        
        grid_min = [-2.0, -2.0, -2.0]
        voxel_size = 0.0625
        grid_size = [128, 128, 128]
        
        # Warm-up
        for _ in range(10):
            out = robocache.voxelize_pointcloud(
                points,
                grid_min=grid_min,
                voxel_size=voxel_size,
                grid_size=grid_size,
                mode='occupancy'
            )
            # Aggregate voxel grids from all GPUs
            dist.all_reduce(out, op=dist.ReduceOp.MAX)
        
        torch.cuda.synchronize(self.device)
        dist.barrier()
        
        # Benchmark
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            
            out = robocache.voxelize_pointcloud(
                points,
                grid_min=grid_min,
                voxel_size=voxel_size,
                grid_size=grid_size,
                mode='occupancy'
            )
            
            dist.all_reduce(out, op=dist.ReduceOp.MAX)
            
            torch.cuda.synchronize(self.device)
            times.append(time.perf_counter() - start)
        
        dist.barrier()
        
        # Gather results
        times_tensor = torch.tensor(times, device=self.device)
        gathered = [torch.zeros_like(times_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, times_tensor)
        
        if self.local_rank == 0:
            all_times = [t.cpu().numpy() for t in gathered]
            max_times = np.max(all_times, axis=0)
            
            total_points = num_points * self.world_size
            throughput = total_points / np.mean(max_times) / 1e9
            
            return {
                'p50_latency_ms': float(np.percentile(max_times, 50) * 1000),
                'p99_latency_ms': float(np.percentile(max_times, 99) * 1000),
                'mean_latency_ms': float(np.mean(max_times) * 1000),
                'throughput_billion_pts_per_sec': float(throughput),
            }
        else:
            return {}
    
    def benchmark_communication_overhead(self) -> Dict:
        """Measure DDP communication overhead"""
        if self.local_rank == 0:
            print(f"[3/3] Communication Overhead...")
        
        sizes = [1_000_000, 10_000_000, 100_000_000]  # 1M, 10M, 100M elements
        results = {}
        
        for size in sizes:
            tensor = torch.randn(size, device=self.device, dtype=torch.float32)
            
            # Warm-up
            for _ in range(10):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(self.device)
            dist.barrier()
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.perf_counter()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(self.device)
                times.append(time.perf_counter() - start)
            
            dist.barrier()
            
            if self.local_rank == 0:
                bytes_transferred = size * 4  # float32
                bandwidth_gb_s = bytes_transferred / np.mean(times) / 1e9
                
                results[f'{size//1_000_000}M'] = {
                    'latency_ms': float(np.mean(times) * 1000),
                    'bandwidth_gb_per_sec': float(bandwidth_gb_s),
                }
        
        return results if self.local_rank == 0 else {}
    
    def run_all(self) -> Dict:
        """Run all benchmarks"""
        if self.local_rank == 0:
            results = {
                'world_size': self.world_size,
                'architecture': self.arch,
                'devices': [torch.cuda.get_device_name(i) for i in range(self.world_size)],
            }
        else:
            results = {}
        
        mf_results = self.benchmark_multimodal_fusion_ddp()
        if self.local_rank == 0:
            results['multimodal_fusion'] = mf_results
        
        vox_results = self.benchmark_voxelization_ddp()
        if self.local_rank == 0:
            results['voxelization'] = vox_results
        
        comm_results = self.benchmark_communication_overhead()
        if self.local_rank == 0:
            results['communication_overhead'] = comm_results
        
        return results


def print_summary(results: Dict, single_gpu_results: Dict = None):
    """Print benchmark summary"""
    print(f"\n{'='*70}")
    print(f"{results['world_size']}-GPU BENCHMARK SUMMARY")
    print(f"{'='*70}\n")
    
    for i, device in enumerate(results['devices']):
        print(f"GPU {i}: {device}")
    
    print(f"\n📊 Multimodal Fusion (DDP):")
    mf = results['multimodal_fusion']
    print(f"  P50 Latency: {mf['p50_latency_ms']:.3f} ms")
    print(f"  P99 Latency: {mf['p99_latency_ms']:.3f} ms")
    print(f"  Throughput: {mf['throughput_batches_per_sec']:.1f} batches/sec")
    
    if single_gpu_results:
        single_throughput = single_gpu_results['multimodal_fusion']['throughput_batches_per_sec']
        speedup = mf['throughput_batches_per_sec'] / single_throughput
        efficiency = speedup / results['world_size'] * 100
        print(f"  Speedup vs 1 GPU: {speedup:.2f}x")
        print(f"  Scaling Efficiency: {efficiency:.1f}%")
    
    print(f"\n📊 Voxelization (DDP, occupancy mode):")
    vox = results['voxelization']
    print(f"  P50 Latency: {vox['p50_latency_ms']:.3f} ms")
    print(f"  Throughput: {vox['throughput_billion_pts_per_sec']:.2f} B pts/sec")
    
    if single_gpu_results:
        single_throughput = single_gpu_results['voxelization']['occupancy']['throughput_billion_pts_per_sec']
        speedup = vox['throughput_billion_pts_per_sec'] / single_throughput
        efficiency = speedup / results['world_size'] * 100
        print(f"  Speedup vs 1 GPU: {speedup:.2f}x")
        print(f"  Scaling Efficiency: {efficiency:.1f}%")
    
    print(f"\n📊 Communication Overhead (all_reduce):")
    for size, data in results['communication_overhead'].items():
        print(f"  {size:>4}: {data['latency_ms']:.2f} ms | {data['bandwidth_gb_per_sec']:.1f} GB/s")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU scaling benchmarks')
    parser.add_argument('--arch', type=str, required=True,
                        help='GPU architecture (for labeling)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--single-gpu-results', type=str, default=None,
                        help='Single GPU baseline JSON for comparison')
    
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, world_size = setup_distributed()
    
    # Load single GPU results if provided
    single_gpu_results = None
    if args.single_gpu_results and local_rank == 0:
        with open(args.single_gpu_results, 'r') as f:
            single_gpu_results = json.load(f)
    
    # Run benchmarks
    benchmark = MultiGPUBenchmark(local_rank, world_size, args.arch)
    results = benchmark.run_all()
    
    # Print and save results (rank 0 only)
    if local_rank == 0:
        print_summary(results, single_gpu_results)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {args.output}")
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0


if __name__ == '__main__':
    exit(main())

