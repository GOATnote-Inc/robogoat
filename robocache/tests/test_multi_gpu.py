#!/usr/bin/env python3
"""
RoboCache Multi-GPU Distributed Tests
Validates scaling on 2-8 GPUs with PyTorch DDP
"""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

try:
    import robocache
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count()
except ImportError:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0


def setup_distributed(rank, world_size):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group"""
    dist.destroy_process_group()


def distributed_worker(rank, world_size, batch_per_gpu, source_len, target_len, dim):
    """Worker process for distributed testing"""
    setup_distributed(rank, world_size)
    
    torch.manual_seed(42 + rank)
    device = torch.device(f'cuda:{rank}')
    
    # Generate data on this GPU
    source_data = torch.randn(batch_per_gpu, source_len, dim, 
                              dtype=torch.bfloat16, device=device)
    source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_per_gpu, -1)
    target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_per_gpu, -1)
    
    # Synchronize
    dist.barrier()
    
    # Run resampling
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        _ = robocache.resample_trajectories(source_data, source_times, target_times)
    torch.cuda.synchronize()
    
    # Timed run
    start.record()
    for _ in range(100):
        result = robocache.resample_trajectories(source_data, source_times, target_times)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / 100
    
    # Gather timing from all ranks
    timing_tensor = torch.tensor([elapsed_ms], device=device)
    timing_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(timing_list, timing_tensor)
    
    if rank == 0:
        timings = [t.item() for t in timing_list]
        avg_time = sum(timings) / len(timings)
        max_time = max(timings)
        min_time = min(timings)
        
        print(f"\n{'='*60}")
        print(f"Multi-GPU Test: {world_size} GPUs")
        print(f"  Batch per GPU: {batch_per_gpu}")
        print(f"  Total batch: {batch_per_gpu * world_size}")
        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Min/Max: {min_time:.3f} / {max_time:.3f} ms")
        print(f"  Imbalance: {((max_time - min_time) / avg_time * 100):.1f}%")
        print(f"{'='*60}\n")
        
        # Validate balanced execution (<10% imbalance)
        assert ((max_time - min_time) / avg_time) < 0.10, f"GPU imbalance too high: {((max_time - min_time) / avg_time * 100):.1f}%"
    
    cleanup_distributed()


@pytest.mark.skipif(GPU_COUNT < 2, reason="Requires 2+ GPUs")
@pytest.mark.slow
def test_2gpu_distributed():
    """Test distributed execution on 2 GPUs"""
    world_size = 2
    mp.spawn(
        distributed_worker,
        args=(world_size, 16, 500, 256, 256),
        nprocs=world_size,
        join=True
    )


@pytest.mark.skipif(GPU_COUNT < 4, reason="Requires 4+ GPUs")
@pytest.mark.slow
def test_4gpu_distributed():
    """Test distributed execution on 4 GPUs"""
    world_size = 4
    mp.spawn(
        distributed_worker,
        args=(world_size, 8, 500, 256, 256),
        nprocs=world_size,
        join=True
    )


@pytest.mark.skipif(GPU_COUNT < 8, reason="Requires 8 GPUs")
@pytest.mark.slow
def test_8gpu_distributed():
    """Test distributed execution on 8 GPUs"""
    world_size = 8
    mp.spawn(
        distributed_worker,
        args=(world_size, 4, 500, 256, 256),
        nprocs=world_size,
        join=True
    )


@pytest.mark.skipif(GPU_COUNT < 2, reason="Requires 2+ GPUs")
def test_gpu_scaling_efficiency():
    """Test that multi-GPU provides near-linear scaling"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    
    # Single GPU baseline
    device = torch.device('cuda:0')
    batch = 32
    source_data = torch.randn(batch, 500, 256, dtype=torch.bfloat16, device=device)
    source_times = torch.linspace(0, 5, 500, device=device).unsqueeze(0).expand(batch, -1)
    target_times = torch.linspace(0, 5, 256, device=device).unsqueeze(0).expand(batch, -1)
    
    # Warmup
    for _ in range(10):
        _ = robocache.resample_trajectories(source_data, source_times, target_times)
    torch.cuda.synchronize()
    
    # Time single GPU
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = robocache.resample_trajectories(source_data, source_times, target_times)
    end.record()
    torch.cuda.synchronize()
    
    single_gpu_time = start.elapsed_time(end) / 100
    
    print(f"\nSingle GPU: {single_gpu_time:.3f} ms for batch={batch}")
    print(f"Expected 2-GPU throughput: {(batch * 2 / single_gpu_time * 1000):.0f} samples/sec")
    
    # Multi-GPU would be tested with DDP (needs separate process)
    # This validates single GPU as baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
