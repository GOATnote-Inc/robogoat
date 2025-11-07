"""Multi-GPU tests with DDP"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def init_process(rank, world_size, backend='nccl'):
    """Initialize distributed process group"""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed process group"""
    dist.destroy_process_group()


def run_multimodal_ddp(rank, world_size):
    """Test multimodal fusion with DDP"""
    import robocache
    
    init_process(rank, world_size)
    torch.cuda.set_device(rank)
    
    batch = 2  # Per GPU
    vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device=f'cuda:{rank}')
    vision_times = torch.linspace(0, 1, 30, device=f'cuda:{rank}').expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device=f'cuda:{rank}')
    proprio_times = torch.linspace(0, 1, 100, device=f'cuda:{rank}').expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device=f'cuda:{rank}')
    imu_times = torch.linspace(0, 1, 200, device=f'cuda:{rank}').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device=f'cuda:{rank}').expand(batch, -1)
    
    # Each GPU processes independently
    out = robocache.fuse_multimodal(
        vision, vision_times, proprio, proprio_times, imu, imu_times, target
    )
    
    # All-reduce to aggregate
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    
    # Verify output
    assert out.shape == (batch, 50, 588), f"Rank {rank} wrong shape: {out.shape}"
    
    cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
def test_multimodal_ddp():
    """Test multimodal fusion with DDP on 2 GPUs"""
    world_size = 2
    mp.spawn(run_multimodal_ddp, args=(world_size,), nprocs=world_size, join=True)


def run_voxelization_ddp(rank, world_size):
    """Test voxelization with DDP"""
    import robocache
    
    init_process(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Each GPU processes different points
    points = torch.rand(100000, 3, device=f'cuda:{rank}') * 4.0 - 2.0
    
    grid = robocache.voxelize_pointcloud(
        points, grid_min=[-2, -2, -2], voxel_size=0.0625,
        grid_size=[128, 128, 128], mode='occupancy'
    )
    
    # Aggregate grids (max occupancy)
    dist.all_reduce(grid, op=dist.ReduceOp.MAX)
    
    # Verify
    assert grid.shape == (128, 128, 128), f"Rank {rank} wrong shape: {grid.shape}"
    
    cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
def test_voxelization_ddp():
    """Test voxelization with DDP on 2 GPUs"""
    world_size = 2
    mp.spawn(run_voxelization_ddp, args=(world_size,), nprocs=world_size, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_gpu_baseline():
    """Baseline test for single GPU (always runs)"""
    import robocache
    
    batch = 4
    vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
    vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
    imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)
    
    out = robocache.fuse_multimodal(
        vision, vision_times, proprio, proprio_times, imu, imu_times, target
    )
    
    assert out.shape == (batch, 50, 588)
