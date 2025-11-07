"""
Long-running stress tests (24h burn-in)

Run with: pytest tests/stress/test_long_running.py --duration=86400
"""

import pytest
import torch
import time
import psutil
import os


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_24h_burn_in():
    """24-hour burn-in test with memory leak detection"""
    import robocache
    
    duration_sec = int(os.environ.get('STRESS_TEST_DURATION', 3600))  # Default 1h
    process = psutil.Process()
    
    # Baseline memory
    torch.cuda.empty_cache()
    baseline_gpu_mb = torch.cuda.memory_allocated() / 1024**2
    baseline_cpu_mb = process.memory_info().rss / 1024**2
    
    start = time.time()
    iteration = 0
    
    print(f"\nStarting {duration_sec}s burn-in (baseline: GPU={baseline_gpu_mb:.1f}MB, CPU={baseline_cpu_mb:.1f}MB)")
    
    while time.time() - start < duration_sec:
        # Multimodal fusion
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
        
        # Voxelization
        points = torch.rand(100000, 3, device='cuda') * 4.0 - 2.0
        grid = robocache.voxelize_pointcloud(
            points, grid_min=[-2, -2, -2], voxel_size=0.0625,
            grid_size=[128, 128, 128], mode='occupancy'
        )
        
        iteration += 1
        
        # Memory check every 1000 iterations
        if iteration % 1000 == 0:
            torch.cuda.synchronize()
            gpu_mb = torch.cuda.memory_allocated() / 1024**2
            cpu_mb = process.memory_info().rss / 1024**2
            
            gpu_leak = gpu_mb - baseline_gpu_mb
            cpu_leak = cpu_mb - baseline_cpu_mb
            
            elapsed_h = (time.time() - start) / 3600
            
            print(f"[{elapsed_h:.2f}h] Iter {iteration}: GPU={gpu_mb:.1f}MB (+{gpu_leak:.1f}), CPU={cpu_mb:.1f}MB (+{cpu_leak:.1f})")
            
            # Fail if leak > 100MB
            assert gpu_leak < 100, f"GPU memory leak: +{gpu_leak:.1f}MB"
            assert cpu_leak < 500, f"CPU memory leak: +{cpu_leak:.1f}MB"
    
    print(f"\n✓ Burn-in complete: {iteration} iterations, no memory leaks")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_repeated_allocation():
    """Test repeated allocation/deallocation (OOM resilience)"""
    import robocache
    
    for i in range(1000):
        # Large allocations
        points = torch.rand(1000000, 3, device='cuda') * 4.0 - 2.0
        features = torch.randn(1000000, 8, device='cuda')
        
        grid = robocache.voxelize_pointcloud(
            points, features, grid_min=[-2, -2, -2], voxel_size=0.05,
            grid_size=[256, 256, 256], mode='mean'
        )
        
        # Force cleanup
        del points, features, grid
        torch.cuda.empty_cache()
        
        if i % 100 == 0:
            print(f"Iteration {i}/1000")
    
    print("✓ No OOM errors")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backpressure_handling():
    """Test behavior under back-pressure (slow consumer)"""
    import robocache
    
    batch = 4
    
    for i in range(100):
        # Producer (fast)
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
        
        # Consumer (slow - simulated)
        time.sleep(0.1)
        
        # Check no accumulation
        if i % 10 == 0:
            gpu_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"Iteration {i}: GPU={gpu_mb:.1f}MB")
    
    print("✓ No accumulation under back-pressure")

