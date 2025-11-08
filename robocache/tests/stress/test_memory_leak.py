"""
Memory leak detection for long-running GPU operations.

Runs 10K iterations and monitors memory growth to detect leaks.
"""

import pytest
import torch
import robocache
import psutil
import time


@pytest.mark.stress
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_no_memory_leak_trajectory_resample():
    """Test that trajectory resampling doesn't leak memory"""
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / 1024 / 1024
    
    batch, src_len, tgt_len, dim = 32, 500, 250, 256
    
    # Run 10K iterations
    for i in range(10000):
        # Create tensors
        data = torch.randn(batch, src_len, dim, device='cuda', dtype=torch.bfloat16)
        src_times = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1)
        tgt_times = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1)
        
        # Run kernel
        result = robocache.resample_trajectories(data, src_times, tgt_times, backend="cuda")
        
        # Explicit cleanup
        del data, src_times, tgt_times, result
        
        # Check memory every 1000 iterations
        if i % 1000 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            current_mem_mb = process.memory_info().rss / 1024 / 1024
            growth_mb = current_mem_mb - initial_mem_mb
            
            print(f"Iteration {i}: Memory growth = {growth_mb:.1f} MB")
            
            # Allow 100 MB growth (buffers, caching)
            assert growth_mb < 100, f"Memory leak detected: {growth_mb:.1f} MB growth after {i} iterations"


@pytest.mark.stress
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_no_memory_leak_voxelization():
    """Test that voxelization doesn't leak memory"""
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / 1024 / 1024
    
    num_points = 500000
    
    # Run 10K iterations
    for i in range(10000):
        # Create point cloud
        points = torch.rand(num_points, 3, device='cuda') * 20.0 - 10.0
        
        # Voxelize
        voxel_grid = robocache.voxelize_pointcloud(
            points,
            grid_min=[-10.0, -10.0, -10.0],
            voxel_size=0.05,
            grid_size=[128, 128, 128],
            mode='occupancy'
        )
        
        # Explicit cleanup
        del points, voxel_grid
        
        # Check memory every 1000 iterations
        if i % 1000 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            current_mem_mb = process.memory_info().rss / 1024 / 1024
            growth_mb = current_mem_mb - initial_mem_mb
            
            print(f"Iteration {i}: Memory growth = {growth_mb:.1f} MB")
            
            # Allow 100 MB growth
            assert growth_mb < 100, f"Memory leak detected: {growth_mb:.1f} MB growth after {i} iterations"


@pytest.mark.stress
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_no_memory_leak_multimodal_fusion():
    """Test that multimodal fusion doesn't leak memory"""
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / 1024 / 1024
    
    batch = 32
    
    # Run 10K iterations
    for i in range(10000):
        # Create multimodal data
        vision = torch.randn(batch, 150, 512, device='cuda', dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 1, 150, device='cuda').unsqueeze(0).expand(batch, -1)
        
        proprio = torch.randn(batch, 500, 14, device='cuda', dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 1, 500, device='cuda').unsqueeze(0).expand(batch, -1)
        
        force = torch.randn(batch, 500, 6, device='cuda', dtype=torch.bfloat16)
        force_times = torch.linspace(0, 1, 500, device='cuda').unsqueeze(0).expand(batch, -1)
        
        target_times = torch.linspace(0, 1, 250, device='cuda').unsqueeze(0).expand(batch, -1)
        
        # Fuse
        fused = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            force, force_times,
            target_times
        )
        
        # Explicit cleanup
        del vision, vision_times, proprio, proprio_times, force, force_times, target_times, fused
        
        # Check memory every 1000 iterations
        if i % 1000 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            current_mem_mb = process.memory_info().rss / 1024 / 1024
            growth_mb = current_mem_mb - initial_mem_mb
            
            print(f"Iteration {i}: Memory growth = {growth_mb:.1f} MB")
            
            # Allow 100 MB growth
            assert growth_mb < 100, f"Memory leak detected: {growth_mb:.1f} MB growth after {i} iterations"


if __name__ == '__main__':
    """Run memory leak tests standalone"""
    print("=" * 70)
    print("Memory Leak Detection Tests")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - tests skipped")
        exit(0)
    
    print("\nRunning trajectory resample leak test...")
    test_no_memory_leak_trajectory_resample()
    print("✅ No memory leak detected")
    
    print("\nRunning voxelization leak test...")
    test_no_memory_leak_voxelization()
    print("✅ No memory leak detected")
    
    print("\nRunning multimodal fusion leak test...")
    test_no_memory_leak_multimodal_fusion()
    print("✅ No memory leak detected")
    
    print("\n" + "=" * 70)
    print("✅ ALL MEMORY LEAK TESTS PASSED")
    print("=" * 70)

