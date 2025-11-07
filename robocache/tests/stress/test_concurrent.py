"""
Concurrent stress tests - multiple threads/streams
"""

import pytest
import torch
import threading
import concurrent.futures


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multithreaded_inference():
    """Test concurrent inference from multiple threads"""
    import robocache
    
    def worker(thread_id):
        """Worker thread"""
        torch.cuda.set_device(0)
        
        for i in range(100):
            batch = 2
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
        
        return f"Thread {thread_id} complete"
    
    # Run 4 threads concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        results = [f.result() for f in futures]
    
    print(results)
    print("✓ Multithreaded stress test passed")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multistream_inference():
    """Test concurrent inference on multiple CUDA streams"""
    import robocache
    
    streams = [torch.cuda.Stream() for _ in range(4)]
    
    for i in range(100):
        for stream in streams:
            with torch.cuda.stream(stream):
                batch = 2
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
        
        # Sync all streams
        for stream in streams:
            stream.synchronize()
    
    print("✓ Multistream stress test passed")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exception_handling():
    """Test exception handling under stress"""
    import robocache
    
    errors = 0
    successes = 0
    
    for i in range(1000):
        try:
            # Invalid inputs every 10th iteration
            if i % 10 == 0:
                vision = torch.randn(4, 30, 512, dtype=torch.bfloat16, device='cuda')
                vision_times = torch.linspace(0, 1, 20, device='cuda').expand(4, -1)  # Wrong size
                proprio = torch.randn(4, 100, 64, dtype=torch.bfloat16, device='cuda')
                proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(4, -1)
                imu = torch.randn(4, 200, 12, dtype=torch.bfloat16, device='cuda')
                imu_times = torch.linspace(0, 1, 200, device='cuda').expand(4, -1)
                target = torch.linspace(0, 1, 50, device='cuda').expand(4, -1)
                
                out = robocache.fuse_multimodal(
                    vision, vision_times, proprio, proprio_times, imu, imu_times, target
                )
            else:
                # Valid inputs
                vision = torch.randn(4, 30, 512, dtype=torch.bfloat16, device='cuda')
                vision_times = torch.linspace(0, 1, 30, device='cuda').expand(4, -1)
                proprio = torch.randn(4, 100, 64, dtype=torch.bfloat16, device='cuda')
                proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(4, -1)
                imu = torch.randn(4, 200, 12, dtype=torch.bfloat16, device='cuda')
                imu_times = torch.linspace(0, 1, 200, device='cuda').expand(4, -1)
                target = torch.linspace(0, 1, 50, device='cuda').expand(4, -1)
                
                out = robocache.fuse_multimodal(
                    vision, vision_times, proprio, proprio_times, imu, imu_times, target
                )
            
            successes += 1
        except Exception as e:
            errors += 1
    
    print(f"Successes: {successes}, Errors: {errors}")
    assert errors == 100, f"Expected 100 errors, got {errors}"
    assert successes == 900, f"Expected 900 successes, got {successes}"
    print("✓ Exception handling test passed")

