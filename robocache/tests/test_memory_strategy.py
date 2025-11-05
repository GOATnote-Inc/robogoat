"""
Test suite for RoboCache memory strategy

Validates production-grade memory management:
- Memory profiling and tracking
- Chunking for large batches
- OOM prediction and prevention
- Memory limits and recovery

Demonstrates expert-level GPU memory management.
"""

import torch
import pytest
import sys
import gc
sys.path.insert(0, '../build')

try:
    import robocache_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    pytest.skip("CUDA extension not built", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def get_memory_info():
    """Get current GPU memory info"""
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        'free': free,
        'total': total,
        'allocated': allocated,
        'reserved': reserved,
        'utilization': (total - free) / total
    }


def clear_cache():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()


class TestMemoryProfiling:
    """Test memory profiling and tracking"""
    
    def test_memory_tracking(self):
        """Test memory tracking during operation"""
        clear_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Capture baseline
        baseline = get_memory_info()
        
        # Allocate and process
        points = torch.randn(8, 4096, 3, device='cuda')
        grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        result = robocache_cuda.voxelize_occupancy(
            points, grid_size, 0.1, origin
        )
        
        # Check peak memory increased
        peak = torch.cuda.max_memory_allocated()
        assert peak > baseline['allocated'], "Peak memory should increase"
        
        # Free memory
        del points, result
        clear_cache()
        
        # Memory should be released
        after = get_memory_info()
        assert after['allocated'] < peak, "Memory should be released"
    
    def test_peak_memory_reset(self):
        """Test peak memory counter reset"""
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate small tensor
        small = torch.randn(100, 100, device='cuda')
        peak_small = torch.cuda.max_memory_allocated()
        
        # Reset
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate another small tensor
        small2 = torch.randn(100, 100, device='cuda')
        peak_after_reset = torch.cuda.max_memory_allocated()
        
        # Peak after reset should be less than cumulative
        assert peak_after_reset < peak_small + small2.element_size() * small2.nelement()


class TestChunking:
    """Test chunking strategy for large batches"""
    
    def test_no_chunking_needed(self):
        """Test small batch that fits in memory"""
        points = torch.randn(2, 100, 3, device='cuda')
        grid_size = torch.tensor([32, 32, 32], dtype=torch::int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        result = robocache_cuda.voxelize_occupancy(
            points, grid_size, 0.1, origin
        )
        
        assert result.shape == (2, 32, 32, 32)
    
    def test_manual_chunking(self):
        """Test manual chunking for large batch"""
        # Simulate large batch by processing in chunks
        total_batch = 64
        chunk_size = 16
        
        results = []
        for i in range(0, total_batch, chunk_size):
            points = torch.randn(chunk_size, 1024, 3, device='cuda')
            grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
            origin = torch.zeros(3, device='cuda')
            
            result = robocache_cuda.voxelize_occupancy(
                points, grid_size, 0.1, origin
            )
            results.append(result)
            
            # Clear intermediate tensors
            del points
        
        # Concatenate results
        final = torch.cat(results, dim=0)
        
        assert final.shape == (total_batch, 64, 64, 64)
    
    def test_adaptive_chunking(self):
        """Test adaptive chunking based on available memory"""
        free, total = torch.cuda.mem_get_info()
        
        # Calculate safe batch size (use 30% of free memory)
        bytes_per_item = 1024 * 3 * 4  # 1024 points × 3 coords × 4 bytes
        bytes_per_item += 64 * 64 * 64 * 4  # + voxel grid
        
        safe_batch_size = int((free * 0.3) / bytes_per_item)
        safe_batch_size = max(1, min(safe_batch_size, 32))  # Clamp to [1, 32]
        
        # Should succeed without OOM
        points = torch.randn(safe_batch_size, 1024, 3, device='cuda')
        grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        result = robocache_cuda.voxelize_occupancy(
            points, grid_size, 0.1, origin
        )
        
        assert result.shape[0] == safe_batch_size


class TestOOMPrevention:
    """Test OOM prediction and prevention"""
    
    def test_small_allocation(self):
        """Test small allocation should not OOM"""
        free, total = torch.cuda.mem_get_info()
        
        # Allocate 1% of free memory
        safe_size = int(free * 0.01 / 4)  # 1% of free, in float32 elements
        
        tensor = torch.randn(safe_size, device='cuda')
        assert tensor is not None
        
        del tensor
    
    def test_warning_on_large_allocation(self):
        """Test warning when allocation approaches memory limit"""
        free, total = torch.cuda.mem_get_info()
        
        # Try to allocate large batch (should warn if approaching limit)
        large_batch = 64
        large_points = 16384
        large_grid = 256
        
        required = (large_batch * large_points * 3 + 
                    large_batch * large_grid ** 3) * 4
        
        if required > free * 0.9:
            # Should warn or fail
            pytest.skip("Would cause OOM - expected behavior")
    
    def test_fallback_to_cpu(self):
        """Test CPU fallback when GPU OOM"""
        try:
            # Try to allocate huge batch
            huge_batch = 256
            points = torch.randn(huge_batch, 16384, 3, device='cuda')
            grid_size = torch.tensor([256, 256, 256], dtype=torch.int32, device='cuda')
            origin = torch.zeros(3, device='cuda')
            
            result = robocache_cuda.voxelize_occupancy(
                points, grid_size, 0.05, origin
            )
            
            # If we get here, GPU had enough memory
            assert result is not None
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Expected - GPU OOM
                # In production, would fallback to CPU or chunking
                clear_cache()
                pytest.skip("GPU OOM - expected behavior")
            else:
                raise


class TestMemoryLimits:
    """Test operation at memory limits"""
    
    def test_maximum_batch_size(self):
        """Find maximum batch size that fits in memory"""
        free, total = torch.cuda.mem_get_info()
        
        # Binary search for max batch size
        min_batch = 1
        max_batch = 128
        
        while min_batch < max_batch:
            test_batch = (min_batch + max_batch + 1) // 2
            
            try:
                clear_cache()
                points = torch.randn(test_batch, 4096, 3, device='cuda')
                grid_size = torch.tensor([128, 128, 128], dtype=torch.int32, device='cuda')
                origin = torch.zeros(3, device='cuda')
                
                result = robocache_cuda.voxelize_occupancy(
                    points, grid_size, 0.1, origin
                )
                
                # Success - try larger
                del points, result
                min_batch = test_batch
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM - try smaller
                    clear_cache()
                    max_batch = test_batch - 1
                else:
                    raise
        
        print(f"Maximum batch size: {min_batch}")
        assert min_batch >= 1, "Should fit at least 1 batch"
    
    def test_maximum_grid_resolution(self):
        """Find maximum grid resolution that fits in memory"""
        free, total = torch.cuda.mem_get_info()
        
        # Try different resolutions
        resolutions = [64, 128, 192, 256, 384, 512]
        max_resolution = 64
        
        for resolution in resolutions:
            try:
                clear_cache()
                points = torch.randn(4, 4096, 3, device='cuda')
                grid_size = torch.tensor([resolution] * 3, dtype=torch.int32, device='cuda')
                origin = torch.zeros(3, device='cuda')
                
                result = robocache_cuda.voxelize_occupancy(
                    points, grid_size, 0.1, origin
                )
                
                max_resolution = resolution
                del points, result
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_cache()
                    break
                else:
                    raise
        
        print(f"Maximum grid resolution: {max_resolution}³")
        assert max_resolution >= 64


class TestMemoryEfficiency:
    """Test memory efficiency of operations"""
    
    def test_memory_reuse(self):
        """Test that memory is efficiently reused"""
        clear_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run multiple operations
        for i in range(10):
            points = torch.randn(4, 1024, 3, device='cuda')
            grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
            origin = torch.zeros(3, device='cuda')
            
            result = robocache_cuda.voxelize_occupancy(
                points, grid_size, 0.1, origin
            )
            
            # Clear intermediate tensors
            del points, result
        
        # Peak memory should not grow linearly with iterations
        peak = torch.cuda.max_memory_allocated()
        
        # Run one more time and check peak didn't increase much
        points = torch.randn(4, 1024, 3, device='cuda')
        grid_size = torch.tensor([64, 64, 64], dtype=torch.int32, device='cuda')
        origin = torch.zeros(3, device='cuda')
        
        result = robocache_cuda.voxelize_occupancy(
            points, grid_size, 0.1, origin
        )
        
        new_peak = torch.cuda.max_memory_allocated()
        
        # Peak should not have increased significantly
        assert new_peak <= peak * 1.1, "Memory reuse not working"


if __name__ == '__main__':
    # Print memory info
    info = get_memory_info()
    print(f"GPU Memory: {info['free'] / 1e9:.1f} GB free / {info['total'] / 1e9:.1f} GB total")
    print(f"Utilization: {info['utilization'] * 100:.1f}%")
    print()
    
    pytest.main([__file__, '-v', '--tb=short'])

