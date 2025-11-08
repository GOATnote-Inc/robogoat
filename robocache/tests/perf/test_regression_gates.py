"""
Performance Regression Gates

Ensures no performance degradation in core operations.
Fails CI if performance drops below thresholds.
"""

import pytest
import torch
import time
import numpy as np

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False


# Performance thresholds (samples/sec)
THRESHOLDS = {
    'trajectory_resample_medium': {
        'min_throughput': 60_000_000,  # 60M samples/sec (H100: 77M, A100: 126M)
        'max_latency_ms': 0.15,
    },
    'voxelization_large': {
        'min_throughput': 5_000_000_000,  # 5B points/sec (H100: 11.8B, A100: 11.76B)
        'max_latency_ms': 0.15,
    },
    'multimodal_fusion': {
        'min_throughput': 40_000_000,  # 40M samples/sec (H100: 60M, A100: 56.9M)
        'max_latency_ms': 0.2,
    }
}


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRegressionGates:
    """Performance regression tests"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def test_trajectory_resample_no_regression(self):
        """Ensure trajectory resampling meets performance threshold"""
        batch, src_len, tgt_len, dim = 32, 500, 250, 256
        
        data = torch.randn(batch, src_len, dim, device='cuda', dtype=torch.bfloat16)
        src_times = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1)
        tgt_times = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1)
        
        # Warmup
        for _ in range(50):
            _ = robocache.resample_trajectories(data, src_times, tgt_times, backend='cuda')
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(200):
            start = time.perf_counter()
            result = robocache.resample_trajectories(data, src_times, tgt_times, backend='cuda')
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time = np.median(times)
        median_time_ms = median_time * 1000
        throughput = (batch * tgt_len) / median_time
        
        print(f"\nTrajectory Resample Performance:")
        print(f"  Latency: {median_time_ms:.3f} ms")
        print(f"  Throughput: {throughput / 1e6:.1f} M samples/sec")
        print(f"  Threshold: {THRESHOLDS['trajectory_resample_medium']['min_throughput'] / 1e6:.1f} M samples/sec")
        
        # Regression check
        assert throughput >= THRESHOLDS['trajectory_resample_medium']['min_throughput'], \
            f"REGRESSION: Throughput {throughput / 1e6:.1f}M < threshold {THRESHOLDS['trajectory_resample_medium']['min_throughput'] / 1e6:.1f}M"
        
        assert median_time_ms <= THRESHOLDS['trajectory_resample_medium']['max_latency_ms'], \
            f"REGRESSION: Latency {median_time_ms:.3f}ms > threshold {THRESHOLDS['trajectory_resample_medium']['max_latency_ms']:.3f}ms"
    
    def test_voxelization_no_regression(self):
        """Ensure voxelization meets performance threshold"""
        num_points = 500_000
        grid_size = 128
        
        points = torch.rand(num_points, 3, device='cuda') * 10.0 - 5.0
        
        # Warmup
        for _ in range(50):
            _ = robocache.voxelize_pointcloud(
                points,
                grid_min=[-5.0, -5.0, -5.0],
                voxel_size=0.1,
                grid_size=[grid_size, grid_size, grid_size],
                mode='occupancy',
                backend='cuda'
            )
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(200):
            start = time.perf_counter()
            voxel_grid = robocache.voxelize_pointcloud(
                points,
                grid_min=[-5.0, -5.0, -5.0],
                voxel_size=0.1,
                grid_size=[grid_size, grid_size, grid_size],
                mode='occupancy',
                backend='cuda'
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time = np.median(times)
        median_time_ms = median_time * 1000
        throughput = num_points / median_time
        
        print(f"\nVoxelization Performance:")
        print(f"  Latency: {median_time_ms:.3f} ms")
        print(f"  Throughput: {throughput / 1e9:.2f} B pts/sec")
        print(f"  Threshold: {THRESHOLDS['voxelization_large']['min_throughput'] / 1e9:.1f} B pts/sec")
        
        # Regression check
        assert throughput >= THRESHOLDS['voxelization_large']['min_throughput'], \
            f"REGRESSION: Throughput {throughput / 1e9:.2f}B < threshold {THRESHOLDS['voxelization_large']['min_throughput'] / 1e9:.1f}B"
        
        assert median_time_ms <= THRESHOLDS['voxelization_large']['max_latency_ms'], \
            f"REGRESSION: Latency {median_time_ms:.3f}ms > threshold {THRESHOLDS['voxelization_large']['max_latency_ms']:.3f}ms"
    
    def test_multimodal_fusion_no_regression(self):
        """Ensure multimodal fusion meets performance threshold"""
        batch = 32
        
        vision = torch.randn(batch, 150, 512, device='cuda', dtype=torch.bfloat16)
        vision_t = torch.linspace(0, 1, 150, device='cuda').unsqueeze(0).expand(batch, -1)
        proprio = torch.randn(batch, 500, 14, device='cuda', dtype=torch.bfloat16)
        proprio_t = torch.linspace(0, 1, 500, device='cuda').unsqueeze(0).expand(batch, -1)
        force = torch.randn(batch, 500, 6, device='cuda', dtype=torch.bfloat16)
        force_t = torch.linspace(0, 1, 500, device='cuda').unsqueeze(0).expand(batch, -1)
        target_t = torch.linspace(0, 1, 250, device='cuda').unsqueeze(0).expand(batch, -1)
        
        # Warmup
        for _ in range(50):
            _ = robocache.fuse_multimodal(
                vision, vision_t, proprio, proprio_t, force, force_t, target_t
            )
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(200):
            start = time.perf_counter()
            fused = robocache.fuse_multimodal(
                vision, vision_t, proprio, proprio_t, force, force_t, target_t
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time = np.median(times)
        median_time_ms = median_time * 1000
        throughput = (batch * 250) / median_time
        
        print(f"\nMultimodal Fusion Performance:")
        print(f"  Latency: {median_time_ms:.3f} ms")
        print(f"  Throughput: {throughput / 1e6:.1f} M samples/sec")
        print(f"  Threshold: {THRESHOLDS['multimodal_fusion']['min_throughput'] / 1e6:.1f} M samples/sec")
        
        # Regression check
        assert throughput >= THRESHOLDS['multimodal_fusion']['min_throughput'], \
            f"REGRESSION: Throughput {throughput / 1e6:.1f}M < threshold {THRESHOLDS['multimodal_fusion']['min_throughput'] / 1e6:.1f}M"
        
        assert median_time_ms <= THRESHOLDS['multimodal_fusion']['max_latency_ms'], \
            f"REGRESSION: Latency {median_time_ms:.3f}ms > threshold {THRESHOLDS['multimodal_fusion']['max_latency_ms']:.3f}ms"
    
    def test_memory_no_regression(self):
        """Ensure memory usage doesn't increase"""
        import psutil
        process = psutil.Process()
        
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run 1000 iterations
        for _ in range(1000):
            batch, src_len, tgt_len, dim = 32, 500, 250, 256
            data = torch.randn(batch, src_len, dim, device='cuda', dtype=torch.bfloat16)
            src_times = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1)
            tgt_times = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1)
            result = robocache.resample_trajectories(data, src_times, tgt_times, backend='cuda')
            del data, src_times, tgt_times, result
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        growth = final_mem - initial_mem
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_mem:.1f} MB")
        print(f"  Final: {final_mem:.1f} MB")
        print(f"  Growth: {growth:.1f} MB")
        
        # Allow 50 MB growth for caching
        assert growth < 50, f"REGRESSION: Memory growth {growth:.1f}MB > 50MB threshold"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

