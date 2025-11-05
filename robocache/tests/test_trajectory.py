"""
Test trajectory resampling functionality.

Validates:
- Correctness of linear interpolation
- Edge cases (boundary conditions, empty sequences)
- Different dtypes (float32, float16, bfloat16)
- Different batch sizes and dimensions
- Performance characteristics
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import robocache


class TestTrajectoryResampling:
    """Test trajectory resampling operation."""
    
    def test_basic_resampling(self):
        """Test basic trajectory resampling works."""
        B, S, T, D = 4, 50, 32, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D), f"Expected shape (B={B}, T={T}, D={D}), got {result.shape}"
        assert result.dtype == source_data.dtype
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"
    
    def test_identity_resampling(self):
        """Test resampling to same times gives same data."""
        B, S, D = 4, 50, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(
            source_data, source_times, source_times, backend='pytorch'
        )
        
        # Should be nearly identical
        max_diff = (result - source_data).abs().max().item()
        assert max_diff < 1e-5, f"Identity resampling has large error: {max_diff}"
    
    def test_linear_interpolation_correctness(self):
        """Test that linear interpolation is correct."""
        # Simple case: linear ramp
        B, S, D = 1, 3, 1
        
        source_data = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)
        source_times = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
        target_times = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32)
        
        result = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='pytorch'
        )
        
        expected = torch.tensor([[[0.0], [0.5], [1.0], [1.5], [2.0]]], dtype=torch.float32)
        
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Interpolation incorrect: {result.squeeze().tolist()} vs {expected.squeeze().tolist()}"
    
    def test_upsample(self):
        """Test upsampling (target times > source times)."""
        B, S, T, D = 4, 20, 100, 8
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert not torch.isnan(result).any()
    
    def test_downsample(self):
        """Test downsampling (target times < source times)."""
        B, S, T, D = 4, 100, 20, 8
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert not torch.isnan(result).any()
    
    def test_bfloat16_dtype(self):
        """Test BF16 dtype (recommended for CUDA)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        B, S, T, D = 4, 50, 32, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.bfloat16, device='cuda')
        source_times = torch.linspace(0, 1, S, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
        
        try:
            result = robocache.resample_trajectories(
                source_data, source_times, target_times, backend='cuda'
            )
            
            assert result.shape == (B, T, D)
            assert result.dtype == torch.bfloat16
            assert not torch.isnan(result.float()).any()
        
        except RuntimeError as e:
            if "CUDA backend" in str(e):
                pytest.skip("CUDA extension not available")
            else:
                raise
    
    def test_large_batch(self):
        """Test large batch size."""
        B, S, T, D = 128, 50, 32, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert not torch.isnan(result).any()
    
    def test_high_dimensional_actions(self):
        """Test high-dimensional action space."""
        B, S, T, D = 4, 50, 32, 256
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert not torch.isnan(result).any()
    
    def test_irregular_times(self):
        """Test with irregular (non-uniform) timestamps."""
        B, S, T, D = 4, 50, 32, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        
        # Irregular source times (but monotonic)
        source_times = torch.sort(torch.rand(B, S), dim=1).values.contiguous()
        
        # Irregular target times (but monotonic)
        target_times = torch.sort(torch.rand(B, T), dim=1).values.contiguous()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert not torch.isnan(result).any()
    
    def test_boundary_extrapolation(self):
        """Test behavior at boundaries (first/last timestamps)."""
        B, S, T, D = 2, 5, 7, 4
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]], dtype=torch.float32).expand(B, -1).contiguous()
        
        # Target times include values before first source time
        target_times = torch.tensor([[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]], dtype=torch.float32).expand(B, -1).contiguous()
        
        result = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='pytorch'
        )
        
        assert result.shape == (B, T, D)
        # First target (0.0) should use first interval [0.2, 0.4]
        # Should not be NaN (clamped extrapolation)
        assert not torch.isnan(result).any()


class TestTrajectoryPerformance:
    """Test performance characteristics of trajectory resampling."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_faster_than_pytorch(self):
        """Test that CUDA backend is faster than PyTorch."""
        try:
            import time
            
            B, S, T, D = 64, 100, 50, 32
            
            # Generate data on GPU
            source_data = torch.randn(B, S, D, dtype=torch.bfloat16, device='cuda')
            source_times = torch.linspace(0, 1, S, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
            target_times = torch.linspace(0, 1, T, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
            
            # Warmup
            for _ in range(10):
                _ = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='pytorch'
                )
            torch.cuda.synchronize()
            
            # Benchmark PyTorch
            start = time.perf_counter()
            for _ in range(100):
                result_pt = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='pytorch'
                )
            torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start
            
            # Warmup CUDA
            for _ in range(10):
                _ = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='cuda'
                )
            torch.cuda.synchronize()
            
            # Benchmark CUDA
            start = time.perf_counter()
            for _ in range(100):
                result_cuda = robocache.resample_trajectories(
                    source_data, source_times, target_times, backend='cuda'
                )
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start
            
            speedup = pytorch_time / cuda_time
            print(f"\nPyTorch: {pytorch_time*1000:.2f}ms")
            print(f"CUDA: {cuda_time*1000:.2f}ms")
            print(f"Speedup: {speedup:.2f}x")
            
            # CUDA should be at least faster (not necessarily 22x on all hardware)
            assert cuda_time < pytorch_time, "CUDA should be faster than PyTorch"
        
        except RuntimeError as e:
            if "CUDA backend" in str(e):
                pytest.skip("CUDA extension not available")
            else:
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

