"""
Comprehensive test suite for multimodal sensor fusion

Tests correctness, edge cases, and multi-backend consistency for the
fuse_multimodal operation across CUDA and PyTorch backends.

Test Coverage:
- CPU golden reference validation
- CUDA vs PyTorch backend parity
- Edge cases (empty data, single points, mismatched frequencies)
- Boundary conditions
- Dtype support (FP32, FP16, BF16)
- Error handling

Author: RoboCache Team
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False

# Check backend availability
if ROBOCACHE_AVAILABLE:
    info = robocache.check_installation()
    CUDA_AVAILABLE = info['cuda_extension_available']
    PYTORCH_AVAILABLE = info['pytorch_available']
else:
    CUDA_AVAILABLE = False
    PYTORCH_AVAILABLE = False


def cpu_golden_fuse_multimodal(
    primary_data,
    primary_times,
    secondary_data,
    secondary_times
):
    """
    CPU golden reference for multimodal fusion.
    
    Uses linear interpolation to align secondary data to primary timestamps,
    then concatenates features.
    """
    batch_size, primary_len, primary_dim = primary_data.shape
    _, secondary_len, secondary_dim = secondary_data.shape
    
    # Convert to float32 on CPU for reference computation
    p_data_cpu = primary_data.cpu().float()
    p_times_cpu = primary_times.cpu().float()
    s_data_cpu = secondary_data.cpu().float()
    s_times_cpu = secondary_times.cpu().float()
    
    # Allocate output
    fused = torch.zeros(
        (batch_size, primary_len, primary_dim + secondary_dim),
        dtype=torch.float32
    )
    
    # Copy primary data directly
    fused[:, :, :primary_dim] = p_data_cpu
    
    # Resample secondary data to primary timestamps
    for b in range(batch_size):
        p_t = p_times_cpu[b]  # [primary_len]
        s_t = s_times_cpu[b]  # [secondary_len]
        s_d = s_data_cpu[b]   # [secondary_len, secondary_dim]
        
        for i in range(primary_len):
            target_time = p_t[i].item()
            
            # Binary search for bracketing indices
            left_idx = 0
            right_idx = secondary_len - 1
            
            # Find indices where s_t[left_idx] <= target_time < s_t[right_idx+1]
            for j in range(secondary_len - 1):
                if s_t[j] <= target_time < s_t[j + 1]:
                    left_idx = j
                    right_idx = j + 1
                    break
            else:
                # Handle edge cases
                if target_time <= s_t[0]:
                    left_idx = 0
                    right_idx = 1 if secondary_len > 1 else 0
                elif target_time >= s_t[-1]:
                    left_idx = max(secondary_len - 2, 0)
                    right_idx = secondary_len - 1
            
            # Interpolate
            if left_idx == right_idx:
                # No interpolation needed
                fused[b, i, primary_dim:] = s_d[left_idx]
            else:
                left_time = s_t[left_idx].item()
                right_time = s_t[right_idx].item()
                
                if abs(right_time - left_time) < 1e-8:
                    weight = 0.5
                else:
                    weight = (target_time - left_time) / (right_time - left_time)
                    weight = max(0.0, min(1.0, weight))
                
                interpolated = s_d[left_idx] + weight * (s_d[right_idx] - s_d[left_idx])
                fused[b, i, primary_dim:] = interpolated
    
    return fused


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestMultimodalFusionCorrectness:
    """Test correctness against CPU golden reference"""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("primary_len,secondary_len", [(10, 20), (50, 100), (100, 50)])
    @pytest.mark.parametrize("primary_dim,secondary_dim", [(32, 16), (64, 64), (128, 32)])
    def test_correctness_various_shapes(self, batch_size, primary_len, secondary_len, primary_dim, secondary_dim):
        """Test correctness across various tensor shapes"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        # Generate test data
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim, dtype=torch.float32)
        primary_times = torch.linspace(0, 1, primary_len).expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim, dtype=torch.float32)
        secondary_times = torch.linspace(0, 1, secondary_len).expand(batch_size, -1)
        
        # Compute golden reference
        golden = cpu_golden_fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times
        )
        
        # Compute using PyTorch backend
        result = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        # Compare
        result_cpu = result.cpu().float()
        max_diff = (golden - result_cpu).abs().max().item()
        mean_diff = (golden - result_cpu).abs().mean().item()
        
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
        assert mean_diff < 1e-5, f"Mean diff {mean_diff} exceeds tolerance"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_pytorch_parity(self, dtype):
        """Test CUDA and PyTorch backends produce consistent results"""
        batch_size, primary_len, secondary_len = 8, 50, 100
        primary_dim, secondary_dim = 64, 32
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim, dtype=dtype, device='cuda')
        primary_times = torch.linspace(0, 1, primary_len, device='cuda').expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim, dtype=dtype, device='cuda')
        secondary_times = torch.linspace(0, 1, secondary_len, device='cuda').expand(batch_size, -1)
        
        # Compute using both backends
        result_cuda = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='cuda'
        )
        
        result_pytorch = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        # Compare (convert to FP32 for comparison)
        cuda_fp32 = result_cuda.float()
        pytorch_fp32 = result_pytorch.float()
        
        max_diff = (cuda_fp32 - pytorch_fp32).abs().max().item()
        mean_diff = (cuda_fp32 - pytorch_fp32).abs().mean().item()
        
        # Tolerance depends on dtype
        if dtype == torch.float32:
            tol = 1e-4
        elif dtype == torch.float16:
            tol = 1e-2
        else:  # bfloat16
            tol = 1e-2
        
        assert max_diff < tol, f"CUDA/PyTorch max diff {max_diff} exceeds tolerance {tol}"
        assert mean_diff < tol / 10, f"CUDA/PyTorch mean diff {mean_diff} exceeds tolerance"


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestMultimodalFusionEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_identical_frequencies(self):
        """Test when primary and secondary have identical frequencies"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size, seq_len = 4, 50
        primary_dim, secondary_dim = 32, 16
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, seq_len, primary_dim)
        secondary_data = torch.randn(batch_size, seq_len, secondary_dim)
        times = torch.linspace(0, 1, seq_len).expand(batch_size, -1)
        
        result = robocache.fuse_multimodal(
            primary_data, times,
            secondary_data, times,
            backend='pytorch'
        )
        
        # When frequencies match, fusion should be simple concatenation
        expected = torch.cat([primary_data, secondary_data], dim=2)
        
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Identical frequency fusion failed: max diff {max_diff}"
    
    def test_upsampling(self):
        """Test upsampling (secondary frequency < primary frequency)"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 2
        primary_len, secondary_len = 100, 50
        primary_dim, secondary_dim = 32, 16
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim)
        primary_times = torch.linspace(0, 1, primary_len).expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim)
        secondary_times = torch.linspace(0, 1, secondary_len).expand(batch_size, -1)
        
        result = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        assert result.shape == (batch_size, primary_len, primary_dim + secondary_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_downsampling(self):
        """Test downsampling (secondary frequency > primary frequency)"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 2
        primary_len, secondary_len = 50, 100
        primary_dim, secondary_dim = 32, 16
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim)
        primary_times = torch.linspace(0, 1, primary_len).expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim)
        secondary_times = torch.linspace(0, 1, secondary_len).expand(batch_size, -1)
        
        result = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        assert result.shape == (batch_size, primary_len, primary_dim + secondary_dim)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_single_timestep(self):
        """Test with single timestep (no interpolation)"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 1
        primary_dim, secondary_dim = 32, 16
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, 1, primary_dim)
        primary_times = torch.tensor([[0.5]])
        
        secondary_data = torch.randn(batch_size, 1, secondary_dim)
        secondary_times = torch.tensor([[0.5]])
        
        result = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        expected = torch.cat([primary_data, secondary_data], dim=2)
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5
    
    def test_non_overlapping_times(self):
        """Test when primary and secondary times don't overlap"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 2
        primary_len, secondary_len = 10, 10
        primary_dim, secondary_dim = 32, 16
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim)
        primary_times = torch.linspace(0, 0.5, primary_len).expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim)
        secondary_times = torch.linspace(0.6, 1.0, secondary_len).expand(batch_size, -1)
        
        result = robocache.fuse_multimodal(
            primary_data, primary_times,
            secondary_data, secondary_times,
            backend='pytorch'
        )
        
        # Should extrapolate (nearest neighbor)
        assert result.shape == (batch_size, primary_len, primary_dim + secondary_dim)
        assert not torch.isnan(result).any()


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestMultimodalFusionErrorHandling:
    """Test error handling and validation"""
    
    def test_invalid_backend(self):
        """Test error handling for invalid backend"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 2
        primary_data = torch.randn(batch_size, 10, 32)
        primary_times = torch.linspace(0, 1, 10).expand(batch_size, -1)
        secondary_data = torch.randn(batch_size, 20, 16)
        secondary_times = torch.linspace(0, 1, 20).expand(batch_size, -1)
        
        with pytest.raises((ValueError, RuntimeError)):
            robocache.fuse_multimodal(
                primary_data, primary_times,
                secondary_data, secondary_times,
                backend='invalid_backend'
            )
    
    def test_shape_mismatch(self):
        """Test error handling for shape mismatches"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch backend not available")
        
        batch_size = 2
        primary_data = torch.randn(batch_size, 10, 32)
        primary_times = torch.linspace(0, 1, 10).expand(batch_size, -1)
        secondary_data = torch.randn(batch_size + 1, 20, 16)  # Wrong batch size
        secondary_times = torch.linspace(0, 1, 20).expand(batch_size + 1, -1)
        
        with pytest.raises((ValueError, RuntimeError)):
            robocache.fuse_multimodal(
                primary_data, primary_times,
                secondary_data, secondary_times,
                backend='pytorch'
            )


@pytest.mark.skipif(not ROBOCACHE_AVAILABLE, reason="RoboCache not available")
class TestMultimodalFusionPerformance:
    """Performance regression tests"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
    def test_cuda_faster_than_pytorch(self):
        """Verify CUDA backend is faster than PyTorch"""
        import time
        
        batch_size, primary_len, secondary_len = 128, 100, 200
        primary_dim, secondary_dim = 256, 128
        
        torch.manual_seed(42)
        primary_data = torch.randn(batch_size, primary_len, primary_dim, dtype=torch.bfloat16, device='cuda')
        primary_times = torch.linspace(0, 1, primary_len, device='cuda').expand(batch_size, -1)
        
        secondary_data = torch.randn(batch_size, secondary_len, secondary_dim, dtype=torch.bfloat16, device='cuda')
        secondary_times = torch.linspace(0, 1, secondary_len, device='cuda').expand(batch_size, -1)
        
        # Warmup
        for _ in range(10):
            _ = robocache.fuse_multimodal(primary_data, primary_times, secondary_data, secondary_times, backend='cuda')
            _ = robocache.fuse_multimodal(primary_data, primary_times, secondary_data, secondary_times, backend='pytorch')
        
        torch.cuda.synchronize()
        
        # Benchmark CUDA
        start = time.time()
        for _ in range(100):
            _ = robocache.fuse_multimodal(primary_data, primary_times, secondary_data, secondary_times, backend='cuda')
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(100):
            _ = robocache.fuse_multimodal(primary_data, primary_times, secondary_data, secondary_times, backend='pytorch')
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        speedup = pytorch_time / cuda_time
        print(f"\nMultimodal Fusion Speedup: {speedup:.2f}x (CUDA: {cuda_time:.4f}s, PyTorch: {pytorch_time:.4f}s)")
        
        # CUDA should be at least 2x faster
        assert speedup > 2.0, f"CUDA backend not faster than PyTorch (speedup: {speedup:.2f}x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

