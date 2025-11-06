"""
Correctness tests for trajectory resampling.

Compares CUDA implementation against PyTorch CPU reference with numerical tolerances.
"""

import pytest
import torch


def resample_pytorch_reference(data, src_times, tgt_times):
    """
    PyTorch CPU reference implementation for trajectory resampling.
    
    Uses binary search + linear interpolation (same algorithm as CUDA kernel).
    """
    B, S, D = data.shape
    T = tgt_times.shape[1]
    result = torch.zeros(B, T, D, dtype=data.dtype, device=data.device)
    
    for b in range(B):
        for t in range(T):
            tgt_t = tgt_times[b, t].item()
            
            # Binary search for insertion point
            idx = torch.searchsorted(src_times[b], tgt_t)
            
            if idx == 0:
                # Before first timestamp: clamp to first
                result[b, t] = data[b, 0]
            elif idx >= S:
                # After last timestamp: clamp to last
                result[b, t] = data[b, -1]
            else:
                # Linear interpolation
                t0 = src_times[b, idx - 1].item()
                t1 = src_times[b, idx].item()
                alpha = (tgt_t - t0) / (t1 - t0 + 1e-9)
                result[b, t] = (1 - alpha) * data[b, idx - 1] + alpha * data[b, idx]
    
    return result


@pytest.fixture
def robocache_module():
    """Import robocache."""
    try:
        import robocache
        return robocache
    except ImportError:
        pytest.skip("robocache not installed")


class TestTrajectoryCorrectness:
    """Correctness tests comparing CUDA vs CPU reference."""
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    @pytest.mark.parametrize("source_len", [100, 500])
    @pytest.mark.parametrize("target_len", [50, 250])
    @pytest.mark.parametrize("dim", [64, 256])
    def test_correctness_parametric(
        self,
        robocache_module,
        batch_size,
        source_len,
        target_len,
        dim
    ):
        """Test correctness across various configurations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Generate test data
        torch.manual_seed(42)
        source_data = torch.randn(batch_size, source_len, dim, dtype=torch.float32)
        source_times = torch.linspace(0, 5, source_len).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len).unsqueeze(0).expand(batch_size, -1)
        
        # CPU reference
        cpu_result = resample_pytorch_reference(source_data, source_times, target_times)
        
        # CUDA implementation
        cuda_result = robocache_module.resample_trajectories(
            source_data.cuda(),
            source_times.cuda(),
            target_times.cuda()
        ).cpu()
        
        # Compare with tight tolerances
        torch.testing.assert_close(
            cuda_result,
            cpu_result,
            rtol=1e-4,
            atol=1e-6,
            msg=f"Mismatch for shape ({batch_size}, {source_len}, {target_len}, {dim})"
        )
    
    def test_boundary_conditions(self, robocache_module):
        """Test edge cases: extrapolation, single point, etc."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.manual_seed(123)
        
        # Test 1: Extrapolation before first timestamp
        source_data = torch.randn(2, 10, 32, dtype=torch.float32)
        source_times = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]] * 2)
        target_times = torch.tensor([[0.5, 1.5, 2.5]] * 2)  # 0.5 is before first
        
        cpu_result = resample_pytorch_reference(source_data, source_times, target_times)
        cuda_result = robocache_module.resample_trajectories(
            source_data.cuda(), source_times.cuda(), target_times.cuda()
        ).cpu()
        
        torch.testing.assert_close(cuda_result, cpu_result, rtol=1e-4, atol=1e-6)
        
        # Verify first target time (0.5) matches first source data
        assert torch.allclose(cuda_result[:, 0], source_data[:, 0], rtol=1e-4)
        
        # Test 2: Extrapolation after last timestamp
        target_times = torch.tensor([[5.0, 9.5, 11.0]] * 2)  # 11.0 is after last
        
        cpu_result = resample_pytorch_reference(source_data, source_times, target_times)
        cuda_result = robocache_module.resample_trajectories(
            source_data.cuda(), source_times.cuda(), target_times.cuda()
        ).cpu()
        
        torch.testing.assert_close(cuda_result, cpu_result, rtol=1e-4, atol=1e-6)
        
        # Verify last target time (11.0) matches last source data
        assert torch.allclose(cuda_result[:, -1], source_data[:, -1], rtol=1e-4)
    
    def test_bf16_precision(self, robocache_module):
        """Test BF16 precision maintains reasonable accuracy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.manual_seed(456)
        batch_size, source_len, target_len, dim = 16, 500, 250, 128
        
        # FP32 reference
        source_data_fp32 = torch.randn(batch_size, source_len, dim, dtype=torch.float32)
        source_times = torch.linspace(0, 5, source_len).unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len).unsqueeze(0).expand(batch_size, -1)
        
        fp32_result = robocache_module.resample_trajectories(
            source_data_fp32.cuda(),
            source_times.cuda(),
            target_times.cuda()
        ).cpu()
        
        # BF16 test
        source_data_bf16 = source_data_fp32.to(torch.bfloat16)
        bf16_result = robocache_module.resample_trajectories(
            source_data_bf16.cuda(),
            source_times.cuda(),
            target_times.cuda()
        ).cpu().to(torch.float32)
        
        # BF16 should be within ~1% relative error
        torch.testing.assert_close(
            bf16_result,
            fp32_result,
            rtol=0.01,
            atol=1e-3,
            msg="BF16 precision loss too high"
        )
    
    def test_gradient_flow(self, robocache_module):
        """Test that gradients flow correctly through the operation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.manual_seed(789)
        batch_size, source_len, target_len, dim = 4, 100, 50, 64
        
        source_data = torch.randn(
            batch_size, source_len, dim,
            dtype=torch.float32,
            device="cuda",
            requires_grad=True
        )
        source_times = torch.linspace(0, 5, source_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
        target_times = torch.linspace(0, 5, target_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass
        result = robocache_module.resample_trajectories(source_data, source_times, target_times)
        
        # Backward pass
        loss = result.sum()
        loss.backward()
        
        # Check gradients exist and are reasonable
        assert source_data.grad is not None
        assert not torch.isnan(source_data.grad).any()
        assert not torch.isinf(source_data.grad).any()
        
        # Gradient should be sparse (only affected source timestamps get gradients)
        grad_norm = source_data.grad.norm().item()
        assert grad_norm > 0, "Gradient should not be zero"
        assert grad_norm < 1000, "Gradient should not explode"

