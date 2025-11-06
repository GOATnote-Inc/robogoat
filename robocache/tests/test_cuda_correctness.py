#!/usr/bin/env python3
"""
RoboCache CUDA Correctness Tests
Validates GPU kernels against CPU reference with tight tolerances
"""
import pytest
import torch
import numpy as np

try:
    import robocache
    CUDA_AVAILABLE = robocache.is_cuda_available() and torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Test configurations
BATCH_SIZES = [1, 8, 32]
SOURCE_LENS = [50, 100, 500]
TARGET_LENS = [25, 50, 256]
DIMS = [64, 128, 256]
DTYPES = [torch.float32, torch.bfloat16]


def cpu_reference_resample(source_data, source_times, target_times):
    """CPU reference implementation for correctness validation"""
    B, S, D = source_data.shape
    T = target_times.shape[1]
    
    result = torch.zeros(B, T, D, dtype=source_data.dtype)
    
    for b in range(B):
        for t in range(T):
            tgt_t = target_times[b, t].item()
            
            # Binary search
            idx = torch.searchsorted(source_times[b], tgt_t).item()
            
            if idx == 0:
                result[b, t] = source_data[b, 0]
            elif idx >= S:
                result[b, t] = source_data[b, -1]
            else:
                t0 = source_times[b, idx - 1].item()
                t1 = source_times[b, idx].item()
                alpha = (tgt_t - t0) / (t1 - t0 + 1e-8)
                result[b, t] = (1 - alpha) * source_data[b, idx - 1] + alpha * source_data[b, idx]
    
    return result


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("source_len", SOURCE_LENS)
@pytest.mark.parametrize("target_len", TARGET_LENS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_resample_correctness(batch_size, source_len, target_len, dim, dtype):
    """Test CUDA kernel matches CPU reference"""
    torch.manual_seed(42)
    
    # Generate data
    source_data = torch.randn(batch_size, source_len, dim, dtype=dtype)
    source_times = torch.linspace(0, 5, source_len).unsqueeze(0).expand(batch_size, -1)
    target_times = torch.linspace(0, 5, target_len).unsqueeze(0).expand(batch_size, -1)
    
    # CPU reference
    cpu_result = cpu_reference_resample(
        source_data.float(),
        source_times,
        target_times
    )
    
    # GPU result
    gpu_result = robocache.resample_trajectories(
        source_data.cuda(),
        source_times.cuda(),
        target_times.cuda()
    ).cpu().float()
    
    # Validate
    rtol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-4 if dtype == torch.bfloat16 else 1e-6
    
    torch.testing.assert_close(
        gpu_result, cpu_result.to(gpu_result.dtype),
        rtol=rtol, atol=atol,
        msg=f"Mismatch: batch={batch_size}, src={source_len}, tgt={target_len}, dim={dim}, dtype={dtype}"
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
def test_resample_boundary_cases():
    """Test edge cases: extrapolation, single point, etc."""
    torch.manual_seed(42)
    
    # Test 1: Target times outside source range
    source_data = torch.randn(2, 10, 8, dtype=torch.float32).cuda()
    source_times = torch.linspace(1, 2, 10).unsqueeze(0).expand(2, -1).cuda()
    target_times = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5], [0.5, 1.0, 1.5, 2.0, 2.5]]).cuda()
    
    result = robocache.resample_trajectories(source_data, source_times, target_times)
    
    # First value should clamp to start
    assert torch.allclose(result[:, 0], source_data[:, 0], rtol=1e-5)
    # Last value should clamp to end
    assert torch.allclose(result[:, -1], source_data[:, -1], rtol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
def test_resample_identical_times():
    """Test when source and target times are identical"""
    torch.manual_seed(42)
    
    source_data = torch.randn(4, 20, 16, dtype=torch.float32).cuda()
    times = torch.linspace(0, 3, 20).unsqueeze(0).expand(4, -1).cuda()
    
    result = robocache.resample_trajectories(source_data, times, times)
    
    torch.testing.assert_close(result, source_data, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
def test_resample_dtype_preservation():
    """Test that output dtype matches input dtype"""
    for dtype in [torch.float32, torch.bfloat16]:
        source_data = torch.randn(2, 10, 8, dtype=dtype).cuda()
        source_times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1).cuda()
        target_times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1).cuda()
        
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        
        assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
def test_resample_device_handling():
    """Test automatic device handling"""
    source_data = torch.randn(2, 10, 8).cuda()
    source_times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1).cuda()
    target_times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1).cuda()
    
    result = robocache.resample_trajectories(source_data, source_times, target_times)
    
    assert result.is_cuda
    assert result.device == source_data.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fallback_when_cuda_kernels_unavailable():
    """Test PyTorch fallback works"""
    # This tests the fallback path regardless of CUDA kernel availability
    source_data = torch.randn(2, 10, 8).cuda()
    source_times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1).cuda()
    target_times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1).cuda()
    
    # Force fallback
    result = robocache._resample_pytorch(source_data, source_times, target_times)
    
    assert result.shape == (2, 5, 8)
    assert result.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

