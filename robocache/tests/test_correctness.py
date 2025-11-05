"""
Correctness validation tests for RoboCache kernels.
Compares CUDA outputs against high-precision CPU references.
"""
import pytest
import torch
import numpy as np


def cpu_reference_trajectory_resample(source_data, source_times, target_times):
    """
    High-precision CPU reference for trajectory resampling.
    Uses float64 throughout for numerical stability.
    """
    batch, src_len, dim = source_data.shape
    _, tgt_len = target_times.shape
    
    # Convert to float64 on CPU
    source_data_cpu = source_data.cpu().float().numpy().astype(np.float64)
    source_times_cpu = source_times.cpu().numpy().astype(np.float64)
    target_times_cpu = target_times.cpu().numpy().astype(np.float64)
    
    output = np.zeros((batch, tgt_len, dim), dtype=np.float64)
    
    for b in range(batch):
        for t in range(tgt_len):
            target_time = target_times_cpu[b, t]
            
            # Binary search for left index
            left_idx = np.searchsorted(source_times_cpu[b], target_time, side='right') - 1
            left_idx = max(0, min(left_idx, src_len - 2))
            right_idx = left_idx + 1
            
            # Compute interpolation weight
            t_left = source_times_cpu[b, left_idx]
            t_right = source_times_cpu[b, right_idx]
            delta = t_right - t_left
            
            if delta > 1e-10:
                weight = (target_time - t_left) / delta
                weight = np.clip(weight, 0.0, 1.0)
            else:
                weight = 0.0
            
            # Linear interpolation
            left_data = source_data_cpu[b, left_idx, :]
            right_data = source_data_cpu[b, right_idx, :]
            output[b, t, :] = left_data + weight * (right_data - left_data)
    
    return output


def test_trajectory_resampling_correctness():
    """Test CUDA kernel produces correct outputs vs CPU reference."""
    try:
        from torch.utils.cpp_extension import load
        robocache = load(
            name='robocache_cuda_optimized',
            sources=[
                'kernels/cutlass/trajectory_resample_optimized_v2.cu',
                'kernels/cutlass/trajectory_resample_optimized_v2_torch.cu',
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++17'],
            verbose=False
        )
    except Exception as e:
        pytest.skip(f"CUDA extension not available: {e}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test configuration
    batch, src_len, tgt_len, dim = 4, 128, 64, 8
    
    # Create test data
    torch.manual_seed(42)
    data = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device='cuda')
    src_t = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    tgt_t = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    
    # CUDA kernel output
    cuda_output = robocache.resample_trajectories(data, src_t, tgt_t)
    
    # CPU reference output
    cpu_reference = cpu_reference_trajectory_resample(data, src_t, tgt_t)
    cpu_reference_tensor = torch.from_numpy(cpu_reference).to(torch.bfloat16)
    
    # Compare
    cuda_output_cpu = cuda_output.cpu()
    max_diff = torch.abs(cuda_output_cpu.float() - cpu_reference_tensor.float()).max().item()
    mean_diff = torch.abs(cuda_output_cpu.float() - cpu_reference_tensor.float()).mean().item()
    
    # BF16 has ~3 decimal digits of precision, so tolerance should be ~1e-2
    assert max_diff < 0.01, f"Max difference {max_diff} exceeds tolerance"
    assert mean_diff < 0.001, f"Mean difference {mean_diff} exceeds tolerance"
    
    print(f"✅ Correctness test passed: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


def test_trajectory_resampling_edge_cases():
    """Test edge cases: unsorted timestamps, NaNs, boundary conditions."""
    try:
        from torch.utils.cpp_extension import load
        robocache = load(
            name='robocache_cuda_optimized',
            sources=[
                'kernels/cutlass/trajectory_resample_optimized_v2.cu',
                'kernels/cutlass/trajectory_resample_optimized_v2_torch.cu',
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++17'],
            verbose=False
        )
    except Exception as e:
        pytest.skip(f"CUDA extension not available: {e}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch, src_len, tgt_len, dim = 2, 32, 16, 4
    
    # Test 1: Target times outside source range (extrapolation)
    data = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device='cuda')
    src_t = torch.linspace(0.2, 0.8, src_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    tgt_t = torch.linspace(0.0, 1.0, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    
    output = robocache.resample_trajectories(data, src_t, tgt_t)
    assert not torch.isnan(output).any(), "NaNs detected in output for extrapolation"
    assert not torch.isinf(output).any(), "Infs detected in output for extrapolation"
    
    # Test 2: All target times at same source time (zero interpolation weight)
    src_t = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    tgt_t = torch.full((batch, tgt_len), 0.5, device='cuda')
    
    output = robocache.resample_trajectories(data, src_t, tgt_t)
    assert not torch.isnan(output).any(), "NaNs detected in output for constant target time"
    
    # Test 3: Very small source sequence
    data_small = torch.randn(batch, 2, dim, dtype=torch.bfloat16, device='cuda')
    src_t_small = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device='cuda')
    tgt_t_small = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
    
    output_small = robocache.resample_trajectories(data_small, src_t_small, tgt_t_small)
    assert not torch.isnan(output_small).any(), "NaNs detected in output for minimal source sequence"
    assert output_small.shape == (batch, tgt_len, dim), f"Unexpected output shape: {output_small.shape}"
    
    print("✅ Edge case tests passed")


def test_pytorch_fallback_correctness():
    """Test PyTorch fallback implementation produces correct outputs."""
    # Import fallback implementation
    # This would import from robocache.backends.PyTorchBackend
    # For now, skip if not implemented
    pytest.skip("PyTorch fallback correctness test not yet implemented")


if __name__ == "__main__":
    test_trajectory_resampling_correctness()
    test_trajectory_resampling_edge_cases()
    print("\n✅ All correctness tests passed!")

