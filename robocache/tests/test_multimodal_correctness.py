# Copyright (c) 2025 GOATnote Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Correctness tests for multimodal sensor fusion.
Compares GPU implementation against CPU reference.
"""

import pytest
import torch
import numpy as np


def cpu_reference_multimodal_fusion(
    stream1_data, stream1_times,
    stream2_data, stream2_times,
    stream3_data, stream3_times,
    target_times
):
    """CPU reference implementation using PyTorch interpolation"""
    B, T = target_times.shape
    
    # Interpolate each stream independently
    s1_interp = torch.nn.functional.interpolate(
        stream1_data.transpose(1, 2).float(),
        size=T,
        mode='linear',
        align_corners=True
    ).transpose(1, 2)
    
    s2_interp = torch.nn.functional.interpolate(
        stream2_data.transpose(1, 2).float(),
        size=T,
        mode='linear',
        align_corners=True
    ).transpose(1, 2)
    
    s3_interp = torch.nn.functional.interpolate(
        stream3_data.transpose(1, 2).float(),
        size=T,
        mode='linear',
        align_corners=True
    ).transpose(1, 2)
    
    # Concatenate features
    return torch.cat([s1_interp, s2_interp, s3_interp], dim=2)


@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("target_len", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multimodal_fusion_correctness(batch_size, target_len, dtype):
    """Test multimodal fusion correctness against CPU reference"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Stream configurations
    s1_len, d1 = 30, 512   # Vision @ 30Hz
    s2_len, d2 = 100, 64   # Proprioception @ 100Hz
    s3_len, d3 = 200, 12   # IMU @ 200Hz
    
    # Generate test data
    torch.manual_seed(42)
    stream1_data = torch.randn(batch_size, s1_len, d1, dtype=dtype, device='cuda')
    stream1_times = torch.linspace(0, 1, s1_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    stream2_data = torch.randn(batch_size, s2_len, d2, dtype=dtype, device='cuda')
    stream2_times = torch.linspace(0, 1, s2_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    stream3_data = torch.randn(batch_size, s3_len, d3, dtype=dtype, device='cuda')
    stream3_times = torch.linspace(0, 1, s3_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    target_times = torch.linspace(0, 1, target_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    # GPU result
    try:
        gpu_result = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # CPU reference
    cpu_result = cpu_reference_multimodal_fusion(
        stream1_data, stream1_times,
        stream2_data, stream2_times,
        stream3_data, stream3_times,
        target_times
    )
    
    # Tolerances based on dtype
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    else:  # bfloat16
        rtol, atol = 1e-2, 1e-3
    
    # Compare
    torch.testing.assert_close(
        gpu_result.float().cpu(),
        cpu_result.float().cpu(),
        rtol=rtol,
        atol=atol,
        msg=f"Mismatch for batch={batch_size}, target_len={target_len}, dtype={dtype}"
    )


@pytest.mark.cuda
def test_multimodal_fusion_shape():
    """Test output shape is correct"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B, T = 4, 50
    s1, d1 = 30, 512
    s2, d2 = 100, 64
    s3, d3 = 200, 12
    
    stream1_data = torch.randn(B, s1, d1, device='cuda')
    stream1_times = torch.linspace(0, 1, s1, device='cuda').expand(B, -1)
    stream2_data = torch.randn(B, s2, d2, device='cuda')
    stream2_times = torch.linspace(0, 1, s2, device='cuda').expand(B, -1)
    stream3_data = torch.randn(B, s3, d3, device='cuda')
    stream3_times = torch.linspace(0, 1, s3, device='cuda').expand(B, -1)
    target_times = torch.linspace(0, 1, T, device='cuda').expand(B, -1)
    
    try:
        result = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    expected_shape = (B, T, d1 + d2 + d3)
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"


@pytest.mark.cuda
def test_multimodal_fusion_boundary_cases():
    """Test edge cases: extrapolation, single point, etc."""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test 1: Target times outside source range (extrapolation)
    stream1_data = torch.randn(2, 10, 8, device='cuda')
    stream1_times = torch.linspace(0.2, 0.8, 10, device='cuda').expand(2, -1)
    stream2_data = torch.randn(2, 20, 4, device='cuda')
    stream2_times = torch.linspace(0.2, 0.8, 20, device='cuda').expand(2, -1)
    stream3_data = torch.randn(2, 30, 2, device='cuda')
    stream3_times = torch.linspace(0.2, 0.8, 30, device='cuda').expand(2, -1)
    target_times = torch.linspace(0.0, 1.0, 15, device='cuda').expand(2, -1)
    
    try:
        result = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
        assert result.shape == (2, 15, 14), "Extrapolation case failed"
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise


@pytest.mark.cuda
@pytest.mark.parametrize("device_id", [0])
def test_multimodal_fusion_device_placement(device_id):
    """Test that kernel runs on specified device"""
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    if torch.cuda.device_count() <= device_id:
        pytest.skip(f"GPU {device_id} not available")
    
    device = f'cuda:{device_id}'
    
    stream1_data = torch.randn(2, 10, 8, device=device)
    stream1_times = torch.linspace(0, 1, 10, device=device).expand(2, -1)
    stream2_data = torch.randn(2, 20, 4, device=device)
    stream2_times = torch.linspace(0, 1, 20, device=device).expand(2, -1)
    stream3_data = torch.randn(2, 30, 2, device=device)
    stream3_times = torch.linspace(0, 1, 30, device=device).expand(2, -1)
    target_times = torch.linspace(0, 1, 15, device=device).expand(2, -1)
    
    try:
        result = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
        assert result.device.type == 'cuda'
        assert result.device.index == device_id
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
