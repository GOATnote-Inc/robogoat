# Copyright (c) 2025 GOATnote Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Timestamp-aware correctness tests for multimodal fusion.

These tests specifically validate that temporal alignment respects actual
timestamps, not just indices. They test non-uniform, phase-shifted, and
jittered timestamp grids to ensure robust interpolation.

Addresses Codex Issue #2: Previous tests used torch.nn.functional.interpolate
which interpolates by index, completely ignoring timestamp tensors.
"""

import pytest
import torch
import numpy as np


def timestamp_aware_interpolation_reference(
    features: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation that actually uses timestamps.
    
    Uses binary search + linear interpolation based on actual time values.
    """
    batch_size, T_src, D = features.shape
    T_tgt = target_times.shape[1]
    
    result = []
    for b in range(batch_size):
        # For each target time, find surrounding source times
        indices = torch.searchsorted(source_times[b], target_times[b])
        indices = torch.clamp(indices, 1, T_src - 1)
        
        idx_left = indices - 1
        idx_right = indices
        
        t_left = source_times[b][idx_left]
        t_right = source_times[b][idx_right]
        
        # Linear interpolation weight
        alpha = (target_times[b] - t_left) / (t_right - t_left + 1e-8)
        alpha = alpha.clamp(0, 1).unsqueeze(-1)  # (T_tgt, 1)
        
        feat_left = features[b][idx_left]
        feat_right = features[b][idx_right]
        
        interp = feat_left * (1 - alpha) + feat_right * alpha
        result.append(interp)
    
    return torch.stack(result, dim=0)


@pytest.mark.cuda
def test_multimodal_nonuniform_timestamps():
    """
    Test with non-uniform timestamp grids.
    
    This catches bugs where interpolation uses indices instead of timestamps.
    """
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 4
    
    # Non-uniform timestamps: clustered at beginning and end
    vision_times = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.2, 0.7, 0.8, 0.9, 0.95, 1.0], device='cuda').expand(B, -1)
    proprio_times = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device='cuda').expand(B, -1)
    imu_times = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], device='cuda').expand(B, -1)
    
    # Target times: different non-uniform grid
    target_times = torch.tensor([0.0, 0.12, 0.25, 0.35, 0.5, 0.75, 0.88, 1.0], device='cuda').expand(B, -1)
    
    # Generate features
    vision = torch.randn(B, 10, 512, device='cuda')
    proprio = torch.randn(B, 11, 64, device='cuda')
    imu = torch.randn(B, 10, 12, device='cuda')
    
    # GPU result
    try:
        gpu_result = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # CPU reference using actual timestamps
    vision_ref = timestamp_aware_interpolation_reference(vision, vision_times, target_times)
    proprio_ref = timestamp_aware_interpolation_reference(proprio, proprio_times, target_times)
    imu_ref = timestamp_aware_interpolation_reference(imu, imu_times, target_times)
    cpu_result = torch.cat([vision_ref, proprio_ref, imu_ref], dim=-1)
    
    # Compare
    torch.testing.assert_close(
        gpu_result.cpu(),
        cpu_result.cpu(),
        rtol=1e-3,
        atol=1e-4,
        msg="Non-uniform timestamp interpolation mismatch"
    )


@pytest.mark.cuda
def test_multimodal_phase_shifted_timestamps():
    """
    Test with phase-shifted timestamp grids.
    
    Streams start at different times, testing temporal alignment robustness.
    """
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 2
    
    # Phase-shifted: each stream starts at different time
    vision_times = torch.linspace(0.0, 1.0, 30, device='cuda').expand(B, -1)
    proprio_times = torch.linspace(0.1, 1.0, 50, device='cuda').expand(B, -1)  # Starts later
    imu_times = torch.linspace(0.0, 0.9, 40, device='cuda').expand(B, -1)      # Ends earlier
    
    target_times = torch.linspace(0.2, 0.8, 25, device='cuda').expand(B, -1)   # Only overlapping region
    
    vision = torch.randn(B, 30, 256, device='cuda')
    proprio = torch.randn(B, 50, 32, device='cuda')
    imu = torch.randn(B, 40, 6, device='cuda')
    
    try:
        gpu_result = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # CPU reference
    vision_ref = timestamp_aware_interpolation_reference(vision, vision_times, target_times)
    proprio_ref = timestamp_aware_interpolation_reference(proprio, proprio_times, target_times)
    imu_ref = timestamp_aware_interpolation_reference(imu, imu_times, target_times)
    cpu_result = torch.cat([vision_ref, proprio_ref, imu_ref], dim=-1)
    
    torch.testing.assert_close(
        gpu_result.cpu(),
        cpu_result.cpu(),
        rtol=1e-3,
        atol=1e-4,
        msg="Phase-shifted timestamp interpolation mismatch"
    )


@pytest.mark.cuda
def test_multimodal_jittered_timestamps():
    """
    Test with jittered (noisy) timestamps.
    
    Simulates real sensor data with timing uncertainty.
    """
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 4
    torch.manual_seed(42)
    
    # Base uniform grid + random jitter
    vision_base = torch.linspace(0, 1, 20, device='cuda')
    vision_jitter = torch.rand(20, device='cuda') * 0.02 - 0.01  # ±10ms jitter
    vision_times = (vision_base + vision_jitter).sort()[0].expand(B, -1)
    
    proprio_base = torch.linspace(0, 1, 50, device='cuda')
    proprio_jitter = torch.rand(50, device='cuda') * 0.01 - 0.005  # ±5ms jitter
    proprio_times = (proprio_base + proprio_jitter).sort()[0].expand(B, -1)
    
    imu_base = torch.linspace(0, 1, 100, device='cuda')
    imu_jitter = torch.rand(100, device='cuda') * 0.005 - 0.0025  # ±2.5ms jitter
    imu_times = (imu_base + imu_jitter).sort()[0].expand(B, -1)
    
    target_times = torch.linspace(0.1, 0.9, 30, device='cuda').expand(B, -1)
    
    vision = torch.randn(B, 20, 128, device='cuda')
    proprio = torch.randn(B, 50, 32, device='cuda')
    imu = torch.randn(B, 100, 6, device='cuda')
    
    try:
        gpu_result = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # CPU reference
    vision_ref = timestamp_aware_interpolation_reference(vision, vision_times, target_times)
    proprio_ref = timestamp_aware_interpolation_reference(proprio, proprio_times, target_times)
    imu_ref = timestamp_aware_interpolation_reference(imu, imu_times, target_times)
    cpu_result = torch.cat([vision_ref, proprio_ref, imu_ref], dim=-1)
    
    torch.testing.assert_close(
        gpu_result.cpu(),
        cpu_result.cpu(),
        rtol=1e-3,
        atol=1e-4,
        msg="Jittered timestamp interpolation mismatch"
    )


@pytest.mark.cuda
def test_multimodal_skewed_sampling_rates():
    """
    Test with vastly different sampling rates per stream.
    
    Vision at 10Hz, proprio at 100Hz, IMU at 500Hz.
    """
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 2
    
    # Different sampling rates
    vision_times = torch.linspace(0, 1, 10, device='cuda').expand(B, -1)   # 10Hz
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(B, -1) # 100Hz
    imu_times = torch.linspace(0, 1, 500, device='cuda').expand(B, -1)     # 500Hz
    
    target_times = torch.linspace(0, 1, 50, device='cuda').expand(B, -1)   # 50Hz output
    
    vision = torch.randn(B, 10, 512, device='cuda')
    proprio = torch.randn(B, 100, 64, device='cuda')
    imu = torch.randn(B, 500, 12, device='cuda')
    
    try:
        gpu_result = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # CPU reference
    vision_ref = timestamp_aware_interpolation_reference(vision, vision_times, target_times)
    proprio_ref = timestamp_aware_interpolation_reference(proprio, proprio_times, target_times)
    imu_ref = timestamp_aware_interpolation_reference(imu, imu_times, target_times)
    cpu_result = torch.cat([vision_ref, proprio_ref, imu_ref], dim=-1)
    
    torch.testing.assert_close(
        gpu_result.cpu(),
        cpu_result.cpu(),
        rtol=1e-3,
        atol=1e-4,
        msg="Skewed sampling rate interpolation mismatch"
    )


@pytest.mark.cuda
def test_multimodal_extrapolation_clamp():
    """
    Test that extrapolation clamps to boundary values.
    
    Target times outside source range should use first/last values.
    """
    try:
        import robocache
    except ImportError:
        pytest.skip("robocache not installed")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 1
    
    # Source spans 0.2 to 0.8, target spans 0.0 to 1.0
    vision_times = torch.linspace(0.2, 0.8, 20, device='cuda').expand(B, -1)
    proprio_times = torch.linspace(0.3, 0.7, 30, device='cuda').expand(B, -1)
    imu_times = torch.linspace(0.25, 0.75, 40, device='cuda').expand(B, -1)
    
    target_times = torch.linspace(0.0, 1.0, 50, device='cuda').expand(B, -1)
    
    # Use constant features to test clamping
    vision = torch.ones(B, 20, 8, device='cuda') * 10.0
    proprio = torch.ones(B, 30, 4, device='cuda') * 20.0
    imu = torch.ones(B, 40, 2, device='cuda') * 30.0
    
    try:
        gpu_result = robocache.fuse_multimodal(
            vision, vision_times,
            proprio, proprio_times,
            imu, imu_times,
            target_times
        )
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("Multimodal fusion kernel not compiled")
        raise
    
    # Extrapolated regions should have same values as boundaries
    # (first and last few timesteps should be constant)
    assert torch.allclose(gpu_result[0, 0, :8], torch.tensor([10.0]*8, device='cuda'), atol=0.1)
    assert torch.allclose(gpu_result[0, -1, :8], torch.tensor([10.0]*8, device='cuda'), atol=0.1)
    
    # Middle should also be constant (since input is constant)
    assert torch.allclose(gpu_result[0, 25, :8], torch.tensor([10.0]*8, device='cuda'), atol=0.1)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

