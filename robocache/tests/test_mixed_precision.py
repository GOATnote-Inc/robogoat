"""Mixed precision accuracy tests - verify numerical behavior across dtypes"""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multimodal_fusion_fp32_bf16_accuracy():
    """FP32 and BF16 results within acceptable tolerance"""
    import robocache
    
    batch = 4
    
    # FP32
    vision_fp32 = torch.randn(batch, 30, 512, dtype=torch.float32, device='cuda')
    vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
    proprio_fp32 = torch.randn(batch, 100, 64, dtype=torch.float32, device='cuda')
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
    imu_fp32 = torch.randn(batch, 200, 12, dtype=torch.float32, device='cuda')
    imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)
    
    out_fp32 = robocache.fuse_multimodal(
        vision_fp32, vision_times, proprio_fp32, proprio_times, imu_fp32, imu_times, target
    )
    
    # BF16 (same data)
    vision_bf16 = vision_fp32.to(torch.bfloat16)
    proprio_bf16 = proprio_fp32.to(torch.bfloat16)
    imu_bf16 = imu_fp32.to(torch.bfloat16)
    
    out_bf16 = robocache.fuse_multimodal(
        vision_bf16, vision_times, proprio_bf16, proprio_times, imu_bf16, imu_times, target
    )
    
    # BF16 has ~3 decimal digits precision, allow larger tolerance
    mae = torch.abs(out_fp32 - out_bf16.float()).mean()
    assert mae < 0.01, f"BF16 accuracy degraded: MAE={mae:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_voxelization_fp32_accuracy():
    """Voxelization maintains accuracy in FP32"""
    import robocache
    
    points = torch.rand(50000, 3, device='cuda') * 4.0 - 2.0
    features = torch.randn(50000, 8, device='cuda')
    
    grid = robocache.voxelize_pointcloud(
        points, features, grid_min=[-2, -2, -2], voxel_size=0.0625,
        grid_size=[128, 128, 128], mode='mean'
    )
    
    # Check no NaNs or Infs
    assert not torch.isnan(grid).any(), "NaN in voxel grid"
    assert not torch.isinf(grid).any(), "Inf in voxel grid"
    
    # Check reasonable value range
    assert grid.abs().max() < 100, "Unreasonable values in grid"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dtype_promotion():
    """Verify dtype promotion rules"""
    import robocache
    
    batch = 2
    
    # Mixed dtypes should work
    vision = torch.randn(batch, 30, 512, dtype=torch.float16, device='cuda')
    vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.float32, device='cuda')
    imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)
    
    out = robocache.fuse_multimodal(
        vision, vision_times, proprio, proprio_times, imu, imu_times, target
    )
    
    # Output should be highest precision input (fp32)
    assert out.dtype == torch.float32, f"Expected fp32, got {out.dtype}"

