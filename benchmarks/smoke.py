"""
Smoke Test for CI

Quick validation that operations meet minimum performance thresholds.
"""

import sys
import torch


# Performance thresholds (P50 latency in milliseconds)
THRESHOLDS = {
    'h100': {
        'multimodal_fusion': 0.10,  # 0.10ms max
        'voxelization': 0.05,        # 0.05ms max
    },
    'a100': {
        'multimodal_fusion': 0.15,  # 0.15ms max
        'voxelization': 0.08,        # 0.08ms max
    },
    'default': {
        'multimodal_fusion': 0.20,  # 0.20ms max
        'voxelization': 0.10,        # 0.10ms max
    }
}

def detect_gpu():
    """Detect GPU type"""
    if not torch.cuda.is_available():
        return 'cpu'
    
    name = torch.cuda.get_device_name(0).lower()
    if 'h100' in name:
        return 'h100'
    elif 'a100' in name:
        return 'a100'
    else:
        return 'default'


def smoke_test_multimodal():
    """Quick multimodal fusion test"""
    import robocache
    
    batch = 2
    vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
    vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
    proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
    proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
    imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
    imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
    target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)
    
    # Warmup
    for _ in range(20):
        _ = robocache.fuse_multimodal(vision, vision_times, proprio, proprio_times, imu, imu_times, target)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = robocache.fuse_multimodal(vision, vision_times, proprio, proprio_times, imu, imu_times, target)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    import numpy as np
    p50 = np.percentile(times, 50)
    
    return p50


def smoke_test_voxelization():
    """Quick voxelization test"""
    import robocache
    
    points = torch.rand(100000, 3, device='cuda') * 4.0 - 2.0
    
    # Warmup
    for _ in range(10):
        _ = robocache.voxelize_pointcloud(points, grid_min=[-2, -2, -2], voxel_size=0.0625, grid_size=[128, 128, 128], mode='occupancy')
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = robocache.voxelize_pointcloud(points, grid_min=[-2, -2, -2], voxel_size=0.0625, grid_size=[128, 128, 128], mode='occupancy')
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    import numpy as np
    p50 = np.percentile(times, 50)
    
    return p50


def main():
    import robocache
    
    print("RoboCache Smoke Test")
    print("===================")
    print(f"Version: {robocache.__version__}")
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0
    
    gpu_type = detect_gpu()
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_type})")
    
    thresholds = THRESHOLDS[gpu_type]
    print(f"Thresholds: {thresholds}")
    print()
    
    # Test 1: Multimodal
    print("[1/2] Multimodal fusion...")
    p50_mm = smoke_test_multimodal()
    threshold_mm = thresholds['multimodal_fusion']
    status_mm = "PASS" if p50_mm < threshold_mm else "FAIL"
    print(f"  P50: {p50_mm:.3f}ms (threshold: {threshold_mm}ms) [{status_mm}]")
    
    # Test 2: Voxelization
    print("[2/2] Voxelization...")
    p50_vox = smoke_test_voxelization()
    threshold_vox = thresholds['voxelization']
    status_vox = "PASS" if p50_vox < threshold_vox else "FAIL"
    print(f"  P50: {p50_vox:.3f}ms (threshold: {threshold_vox}ms) [{status_vox}]")
    
    print()
    if status_mm == "PASS" and status_vox == "PASS":
        print("✓ All tests passed")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

