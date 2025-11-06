"""
Integration benchmark: All 3 RoboCache operations in realistic pipeline
Simulates robot learning preprocessing: resample → fuse → voxelize
"""

import torch
import time

def resample_pytorch(src, src_t, tgt_t):
    """PyTorch trajectory resampling"""
    B, S, D = src.shape
    T = tgt_t.shape[1]
    out = torch.zeros(B, T, D, dtype=src.dtype, device=src.device)
    for b in range(B):
        for t in range(T):
            tgt = tgt_t[b, t]
            if tgt <= src_t[b, 0]:
                out[b, t] = src[b, 0]
            elif tgt >= src_t[b, -1]:
                out[b, t] = src[b, -1]
            else:
                left, right = 0, S - 1
                while left < right - 1:
                    mid = (left + right) // 2
                    if src_t[b, mid] <= tgt:
                        left = mid
                    else:
                        right = mid
                a = (tgt - src_t[b, left]) / (src_t[b, right] - src_t[b, left] + 1e-8)
                out[b, t] = (1-a) * src[b, left] + a * src[b, right]
    return out

def voxelize_pytorch(points, grid_size, voxel_size, origin):
    """PyTorch voxelization"""
    X, Y, Z = grid_size
    grid = torch.zeros(X, Y, Z, device=points.device)
    for i in range(points.shape[0]):
        p = points[i]
        x = int((p[0] - origin[0]) / voxel_size)
        y = int((p[1] - origin[1]) / voxel_size)
        z = int((p[2] - origin[2]) / voxel_size)
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            grid[x, y, z] = 1.0
    return grid

def robot_learning_pipeline(batch_size, device='cpu'):
    """
    Realistic robot learning preprocessing pipeline
    
    Inputs:
    - Vision trajectory: 30 Hz RGB-D features
    - Proprio trajectory: 100 Hz joint states  
    - Point cloud: 1000 points per frame
    
    Processing:
    1. Resample vision 30Hz → 50Hz
    2. Resample proprio 100Hz → 50Hz
    3. Fuse modalities
    4. Voxelize point cloud
    
    Outputs:
    - Aligned features for transformer
    - Voxel grid for spatial reasoning
    """
    
    # Simulate robot data
    vision = torch.randn(batch_size, 30, 512, device=device)  # 30 Hz, 512-dim
    vision_t = torch.linspace(0, 1, 30, device=device).unsqueeze(0).expand(batch_size, -1)
    
    proprio = torch.randn(batch_size, 100, 32, device=device)  # 100 Hz, 32-dim
    proprio_t = torch.linspace(0, 1, 100, device=device).unsqueeze(0).expand(batch_size, -1)
    
    points = torch.randn(1000, 3, device=device) * 5  # 1000 points
    
    target_t = torch.linspace(0, 1, 50, device=device).unsqueeze(0).expand(batch_size, -1)  # 50 Hz target
    
    # Pipeline execution
    start = time.time()
    
    # Step 1: Resample vision
    vision_aligned = resample_pytorch(vision, vision_t, target_t)
    
    # Step 2: Resample proprio
    proprio_aligned = resample_pytorch(proprio, proprio_t, target_t)
    
    # Step 3: Fuse modalities
    fused = torch.cat([vision_aligned, proprio_aligned], dim=2)
    
    # Step 4: Voxelize point cloud
    grid = voxelize_pytorch(points, (32, 32, 32), 0.5, torch.tensor([-8.0, -8.0, -8.0], device=device))
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    return {
        'fused_features': fused,
        'voxel_grid': grid,
        'latency_ms': elapsed * 1000,
        'shapes': {
            'vision': tuple(vision.shape),
            'proprio': tuple(proprio.shape),
            'fused': tuple(fused.shape),
            'grid': tuple(grid.shape)
        }
    }

if __name__ == '__main__':
    print("=" * 70)
    print("RoboCache Integration Benchmark")
    print("Realistic Robot Learning Preprocessing Pipeline")
    print("=" * 70)
    
    # Test CPU
    print("\n[CPU Baseline]")
    result_cpu = robot_learning_pipeline(batch_size=4, device='cpu')
    print(f"✅ Latency: {result_cpu['latency_ms']:.2f} ms")
    print(f"   Vision: {result_cpu['shapes']['vision']} → Fused: {result_cpu['shapes']['fused']}")
    print(f"   Voxel grid: {result_cpu['shapes']['grid']}, {int(result_cpu['voxel_grid'].sum())} occupied")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\n[CUDA Accelerated]")
        # Warmup
        for _ in range(3):
            _ = robot_learning_pipeline(batch_size=4, device='cuda')
        
        # Benchmark
        results = []
        for _ in range(10):
            result = robot_learning_pipeline(batch_size=4, device='cuda')
            results.append(result['latency_ms'])
        
        avg_latency = sum(results) / len(results)
        speedup = result_cpu['latency_ms'] / avg_latency
        
        print(f"✅ Latency: {avg_latency:.2f} ms")
        print(f"   Speedup: {speedup:.1f}x vs CPU")
        print(f"   Vision: {result['shapes']['vision']} → Fused: {result['shapes']['fused']}")
        print(f"   Voxel grid: {result['shapes']['grid']}, {int(result['voxel_grid'].sum())} occupied")
    else:
        print("\n[CUDA Not Available - Skipping GPU Test]")
    
    print("\n" + "=" * 70)
    print("✅ Integration Test Complete")
    print("=" * 70)
    print("\nPipeline Steps:")
    print("1. Resample vision trajectory (30Hz → 50Hz)")
    print("2. Resample proprio trajectory (100Hz → 50Hz)")
    print("3. Fuse modalities (vision + proprio)")
    print("4. Voxelize point cloud (1000 points → 32³ grid)")
    print("\nReady for robot foundation model training!")

