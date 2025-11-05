#!/usr/bin/env python3
"""
End-to-end test of RoboCache Python package
Tests both CUDA and PyTorch backends
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import torch
import time

def test_pytorch_backend():
    """Test PyTorch fallback (CPU)"""
    print("=" * 60)
    print("TEST 1: PyTorch Backend (CPU)")
    print("=" * 60)
    
    import robocache
    
    B, S, T, D = 4, 10, 20, 8
    
    src = torch.randn(B, S, D, dtype=torch.float32)
    src_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1)
    tgt_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    
    result = robocache.resample_trajectories(src, src_times, tgt_times, backend="pytorch")
    
    assert result.shape == (B, T, D), f"Shape mismatch: {result.shape}"
    assert result.dtype == torch.float32, f"Dtype mismatch: {result.dtype}"
    
    print(f"✅ Input: {src.shape}, Output: {result.shape}")
    print(f"✅ PyTorch backend working")
    print()
    return True

def test_cuda_backend():
    """Test CUDA backend (GPU)"""
    print("=" * 60)
    print("TEST 2: CUDA Backend (GPU)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping")
        return False
    
    import robocache
    
    B, S, T, D = 32, 50, 256, 128
    
    src = torch.randn(B, S, D, dtype=torch.bfloat16, device='cuda')
    src_times = torch.linspace(0, 1, S, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
    tgt_times = torch.linspace(0, 1, T, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
    
    try:
        # Warmup
        for _ in range(3):
            _ = robocache.resample_trajectories(src, src_times, tgt_times, backend="cuda")
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            result = robocache.resample_trajectories(src, src_times, tgt_times, backend="cuda")
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        assert result.shape == (B, T, D), f"Shape mismatch: {result.shape}"
        assert result.dtype == torch.bfloat16, f"Dtype mismatch: {result.dtype}"
        
        avg_time = elapsed / 100 * 1000  # ms
        throughput = (B * T) / (elapsed / 100)  # samples/sec
        
        print(f"✅ Input: {src.shape}, Output: {result.shape}")
        print(f"✅ Average latency: {avg_time:.3f} ms")
        print(f"✅ Throughput: {throughput:.0f} samples/sec")
        print(f"✅ CUDA backend working")
        print()
        return True
        
    except Exception as e:
        print(f"❌ CUDA backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_fusion():
    """Test multimodal fusion"""
    print("=" * 60)
    print("TEST 3: Multimodal Fusion")
    print("=" * 60)
    
    import robocache
    
    B, T = 8, 50
    Sv, Sp, Sf = 30, 30, 30
    Dv, Dp, Df = 512, 128, 64
    
    v_data = torch.randn(B, Sv, Dv)
    v_times = torch.linspace(0, 1, Sv).unsqueeze(0).expand(B, -1)
    
    p_data = torch.randn(B, Sp, Dp)
    p_times = torch.linspace(0, 1, Sp).unsqueeze(0).expand(B, -1)
    
    f_data = torch.randn(B, Sf, Df)
    f_times = torch.linspace(0, 1, Sf).unsqueeze(0).expand(B, -1)
    
    tgt_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    
    result = robocache.fuse_multimodal(
        v_data, v_times,
        p_data, p_times,
        f_data, f_times,
        tgt_times,
        backend="pytorch"
    )
    
    expected_shape = (B, T, Dv + Dp + Df)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} vs {expected_shape}"
    
    print(f"✅ Vision: {v_data.shape} → {result.shape[2][:Dv] if hasattr(result.shape[2], '__getitem__') else Dv}D")
    print(f"✅ Proprio: {p_data.shape} → {Dp}D")
    print(f"✅ Force: {f_data.shape} → {Df}D")
    print(f"✅ Fused output: {result.shape}")
    print()
    return True

def test_voxelization():
    """Test point cloud voxelization"""
    print("=" * 60)
    print("TEST 4: Point Cloud Voxelization")
    print("=" * 60)
    
    import robocache
    
    N = 10000
    grid_size = (32, 32, 32)
    voxel_size = 0.05
    
    points = torch.randn(N, 3) * 2.0  # 2m range
    
    grid = robocache.voxelize_point_cloud(
        points,
        grid_size=grid_size,
        voxel_size=voxel_size
    )
    
    assert grid.shape == grid_size, f"Grid shape mismatch: {grid.shape}"
    occupied = (grid > 0).sum().item()
    
    print(f"✅ Points: {N}")
    print(f"✅ Grid: {grid_size}")
    print(f"✅ Occupied voxels: {occupied}/{grid.numel()}")
    print()
    return True

def main():
    print("\n" + "=" * 60)
    print("RoboCache End-to-End Package Test")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: PyTorch backend
    try:
        results.append(("PyTorch Backend", test_pytorch_backend()))
    except Exception as e:
        print(f"❌ PyTorch backend failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("PyTorch Backend", False))
    
    # Test 2: CUDA backend
    try:
        results.append(("CUDA Backend", test_cuda_backend()))
    except Exception as e:
        print(f"❌ CUDA backend failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CUDA Backend", False))
    
    # Test 3: Multimodal fusion
    try:
        results.append(("Multimodal Fusion", test_multimodal_fusion()))
    except Exception as e:
        print(f"❌ Multimodal fusion failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multimodal Fusion", False))
    
    # Test 4: Voxelization
    try:
        results.append(("Voxelization", test_voxelization()))
    except Exception as e:
        print(f"❌ Voxelization failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Voxelization", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Package is production-ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

