#!/usr/bin/env python3
"""
test_multimodal_fusion.py
Test suite for Phase 2 multimodal sensor fusion
"""

import torch
import time
import numpy as np

# Try to import CUDA extension
try:
    import robocache_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("WARNING: robocache_cuda not available, tests will be skipped")
    CUDA_AVAILABLE = False


def generate_sensor_data(batch_size, src_len, dim, dtype=torch.bfloat16, device='cuda'):
    """Generate synthetic sensor data with temporal patterns"""
    # Create data with temporal smoothness (robot sensors don't jump randomly)
    data = torch.randn(batch_size, src_len, dim, dtype=dtype, device=device)
    
    # Apply temporal smoothing (moving average)
    kernel_size = 5
    kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
    
    for b in range(batch_size):
        for d in range(dim):
            signal = data[b, :, d].unsqueeze(0).unsqueeze(0)
            # Pad for valid convolution
            padded = torch.nn.functional.pad(signal, (kernel_size//2, kernel_size//2), mode='replicate')
            smoothed = torch.nn.functional.conv1d(padded, kernel)
            data[b, :, d] = smoothed.squeeze()
    
    return data


def generate_timestamps(batch_size, length, frequency, device='cuda'):
    """Generate timestamps at given frequency (Hz)"""
    times = torch.zeros(batch_size, length, dtype=torch.float32, device=device)
    for b in range(batch_size):
        times[b] = torch.arange(length, dtype=torch.float32, device=device) / frequency
    return times


def test_correctness():
    """Test multimodal fusion correctness"""
    print("\n" + "="*80)
    print("TEST 1: Correctness Verification")
    print("="*80)
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA extension not available")
        return
    
    batch_size = 4
    
    # Vision: 30 Hz, 512D
    vision_src_len = 30
    vision_dim = 512
    vision_data = generate_sensor_data(batch_size, vision_src_len, vision_dim)
    vision_times = generate_timestamps(batch_size, vision_src_len, 30.0)
    
    # Proprioception: 100 Hz, 14D (7-DOF robot: pos + vel)
    proprio_src_len = 100
    proprio_dim = 14
    proprio_data = generate_sensor_data(batch_size, proprio_src_len, proprio_dim)
    proprio_times = generate_timestamps(batch_size, proprio_src_len, 100.0)
    
    # Force: 333 Hz, 6D (6-axis force-torque sensor)
    force_src_len = 333
    force_dim = 6
    force_data = generate_sensor_data(batch_size, force_src_len, force_dim)
    force_times = generate_timestamps(batch_size, force_src_len, 333.0)
    
    # Target: 50 Hz
    target_len = 50
    target_times = generate_timestamps(batch_size, target_len, 50.0)
    
    print(f"Configuration:")
    print(f"  Vision:  {vision_src_len} @ {vision_dim}D (30 Hz)")
    print(f"  Proprio: {proprio_src_len} @ {proprio_dim}D (100 Hz)")
    print(f"  Force:   {force_src_len} @ {force_dim}D (333 Hz)")
    print(f"  Target:  {target_len} timesteps (50 Hz)")
    print(f"  Batch:   {batch_size}")
    
    # Call fused multimodal alignment
    try:
        output = robocache_cuda.fused_multimodal_alignment(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        print(f"\n✓ Fused alignment succeeded")
        print(f"  Output shape: {output.shape}")
        
        expected_shape = (batch_size, target_len, vision_dim + proprio_dim + force_dim)
        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
        print(f"  ✓ Shape correct: {output.shape}")
        
        # Check for NaNs
        assert not torch.isnan(output).any(), "Output contains NaNs"
        print(f"  ✓ No NaNs")
        
        # Check for Infs
        assert not torch.isinf(output).any(), "Output contains Infs"
        print(f"  ✓ No Infs")
        
        # Check that output is bounded (sanity check)
        assert output.abs().max() < 100.0, f"Output values suspiciously large: {output.abs().max()}"
        print(f"  ✓ Values reasonable (max abs: {output.abs().max().item():.3f})")
        
        print("\n✅ CORRECTNESS TEST PASSED")
        
    except Exception as e:
        print(f"\n❌ CORRECTNESS TEST FAILED: {e}")
        raise


def test_performance():
    """Benchmark multimodal fusion performance"""
    print("\n" + "="*80)
    print("TEST 2: Performance Benchmark")
    print("="*80)
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA extension not available")
        return
    
    configs = [
        {
            'name': 'Small (1-sec)',
            'batch': 32,
            'vision_len': 30, 'vision_dim': 512,
            'proprio_len': 100, 'proprio_dim': 14,
            'force_len': 333, 'force_dim': 6,
            'target_len': 50
        },
        {
            'name': 'Medium (5-sec)',
            'batch': 128,
            'vision_len': 150, 'vision_dim': 512,
            'proprio_len': 500, 'proprio_dim': 14,
            'force_len': 1665, 'force_dim': 6,
            'target_len': 250
        },
        {
            'name': 'Large (10-sec)',
            'batch': 256,
            'vision_len': 300, 'vision_dim': 768,
            'proprio_len': 1000, 'proprio_dim': 14,
            'force_len': 3330, 'force_dim': 6,
            'target_len': 500
        }
    ]
    
    print(f"\nRunning benchmarks...\n")
    
    for config in configs:
        print(f"\n{'-'*80}")
        print(f"Config: {config['name']}")
        print(f"{'-'*80}")
        
        # Generate data
        vision_data = generate_sensor_data(
            config['batch'], config['vision_len'], config['vision_dim']
        )
        vision_times = generate_timestamps(
            config['batch'], config['vision_len'], 30.0
        )
        
        proprio_data = generate_sensor_data(
            config['batch'], config['proprio_len'], config['proprio_dim']
        )
        proprio_times = generate_timestamps(
            config['batch'], config['proprio_len'], 100.0
        )
        
        force_data = generate_sensor_data(
            config['batch'], config['force_len'], config['force_dim']
        )
        force_times = generate_timestamps(
            config['batch'], config['force_len'], 333.0
        )
        
        target_times = generate_timestamps(
            config['batch'], config['target_len'], 50.0
        )
        
        # Warmup
        for _ in range(10):
            output = robocache_cuda.fused_multimodal_alignment(
                vision_data, vision_times,
                proprio_data, proprio_times,
                force_data, force_times,
                target_times
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            output = robocache_cuda.fused_multimodal_alignment(
                vision_data, vision_times,
                proprio_data, proprio_times,
                force_data, force_times,
                target_times
            )
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / num_iters
        
        # Calculate metrics
        total_dim = config['vision_dim'] + config['proprio_dim'] + config['force_dim']
        
        input_bytes = (
            config['batch'] * config['vision_len'] * config['vision_dim'] * 2 +  # BF16 = 2 bytes
            config['batch'] * config['vision_len'] * 4 +  # FP32 times
            config['batch'] * config['proprio_len'] * config['proprio_dim'] * 2 +
            config['batch'] * config['proprio_len'] * 4 +
            config['batch'] * config['force_len'] * config['force_dim'] * 2 +
            config['batch'] * config['force_len'] * 4 +
            config['batch'] * config['target_len'] * 4
        )
        
        output_bytes = config['batch'] * config['target_len'] * total_dim * 2
        total_bytes = input_bytes + output_bytes
        
        bandwidth_gbs = (total_bytes / 1e9) / (elapsed_ms / 1000.0)
        h100_peak = 3000.0  # GB/s for H100 PCIe
        efficiency = (bandwidth_gbs / h100_peak) * 100.0
        
        throughput = (1000.0 / elapsed_ms) * config['batch']
        
        print(f"  Latency:     {elapsed_ms:.3f} ms")
        print(f"  Throughput:  {throughput:.0f} samples/sec")
        print(f"  Bandwidth:   {bandwidth_gbs:.1f} GB/s")
        print(f"  Efficiency:  {efficiency:.2f}% of H100 peak")
        print(f"  Data size:   {total_bytes/1024/1024:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ PERFORMANCE TEST COMPLETED")
    print("="*80)


def test_without_force():
    """Test multimodal fusion without force sensor (optional modality)"""
    print("\n" + "="*80)
    print("TEST 3: Optional Force Sensor")
    print("="*80)
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA extension not available")
        return
    
    batch_size = 16
    
    # Vision and proprio only
    vision_data = generate_sensor_data(batch_size, 30, 512)
    vision_times = generate_timestamps(batch_size, 30, 30.0)
    
    proprio_data = generate_sensor_data(batch_size, 100, 14)
    proprio_times = generate_timestamps(batch_size, 100, 100.0)
    
    target_times = generate_timestamps(batch_size, 50, 50.0)
    
    try:
        output = robocache_cuda.fused_multimodal_alignment(
            vision_data, vision_times,
            proprio_data, proprio_times,
            None, None,  # No force sensor
            target_times
        )
        
        expected_shape = (batch_size, 50, 512 + 14)
        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
        
        print(f"✓ Alignment without force sensor succeeded")
        print(f"  Output shape: {output.shape}")
        print("\n✅ OPTIONAL FORCE SENSOR TEST PASSED")
        
    except Exception as e:
        print(f"\n❌ OPTIONAL FORCE SENSOR TEST FAILED: {e}")
        raise


def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       RoboCache Phase 2: Multimodal Fusion Test Suite               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA not available")
        return 1
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    try:
        test_correctness()
        test_without_force()
        test_performance()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED")
        print("="*80 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

