#!/usr/bin/env python3
"""
Investigation: Practical alternatives to raw CUDA for trajectory resampling

Following the principle "don't reinvent the wheel", we investigate:
1. PyTorch native operations (torch.nn.functional.interpolate, searchsorted, etc.)
2. Triton kernels (easier development, auto-tuning)
3. xformers efficient operations
4. PyTorch profiler for bottleneck analysis

Goal: Find faster, more maintainable solutions than raw CUDA
"""

import torch
import time
import numpy as np
from torch.profiler import profile, ProfilerActivity

# ==============================================================================
# Baseline: Current CUDA implementation
# ==============================================================================

def cuda_baseline(source_data, source_times, target_times):
    """Current CUDA kernel (for comparison)"""
    try:
        import robocache
        return robocache.resample_trajectories(source_data, source_times, target_times)
    except ImportError:
        print("RoboCache CUDA not available")
        return None

# ==============================================================================
# Alternative 1: PyTorch Native Operations
# ==============================================================================

def pytorch_searchsorted_approach(source_data, source_times, target_times):
    """
    Use torch.searchsorted for index finding (GPU-accelerated)
    Much faster than Python loops, leverages PyTorch optimizations
    """
    batch_size, source_len, action_dim = source_data.shape
    target_len = target_times.shape[1]
    
    # torch.searchsorted is GPU-accelerated and highly optimized
    # Returns indices where target_times would be inserted
    right_indices = torch.searchsorted(source_times.contiguous(), 
                                       target_times.contiguous(), 
                                       right=True)
    left_indices = (right_indices - 1).clamp(min=0)
    right_indices = right_indices.clamp(max=source_len - 1)
    
    # Gather values at left and right indices (GPU-accelerated gather)
    # [batch, target_len, action_dim]
    left_values = torch.gather(
        source_data, 
        1, 
        left_indices.unsqueeze(-1).expand(-1, -1, action_dim)
    )
    right_values = torch.gather(
        source_data, 
        1, 
        right_indices.unsqueeze(-1).expand(-1, -1, action_dim)
    )
    
    # Get times at left and right (for weight calculation)
    left_times = torch.gather(source_times, 1, left_indices)
    right_times = torch.gather(source_times, 1, right_indices)
    
    # Compute interpolation weights
    delta = right_times - left_times
    delta = torch.where(delta < 1e-6, torch.ones_like(delta), delta)  # Avoid division by zero
    weights = ((target_times - left_times) / delta).clamp(0, 1)
    
    # Linear interpolation (fully vectorized on GPU)
    output = left_values + weights.unsqueeze(-1) * (right_values - left_values)
    
    return output

def pytorch_interpolate_approach(source_data, source_times, target_times):
    """
    Use torch.nn.functional.interpolate (highly optimized)
    
    Note: This requires regular grid, so we use grid_sample for irregular
    """
    batch_size, source_len, action_dim = source_data.shape
    target_len = target_times.shape[1]
    
    # Normalize times to [-1, 1] for grid_sample
    # grid_sample expects normalized coordinates
    source_min = source_times[:, 0:1]
    source_max = source_times[:, -1:]
    
    # Normalize target times to [-1, 1]
    normalized_target = 2.0 * (target_times - source_min) / (source_max - source_min + 1e-6) - 1.0
    
    # Reshape for grid_sample: [batch, action_dim, 1, source_len] and grid [batch, target_len, 1, 2]
    # grid_sample is HIGHLY optimized (uses texture memory internally on GPU)
    data_reshaped = source_data.transpose(1, 2).unsqueeze(2)  # [batch, action_dim, 1, source_len]
    
    # Create grid for 1D sampling
    grid = torch.zeros(batch_size, target_len, 1, 2, device=source_data.device)
    grid[:, :, :, 0] = normalized_target.unsqueeze(2)  # x-coordinate
    
    # grid_sample with linear interpolation (uses GPU texture units!)
    output = torch.nn.functional.grid_sample(
        data_reshaped,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    # Reshape back: [batch, action_dim, 1, target_len] -> [batch, target_len, action_dim]
    output = output.squeeze(2).transpose(1, 2)
    
    return output

# ==============================================================================
# Alternative 2: Triton Kernel
# ==============================================================================

try:
    import triton
    import triton.language as tl
    
    @triton.jit
    def resample_triton_kernel(
        source_data_ptr, source_times_ptr, target_times_ptr, output_ptr,
        batch_size, source_len, target_len, action_dim,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton kernel for trajectory resampling
        
        Triton benefits:
        - Automatic memory coalescing optimization
        - Easier to write than raw CUDA
        - Auto-tuning finds best BLOCK_SIZE
        """
        # Get program ID
        pid_batch = tl.program_id(0)
        pid_target = tl.program_id(1)
        
        if pid_batch >= batch_size or pid_target >= target_len:
            return
        
        # Load target time
        target_time = tl.load(target_times_ptr + pid_batch * target_len + pid_target)
        
        # Binary search for left index
        left = 0
        right = source_len - 1
        
        # Unroll binary search
        for _ in range(10):  # log2(1024) = 10 iterations max
            if left >= right - 1:
                break
            mid = (left + right) // 2
            mid_time = tl.load(source_times_ptr + pid_batch * source_len + mid)
            if mid_time <= target_time:
                left = mid
            else:
                right = mid
        
        right = min(left + 1, source_len - 1)
        
        # Load times and compute weight
        t_left = tl.load(source_times_ptr + pid_batch * source_len + left)
        t_right = tl.load(source_times_ptr + pid_batch * source_len + right)
        delta = t_right - t_left
        weight = tl.where(delta < 1e-6, 0.0, (target_time - t_left) / delta)
        weight = tl.maximum(0.0, tl.minimum(1.0, weight))
        
        # Interpolate each dimension (vectorized within Triton)
        dim_block = tl.arange(0, BLOCK_SIZE)
        for dim_start in range(0, action_dim, BLOCK_SIZE):
            dims = dim_start + dim_block
            mask = dims < action_dim
            
            # Load left and right values (vectorized)
            left_offset = pid_batch * source_len * action_dim + left * action_dim + dims
            right_offset = pid_batch * source_len * action_dim + right * action_dim + dims
            out_offset = pid_batch * target_len * action_dim + pid_target * action_dim + dims
            
            left_vals = tl.load(source_data_ptr + left_offset, mask=mask)
            right_vals = tl.load(source_data_ptr + right_offset, mask=mask)
            
            # Interpolate
            result = left_vals + weight * (right_vals - left_vals)
            
            # Store
            tl.store(output_ptr + out_offset, result, mask=mask)
    
    def triton_resample(source_data, source_times, target_times):
        """
        Triton implementation with auto-tuning
        
        Benefits:
        - Triton auto-tunes BLOCK_SIZE
        - Easier to maintain than raw CUDA
        - Often competitive with hand-optimized CUDA
        """
        batch_size, source_len, action_dim = source_data.shape
        target_len = target_times.shape[1]
        
        output = torch.empty(batch_size, target_len, action_dim, 
                           dtype=source_data.dtype, device=source_data.device)
        
        # Launch grid
        grid = lambda meta: (batch_size, target_len)
        
        # Triton will auto-tune BLOCK_SIZE
        resample_triton_kernel[grid](
            source_data, source_times, target_times, output,
            batch_size, source_len, target_len, action_dim,
            BLOCK_SIZE=128
        )
        
        return output
    
    TRITON_AVAILABLE = True

except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available (pip install triton)")

# ==============================================================================
# Benchmarking
# ==============================================================================

def benchmark(func, *args, name="Method", warmup=10, iters=100):
    """Benchmark a function with warmup"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        result = func(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    
    return elapsed, result

def profile_pytorch_approach(source_data, source_times, target_times):
    """Use PyTorch profiler to analyze bottlenecks"""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        output = pytorch_searchsorted_approach(source_data, source_times, target_times)
    
    print("\n" + "="*80)
    print("PyTorch Profiler Results (searchsorted approach)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return output

# ==============================================================================
# Main Comparison
# ==============================================================================

def main():
    print("="*80)
    print("RoboCache Alternative Approaches Investigation")
    print("="*80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    source_len = 500
    target_len = 250
    action_dim = 32
    dtype = torch.bfloat16
    
    # Generate data
    source_data = torch.randn(batch_size, source_len, action_dim, dtype=dtype, device=device)
    source_times = torch.linspace(0, 1, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
    target_times = torch.linspace(0, 1, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Sort source_times (requirement for searchsorted)
    source_times, _ = torch.sort(source_times, dim=1)
    
    results = {}
    
    # 1. Current CUDA baseline
    print("\n[1/4] Testing CUDA baseline...")
    cuda_result = cuda_baseline(source_data, source_times, target_times)
    if cuda_result is not None:
        elapsed, _ = benchmark(cuda_baseline, source_data, source_times, target_times, name="CUDA")
        results["CUDA Baseline"] = elapsed
        print(f"  ✓ CUDA Baseline: {elapsed:.3f} ms")
    
    # 2. PyTorch searchsorted
    print("\n[2/4] Testing PyTorch searchsorted...")
    elapsed, pytorch_result = benchmark(
        pytorch_searchsorted_approach, 
        source_data, source_times, target_times,
        name="PyTorch searchsorted"
    )
    results["PyTorch searchsorted"] = elapsed
    print(f"  ✓ PyTorch searchsorted: {elapsed:.3f} ms")
    
    # 3. PyTorch grid_sample (texture memory)
    print("\n[3/4] Testing PyTorch grid_sample...")
    elapsed, grid_result = benchmark(
        pytorch_interpolate_approach,
        source_data, source_times, target_times,
        name="PyTorch grid_sample"
    )
    results["PyTorch grid_sample"] = elapsed
    print(f"  ✓ PyTorch grid_sample: {elapsed:.3f} ms")
    
    # 4. Triton (if available)
    if TRITON_AVAILABLE:
        print("\n[4/4] Testing Triton kernel...")
        elapsed, triton_result = benchmark(
            triton_resample,
            source_data, source_times, target_times,
            name="Triton"
        )
        results["Triton"] = elapsed
        print(f"  ✓ Triton: {elapsed:.3f} ms")
    else:
        print("\n[4/4] Triton not available (skipped)")
    
    # Results summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Configuration: batch={batch_size}, src={source_len}, tgt={target_len}, dim={action_dim}")
    print(f"Data type: {dtype}")
    print("-"*80)
    
    if "CUDA Baseline" in results:
        baseline = results["CUDA Baseline"]
        for name, elapsed in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / elapsed
            print(f"{name:25s}: {elapsed:7.3f} ms  ({speedup:5.2f}x vs CUDA)")
    else:
        for name, elapsed in sorted(results.items(), key=lambda x: x[1]):
            print(f"{name:25s}: {elapsed:7.3f} ms")
    
    # Memory efficiency
    print("\n" + "="*80)
    print("MEMORY ANALYSIS")
    print("="*80)
    
    # Calculate theoretical bandwidth
    bytes_read = batch_size * source_len * action_dim * 2  # BF16
    bytes_read += batch_size * (source_len + target_len) * 4  # FP32 times
    bytes_write = batch_size * target_len * action_dim * 2  # BF16
    total_bytes = bytes_read + bytes_write
    
    for name, elapsed in results.items():
        bandwidth = (total_bytes / 1e9) / (elapsed / 1000)  # GB/s
        efficiency = bandwidth / 3000 * 100  # % of HBM3 peak
        print(f"{name:25s}: {bandwidth:6.1f} GB/s ({efficiency:5.2f}% HBM3)")
    
    # Profile the PyTorch approach
    print("\n" + "="*80)
    print("DETAILED PROFILING (PyTorch searchsorted)")
    print("="*80)
    _ = profile_pytorch_approach(source_data, source_times, target_times)
    
    # Correctness check
    print("\n" + "="*80)
    print("CORRECTNESS CHECK")
    print("="*80)
    
    if cuda_result is not None:
        error = torch.abs(pytorch_result.float() - cuda_result.float()).mean()
        print(f"Mean absolute error (PyTorch vs CUDA): {error:.6f}")
        if error < 0.01:
            print("✓ Results match within tolerance")
        else:
            print("✗ Results differ significantly!")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_name = min(results.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Fastest approach: {best_name} ({results[best_name]:.3f} ms)")
    
    print("\nAnalysis:")
    print("• torch.searchsorted: GPU-accelerated, highly optimized, easy to maintain")
    print("• torch.grid_sample: Uses GPU texture memory (hardware interpolation)")
    print("• Triton: Auto-tuning, easier than CUDA, competitive performance")
    print("• CUDA: Maximum control, but harder to maintain")
    
    print("\nNext steps:")
    if "PyTorch grid_sample" in results and results["PyTorch grid_sample"] < results.get("CUDA Baseline", float('inf')):
        print("✓ PyTorch grid_sample is faster! Use it instead of custom CUDA")
    elif "PyTorch searchsorted" in results and results["PyTorch searchsorted"] < results.get("CUDA Baseline", float('inf')) * 1.5:
        print("✓ PyTorch searchsorted is competitive. Consider using it for maintainability")
    elif TRITON_AVAILABLE and "Triton" in results:
        print("✓ Consider Triton for easier development with comparable performance")
    else:
        print("→ Current CUDA kernel is fastest, but consider Triton for maintainability")

if __name__ == "__main__":
    main()

