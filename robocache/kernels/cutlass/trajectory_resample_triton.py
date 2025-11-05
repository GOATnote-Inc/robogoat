"""
Triton implementation of trajectory resampling

Benefits over raw CUDA:
- Easier to write and maintain
- Auto-tuning finds optimal configurations
- Python-based (no C++ compilation complexity)
- Often competitive with hand-tuned CUDA

Target: Match or beat our 0.043ms CUDA kernel
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_D': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_D': 64}, num_warps=8),
    ],
    key=['action_dim'],
)
@triton.jit
def resample_kernel(
    # Pointers
    source_data_ptr,
    source_times_ptr,
    target_times_ptr,
    output_ptr,
    # Shapes
    batch_size,
    source_len,
    target_len,
    action_dim,
    # Block sizes (auto-tuned)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for trajectory resampling with auto-tuning
    
    Grid: (num_batches, num_target_blocks)
    where num_target_blocks = ceil(target_len / BLOCK_SIZE_M)
    """
    # Program IDs
    pid_batch = tl.program_id(axis=0)
    pid_target = tl.program_id(axis=1)
    
    # This program processes BLOCK_SIZE_M target times
    target_start = pid_target * BLOCK_SIZE_M
    target_offsets = target_start + tl.arange(0, BLOCK_SIZE_M)
    target_mask = target_offsets < target_len
    
    # Load target times for this block
    target_time_ptrs = target_times_ptr + pid_batch * target_len + target_offsets
    target_times = tl.load(target_time_ptrs, mask=target_mask, other=0.0)
    
    # Binary search for each target time
    # Note: Triton doesn't have while loops, so we unroll manually
    left = tl.full([BLOCK_SIZE_M], 0, dtype=tl.int32)
    right = tl.full([BLOCK_SIZE_M], source_len - 1, dtype=tl.int32)
    
    # Binary search (10 iterations = log2(1024))
    for _ in range(10):
        mid = (left + right) // 2
        
        # Load mid times for all threads
        mid_time_ptrs = source_times_ptr + pid_batch * source_len + mid
        mid_times = tl.load(mid_time_ptrs, mask=target_mask, other=0.0)
        
        # Update left or right based on comparison
        is_less_equal = mid_times <= target_times
        left = tl.where(is_less_equal, mid, left)
        right = tl.where(is_less_equal, right, mid)
        
        # Early exit if converged (all left >= right - 1)
        converged = left >= right - 1
        if tl.sum(converged.to(tl.int32)) == BLOCK_SIZE_M:
            break
    
    right = tl.minimum(left + 1, source_len - 1)
    
    # Load times at left and right indices
    left_time_ptrs = source_times_ptr + pid_batch * source_len + left
    right_time_ptrs = source_times_ptr + pid_batch * source_len + right
    
    left_times = tl.load(left_time_ptrs, mask=target_mask, other=0.0)
    right_times = tl.load(right_time_ptrs, mask=target_mask, other=0.0)
    
    # Compute interpolation weights
    delta = right_times - left_times
    delta = tl.where(delta < 1e-6, 1.0, delta)  # Avoid division by zero
    weights = tl.maximum(0.0, tl.minimum(1.0, (target_times - left_times) / delta))
    
    # Interpolate all dimensions in blocks
    for dim_start in range(0, action_dim, BLOCK_SIZE_D):
        dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_D)
        dim_mask = dim_offsets < action_dim
        
        # Compute pointers for this dimension block
        # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_D]
        left_ptrs = (source_data_ptr + 
                    pid_batch * source_len * action_dim + 
                    left[:, None] * action_dim + 
                    dim_offsets[None, :])
        
        right_ptrs = (source_data_ptr + 
                     pid_batch * source_len * action_dim + 
                     right[:, None] * action_dim + 
                     dim_offsets[None, :])
        
        output_ptrs = (output_ptr + 
                      pid_batch * target_len * action_dim + 
                      target_offsets[:, None] * action_dim + 
                      dim_offsets[None, :])
        
        # Create 2D mask
        mask_2d = target_mask[:, None] & dim_mask[None, :]
        
        # Load left and right values
        left_vals = tl.load(left_ptrs, mask=mask_2d, other=0.0)
        right_vals = tl.load(right_ptrs, mask=mask_2d, other=0.0)
        
        # Interpolate
        result = left_vals + weights[:, None] * (right_vals - left_vals)
        
        # Store result
        tl.store(output_ptrs, result, mask=mask_2d)


def resample_trajectories_triton(source_data, source_times, target_times):
    """
    Triton-based trajectory resampling
    
    Args:
        source_data: [batch, source_len, action_dim]
        source_times: [batch, source_len]
        target_times: [batch, target_len]
    
    Returns:
        output: [batch, target_len, action_dim]
    """
    batch_size, source_len, action_dim = source_data.shape
    target_len = target_times.shape[1]
    
    # Allocate output
    output = torch.empty(
        batch_size, target_len, action_dim,
        dtype=source_data.dtype,
        device=source_data.device
    )
    
    # Launch kernel (Triton auto-tunes BLOCK_SIZE_M and BLOCK_SIZE_D)
    def grid(META):
        return (batch_size, triton.cdiv(target_len, META['BLOCK_SIZE_M']))
    
    resample_kernel[grid](
        source_data, source_times, target_times, output,
        batch_size, source_len, target_len, action_dim,
    )
    
    return output


# ==============================================================================
# Benchmark
# ==============================================================================

if __name__ == "__main__":
    import time
    
    print("="*80)
    print("Triton Trajectory Resampling Benchmark")
    print("="*80)
    
    device = "cuda"
    batch_size = 256
    source_len = 500
    target_len = 250
    action_dim = 32
    dtype = torch.bfloat16
    
    # Generate data
    source_data = torch.randn(batch_size, source_len, action_dim, dtype=dtype, device=device)
    source_times = torch.linspace(0, 1, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
    target_times = torch.linspace(0, 1, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
    source_times, _ = torch.sort(source_times, dim=1)
    
    # Warmup (Triton compiles and auto-tunes on first run)
    print("\nCompiling and auto-tuning Triton kernel...")
    for _ in range(5):
        _ = resample_trajectories_triton(source_data, source_times, target_times)
    torch.cuda.synchronize()
    print("✓ Compilation complete")
    
    # Benchmark
    print("\nBenchmarking...")
    start = time.perf_counter()
    for _ in range(100):
        output = resample_trajectories_triton(source_data, source_times, target_times)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000  # ms
    
    # Calculate bandwidth
    bytes_total = (batch_size * source_len * action_dim * 2 + 
                  batch_size * (source_len + target_len) * 4 + 
                  batch_size * target_len * action_dim * 2)
    bandwidth = (bytes_total / 1e9) / (elapsed / 1000)
    efficiency = bandwidth / 3000 * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Triton kernel:     {elapsed:.3f} ms  {bandwidth:6.1f} GB/s  {efficiency:5.2f}% eff")
    print(f"CUDA baseline:     0.043 ms   307.0 GB/s  10.24% eff")
    print(f"PyTorch native:    0.116 ms   112.2 GB/s   3.74% eff")
    print("="*80)
    
    speedup_vs_cuda = 0.043 / elapsed
    speedup_vs_pytorch = 0.116 / elapsed
    
    print(f"\nTriton vs CUDA:    {speedup_vs_cuda:.2f}x")
    print(f"Triton vs PyTorch: {speedup_vs_pytorch:.2f}x")
    
    if elapsed < 0.043:
        print("\n🏆 Triton is FASTER than hand-tuned CUDA!")
        print("   → Use Triton for best of both worlds: speed + maintainability")
    elif elapsed < 0.05:
        print("\n✓ Triton matches CUDA performance!")
        print("  → Use Triton for easier development and maintenance")
    elif elapsed < 0.116:
        print("\n✓ Triton beats PyTorch native")
        print("  → Consider Triton as middle ground: faster than PyTorch, easier than CUDA")
    else:
        print("\n→ Hand-tuned CUDA is still best for maximum performance")
        print("  But Triton is a good option for rapid development")

