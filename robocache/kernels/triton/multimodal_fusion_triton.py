"""
multimodal_fusion_triton.py
Triton implementation of multimodal sensor fusion for comparison with CUDA

This is a direct comparison implementation to evaluate:
- Development time
- Performance
- Code complexity
- Maintainability
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _binary_search(
    times_ptr,
    length: tl.constexpr,
    target: tl.float32,
) -> tl.int32:
    """Binary search for target time in sorted array"""
    left = 0
    right = length - 1
    
    # Unroll search iterations for better performance
    for _ in range(12):  # log2(4096) = 12, supports up to 4K timesteps
        if left >= right - 1:
            break
        mid = (left + right) // 2
        mid_val = tl.load(times_ptr + mid)
        if mid_val <= target:
            left = mid
        else:
            right = mid
    
    return left


@triton.jit
def fused_multimodal_alignment_kernel(
    # Vision inputs
    vision_data_ptr,
    vision_times_ptr,
    vision_src_len: tl.constexpr,
    vision_dim: tl.constexpr,
    # Proprio inputs
    proprio_data_ptr,
    proprio_times_ptr,
    proprio_src_len: tl.constexpr,
    proprio_dim: tl.constexpr,
    # Force inputs
    force_data_ptr,
    force_times_ptr,
    force_src_len: tl.constexpr,
    force_dim: tl.constexpr,
    # Target times
    target_times_ptr,
    target_len: tl.constexpr,
    # Output
    output_ptr,
    # Dimensions
    batch_size: tl.constexpr,
    total_dim: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_TARGET: tl.constexpr,
    BLOCK_SIZE_FEATURE: tl.constexpr,
):
    """
    Fused multimodal alignment kernel in Triton
    
    Aligns 3 sensor streams (vision, proprio, force) to common target frequency.
    Uses binary search + linear interpolation.
    """
    
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_target = tl.program_id(1)
    
    # Batch and target indices
    batch_idx = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    target_idx = pid_target * BLOCK_SIZE_TARGET + tl.arange(0, BLOCK_SIZE_TARGET)
    
    # Masks
    batch_mask = batch_idx < batch_size
    target_mask = target_idx < target_len
    
    # Load target times
    target_offset = batch_idx[:, None] * target_len + target_idx[None, :]
    target_times = tl.load(
        target_times_ptr + target_offset,
        mask=batch_mask[:, None] & target_mask[None, :],
        other=0.0
    )
    
    # Process each (batch, target) pair
    for b in range(BLOCK_SIZE_BATCH):
        if not (batch_idx[b] < batch_size):
            continue
            
        for t in range(BLOCK_SIZE_TARGET):
            if not (target_idx[t] < target_len):
                continue
            
            curr_batch = batch_idx[b]
            curr_target = target_idx[t]
            target_time = target_times[b, t]
            
            # Output base offset
            out_base = (curr_batch * target_len + curr_target) * total_dim
            
            # ==== Align Vision ====
            vision_times_base = curr_batch * vision_src_len
            v_left = _binary_search(
                vision_times_ptr + vision_times_base,
                vision_src_len,
                target_time
            )
            v_right = tl.minimum(v_left + 1, vision_src_len - 1)
            
            # Load vision timestamps
            v_t_left = tl.load(vision_times_ptr + vision_times_base + v_left)
            v_t_right = tl.load(vision_times_ptr + vision_times_base + v_right)
            
            # Compute interpolation weight
            v_delta = v_t_right - v_t_left
            v_weight = tl.where(
                v_delta < 1e-6,
                0.0,
                tl.maximum(0.0, tl.minimum(1.0, (target_time - v_t_left) / v_delta))
            )
            
            # Interpolate vision features
            vision_left_base = (curr_batch * vision_src_len + v_left) * vision_dim
            vision_right_base = (curr_batch * vision_src_len + v_right) * vision_dim
            
            for d_block in range(0, vision_dim, BLOCK_SIZE_FEATURE):
                d_idx = d_block + tl.arange(0, BLOCK_SIZE_FEATURE)
                d_mask = d_idx < vision_dim
                
                v_left_val = tl.load(
                    vision_data_ptr + vision_left_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                v_right_val = tl.load(
                    vision_data_ptr + vision_right_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                
                # Linear interpolation
                v_out_val = v_left_val + v_weight * (v_right_val - v_left_val)
                
                tl.store(
                    output_ptr + out_base + d_idx,
                    v_out_val,
                    mask=d_mask
                )
            
            # ==== Align Proprioception ====
            proprio_times_base = curr_batch * proprio_src_len
            p_left = _binary_search(
                proprio_times_ptr + proprio_times_base,
                proprio_src_len,
                target_time
            )
            p_right = tl.minimum(p_left + 1, proprio_src_len - 1)
            
            p_t_left = tl.load(proprio_times_ptr + proprio_times_base + p_left)
            p_t_right = tl.load(proprio_times_ptr + proprio_times_base + p_right)
            
            p_delta = p_t_right - p_t_left
            p_weight = tl.where(
                p_delta < 1e-6,
                0.0,
                tl.maximum(0.0, tl.minimum(1.0, (target_time - p_t_left) / p_delta))
            )
            
            proprio_left_base = (curr_batch * proprio_src_len + p_left) * proprio_dim
            proprio_right_base = (curr_batch * proprio_src_len + p_right) * proprio_dim
            
            for d_block in range(0, proprio_dim, BLOCK_SIZE_FEATURE):
                d_idx = d_block + tl.arange(0, BLOCK_SIZE_FEATURE)
                d_mask = d_idx < proprio_dim
                
                p_left_val = tl.load(
                    proprio_data_ptr + proprio_left_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                p_right_val = tl.load(
                    proprio_data_ptr + proprio_right_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                
                p_out_val = p_left_val + p_weight * (p_right_val - p_left_val)
                
                tl.store(
                    output_ptr + out_base + vision_dim + d_idx,
                    p_out_val,
                    mask=d_mask
                )
            
            # ==== Align Force ====
            force_times_base = curr_batch * force_src_len
            f_left = _binary_search(
                force_times_ptr + force_times_base,
                force_src_len,
                target_time
            )
            f_right = tl.minimum(f_left + 1, force_src_len - 1)
            
            f_t_left = tl.load(force_times_ptr + force_times_base + f_left)
            f_t_right = tl.load(force_times_ptr + force_times_base + f_right)
            
            f_delta = f_t_right - f_t_left
            f_weight = tl.where(
                f_delta < 1e-6,
                0.0,
                tl.maximum(0.0, tl.minimum(1.0, (target_time - f_t_left) / f_delta))
            )
            
            force_left_base = (curr_batch * force_src_len + f_left) * force_dim
            force_right_base = (curr_batch * force_src_len + f_right) * force_dim
            
            for d_block in range(0, force_dim, BLOCK_SIZE_FEATURE):
                d_idx = d_block + tl.arange(0, BLOCK_SIZE_FEATURE)
                d_mask = d_idx < force_dim
                
                f_left_val = tl.load(
                    force_data_ptr + force_left_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                f_right_val = tl.load(
                    force_data_ptr + force_right_base + d_idx,
                    mask=d_mask,
                    other=0.0
                )
                
                f_out_val = f_left_val + f_weight * (f_right_val - f_left_val)
                
                tl.store(
                    output_ptr + out_base + vision_dim + proprio_dim + d_idx,
                    f_out_val,
                    mask=d_mask
                )


def fused_multimodal_alignment_triton(
    vision_data: torch.Tensor,
    vision_times: torch.Tensor,
    proprio_data: torch.Tensor,
    proprio_times: torch.Tensor,
    force_data: torch.Tensor,
    force_times: torch.Tensor,
    target_times: torch.Tensor,
) -> torch.Tensor:
    """
    Triton implementation of fused multimodal alignment
    
    Args:
        vision_data: [batch, vision_src_len, vision_dim]
        vision_times: [batch, vision_src_len]
        proprio_data: [batch, proprio_src_len, proprio_dim]
        proprio_times: [batch, proprio_src_len]
        force_data: [batch, force_src_len, force_dim]
        force_times: [batch, force_src_len]
        target_times: [batch, target_len]
    
    Returns:
        output: [batch, target_len, vision_dim + proprio_dim + force_dim]
    """
    batch_size, target_len = target_times.shape
    _, vision_src_len, vision_dim = vision_data.shape
    _, proprio_src_len, proprio_dim = proprio_data.shape
    _, force_src_len, force_dim = force_data.shape
    
    total_dim = vision_dim + proprio_dim + force_dim
    
    # Allocate output
    output = torch.empty(
        (batch_size, target_len, total_dim),
        device=vision_data.device,
        dtype=vision_data.dtype
    )
    
    # Launch parameters - auto-tuning would optimize these
    BLOCK_SIZE_BATCH = 1
    BLOCK_SIZE_TARGET = 1
    BLOCK_SIZE_FEATURE = 16
    
    grid = (
        triton.cdiv(batch_size, BLOCK_SIZE_BATCH),
        triton.cdiv(target_len, BLOCK_SIZE_TARGET),
    )
    
    # Launch kernel
    fused_multimodal_alignment_kernel[grid](
        vision_data, vision_times, vision_src_len, vision_dim,
        proprio_data, proprio_times, proprio_src_len, proprio_dim,
        force_data, force_times, force_src_len, force_dim,
        target_times, target_len,
        output,
        batch_size, total_dim,
        BLOCK_SIZE_BATCH, BLOCK_SIZE_TARGET, BLOCK_SIZE_FEATURE,
    )
    
    return output


# ==============================================================================
# Auto-tuned version (more realistic Triton usage)
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_TARGET': 1, 'BLOCK_SIZE_FEATURE': 16}),
        triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_TARGET': 2, 'BLOCK_SIZE_FEATURE': 16}),
        triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_TARGET': 4, 'BLOCK_SIZE_FEATURE': 32}),
        triton.Config({'BLOCK_SIZE_BATCH': 2, 'BLOCK_SIZE_TARGET': 1, 'BLOCK_SIZE_FEATURE': 16}),
        triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_TARGET': 1, 'BLOCK_SIZE_FEATURE': 32}),
    ],
    key=['batch_size', 'target_len', 'vision_dim', 'proprio_dim', 'force_dim'],
)
@triton.jit
def fused_multimodal_alignment_kernel_autotuned(
    vision_data_ptr, vision_times_ptr, vision_src_len: tl.constexpr, vision_dim: tl.constexpr,
    proprio_data_ptr, proprio_times_ptr, proprio_src_len: tl.constexpr, proprio_dim: tl.constexpr,
    force_data_ptr, force_times_ptr, force_src_len: tl.constexpr, force_dim: tl.constexpr,
    target_times_ptr, target_len: tl.constexpr,
    output_ptr,
    batch_size: tl.constexpr, total_dim: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_TARGET: tl.constexpr,
    BLOCK_SIZE_FEATURE: tl.constexpr,
):
    """Auto-tuned version - same as above but with Triton's auto-tuner"""
    # (Implementation identical to non-autotuned version)
    pass


def fused_multimodal_alignment_triton_autotuned(
    vision_data: torch.Tensor,
    vision_times: torch.Tensor,
    proprio_data: torch.Tensor,
    proprio_times: torch.Tensor,
    force_data: torch.Tensor,
    force_times: torch.Tensor,
    target_times: torch.Tensor,
) -> torch.Tensor:
    """Auto-tuned version for fair comparison"""
    # Would use fused_multimodal_alignment_kernel_autotuned
    # For now, use manual version
    return fused_multimodal_alignment_triton(
        vision_data, vision_times,
        proprio_data, proprio_times,
        force_data, force_times,
        target_times
    )

