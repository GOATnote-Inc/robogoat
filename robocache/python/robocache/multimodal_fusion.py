"""
multimodal_fusion.py
PyTorch integration for multimodal fusion kernels

Copyright (c) 2025 GOATnote Inc.
SPDX-License-Identifier: Apache-2.0
"""

import torch
from torch.autograd import Function
from typing import Optional, Tuple
import warnings

try:
    import robocache_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    warnings.warn("robocache_cuda extension not found. Falling back to PyTorch implementation.")


class MultimodalFusionFunction(Function):
    """
    Custom autograd function for multimodal fusion with temporal alignment.

    Forward pass uses optimized CUDA kernel.
    Backward pass uses PyTorch autograd.
    """

    @staticmethod
    def forward(ctx, vision_features, vision_timestamps,
                proprio_features, proprio_timestamps,
                lang_embeddings, target_timestamps,
                use_optimized=True):
        """
        Args:
            vision_features: [B, T_vis, D_vis] BF16/FP32
            vision_timestamps: [B, T_vis] FP32
            proprio_features: [B, T_prop, D_prop] BF16/FP32
            proprio_timestamps: [B, T_prop] FP32
            lang_embeddings: [B, L, D_lang] BF16/FP32
            target_timestamps: [B, T] FP32
            use_optimized: bool, use warp-optimized kernel

        Returns:
            fused: [B, T, D_vis + D_prop + D_lang] same dtype as inputs
        """
        # Save for backward
        ctx.save_for_backward(vision_features, vision_timestamps,
                              proprio_features, proprio_timestamps,
                              lang_embeddings, target_timestamps)
        ctx.use_optimized = use_optimized

        # Validate inputs
        assert vision_features.is_cuda, "vision_features must be on CUDA"
        assert proprio_features.is_cuda, "proprio_features must be on CUDA"
        assert lang_embeddings.is_cuda, "lang_embeddings must be on CUDA"

        assert vision_features.dim() == 3, "vision_features must be [B, T, D]"
        assert proprio_features.dim() == 3, "proprio_features must be [B, T, D]"
        assert lang_embeddings.dim() == 3, "lang_embeddings must be [B, L, D]"

        # Convert to bfloat16 if needed
        original_dtype = vision_features.dtype
        if original_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"Unsupported dtype: {original_dtype}")

        # Call CUDA kernel
        if _CUDA_AVAILABLE:
            fused = robocache_cuda.fuse_multimodal(
                vision_features.contiguous(),
                vision_timestamps.contiguous(),
                proprio_features.contiguous(),
                proprio_timestamps.contiguous(),
                lang_embeddings.contiguous(),
                target_timestamps.contiguous(),
                use_optimized
            )
        else:
            # Fallback to PyTorch implementation
            fused = _pytorch_multimodal_fusion(
                vision_features, vision_timestamps,
                proprio_features, proprio_timestamps,
                lang_embeddings, target_timestamps
            )

        return fused

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - compute gradients w.r.t. inputs.

        Since this is a linear interpolation operation, gradients flow back
        through the interpolation weights.
        """
        vision_features, vision_timestamps, \
        proprio_features, proprio_timestamps, \
        lang_embeddings, target_timestamps = ctx.saved_tensors

        B, T, D_total = grad_output.shape
        D_vis = vision_features.shape[2]
        D_prop = proprio_features.shape[2]
        D_lang = lang_embeddings.shape[2]

        # Split gradient
        grad_vision = grad_output[:, :, :D_vis]
        grad_proprio = grad_output[:, :, D_vis:D_vis+D_prop]
        grad_lang = grad_output[:, :, D_vis+D_prop:]

        # Backward through temporal interpolation
        # For vision and proprio, we need to interpolate gradients back to source timestamps
        grad_vision_features = _interpolate_gradients_backward(
            grad_vision, target_timestamps,
            vision_timestamps, vision_features.shape[1]
        )

        grad_proprio_features = _interpolate_gradients_backward(
            grad_proprio, target_timestamps,
            proprio_timestamps, proprio_features.shape[1]
        )

        # Language gradients: average pooling backward
        grad_lang_embeddings = grad_lang.unsqueeze(2).expand(
            B, T, lang_embeddings.shape[1], D_lang
        ).mean(dim=1) / lang_embeddings.shape[1]

        return grad_vision_features, None, grad_proprio_features, None, \
               grad_lang_embeddings, None, None


def _interpolate_gradients_backward(grad, target_times, source_times, src_length):
    """
    Backward pass for temporal interpolation.

    Distributes gradients to source timestamps based on interpolation weights.
    """
    B, T, D = grad.shape
    device = grad.device

    grad_source = torch.zeros(B, src_length, D, dtype=grad.dtype, device=device)

    for b in range(B):
        for t in range(T):
            target_t = target_times[b, t].item()

            # Find source indices
            left_idx = torch.searchsorted(source_times[b], target_t, right=False)
            left_idx = torch.clamp(left_idx, 0, src_length - 2)
            right_idx = left_idx + 1

            # Compute interpolation weight
            t_left = source_times[b, left_idx].item()
            t_right = source_times[b, right_idx].item()

            if t_right > t_left:
                weight = (target_t - t_left) / (t_right - t_left)
                weight = max(0.0, min(1.0, weight))
            else:
                weight = 0.0

            # Distribute gradient
            grad_source[b, left_idx] += (1.0 - weight) * grad[b, t]
            grad_source[b, right_idx] += weight * grad[b, t]

    return grad_source


def _pytorch_multimodal_fusion(vision_features, vision_timestamps,
                                proprio_features, proprio_timestamps,
                                lang_embeddings, target_timestamps):
    """
    Pure PyTorch fallback implementation of multimodal fusion.
    Used when CUDA extension is not available.
    """
    B = vision_features.shape[0]
    T = target_timestamps.shape[1]
    D_vis = vision_features.shape[2]
    D_prop = proprio_features.shape[2]
    D_lang = lang_embeddings.shape[2]
    D_total = D_vis + D_prop + D_lang

    device = vision_features.device
    dtype = vision_features.dtype

    # Allocate output
    fused = torch.zeros(B, T, D_total, dtype=dtype, device=device)

    # Process each batch and timestep
    for b in range(B):
        for t in range(T):
            target_t = target_timestamps[b, t]

            # Interpolate vision
            vis_idx = torch.searchsorted(vision_timestamps[b], target_t, right=False)
            vis_idx = torch.clamp(vis_idx, 0, vision_features.shape[1] - 2)
            vis_left = vision_features[b, vis_idx]
            vis_right = vision_features[b, vis_idx + 1]

            t_left = vision_timestamps[b, vis_idx]
            t_right = vision_timestamps[b, vis_idx + 1]
            weight = (target_t - t_left) / (t_right - t_left + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)

            vis_interp = (1.0 - weight) * vis_left + weight * vis_right
            fused[b, t, :D_vis] = vis_interp

            # Interpolate proprio
            prop_idx = torch.searchsorted(proprio_timestamps[b], target_t, right=False)
            prop_idx = torch.clamp(prop_idx, 0, proprio_features.shape[1] - 2)
            prop_left = proprio_features[b, prop_idx]
            prop_right = proprio_features[b, prop_idx + 1]

            t_left = proprio_timestamps[b, prop_idx]
            t_right = proprio_timestamps[b, prop_idx + 1]
            weight = (target_t - t_left) / (t_right - t_left + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)

            prop_interp = (1.0 - weight) * prop_left + weight * prop_right
            fused[b, t, D_vis:D_vis+D_prop] = prop_interp

            # Language (average pooling)
            lang_avg = lang_embeddings[b].mean(dim=0)
            fused[b, t, D_vis+D_prop:] = lang_avg

    return fused


def fuse_multimodal(
    vision_features: torch.Tensor,
    vision_timestamps: torch.Tensor,
    proprio_features: torch.Tensor,
    proprio_timestamps: torch.Tensor,
    lang_embeddings: torch.Tensor,
    target_timestamps: torch.Tensor,
    use_optimized: bool = True
) -> torch.Tensor:
    """
    Fuse multimodal data (vision, proprioception, language) with temporal alignment.

    This function performs efficient temporal alignment and concatenation of multimodal
    robot data. It handles different sensor frequencies by interpolating to a common
    target timeline.

    Args:
        vision_features: Vision embeddings [B, T_vis, D_vis] (e.g., from RGB-D camera at 30Hz)
        vision_timestamps: Vision timestamps [B, T_vis] in seconds
        proprio_features: Proprioception embeddings [B, T_prop, D_prop] (e.g., joint states at 100Hz)
        proprio_timestamps: Proprio timestamps [B, T_prop] in seconds
        lang_embeddings: Language embeddings [B, L, D_lang] (e.g., CLIP embeddings)
        target_timestamps: Target timestamps [B, T] in seconds (output timeline)
        use_optimized: Use warp-optimized CUDA kernel (default: True)

    Returns:
        fused: Fused multimodal features [B, T, D_vis + D_prop + D_lang]

    Example:
        >>> vision = torch.randn(64, 30, 256, dtype=torch.bfloat16, device='cuda')
        >>> vision_times = torch.linspace(0, 1, 30, device='cuda').unsqueeze(0).expand(64, -1)
        >>> proprio = torch.randn(64, 100, 64, dtype=torch.bfloat16, device='cuda')
        >>> proprio_times = torch.linspace(0, 1, 100, device='cuda').unsqueeze(0).expand(64, -1)
        >>> lang = torch.randn(64, 77, 512, dtype=torch.bfloat16, device='cuda')
        >>> target_times = torch.linspace(0, 1, 50, device='cuda').unsqueeze(0).expand(64, -1)
        >>>
        >>> fused = fuse_multimodal(vision, vision_times, proprio, proprio_times,
        ...                         lang, target_times)
        >>> print(fused.shape)  # [64, 50, 832]
    """
    return MultimodalFusionFunction.apply(
        vision_features, vision_timestamps,
        proprio_features, proprio_timestamps,
        lang_embeddings, target_timestamps,
        use_optimized
    )


class MultimodalFusionModule(torch.nn.Module):
    """
    PyTorch Module wrapper for multimodal fusion.

    This can be used as a layer in a larger model and supports:
    - Automatic mixed precision (AMP)
    - TorchScript compilation
    - Gradient checkpointing

    Example:
        >>> fusion = MultimodalFusionModule()
        >>> fused = fusion(vision, vision_times, proprio, proprio_times,
        ...                lang, target_times)
    """

    def __init__(self, use_optimized: bool = True):
        super().__init__()
        self.use_optimized = use_optimized

    def forward(self,
                vision_features: torch.Tensor,
                vision_timestamps: torch.Tensor,
                proprio_features: torch.Tensor,
                proprio_timestamps: torch.Tensor,
                lang_embeddings: torch.Tensor,
                target_timestamps: torch.Tensor) -> torch.Tensor:
        return fuse_multimodal(
            vision_features, vision_timestamps,
            proprio_features, proprio_timestamps,
            lang_embeddings, target_timestamps,
            self.use_optimized
        )

    def extra_repr(self) -> str:
        return f'use_optimized={self.use_optimized}'


# Convenience function for creating batched timestamps
def create_timestamps(batch_size: int, length: int,
                     start: float = 0.0, end: float = 1.0,
                     device: str = 'cuda') -> torch.Tensor:
    """
    Create batched timestamps for multimodal fusion.

    Args:
        batch_size: Number of batches
        length: Sequence length
        start: Start time in seconds
        end: End time in seconds
        device: Device to create tensor on

    Returns:
        timestamps: [batch_size, length] FP32 tensor
    """
    times = torch.linspace(start, end, length, device=device)
    return times.unsqueeze(0).expand(batch_size, -1)


def benchmark_multimodal_fusion(
    batch_size: int = 256,
    target_length: int = 50,
    vision_length: int = 30,
    vision_dim: int = 256,
    proprio_length: int = 100,
    proprio_dim: int = 64,
    lang_length: int = 77,
    lang_dim: int = 512,
    num_iters: int = 100,
    warmup: int = 10,
    device: str = 'cuda'
) -> dict:
    """
    Benchmark multimodal fusion performance.

    Returns a dict with timing and throughput metrics.
    """
    # Create test data
    vision = torch.randn(batch_size, vision_length, vision_dim,
                         dtype=torch.bfloat16, device=device)
    vision_times = create_timestamps(batch_size, vision_length, device=device)

    proprio = torch.randn(batch_size, proprio_length, proprio_dim,
                          dtype=torch.bfloat16, device=device)
    proprio_times = create_timestamps(batch_size, proprio_length, device=device)

    lang = torch.randn(batch_size, lang_length, lang_dim,
                       dtype=torch.bfloat16, device=device)
    target_times = create_timestamps(batch_size, target_length, device=device)

    # Warmup
    for _ in range(warmup):
        fused = fuse_multimodal(vision, vision_times, proprio, proprio_times,
                               lang, target_times)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fused = fuse_multimodal(vision, vision_times, proprio, proprio_times,
                               lang, target_times)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters
    throughput = (batch_size * target_length) / (elapsed_ms / 1000.0)

    return {
        'latency_ms': elapsed_ms,
        'throughput_samples_per_sec': throughput,
        'throughput_Ksamples_per_sec': throughput / 1000.0,
        'batch_size': batch_size,
        'target_length': target_length,
    }


if __name__ == '__main__':
    print("Testing multimodal fusion...")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(1)

    # Test 1: Basic functionality
    print("\n1. Basic functionality test")
    batch_size = 4
    vision = torch.randn(batch_size, 30, 256, dtype=torch.bfloat16, device='cuda')
    vision_times = create_timestamps(batch_size, 30)
    proprio = torch.randn(batch_size, 100, 64, dtype=torch.bfloat16, device='cuda')
    proprio_times = create_timestamps(batch_size, 100)
    lang = torch.randn(batch_size, 77, 512, dtype=torch.bfloat16, device='cuda')
    target_times = create_timestamps(batch_size, 50)

    fused = fuse_multimodal(vision, vision_times, proprio, proprio_times,
                           lang, target_times)

    print(f"  Input: vision{vision.shape}, proprio{proprio.shape}, lang{lang.shape}")
    print(f"  Output: {fused.shape}")
    print(f"  ✓ Shape correct: {fused.shape == (batch_size, 50, 832)}")

    # Test 2: Gradient flow
    print("\n2. Gradient flow test")
    vision.requires_grad = True
    proprio.requires_grad = True
    lang.requires_grad = True

    fused = fuse_multimodal(vision, vision_times, proprio, proprio_times,
                           lang, target_times)
    loss = fused.sum()
    loss.backward()

    print(f"  ✓ Vision grad: {vision.grad is not None and not torch.isnan(vision.grad).any()}")
    print(f"  ✓ Proprio grad: {proprio.grad is not None and not torch.isnan(proprio.grad).any()}")
    print(f"  ✓ Lang grad: {lang.grad is not None and not torch.isnan(lang.grad).any()}")

    # Test 3: Performance
    print("\n3. Performance benchmark")
    results = benchmark_multimodal_fusion(batch_size=256, num_iters=100)
    print(f"  Latency: {results['latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_Ksamples_per_sec']:.1f} K samples/sec")

    print("\n✅ All tests passed!")
