#!/usr/bin/env python3
"""
Basic usage example for RoboCache trajectory resampling

This example demonstrates:
1. Loading robot trajectories at different frequencies
2. Resampling to a uniform frequency for batched training
3. Performance comparison with PyTorch CPU baseline
4. Integration with robot learning datasets
"""

import torch
import time
import numpy as np

# Check if RoboCache is installed
try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    print("WARNING: RoboCache not found. Install with: pip install -e .")
    ROBOCACHE_AVAILABLE = False


def generate_robot_trajectory(
    num_frames: int,
    action_dim: int,
    frequency_hz: float = 100.0,
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic robot trajectory with smooth motion.

    Args:
        num_frames: Number of timesteps
        action_dim: Dimension of action space (e.g., 7 for robot arm)
        frequency_hz: Sampling frequency
        device: 'cuda' or 'cpu'

    Returns:
        actions: Tensor of shape [num_frames, action_dim]
        times: Tensor of shape [num_frames] (in seconds)
    """
    dt = 1.0 / frequency_hz
    times = torch.arange(num_frames, device=device, dtype=torch.float32) * dt

    # Generate smooth trajectories using multiple sine waves
    actions = torch.zeros(num_frames, action_dim, device=device)
    for i in range(action_dim):
        freq = 0.5 + 0.3 * np.random.randn()  # Random frequency
        phase = 2 * np.pi * np.random.rand()   # Random phase
        amplitude = 0.5 + 0.5 * np.random.rand()  # Random amplitude
        actions[:, i] = amplitude * torch.sin(2 * np.pi * freq * times + phase)

    return actions, times


def pytorch_baseline_resample(
    source_data: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch CPU baseline for trajectory resampling (for comparison).

    This is the typical approach without RoboCache.
    Much slower, especially for large batches.
    """
    batch_size = source_data.shape[0]
    target_length = target_times.shape[1]
    action_dim = source_data.shape[2]

    output = torch.zeros(batch_size, target_length, action_dim, dtype=source_data.dtype)

    # Move to CPU for interpolation
    source_data_cpu = source_data.cpu()
    source_times_cpu = source_times.cpu()
    target_times_cpu = target_times.cpu()

    # Interpolate each batch and dimension separately (slow!)
    for b in range(batch_size):
        for d in range(action_dim):
            output[b, :, d] = torch.from_numpy(
                np.interp(
                    target_times_cpu[b].numpy(),
                    source_times_cpu[b].numpy(),
                    source_data_cpu[b, :, d].numpy()
                )
            )

    return output.to(source_data.device)


def main():
    print("=" * 80)
    print("RoboCache Basic Usage Example")
    print("=" * 80)
    print()

    # Check installation
    if ROBOCACHE_AVAILABLE:
        robocache.print_installation_info()
        print()
    else:
        print("ERROR: RoboCache not available. Please build and install first.")
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. RoboCache requires a CUDA GPU.")
        return

    # ==============================================================================
    # Example 1: Basic trajectory resampling
    # ==============================================================================

    print("=" * 80)
    print("Example 1: Basic Trajectory Resampling")
    print("=" * 80)
    print()

    # Simulate heterogeneous robot data at different frequencies
    batch_size = 64
    action_dim = 32  # e.g., 7 DOF arm + gripper + 6D pose

    # Create trajectories at different frequencies (like real robot data)
    print("Generating robot trajectories at different frequencies...")
    print(f"  Batch size: {batch_size}")
    print(f"  Action dimension: {action_dim}")
    print()

    trajectories = []
    timestamps = []

    frequencies = [30, 50, 100, 125]  # Different robot types
    for i in range(batch_size):
        freq = frequencies[i % len(frequencies)]
        num_frames = int(freq * 2)  # 2 seconds of data
        actions, times = generate_robot_trajectory(num_frames, action_dim, freq)
        trajectories.append(actions)
        timestamps.append(times)

    print("Generated trajectories:")
    for i, (traj, times) in enumerate(zip(trajectories[:4], timestamps[:4])):
        freq = frequencies[i % len(frequencies)]
        print(f"  Trajectory {i}: {traj.shape[0]} frames at {freq} Hz")

    # Pad to same length for batching (max length)
    max_len = max(t.shape[0] for t in trajectories)
    source_data = torch.zeros(batch_size, max_len, action_dim, dtype=torch.bfloat16, device='cuda')
    source_times = torch.zeros(batch_size, max_len, dtype=torch.float32, device='cuda')

    for i, (traj, times) in enumerate(zip(trajectories, timestamps)):
        length = traj.shape[0]
        source_data[i, :length] = traj.to(torch.bfloat16)
        source_times[i, :length] = times
        # Fill remaining with last values
        if length < max_len:
            source_data[i, length:] = source_data[i, length-1]
            source_times[i, length:] = source_times[i, length-1]

    # Create target times (uniform 50 Hz for all trajectories)
    target_freq = 50.0
    target_length = int(target_freq * 2)  # 2 seconds at 50 Hz
    target_times = torch.linspace(0, 2.0, target_length, device='cuda').unsqueeze(0).expand(batch_size, -1)

    print()
    print(f"Resampling all trajectories to uniform {target_freq} Hz...")

    # Resample using RoboCache
    with torch.no_grad():
        resampled = robocache.resample_trajectories(source_data, source_times, target_times)

    print(f"  Input shape:  {source_data.shape}")
    print(f"  Output shape: {resampled.shape}")
    print(f"  All trajectories now at {target_freq} Hz!")
    print()

    # ==============================================================================
    # Example 2: Performance comparison
    # ==============================================================================

    print("=" * 80)
    print("Example 2: Performance Comparison")
    print("=" * 80)
    print()

    # Create test data
    test_batch = 256
    test_source_len = 100
    test_target_len = 50
    test_action_dim = 32

    test_data = torch.randn(test_batch, test_source_len, test_action_dim,
                           dtype=torch.bfloat16, device='cuda')
    test_src_times = torch.linspace(0, 1, test_source_len, device='cuda').unsqueeze(0).expand(test_batch, -1)
    test_tgt_times = torch.linspace(0, 1, test_target_len, device='cuda').unsqueeze(0).expand(test_batch, -1)

    print(f"Test configuration:")
    print(f"  Batch size: {test_batch}")
    print(f"  Source length: {test_source_len} frames")
    print(f"  Target length: {test_target_len} frames")
    print(f"  Action dim: {test_action_dim}")
    print()

    # Warmup
    for _ in range(10):
        _ = robocache.resample_trajectories(test_data, test_src_times, test_tgt_times)
    torch.cuda.synchronize()

    # Benchmark RoboCache
    num_iterations = 1000
    start = time.time()
    for _ in range(num_iterations):
        _ = robocache.resample_trajectories(test_data, test_src_times, test_tgt_times)
    torch.cuda.synchronize()
    robocache_time = (time.time() - start) / num_iterations

    print(f"RoboCache Performance:")
    print(f"  Time per batch: {robocache_time * 1000:.3f} ms")
    print(f"  Throughput: {test_batch * test_target_len / robocache_time:.0f} samples/sec")
    print(f"  Throughput: {test_batch * test_target_len / robocache_time / 1000:.1f} K samples/sec")
    print()

    # Benchmark PyTorch baseline (smaller batch for speed)
    print("PyTorch CPU Baseline (using smaller batch for comparison):")
    small_batch = 16
    small_data = test_data[:small_batch].float()  # Use FP32 for CPU
    small_src_times = test_src_times[:small_batch]
    small_tgt_times = test_tgt_times[:small_batch]

    start = time.time()
    _ = pytorch_baseline_resample(small_data, small_src_times, small_tgt_times)
    baseline_time = time.time() - start

    print(f"  Time per batch: {baseline_time * 1000:.3f} ms (batch={small_batch})")
    print(f"  Throughput: {small_batch * test_target_len / baseline_time:.0f} samples/sec")
    print()

    # Estimate speedup (scale baseline to full batch)
    estimated_baseline_time = baseline_time * (test_batch / small_batch)
    speedup = estimated_baseline_time / robocache_time

    print(f"Speedup: {speedup:.1f}x faster than PyTorch CPU")
    print()

    # ==============================================================================
    # Example 3: Integration with training loop
    # ==============================================================================

    print("=" * 80)
    print("Example 3: Integration with Training Loop")
    print("=" * 80)
    print()

    print("Typical usage in robot learning training:")
    print("""
    # In your training loop:
    for batch in dataloader:
        # Heterogeneous robot data at different frequencies
        source_data = batch['trajectories']  # [B, T_var, D]
        source_times = batch['timestamps']   # [B, T_var]

        # Resample to uniform frequency for batching
        target_times = torch.linspace(0, T_max, T_uniform).expand(B, -1)
        resampled = robocache.resample_trajectories(
            source_data, source_times, target_times
        )

        # Now all trajectories are uniform length - ready for model!
        output = model(resampled)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    """)

    print("=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
