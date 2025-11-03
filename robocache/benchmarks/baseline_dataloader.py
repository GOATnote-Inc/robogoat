#!/usr/bin/env python3
"""
Baseline PyTorch DataLoader for robot learning data.

This is the standard approach that most researchers use:
- CPU-based preprocessing
- NumPy interpolation
- Multiple worker processes
- CPU-GPU transfer overhead

This is SLOW but represents the current state-of-the-art.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import time
import argparse
from pathlib import Path


class BaselineRobotDataset(Dataset):
    """
    Standard PyTorch dataset with CPU preprocessing.

    This represents the typical implementation used in robot learning research.
    All preprocessing happens on CPU, which is slow but straightforward.
    """

    def __init__(self, h5_path, target_fps=50, max_duration=4.0):
        self.h5_path = h5_path
        self.target_fps = target_fps
        self.max_duration = max_duration

        # Open file to read metadata
        with h5py.File(h5_path, 'r') as f:
            self.num_trajectories = len(f['lengths'])
            self.lengths = f['lengths'][:]

        print(f"Loaded dataset: {self.num_trajectories} trajectories")

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        """
        Load and preprocess one trajectory.

        Everything happens on CPU:
        1. Load from disk (slow)
        2. Resample with NumPy (slow)
        3. Convert to tensors (slow)
        4. Will be transferred to GPU later (slow)
        """
        with h5py.File(self.h5_path, 'r') as f:
            traj_len = self.lengths[idx]

            # Load data from disk
            actions = f['actions'][idx, :traj_len].astype(np.float32)
            proprio = f['proprioception'][idx, :traj_len].astype(np.float32)
            timestamps = f['timestamps'][idx, :traj_len].astype(np.float32)

            # Resample to target frequency using NumPy (CPU-based, slow)
            duration = timestamps[-1]
            target_len = int(duration * self.target_fps)
            target_times = np.linspace(timestamps[0], timestamps[-1], target_len)

            # Linear interpolation for each dimension (very slow)
            actions_resampled = np.stack([
                np.interp(target_times, timestamps, actions[:, i])
                for i in range(actions.shape[1])
            ], axis=1)

            proprio_resampled = np.stack([
                np.interp(target_times, timestamps, proprio[:, i])
                for i in range(proprio.shape[1])
            ], axis=1)

            # Convert to PyTorch tensors (additional copy)
            actions_tensor = torch.from_numpy(actions_resampled).float()
            proprio_tensor = torch.from_numpy(proprio_resampled).float()

        return {
            'actions': actions_tensor,
            'proprio': proprio_tensor,
        }


def benchmark_baseline(
    data_path,
    batch_size=32,
    num_workers=8,
    num_batches=100,
    target_fps=50
):
    """
    Benchmark standard PyTorch DataLoader.

    This uses the typical setup:
    - Multiple CPU workers
    - Pin memory for faster CPU->GPU transfer
    - Prefetching for overlap

    Despite these optimizations, it's still slow because:
    - CPU preprocessing is the bottleneck
    - Limited parallelism (GIL, process overhead)
    - Slow disk I/O
    """

    print("=" * 80)
    print("BASELINE: Standard PyTorch DataLoader")
    print("=" * 80)
    print()

    dataset = BaselineRobotDataset(data_path, target_fps=target_fps)

    # Standard DataLoader configuration
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Configuration:")
    print(f"  Dataset: {len(dataset)} trajectories")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Target FPS: {target_fps}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Warmup
    print("Warming up (5 batches)...")
    for i, batch in enumerate(loader):
        if torch.cuda.is_available():
            # Simulate moving to GPU (typical in training)
            _ = batch['actions'].cuda(non_blocking=True)
            _ = batch['proprio'].cuda(non_blocking=True)
        if i >= 4:
            break

    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    total_samples = 0
    total_frames = 0

    for i, batch in enumerate(loader):
        if torch.cuda.is_available():
            # Transfer to GPU
            actions = batch['actions'].cuda(non_blocking=True)
            proprio = batch['proprio'].cuda(non_blocking=True)
        else:
            actions = batch['actions']
            proprio = batch['proprio']

        batch_size_actual = actions.shape[0]
        seq_len = actions.shape[1]

        total_samples += batch_size_actual
        total_frames += batch_size_actual * seq_len

        if i >= num_batches - 1:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    throughput_trajs = total_samples / elapsed
    throughput_frames = total_frames / elapsed
    time_per_batch = elapsed / num_batches * 1000

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Total time:              {elapsed:.2f} seconds")
    print(f"Total trajectories:      {total_samples}")
    print(f"Total frames:            {total_frames}")
    print(f"Throughput:              {throughput_trajs:.1f} trajectories/sec")
    print(f"Throughput:              {throughput_frames:.0f} frames/sec")
    print(f"Time per batch:          {time_per_batch:.1f} ms")
    print("=" * 80)
    print()

    return {
        'elapsed': elapsed,
        'total_samples': total_samples,
        'total_frames': total_frames,
        'throughput_trajs': throughput_trajs,
        'throughput_frames': throughput_frames,
        'time_per_batch_ms': time_per_batch,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark baseline PyTorch DataLoader')
    parser.add_argument('--data', type=str, default='./data/robot_learning/robot_synthetic.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of DataLoader workers (default: 8)')
    parser.add_argument('--num-batches', type=int, default=100,
                       help='Number of batches to benchmark (default: 100)')
    parser.add_argument('--target-fps', type=int, default=50,
                       help='Target resampling frequency in Hz (default: 50)')

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found at {args.data}")
        print("Run download_data.py first to generate the dataset.")
        return

    results = benchmark_baseline(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        target_fps=args.target_fps
    )

    print("\n💡 Bottlenecks in this baseline implementation:")
    print("   1. CPU-based NumPy interpolation (np.interp)")
    print("   2. Multiple process overhead (multiprocessing)")
    print("   3. Disk I/O for each sample")
    print("   4. CPU-to-GPU data transfer")
    print("\n🚀 RoboCache solves this by moving everything to GPU!")
    print()


if __name__ == '__main__':
    main()
