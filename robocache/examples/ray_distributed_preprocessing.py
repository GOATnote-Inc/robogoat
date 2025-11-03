#!/usr/bin/env python3
"""
Ray Distributed Preprocessing Example for RoboCache

This example demonstrates how to use Ray for distributed GPU-accelerated
preprocessing of large-scale robot trajectory datasets (1M+ trajectories).

Architecture:
- Ray cluster with multiple GPU nodes (e.g., DGX H100 with 8 GPUs)
- Each GPU runs a TrajectoryPreprocessor actor
- Work is distributed across GPUs for parallel processing
- Results are aggregated and optionally saved to distributed storage

Use cases:
- Offline dataset preprocessing (RT-X, Open-X Embodiment, custom datasets)
- Real-time data augmentation during training
- Multi-terabyte trajectory resampling pipelines

Requirements:
- ray[default] >= 2.9.0
- robocache
- torch with CUDA support
"""

import ray
import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    print("ERROR: RoboCache not installed. Run: pip install -e .")
    ROBOCACHE_AVAILABLE = False


@dataclass
class PreprocessingConfig:
    """Configuration for trajectory preprocessing pipeline"""
    target_frequency: float = 50.0  # Hz
    target_length: int = 100  # frames
    batch_size: int = 256
    num_workers: int = 8  # Number of GPU workers
    dtype: str = 'bfloat16'  # 'bfloat16', 'float16', or 'float32'


@ray.remote(num_gpus=1)
class TrajectoryPreprocessor:
    """
    Ray actor for GPU-accelerated trajectory preprocessing.

    Each actor runs on a single GPU and processes batches of trajectories
    using RoboCache kernels. Includes telemetry for monitoring throughput.
    """

    def __init__(self, gpu_id: int, config: PreprocessingConfig):
        """
        Initialize preprocessor actor on specified GPU.

        Args:
            gpu_id: CUDA device ID to use
            config: Preprocessing configuration
        """
        self.gpu_id = gpu_id
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')

        # Set CUDA device for this actor
        torch.cuda.set_device(self.device)

        # Initialize metrics
        self.trajectories_processed = 0
        self.total_time = 0.0
        self.batch_count = 0

        print(f"[GPU {gpu_id}] TrajectoryPreprocessor initialized")
        print(f"[GPU {gpu_id}] Device: {torch.cuda.get_device_name(gpu_id)}")

    def process_batch(
        self,
        source_data: np.ndarray,
        source_times: np.ndarray,
        trajectory_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a batch of trajectories using GPU acceleration.

        Args:
            source_data: [batch, source_len, action_dim] NumPy array
            source_times: [batch, source_len] NumPy array
            trajectory_ids: Optional list of trajectory IDs for tracking

        Returns:
            resampled_data: [batch, target_len, action_dim] NumPy array
            metrics: Dictionary of performance metrics
        """
        batch_size = source_data.shape[0]
        source_len = source_data.shape[1]
        action_dim = source_data.shape[2]

        # Convert to PyTorch tensors on GPU
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        dtype = dtype_map[self.config.dtype]

        source_data_gpu = torch.from_numpy(source_data).to(
            device=self.device, dtype=dtype
        )
        source_times_gpu = torch.from_numpy(source_times).to(
            device=self.device, dtype=torch.float32
        )

        # Generate target timestamps (uniform frequency)
        max_time = source_times_gpu.max()
        target_length = self.config.target_length
        target_times_gpu = torch.linspace(
            0, max_time.item(), target_length, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)

        # GPU-accelerated resampling
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            resampled_gpu = robocache.resample_trajectories(
                source_data_gpu, source_times_gpu, target_times_gpu
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # Convert back to NumPy (for compatibility with storage systems)
        resampled_np = resampled_gpu.cpu().numpy()

        # Update metrics
        self.trajectories_processed += batch_size
        self.total_time += elapsed
        self.batch_count += 1

        # Calculate bandwidth
        dtype_bytes = 2 if dtype in [torch.bfloat16, torch.float16] else 4
        bytes_read = batch_size * source_len * action_dim * dtype_bytes
        bytes_written = batch_size * target_length * action_dim * dtype_bytes
        bandwidth_gbps = (bytes_read + bytes_written) / elapsed / 1e9

        metrics = {
            'batch_size': batch_size,
            'elapsed_seconds': elapsed,
            'throughput_traj_per_sec': batch_size / elapsed,
            'bandwidth_gbps': bandwidth_gbps,
            'gpu_id': self.gpu_id,
        }

        return resampled_np, metrics

    def get_stats(self) -> Dict[str, float]:
        """
        Get aggregate statistics for this worker.

        Returns:
            Dictionary of cumulative statistics
        """
        avg_time = self.total_time / self.batch_count if self.batch_count > 0 else 0
        avg_throughput = (
            self.trajectories_processed / self.total_time
            if self.total_time > 0 else 0
        )

        return {
            'gpu_id': self.gpu_id,
            'trajectories_processed': self.trajectories_processed,
            'total_time_seconds': self.total_time,
            'batch_count': self.batch_count,
            'avg_time_per_batch_seconds': avg_time,
            'avg_throughput_traj_per_sec': avg_throughput,
        }


def generate_synthetic_trajectories(
    num_trajectories: int,
    source_length: int,
    action_dim: int,
    frequency_hz: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic robot trajectories for testing.

    Args:
        num_trajectories: Number of trajectories to generate
        source_length: Number of frames per trajectory
        action_dim: Dimensionality of action space
        frequency_hz: Sampling frequency

    Returns:
        data: [num_trajectories, source_length, action_dim]
        times: [num_trajectories, source_length]
    """
    dt = 1.0 / frequency_hz
    times = np.arange(source_length) * dt
    times = np.tile(times, (num_trajectories, 1))

    # Generate smooth trajectories using sine waves
    data = np.zeros((num_trajectories, source_length, action_dim), dtype=np.float32)
    for i in range(num_trajectories):
        for d in range(action_dim):
            freq = 0.5 + 0.3 * np.random.randn()
            phase = 2 * np.pi * np.random.rand()
            amplitude = 0.5 + 0.5 * np.random.rand()
            data[i, :, d] = amplitude * np.sin(2 * np.pi * freq * times[i] + phase)

    return data, times


class DistributedPreprocessingPipeline:
    """
    Distributed preprocessing pipeline using Ray for GPU parallelization.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize distributed preprocessing pipeline.

        Args:
            config: Preprocessing configuration
        """
        self.config = config

        # Initialize Ray cluster
        if not ray.is_initialized():
            ray.init()

        # Get available GPU resources
        resources = ray.cluster_resources()
        num_gpus = int(resources.get('GPU', 0))

        if num_gpus == 0:
            raise RuntimeError("No GPUs available in Ray cluster")

        # Limit workers to available GPUs
        self.num_workers = min(config.num_workers, num_gpus)

        print(f"Initializing {self.num_workers} GPU workers...")

        # Create worker actors (one per GPU)
        self.workers = [
            TrajectoryPreprocessor.remote(gpu_id=i, config=config)
            for i in range(self.num_workers)
        ]

        print(f"✓ {self.num_workers} workers initialized")

    def process_dataset(
        self,
        dataset_data: np.ndarray,
        dataset_times: np.ndarray,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process entire dataset using distributed GPU workers.

        Args:
            dataset_data: [N, source_len, action_dim] full dataset
            dataset_times: [N, source_len] timestamps
            verbose: Print progress updates

        Returns:
            resampled_dataset: [N, target_len, action_dim]
            aggregate_metrics: Performance statistics
        """
        num_trajectories = dataset_data.shape[0]
        batch_size = self.config.batch_size

        # Split dataset into batches
        num_batches = (num_trajectories + batch_size - 1) // batch_size

        if verbose:
            print(f"\nProcessing {num_trajectories} trajectories in {num_batches} batches")
            print(f"Batch size: {batch_size}")
            print(f"Workers: {self.num_workers}")
            print()

        # Distribute batches across workers using round-robin
        batch_futures = []
        batch_indices = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_trajectories)

            batch_data = dataset_data[start_idx:end_idx]
            batch_times = dataset_times[start_idx:end_idx]

            # Assign to worker (round-robin distribution)
            worker_id = batch_idx % self.num_workers

            # Submit task to Ray
            future = self.workers[worker_id].process_batch.remote(
                batch_data, batch_times
            )
            batch_futures.append(future)
            batch_indices.append((start_idx, end_idx))

        # Collect results
        start_time = time.time()
        results = ray.get(batch_futures)
        total_time = time.time() - start_time

        # Aggregate results
        resampled_batches = []
        all_metrics = []

        for (start_idx, end_idx), (resampled_batch, metrics) in zip(
            batch_indices, results
        ):
            resampled_batches.append(resampled_batch)
            all_metrics.append(metrics)

            if verbose and len(resampled_batches) % 10 == 0:
                print(f"  Processed {len(resampled_batches)}/{num_batches} batches "
                      f"({len(resampled_batches) * batch_size} trajectories)")

        # Concatenate all batches
        resampled_dataset = np.concatenate(resampled_batches, axis=0)

        # Calculate aggregate metrics
        total_trajectories = sum(m['batch_size'] for m in all_metrics)
        avg_throughput = total_trajectories / total_time
        avg_bandwidth = np.mean([m['bandwidth_gbps'] for m in all_metrics])

        aggregate_metrics = {
            'total_trajectories': total_trajectories,
            'total_time_seconds': total_time,
            'avg_throughput_traj_per_sec': avg_throughput,
            'avg_bandwidth_gbps': avg_bandwidth,
            'num_workers': self.num_workers,
        }

        if verbose:
            print(f"\n✓ Processing complete!")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {avg_throughput:.1f} trajectories/sec")
            print(f"  Bandwidth: {avg_bandwidth:.1f} GB/s (aggregate)")
            print(f"  Per-GPU throughput: {avg_throughput / self.num_workers:.1f} traj/s")

        return resampled_dataset, aggregate_metrics

    def get_worker_stats(self) -> List[Dict[str, float]]:
        """
        Get statistics from all workers.

        Returns:
            List of worker statistics dictionaries
        """
        stats_futures = [worker.get_stats.remote() for worker in self.workers]
        return ray.get(stats_futures)

    def shutdown(self):
        """Shutdown Ray cluster and cleanup resources."""
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Ray distributed preprocessing example for RoboCache'
    )
    parser.add_argument(
        '--num-trajectories', type=int, default=10000,
        help='Number of trajectories to process (default: 10000)'
    )
    parser.add_argument(
        '--source-length', type=int, default=100,
        help='Length of source trajectories (default: 100)'
    )
    parser.add_argument(
        '--action-dim', type=int, default=32,
        help='Action space dimension (default: 32)'
    )
    parser.add_argument(
        '--target-frequency', type=float, default=50.0,
        help='Target resampling frequency in Hz (default: 50.0)'
    )
    parser.add_argument(
        '--target-length', type=int, default=100,
        help='Length of resampled trajectories (default: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size per worker (default: 256)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='Number of GPU workers (default: 8)'
    )
    parser.add_argument(
        '--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16',
        help='Data type for processing (default: bfloat16)'
    )

    args = parser.parse_args()

    if not ROBOCACHE_AVAILABLE:
        print("ERROR: RoboCache not available. Exiting.")
        return

    print("=" * 80)
    print("RoboCache Ray Distributed Preprocessing Example")
    print("=" * 80)
    print()

    # Configuration
    config = PreprocessingConfig(
        target_frequency=args.target_frequency,
        target_length=args.target_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dtype=args.dtype,
    )

    print("Configuration:")
    print(f"  Num trajectories: {args.num_trajectories:,}")
    print(f"  Source length: {args.source_length}")
    print(f"  Target length: {args.target_length}")
    print(f"  Action dim: {args.action_dim}")
    print(f"  Target frequency: {args.target_frequency} Hz")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Data type: {args.dtype}")
    print()

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset_data, dataset_times = generate_synthetic_trajectories(
        num_trajectories=args.num_trajectories,
        source_length=args.source_length,
        action_dim=args.action_dim,
        frequency_hz=100.0,
    )
    print(f"  Dataset shape: {dataset_data.shape}")
    print(f"  Dataset size: {dataset_data.nbytes / 1e9:.2f} GB")
    print()

    # Initialize distributed pipeline
    pipeline = DistributedPreprocessingPipeline(config)

    # Process dataset
    resampled_dataset, metrics = pipeline.process_dataset(
        dataset_data, dataset_times, verbose=True
    )

    # Print worker statistics
    print("\nWorker Statistics:")
    print("-" * 80)
    worker_stats = pipeline.get_worker_stats()
    for stats in worker_stats:
        print(f"  GPU {stats['gpu_id']}:")
        print(f"    Trajectories processed: {stats['trajectories_processed']:,}")
        print(f"    Total time: {stats['total_time_seconds']:.2f}s")
        print(f"    Avg throughput: {stats['avg_throughput_traj_per_sec']:.1f} traj/s")

    # Calculate scaling efficiency
    print("\nScaling Analysis:")
    print("-" * 80)
    single_gpu_throughput = np.mean([
        s['avg_throughput_traj_per_sec'] for s in worker_stats
    ])
    aggregate_throughput = metrics['avg_throughput_traj_per_sec']
    ideal_scaling = single_gpu_throughput * config.num_workers
    scaling_efficiency = (aggregate_throughput / ideal_scaling) * 100

    print(f"  Single GPU throughput: {single_gpu_throughput:.1f} traj/s")
    print(f"  Aggregate throughput: {aggregate_throughput:.1f} traj/s")
    print(f"  Ideal linear scaling: {ideal_scaling:.1f} traj/s")
    print(f"  Scaling efficiency: {scaling_efficiency:.1f}%")

    if scaling_efficiency > 90:
        print("  ✓ Excellent scaling (>90% efficiency)")
    elif scaling_efficiency > 75:
        print("  ✓ Good scaling (75-90% efficiency)")
    else:
        print("  ⚠ Suboptimal scaling (<75% efficiency)")
        print("    Consider: larger batch sizes, faster storage, network optimization")

    print()

    # Cleanup
    pipeline.shutdown()

    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
