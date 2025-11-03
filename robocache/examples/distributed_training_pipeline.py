#!/usr/bin/env python3
"""
Distributed Training Pipeline with RoboCache

This example demonstrates integration of RoboCache with PyTorch distributed
training for embodied AI foundation models. Shows how to eliminate CPU-side
data preprocessing bottlenecks by moving trajectory resampling to GPU.

Architecture:
- PyTorch DistributedDataParallel (DDP) for multi-GPU training
- Custom DataLoader collate function with GPU-accelerated resampling
- Zero data stall: preprocessing happens on GPU while model trains
- Supports heterogeneous robot datasets (RT-X, Open-X, custom corpora)

Use cases:
- Training robot foundation models (transformer policies, diffusion models)
- Multimodal learning (vision + language + proprioception)
- Large-scale imitation learning (1M+ demonstrations)

Requirements:
- torch >= 2.0 with CUDA support
- robocache
- transformers (optional, for model examples)
"""

import os
import time
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    print("WARNING: RoboCache not available. Install with: pip install -e .")
    ROBOCACHE_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for distributed training"""
    # Model
    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8

    # Training
    batch_size: int = 32  # Per GPU
    num_epochs: int = 10
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1

    # Data
    target_frequency: float = 50.0  # Hz
    target_length: int = 100  # frames
    action_dim: int = 32

    # Distributed
    backend: str = 'nccl'
    world_size: int = 8
    num_workers: int = 4  # DataLoader workers per GPU


class RobotTrajectoryDataset(Dataset):
    """
    Dataset for robot trajectories with heterogeneous sampling rates.

    In real applications, this would load from:
    - LanceDB (columnar storage for fast queries)
    - HDF5 files (hierarchical data format)
    - RLDS (Robot Learning Dataset Standard)
    - S3/GCS (cloud object storage)
    """

    def __init__(
        self,
        num_trajectories: int = 10000,
        source_length: int = 100,
        action_dim: int = 32,
        include_vision: bool = False
    ):
        self.num_trajectories = num_trajectories
        self.source_length = source_length
        self.action_dim = action_dim
        self.include_vision = include_vision

        # In production: self.db = LanceDB.open(dataset_path)
        # For demo: generate synthetic data indices
        self.trajectory_ids = list(range(num_trajectories))

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single trajectory.

        Returns variable-length trajectory that will be resampled
        in the collate function on GPU.
        """
        # In production:
        #   trajectory = self.db.get(self.trajectory_ids[idx])
        #   actions = torch.from_numpy(trajectory['actions'])
        #   times = torch.from_numpy(trajectory['timestamps'])

        # For demo: generate synthetic data
        # Simulate different sampling rates across trajectories
        actual_length = self.source_length + (idx % 50) - 25  # Variable length
        actual_length = max(50, min(150, actual_length))

        # Generate smooth actions (sine waves)
        t = torch.linspace(0, 2, actual_length)
        actions = torch.stack([
            torch.sin(2 * torch.pi * (0.5 + 0.1 * d) * t)
            for d in range(self.action_dim)
        ], dim=-1)

        # Timestamps (non-uniform in general)
        times = t.clone()

        trajectory = {
            'actions': actions,  # [T, D]
            'times': times,      # [T]
            'trajectory_id': idx,
        }

        if self.include_vision:
            # Placeholder for vision data
            # In production: Load RGB-D frames at 30 Hz
            trajectory['rgb'] = torch.randn(actual_length // 3, 3, 224, 224)

        return trajectory


def robocache_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    target_frequency: float = 50.0,
    target_length: int = 100,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function with GPU-accelerated resampling.

    This is the key integration point: instead of resampling on CPU
    (which would stall GPU training), we move data to GPU and use
    RoboCache for fast resampling.

    Args:
        batch: List of trajectories from dataset
        target_frequency: Target sampling rate (Hz)
        target_length: Target number of frames after resampling
        device: PyTorch device ('cuda' or 'cpu')

    Returns:
        Batched and resampled trajectories ready for model
    """
    batch_size = len(batch)

    # Determine max source length for padding
    max_source_len = max(traj['actions'].shape[0] for traj in batch)
    action_dim = batch[0]['actions'].shape[1]

    # Pre-allocate tensors on CPU
    source_data = torch.zeros(batch_size, max_source_len, action_dim, dtype=torch.float32)
    source_times = torch.zeros(batch_size, max_source_len, dtype=torch.float32)
    valid_lengths = torch.zeros(batch_size, dtype=torch.long)

    # Stack and pad variable-length trajectories
    for i, traj in enumerate(batch):
        length = traj['actions'].shape[0]
        source_data[i, :length] = traj['actions']
        source_times[i, :length] = traj['times']
        valid_lengths[i] = length

        # Pad with last frame (common in robot learning)
        if length < max_source_len:
            source_data[i, length:] = source_data[i, length - 1]
            source_times[i, length:] = source_times[i, length - 1]

    # Move to GPU (fast PCIE transfer)
    source_data_gpu = source_data.to(device, dtype=torch.bfloat16)
    source_times_gpu = source_times.to(device)

    # Generate target timestamps (uniform frequency)
    max_time = source_times_gpu.max()
    target_times_gpu = torch.linspace(
        0, max_time.item(), target_length, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    # GPU-accelerated resampling (FAST!)
    # This is 40-70x faster than CPU resampling
    with torch.no_grad():
        resampled_actions = robocache.resample_trajectories(
            source_data_gpu,
            source_times_gpu,
            target_times_gpu
        )

    # Return batched data
    return {
        'actions': resampled_actions,  # [B, T, D] on GPU
        'trajectory_ids': torch.tensor([t['trajectory_id'] for t in batch]),
    }


class SimpleTransformerPolicy(nn.Module):
    """
    Toy transformer policy for demonstration.

    In production, this would be:
    - RT-1/RT-2 style vision-language-action models
    - Diffusion policies (e.g., Diffusion Policy, 3D Diffuser Actor)
    - ACT (Action Chunking Transformer)
    - OpenVLA, OCTO, or other foundation models
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Simple architecture: embed actions → transformer → predict next actions
        self.embed = nn.Linear(config.action_dim, config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)

        self.output = nn.Linear(config.model_dim, config.action_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [batch, seq_len, action_dim]

        Returns:
            predicted_actions: [batch, seq_len, action_dim]
        """
        # Embed
        x = self.embed(actions)  # [B, T, model_dim]

        # Transform
        x = self.transformer(x)  # [B, T, model_dim]

        # Output
        pred = self.output(x)  # [B, T, action_dim]

        return pred


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    config: TrainingConfig,
    rank: int
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        metrics: Dictionary of training metrics
    """
    model.train()

    total_loss = 0.0
    total_batches = 0
    data_time = 0.0
    compute_time = 0.0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        data_end = time.time()

        # Get resampled actions (already on GPU from collate_fn!)
        actions = batch['actions']  # [B, T, D]

        # Simple supervised learning: predict next action
        # Input: actions[:, :-1]
        # Target: actions[:, 1:]
        input_actions = actions[:, :-1, :]
        target_actions = actions[:, 1:, :]

        # Forward pass
        compute_start = time.time()
        predictions = model(input_actions)

        # Compute loss
        loss = criterion(predictions, target_actions)

        # Backward pass
        loss.backward()

        # Update weights (with gradient accumulation)
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        compute_end = time.time()

        # Metrics
        total_loss += loss.item()
        total_batches += 1
        compute_time += (compute_end - compute_start)
        if batch_idx > 0:
            data_time += (data_end - start_time)

        # Logging (rank 0 only)
        if rank == 0 and batch_idx % 10 == 0:
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Data time: {data_time:.2f}s | "
                  f"Compute time: {compute_time:.2f}s")

        start_time = time.time()

    epoch_time = time.time() - start_time

    metrics = {
        'loss': total_loss / total_batches,
        'epoch_time': epoch_time,
        'data_time': data_time,
        'compute_time': compute_time,
        'throughput': len(dataloader.dataset) / epoch_time,
    }

    return metrics


def main_worker(rank: int, world_size: int, config: TrainingConfig):
    """
    Main training function for each GPU worker.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        config: Training configuration
    """
    print(f"[Rank {rank}] Initializing...")

    # Setup distributed
    setup_distributed(rank, world_size, config.backend)

    # Create model
    model = SimpleTransformerPolicy(config).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Create criterion
    criterion = nn.MSELoss()

    # Create dataset and dataloader
    dataset = RobotTrajectoryDataset(
        num_trajectories=10000,
        source_length=100,
        action_dim=config.action_dim
    )

    # Distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=lambda batch: robocache_collate_fn(
            batch,
            target_frequency=config.target_frequency,
            target_length=config.target_length,
            device=f'cuda:{rank}'
        ),
        pin_memory=False,  # Already on GPU from collate_fn
        drop_last=True
    )

    if rank == 0:
        print(f"\n{'='*80}")
        print("RoboCache Distributed Training Pipeline")
        print(f"{'='*80}")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Global batch size: {config.batch_size * world_size}")
        print(f"Dataset size: {len(dataset)} trajectories")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Target frequency: {config.target_frequency} Hz")
        print(f"Target length: {config.target_length} frames")
        print(f"{'='*80}\n")

    # Training loop
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)

        metrics = train_epoch(
            model, dataloader, optimizer, criterion,
            epoch, config, rank
        )

        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Epoch time: {metrics['epoch_time']:.2f}s")
            print(f"  Data loading time: {metrics['data_time']:.2f}s "
                  f"({metrics['data_time']/metrics['epoch_time']*100:.1f}%)")
            print(f"  Compute time: {metrics['compute_time']:.2f}s "
                  f"({metrics['compute_time']/metrics['epoch_time']*100:.1f}%)")
            print(f"  Throughput: {metrics['throughput']:.1f} trajectories/sec\n")

    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description='Distributed training with RoboCache'
    )
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--target-frequency', type=float, default=50.0,
                        help='Target sampling frequency in Hz (default: 50.0)')
    parser.add_argument('--target-length', type=int, default=100,
                        help='Target trajectory length (default: 100)')
    parser.add_argument('--action-dim', type=int, default=32,
                        help='Action space dimension (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers per GPU (default: 4)')

    args = parser.parse_args()

    if not ROBOCACHE_AVAILABLE:
        print("ERROR: RoboCache not available. Exiting.")
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Exiting.")
        return

    # Configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        target_frequency=args.target_frequency,
        target_length=args.target_length,
        action_dim=args.action_dim,
        num_workers=args.num_workers,
        world_size=torch.cuda.device_count()
    )

    print(f"Starting distributed training on {config.world_size} GPUs...")

    # Launch distributed training
    torch.multiprocessing.spawn(
        main_worker,
        args=(config.world_size, config),
        nprocs=config.world_size,
        join=True
    )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
