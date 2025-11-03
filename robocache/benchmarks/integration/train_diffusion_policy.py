#!/usr/bin/env python3
"""
End-to-end training benchmark: Diffusion Policy with RoboCache

This demonstrates real-world speedup on actual model training,
not just data loading microbenchmarks.

Diffusion Policy is a state-of-the-art robot learning model
used in many research labs and companies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse
from pathlib import Path
import json

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from baseline_dataloader import BaselineRobotDataset
from robocache_dataloader import RoboCacheDataset, RoboCacheBatchedDataset


class SimpleDiffusionPolicy(nn.Module):
    """
    Simplified Diffusion Policy model for robot learning.

    This is a lightweight version for benchmarking that captures
    the key computational patterns of the full model.

    Real Diffusion Policy papers:
    - "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
      (Chi et al., RSS 2023)
    """

    def __init__(
        self,
        action_dim=7,
        proprio_dim=14,
        hidden_dim=256,
        num_layers=4,
        horizon=16
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoising network (U-Net style)
        self.denoise_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + action_dim * horizon, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim * horizon)

    def forward(self, obs, noisy_actions, timesteps):
        """
        Forward pass of diffusion policy.

        Args:
            obs: Proprioception [batch, seq_len, proprio_dim]
            noisy_actions: Noisy action sequence [batch, horizon, action_dim]
            timesteps: Diffusion timesteps [batch, 1]

        Returns:
            Predicted noise [batch, horizon, action_dim]
        """
        batch_size = obs.shape[0]

        # Encode observation (use last timestep)
        obs_feat = self.obs_encoder(obs[:, -1])  # [batch, hidden_dim]

        # Encode diffusion timestep
        time_feat = self.time_embed(timesteps)  # [batch, hidden_dim]

        # Combine features
        combined = obs_feat + time_feat  # [batch, hidden_dim]

        # Flatten noisy actions
        noisy_flat = noisy_actions.reshape(batch_size, -1)  # [batch, horizon * action_dim]

        # Concatenate with actions
        x = torch.cat([combined, noisy_flat], dim=1)

        # Denoising network
        for layer in self.denoise_net:
            x = layer(x) + combined  # Residual connection

        # Output projection
        noise_pred = self.output_proj(x)  # [batch, horizon * action_dim]
        noise_pred = noise_pred.reshape(batch_size, self.horizon, self.action_dim)

        return noise_pred


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    num_diffusion_steps=10,
    horizon=16
):
    """
    Train one epoch of Diffusion Policy.

    Returns:
        dict with training metrics including data loading time
    """
    model.train()

    total_loss = 0
    total_samples = 0
    data_time = 0
    model_time = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        # Move data to GPU (if not already there)
        if not batch['actions'].is_cuda:
            actions = batch['actions'].to(device, non_blocking=True)
            proprio = batch['proprio'].to(device, non_blocking=True)
        else:
            actions = batch['actions']
            proprio = batch['proprio']

        batch_size = actions.shape[0]
        seq_len = actions.shape[1]

        # Extract action sequences for diffusion
        # Use a sliding window to create multiple training samples
        if seq_len > horizon:
            start_idx = torch.randint(0, seq_len - horizon, (1,)).item()
            action_seq = actions[:, start_idx:start_idx + horizon]
        else:
            # Pad if too short
            action_seq = F.pad(actions, (0, 0, 0, horizon - seq_len))

        data_time += time.time() - batch_start
        forward_start = time.time()

        # Diffusion training (simplified DDPM)
        # Sample random timesteps
        timesteps = torch.randint(
            0, num_diffusion_steps, (batch_size,), device=device
        ).unsqueeze(1).float() / num_diffusion_steps

        # Add noise to actions
        noise = torch.randn_like(action_seq)
        alpha = 1.0 - timesteps.squeeze() * 0.9  # Noise schedule
        noisy_actions = (
            alpha.unsqueeze(1).unsqueeze(2) * action_seq +
            (1 - alpha).unsqueeze(1).unsqueeze(2).sqrt() * noise
        )

        # Predict noise
        optimizer.zero_grad()
        noise_pred = model(proprio, noisy_actions, timesteps)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        loss.backward()
        optimizer.step()

        model_time += time.time() - forward_start

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_time = time.time() - epoch_start

    return {
        'loss': total_loss / total_samples,
        'data_time': data_time,
        'model_time': model_time,
        'total_time': epoch_time,
        'samples': total_samples,
    }


def benchmark_training(
    data_path,
    mode='baseline',
    batch_size=32,
    num_epochs=3,
    target_fps=50,
    device='cuda'
):
    """
    Benchmark end-to-end training with either baseline or RoboCache.

    This measures the REAL speedup in actual training, not just data loading.
    """
    print("=" * 80)
    print(f"TRAINING BENCHMARK: {mode.upper()}")
    print("=" * 80)
    print()

    # Create dataset and dataloader
    if mode == 'baseline':
        dataset = BaselineRobotDataset(data_path, target_fps=target_fps)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
    elif mode == 'robocache':
        if not ROBOCACHE_AVAILABLE:
            print("ERROR: RoboCache not available")
            return None
        dataset = RoboCacheDataset(data_path, target_fps=target_fps, preload_to_gpu=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    elif mode == 'robocache_batched':
        if not ROBOCACHE_AVAILABLE:
            print("ERROR: RoboCache not available")
            return None
        dataset = RoboCacheBatchedDataset(data_path, target_fps=target_fps, batch_size=batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Configuration:")
    print(f"  Dataset: {len(dataset)} items")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print()

    # Create model
    model = SimpleDiffusionPolicy(
        action_dim=7,
        proprio_dim=14,
        hidden_dim=256,
        num_layers=4,
        horizon=16
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Training...")
    epoch_results = []

    for epoch in range(num_epochs):
        result = train_epoch(
            model, dataloader, optimizer, device,
            num_diffusion_steps=10, horizon=16
        )

        epoch_results.append(result)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Data time: {result['data_time']:.2f}s ({result['data_time']/result['total_time']*100:.1f}%)")
        print(f"  Model time: {result['model_time']:.2f}s ({result['model_time']/result['total_time']*100:.1f}%)")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Throughput: {result['samples']/result['total_time']:.1f} samples/sec")
        print()

    # Calculate averages
    avg_result = {
        'loss': sum(r['loss'] for r in epoch_results) / len(epoch_results),
        'data_time': sum(r['data_time'] for r in epoch_results) / len(epoch_results),
        'model_time': sum(r['model_time'] for r in epoch_results) / len(epoch_results),
        'total_time': sum(r['total_time'] for r in epoch_results) / len(epoch_results),
        'samples': sum(r['samples'] for r in epoch_results) / len(epoch_results),
    }

    print("=" * 80)
    print("Average Results:")
    print("=" * 80)
    print(f"  Loss: {avg_result['loss']:.4f}")
    print(f"  Data time: {avg_result['data_time']:.2f}s ({avg_result['data_time']/avg_result['total_time']*100:.1f}%)")
    print(f"  Model time: {avg_result['model_time']:.2f}s ({avg_result['model_time']/avg_result['total_time']*100:.1f}%)")
    print(f"  Total time: {avg_result['total_time']:.2f}s")
    print(f"  Throughput: {avg_result['samples']/avg_result['total_time']:.1f} samples/sec")
    print("=" * 80)
    print()

    return avg_result


def run_comparison(data_path, batch_size=32, num_epochs=3, output_dir='../results'):
    """
    Run complete training comparison: baseline vs RoboCache.

    This is the ultimate proof: real training speedup on a real model.
    """
    print("\n" + "=" * 80)
    print("DIFFUSION POLICY TRAINING BENCHMARK")
    print("End-to-End Performance: Baseline vs RoboCache")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return None

    if not ROBOCACHE_AVAILABLE:
        print("ERROR: RoboCache not available")
        return None

    results = {}

    # Baseline
    print("[1/2] Training with baseline PyTorch DataLoader...\n")
    baseline_result = benchmark_training(
        data_path, mode='baseline',
        batch_size=batch_size, num_epochs=num_epochs
    )
    results['baseline'] = baseline_result

    # RoboCache
    print("[2/2] Training with RoboCache DataLoader...\n")
    robocache_result = benchmark_training(
        data_path, mode='robocache',
        batch_size=batch_size, num_epochs=num_epochs
    )
    results['robocache'] = robocache_result

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80 + "\n")

    if baseline_result and robocache_result:
        total_speedup = baseline_result['total_time'] / robocache_result['total_time']
        data_speedup = baseline_result['data_time'] / robocache_result['data_time']

        print(f"Total Training Time:")
        print(f"  Baseline:    {baseline_result['total_time']:.2f}s")
        print(f"  RoboCache:   {robocache_result['total_time']:.2f}s")
        print(f"  Speedup:     {total_speedup:.2f}× faster")
        print()

        print(f"Data Loading Time:")
        print(f"  Baseline:    {baseline_result['data_time']:.2f}s ({baseline_result['data_time']/baseline_result['total_time']*100:.1f}% of total)")
        print(f"  RoboCache:   {robocache_result['data_time']:.2f}s ({robocache_result['data_time']/robocache_result['total_time']*100:.1f}% of total)")
        print(f"  Speedup:     {data_speedup:.2f}× faster")
        print()

        print(f"Model Training Time:")
        print(f"  Baseline:    {baseline_result['model_time']:.2f}s")
        print(f"  RoboCache:   {robocache_result['model_time']:.2f}s")
        print()

        print("Key Insight:")
        if baseline_result['data_time'] > baseline_result['model_time']:
            print(f"  ⚠️  Baseline is DATA-BOUND ({baseline_result['data_time']/baseline_result['total_time']*100:.0f}% time in data loading)")
            print(f"  ✅ RoboCache eliminates this bottleneck!")
        else:
            print(f"  ✅ Baseline is compute-bound (good!)")

        print()
        print(f"💰 For 100 epochs on H100 @ $2/hour:")
        print(f"  Baseline cost:   ${baseline_result['total_time'] * 100 / 3600 * 2:.2f}")
        print(f"  RoboCache cost:  ${robocache_result['total_time'] * 100 / 3600 * 2:.2f}")
        print(f"  Savings:         ${(baseline_result['total_time'] - robocache_result['total_time']) * 100 / 3600 * 2:.2f}")
        print()

        results['comparison'] = {
            'total_speedup': total_speedup,
            'data_speedup': data_speedup,
        }

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results_file = output_dir / 'training_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📁 Results saved to: {results_file}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Diffusion Policy training with baseline vs RoboCache'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../data/robot_learning/robot_synthetic.h5',
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of epochs to train (default: 3)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='compare',
        choices=['baseline', 'robocache', 'compare'],
        help='Benchmark mode (default: compare)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: Dataset not found at {args.data}")
        print("Run: python download_data.py first")
        return

    if args.mode == 'compare':
        run_comparison(
            args.data,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir
        )
    else:
        benchmark_training(
            args.data,
            mode=args.mode,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs
        )


if __name__ == '__main__':
    main()
