#!/usr/bin/env python3
"""
Train robot policy on RT-X datasets with RoboCache acceleration

Usage:
    # Train on Bridge V2
    python train_with_rtx.py --dataset bridge_dataset --epochs 10
    
    # Train on RT-1
    python train_with_rtx.py --dataset rt_1 --epochs 10
    
    # Use local TFRecords
    python train_with_rtx.py --dataset bridge_dataset --data-dir ./data/bridge_v2
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

try:
    import robocache
    from robocache.datasets import RTXDataLoader
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False
    print("⚠️  RoboCache not installed - install with: pip install robocache")
    exit(1)


class RobotPolicy(nn.Module):
    """Simple vision-language-conditioned policy"""
    
    def __init__(self, action_dim: int = 7):
        super().__init__()
        
        # Vision encoder (CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(14, 64),  # Assuming 14D proprio (7 joints + 7 velocities)
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 16 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Args:
            observations: Dict with 'rgb' (B, T, H, W, C) and 'proprio' (B, T, D)
        
        Returns:
            actions: (B, T, action_dim)
        """
        batch_size, seq_len = observations['rgb'].shape[:2]
        
        # Reshape for CNN: (B, T, H, W, C) -> (B*T, C, H, W)
        rgb = observations['rgb'].view(-1, *observations['rgb'].shape[2:])
        rgb = rgb.permute(0, 3, 1, 2)  # (B*T, C, H, W)
        
        # Vision features
        vision_feat = self.vision_encoder(rgb)  # (B*T, 2048)
        vision_feat = vision_feat.view(batch_size, seq_len, -1)  # (B, T, 2048)
        
        # Proprioception features
        proprio_feat = self.proprio_encoder(observations['proprio'])  # (B, T, 64)
        
        # Concatenate and predict actions
        combined = torch.cat([vision_feat, proprio_feat], dim=-1)  # (B, T, 2112)
        actions = self.policy_head(combined)  # (B, T, action_dim)
        
        return actions


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to GPU
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        actions_gt = batch['actions'].to(device)
        
        # Forward pass
        actions_pred = model(observations)
        
        # Loss (behavior cloning)
        loss = criterion(actions_pred, actions_gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train robot policy on RT-X datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='RT-X dataset name (e.g., bridge_dataset, rt_1, droid)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Local directory with TFRecords (if None, downloads from GCS)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Limit number of episodes (for debugging)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training on RT-X Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print("Loading RT-X dataset...")
    train_loader = RTXDataLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        split='train',
        max_episodes=args.max_episodes,
        sequence_length=50,
    )
    
    # Model
    model = RobotPolicy(action_dim=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("✅ Training Complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

