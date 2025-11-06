"""
RT-X Dataset Loader with RoboCache Acceleration

Implements efficient loading and preprocessing for RT-X datasets using
RoboCache CUDA kernels for trajectory resampling and multimodal fusion.

Performance Target: 95%+ GPU utilization during training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np


class RTXEpisode:
    """Single RT-X episode with multimodal data"""
    
    def __init__(
        self,
        episode_id: str,
        vision: torch.Tensor,  # [T_v, D_v] vision features
        vision_times: torch.Tensor,  # [T_v] timestamps
        actions: torch.Tensor,  # [T_a, D_a] action trajectory
        action_times: torch.Tensor,  # [T_a] timestamps
        proprio: Optional[torch.Tensor] = None,  # [T_p, D_p] proprioception
        proprio_times: Optional[torch.Tensor] = None,  # [T_p]
    ):
        self.episode_id = episode_id
        self.vision = vision
        self.vision_times = vision_times
        self.actions = actions
        self.action_times = action_times
        self.proprio = proprio
        self.proprio_times = proprio_times
    
    def __len__(self):
        return len(self.action_times)


class RTXDataset(Dataset):
    """
    RT-X Dataset for robot foundation model training
    
    Features:
    - Variable-frequency sensor data (30Hz vision, 100Hz actions)
    - Multimodal inputs (vision + actions + optional proprio)
    - Episode-based structure
    - RoboCache preprocessing
    
    Args:
        data_dir: Path to RT-X data
        target_hz: Target resampling frequency (default: 50Hz)
        episode_length_sec: Episode length in seconds (default: 5.0)
        use_robocache: Use CUDA kernels if available (default: True)
    """
    
    def __init__(
        self,
        data_dir: str,
        target_hz: float = 50.0,
        episode_length_sec: float = 5.0,
        use_robocache: bool = True,
        device: str = 'cuda',
    ):
        self.data_dir = Path(data_dir)
        self.target_hz = target_hz
        self.episode_length_sec = episode_length_sec
        self.use_robocache = use_robocache
        self.device = device
        
        self.target_length = int(target_hz * episode_length_sec)
        
        # Load episodes (simulated for now - replace with actual RT-X loading)
        self.episodes = self._load_episodes()
    
    def _load_episodes(self) -> List[RTXEpisode]:
        """
        Load RT-X episodes from disk
        
        TODO: Replace with actual RT-X data loading
        For now, generates synthetic episodes matching RT-X structure
        """
        episodes = []
        num_episodes = 100  # Simulate 100 episodes
        
        for i in range(num_episodes):
            # Simulate RT-X data structure
            # Vision: 30 Hz RGB-D features
            vision_hz = 30
            vision_length = int(vision_hz * self.episode_length_sec)
            vision = torch.randn(vision_length, 512)  # ResNet features
            vision_times = torch.linspace(0, self.episode_length_sec, vision_length)
            
            # Actions: 10 Hz (standard RT-X)
            action_hz = 10
            action_length = int(action_hz * self.episode_length_sec)
            actions = torch.randn(action_length, 7)  # 7-DOF robot
            action_times = torch.linspace(0, self.episode_length_sec, action_length)
            
            # Proprio: 100 Hz joint states
            proprio_hz = 100
            proprio_length = int(proprio_hz * self.episode_length_sec)
            proprio = torch.randn(proprio_length, 14)  # 7 pos + 7 vel
            proprio_times = torch.linspace(0, self.episode_length_sec, proprio_length)
            
            episode = RTXEpisode(
                episode_id=f"episode_{i:04d}",
                vision=vision,
                vision_times=vision_times,
                actions=actions,
                action_times=action_times,
                proprio=proprio,
                proprio_times=proprio_times,
            )
            episodes.append(episode)
        
        return episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get preprocessed episode data
        
        Returns:
            dict with keys:
            - 'vision': [T, D_v] resampled vision features
            - 'proprio': [T, D_p] resampled proprioception
            - 'actions': [T, D_a] resampled actions (targets)
            - 'fused': [T, D_v + D_p] fused features (for transformer)
        """
        episode = self.episodes[idx]
        
        # Target timestamps (uniform sampling at target_hz)
        target_times = torch.linspace(
            0, self.episode_length_sec, self.target_length
        ).unsqueeze(0)  # [1, T]
        
        # Move to device and add batch dimension
        vision = episode.vision.unsqueeze(0).to(self.device)  # [1, T_v, D_v]
        vision_times = episode.vision_times.unsqueeze(0).to(self.device)  # [1, T_v]
        
        actions = episode.actions.unsqueeze(0).to(self.device)  # [1, T_a, D_a]
        action_times = episode.action_times.unsqueeze(0).to(self.device)  # [1, T_a]
        
        target_times = target_times.to(self.device)  # [1, T]
        
        if self.use_robocache:
            # Use RoboCache CUDA kernels
            try:
                import robocache
                
                # Resample vision to target frequency
                vision_resampled = robocache.resample_trajectories(
                    vision, vision_times, target_times
                )  # [1, T, D_v]
                
                # Resample actions to target frequency
                actions_resampled = robocache.resample_trajectories(
                    actions, action_times, target_times
                )  # [1, T, D_a]
                
                # Resample proprio if available
                if episode.proprio is not None:
                    proprio = episode.proprio.unsqueeze(0).to(self.device)
                    proprio_times = episode.proprio_times.unsqueeze(0).to(self.device)
                    proprio_resampled = robocache.resample_trajectories(
                        proprio, proprio_times, target_times
                    )  # [1, T, D_p]
                    
                    # Fuse vision + proprio
                    fused = torch.cat([vision_resampled, proprio_resampled], dim=2)
                else:
                    fused = vision_resampled
                    proprio_resampled = None
                
            except Exception as e:
                print(f"RoboCache failed, falling back to PyTorch: {e}")
                self.use_robocache = False
                return self.__getitem__(idx)  # Retry with PyTorch
        else:
            # PyTorch fallback (slower)
            vision_resampled = self._resample_pytorch(vision, vision_times, target_times)
            actions_resampled = self._resample_pytorch(actions, action_times, target_times)
            
            if episode.proprio is not None:
                proprio = episode.proprio.unsqueeze(0).to(self.device)
                proprio_times = episode.proprio_times.unsqueeze(0).to(self.device)
                proprio_resampled = self._resample_pytorch(proprio, proprio_times, target_times)
                fused = torch.cat([vision_resampled, proprio_resampled], dim=2)
            else:
                fused = vision_resampled
                proprio_resampled = None
        
        # Remove batch dimension for return
        return {
            'vision': vision_resampled.squeeze(0),  # [T, D_v]
            'proprio': proprio_resampled.squeeze(0) if proprio_resampled is not None else None,
            'actions': actions_resampled.squeeze(0),  # [T, D_a]
            'fused': fused.squeeze(0),  # [T, D_v + D_p]
            'episode_id': episode.episode_id,
        }
    
    def _resample_pytorch(
        self,
        data: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback for trajectory resampling"""
        B, S, D = data.shape
        T = target_times.shape[1]
        
        output = torch.zeros(B, T, D, dtype=data.dtype, device=data.device)
        
        for b in range(B):
            for t in range(T):
                tgt = target_times[b, t]
                
                if tgt <= source_times[b, 0]:
                    output[b, t] = data[b, 0]
                elif tgt >= source_times[b, -1]:
                    output[b, t] = data[b, -1]
                else:
                    # Binary search
                    left, right = 0, S - 1
                    while left < right - 1:
                        mid = (left + right) // 2
                        if source_times[b, mid] <= tgt:
                            left = mid
                        else:
                            right = mid
                    
                    # Linear interpolation
                    alpha = (tgt - source_times[b, left]) / (
                        source_times[b, right] - source_times[b, left] + 1e-8
                    )
                    output[b, t] = (1 - alpha) * data[b, left] + alpha * data[b, right]
        
        return output


def create_rtx_dataloader(
    data_dir: str,
    batch_size: int = 32,
    target_hz: float = 50.0,
    episode_length_sec: float = 5.0,
    num_workers: int = 0,
    use_robocache: bool = True,
    device: str = 'cuda',
) -> DataLoader:
    """
    Create RT-X DataLoader with RoboCache preprocessing
    
    Args:
        data_dir: Path to RT-X data
        batch_size: Batch size
        target_hz: Target resampling frequency
        episode_length_sec: Episode length in seconds
        num_workers: Number of DataLoader workers (0 for GPU preprocessing)
        use_robocache: Use CUDA kernels if available
        device: Device for preprocessing
    
    Returns:
        DataLoader with RoboCache preprocessing
    """
    dataset = RTXDataset(
        data_dir=data_dir,
        target_hz=target_hz,
        episode_length_sec=episode_length_sec,
        use_robocache=use_robocache,
        device=device,
    )
    
    # For GPU preprocessing, use num_workers=0 (process in main thread)
    # For CPU preprocessing, can use multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    
    return dataloader


if __name__ == '__main__':
    """Test RT-X dataloader"""
    print("=" * 70)
    print("RT-X DataLoader Test")
    print("=" * 70)
    
    # Create dataloader
    dataloader = create_rtx_dataloader(
        data_dir='/fake/path',  # Synthetic data for now
        batch_size=16,
        target_hz=50.0,
        episode_length_sec=5.0,
        use_robocache=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"\nDataset: {len(dataloader.dataset)} episodes")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Batches: {len(dataloader)}")
    
    # Test loading a batch
    print("\nLoading batch...")
    batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Vision: {batch['vision'].shape}")
    print(f"  Actions: {batch['actions'].shape}")
    print(f"  Fused: {batch['fused'].shape}")
    if batch['proprio'] is not None:
        print(f"  Proprio: {batch['proprio'].shape}")
    
    print("\n✅ RT-X DataLoader working")

