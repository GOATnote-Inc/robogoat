"""
Sim-to-Real Transfer: Domain Randomization and Augmentation

Implements domain randomization for bridging the sim-to-real gap in robot learning.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class DomainRandomizer:
    """
    Domain randomization for sim-to-real transfer.
    
    Applies realistic augmentations to synthetic sensor data to improve
    real-world performance.
    
    Args:
        lighting_range: (min, max) brightness multiplier
        blur_kernel_range: (min, max) blur kernel size (odd numbers)
        lidar_dropout_prob: Probability of dropping LiDAR points
        lidar_noise_std: Standard deviation of Gaussian noise
        camera_noise_std: Standard deviation of camera sensor noise
        latency_range_ms: (min, max) sensor latency in milliseconds
    """
    
    def __init__(
        self,
        lighting_range: Tuple[float, float] = (0.7, 1.3),
        blur_kernel_range: Tuple[int, int] = (3, 7),
        lidar_dropout_prob: float = 0.05,
        lidar_noise_std: float = 0.02,
        camera_noise_std: float = 0.05,
        latency_range_ms: Tuple[float, float] = (10.0, 50.0),
        enable_occlusions: bool = True,
    ):
        self.lighting_range = lighting_range
        self.blur_kernel_range = blur_kernel_range
        self.lidar_dropout_prob = lidar_dropout_prob
        self.lidar_noise_std = lidar_noise_std
        self.camera_noise_std = camera_noise_std
        self.latency_range_ms = latency_range_ms
        self.enable_occlusions = enable_occlusions
    
    def augment_vision(self, vision: torch.Tensor) -> torch.Tensor:
        """
        Apply domain randomization to vision features.
        
        Args:
            vision: [B, T, D] or [B, H, W, C] vision data
            
        Returns:
            Augmented vision data
        """
        # Lighting variation
        lighting_factor = random.uniform(*self.lighting_range)
        vision = vision * lighting_factor
        
        # Sensor noise (Gaussian)
        noise = torch.randn_like(vision) * self.camera_noise_std
        vision = vision + noise
        
        # Blur (if spatial dims exist)
        if vision.dim() == 4:  # [B, H, W, C]
            kernel_size = random.choice(range(
                self.blur_kernel_range[0],
                self.blur_kernel_range[1] + 1,
                2  # Only odd numbers
            ))
            # Simple box blur
            vision = F.avg_pool2d(
                vision.permute(0, 3, 1, 2),  # [B, C, H, W]
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ).permute(0, 2, 3, 1)  # Back to [B, H, W, C]
        
        # Occlusions (random patches)
        if self.enable_occlusions:
            if vision.dim() == 4:
                B, H, W, C = vision.shape
                for b in range(B):
                    if random.random() < 0.3:  # 30% chance of occlusion
                        # Random rectangular occlusion
                        h_size = random.randint(H // 10, H // 4)
                        w_size = random.randint(W // 10, W // 4)
                        h_start = random.randint(0, H - h_size)
                        w_start = random.randint(0, W - w_size)
                        vision[b, h_start:h_start+h_size, w_start:w_start+w_size, :] = 0
        
        return vision
    
    def augment_lidar(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply domain randomization to LiDAR point cloud.
        
        Args:
            points: [N, 3] or [B, N, 3] point cloud (x, y, z)
            
        Returns:
            Augmented point cloud
        """
        # Gaussian noise
        noise = torch.randn_like(points) * self.lidar_noise_std
        points = points + noise
        
        # Random dropout (simulates occlusions, reflective surfaces)
        dropout_mask = torch.rand(points.shape[:-1], device=points.device) > self.lidar_dropout_prob
        points = points * dropout_mask.unsqueeze(-1)
        
        # Random outliers (5% of points)
        outlier_mask = torch.rand(points.shape[:-1], device=points.device) < 0.05
        if points.dim() == 2:
            outliers = torch.rand_like(points) * 20.0 - 10.0  # Random points in [-10, 10]
            points[outlier_mask] = outliers[outlier_mask]
        
        return points
    
    def augment_proprioception(
        self,
        proprio: torch.Tensor,
        noise_std: float = 0.01
    ) -> torch.Tensor:
        """
        Apply domain randomization to proprioceptive data (joint states).
        
        Args:
            proprio: [B, T, D] proprioception (positions, velocities)
            noise_std: Standard deviation of joint sensor noise
            
        Returns:
            Augmented proprioception
        """
        # Joint sensor noise
        noise = torch.randn_like(proprio) * noise_std
        proprio = proprio + noise
        
        return proprio
    
    def __call__(
        self,
        vision: Optional[torch.Tensor] = None,
        lidar: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply domain randomization to all modalities.
        
        Args:
            vision: Optional vision data
            lidar: Optional LiDAR data
            proprio: Optional proprioception data
            
        Returns:
            Tuple of augmented (vision, lidar, proprio)
        """
        if vision is not None:
            vision = self.augment_vision(vision)
        
        if lidar is not None:
            lidar = self.augment_lidar(lidar)
        
        if proprio is not None:
            proprio = self.augment_proprioception(proprio)
        
        return vision, lidar, proprio


class LatencySimulator:
    """
    Simulates real-world sensor latency for sim-to-real transfer.
    
    In real robots, sensors have variable latency (10-50ms typical).
    This simulator delays observations to match real hardware.
    
    Args:
        latency_ms: Sensor latency in milliseconds (mean)
        jitter_ms: Latency jitter (±jitter_ms)
        buffer_size: Maximum buffer size
    """
    
    def __init__(
        self,
        latency_ms: float = 30.0,
        jitter_ms: float = 10.0,
        buffer_size: int = 200,
    ):
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_observation(
        self,
        observation: torch.Tensor,
        timestamp_ms: float
    ):
        """Add observation to buffer with timestamp"""
        self.buffer.append((observation, timestamp_ms))
        
        # Maintain buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_delayed_observation(
        self,
        current_time_ms: float
    ) -> Optional[torch.Tensor]:
        """
        Get observation from `latency` ms ago.
        
        Args:
            current_time_ms: Current simulation time
            
        Returns:
            Delayed observation, or None if buffer empty
        """
        if not self.buffer:
            return None
        
        # Random jitter
        actual_latency = self.latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)
        target_time = current_time_ms - actual_latency
        
        # Find closest observation
        closest_idx = 0
        min_diff = abs(self.buffer[0][1] - target_time)
        
        for i, (obs, timestamp) in enumerate(self.buffer):
            diff = abs(timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        return self.buffer[closest_idx][0]
    
    def clear(self):
        """Clear buffer"""
        self.buffer = []


if __name__ == '__main__':
    """Test domain randomization"""
    print("=" * 70)
    print("Domain Randomization Test")
    print("=" * 70)
    
    # Create randomizer
    randomizer = DomainRandomizer()
    
    # Test vision augmentation
    vision = torch.randn(4, 224, 224, 3)  # [B, H, W, C]
    vision_aug = randomizer.augment_vision(vision)
    print(f"\nVision: {vision.shape} → {vision_aug.shape}")
    print(f"  Mean: {vision.mean():.4f} → {vision_aug.mean():.4f}")
    print(f"  Std: {vision.std():.4f} → {vision_aug.std():.4f}")
    
    # Test LiDAR augmentation
    lidar = torch.rand(500000, 3) * 20.0 - 10.0  # [N, 3]
    lidar_aug = randomizer.augment_lidar(lidar)
    non_zero = (lidar_aug.abs().sum(dim=-1) > 0).float().mean()
    print(f"\nLiDAR: {lidar.shape} → {lidar_aug.shape}")
    print(f"  Non-zero points: {non_zero:.2%} (expected ~95% after dropout)")
    
    # Test proprioception augmentation
    proprio = torch.randn(4, 100, 14)  # [B, T, D]
    proprio_aug = randomizer.augment_proprioception(proprio)
    print(f"\nProprioception: {proprio.shape} → {proprio_aug.shape}")
    
    # Test latency simulator
    print(f"\nLatency Simulator:")
    latency_sim = LatencySimulator(latency_ms=30.0, jitter_ms=10.0)
    for t in range(0, 100, 10):
        obs = torch.randn(3, 512)
        latency_sim.add_observation(obs, timestamp_ms=float(t))
    
    delayed_obs = latency_sim.get_delayed_observation(current_time_ms=100.0)
    print(f"  Delayed observation shape: {delayed_obs.shape if delayed_obs is not None else 'None'}")
    print(f"  Buffer size: {len(latency_sim.buffer)}")
    
    print("\n✅ Domain randomization working")

