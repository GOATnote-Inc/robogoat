#!/usr/bin/env python3
"""
Real-World RT-X Benchmark: RoboCache vs PyTorch Baseline

Measures:
- Wall-clock training time
- GPU utilization
- Throughput (episodes/sec)
- Memory usage

Goal: Prove RoboCache delivers 5-10x speedup on real robot training
"""

import torch
import torch.nn as nn
import time
import subprocess
from pathlib import Path
import json
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TransformerPolicy(nn.Module):
    """Realistic policy network for robot control"""
    
    def __init__(self, vision_dim=512, state_dim=8, action_dim=7, hidden_dim=256, num_layers=4):
        super().__init__()
        
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, vision, state):
        """
        Args:
            vision: [B, T, vision_dim]
            state: [B, T, state_dim]
        Returns:
            actions: [B, T, action_dim]
        """
        v_enc = self.vision_encoder(vision)  # [B, T, hidden]
        s_enc = self.state_encoder(state)     # [B, T, hidden]
        
        fused = v_enc + s_enc  # [B, T, hidden]
        
        out = self.transformer(fused)  # [B, T, hidden]
        actions = self.action_head(out)  # [B, T, action_dim]
        
        return actions


class RTXSyntheticDataset:
    """
    Realistic synthetic RT-X dataset
    
    Simulates:
    - Variable frequency sensors (30Hz vision, 100Hz state, 50Hz actions)
    - Temporal coherence (episode structure)
    - Realistic data sizes
    """
    
    def __init__(self, num_episodes=1000, episode_length_sec=5.0, device='cuda'):
        self.num_episodes = num_episodes
        self.device = device
        
        # Frequencies
        self.vision_hz = 30
        self.state_hz = 100
        self.action_hz = 50
        
        # Dimensions
        self.vision_dim = 512
        self.state_dim = 8
        self.action_dim = 7
        
        # Episode structure
        self.episode_length_sec = episode_length_sec
        self.vision_len = int(self.vision_hz * episode_length_sec)
        self.state_len = int(self.state_hz * episode_length_sec)
        self.action_len = int(self.action_hz * episode_length_sec)
        
        # Target (for model training)
        self.target_hz = 50
        self.target_len = int(self.target_hz * episode_length_sec)
        
        print(f"Dataset: {num_episodes} episodes")
        print(f"  Vision: {self.vision_len} samples @ {self.vision_hz}Hz")
        print(f"  State:  {self.state_len} samples @ {self.state_hz}Hz")
        print(f"  Action: {self.action_len} samples @ {self.action_hz}Hz")
        print(f"  Target: {self.target_len} samples @ {self.target_hz}Hz")
        
        # Pre-generate data on GPU
        self._generate_data()
    
    def _generate_data(self):
        """Pre-generate all episodes on GPU"""
        print("Generating synthetic episodes...")
        
        self.vision_data = torch.randn(
            self.num_episodes, self.vision_len, self.vision_dim,
            device=self.device, dtype=torch.float32
        )
        self.vision_times = torch.linspace(
            0, self.episode_length_sec, self.vision_len,
            device=self.device
        ).unsqueeze(0).expand(self.num_episodes, -1)
        
        self.state_data = torch.randn(
            self.num_episodes, self.state_len, self.state_dim,
            device=self.device, dtype=torch.float32
        )
        self.state_times = torch.linspace(
            0, self.episode_length_sec, self.state_len,
            device=self.device
        ).unsqueeze(0).expand(self.num_episodes, -1)
        
        self.action_data = torch.randn(
            self.num_episodes, self.action_len, self.action_dim,
            device=self.device, dtype=torch.float32
        )
        self.action_times = torch.linspace(
            0, self.episode_length_sec, self.action_len,
            device=self.device
        ).unsqueeze(0).expand(self.num_episodes, -1)
        
        self.target_times = torch.linspace(
            0, self.episode_length_sec, self.target_len,
            device=self.device
        ).unsqueeze(0)
        
        print("✓ Data generated on GPU")
    
    def get_batch_robocache(self, batch_indices):
        """Get batch using RoboCache CUDA kernels"""
        import robocache
        
        # Slice batch
        vision = self.vision_data[batch_indices]  # [B, T_v, D_v]
        vision_times = self.vision_times[batch_indices]  # [B, T_v]
        state = self.state_data[batch_indices]  # [B, T_s, D_s]
        state_times = self.state_times[batch_indices]  # [B, T_s]
        actions = self.action_data[batch_indices]  # [B, T_a, D_a]
        
        target_times = self.target_times.expand(len(batch_indices), -1)  # [B, T]
        
        # RoboCache preprocessing (GPU-accelerated)
        vision_aligned = robocache.resample_trajectories(
            vision, vision_times, target_times
        )  # [B, T, D_v]
        
        state_aligned = robocache.resample_trajectories(
            state, state_times, target_times
        )  # [B, T, D_s]
        
        return vision_aligned, state_aligned, actions
    
    def get_batch_pytorch(self, batch_indices):
        """Get batch using PyTorch (CPU-like preprocessing)"""
        # Slice batch
        vision = self.vision_data[batch_indices]
        vision_times = self.vision_times[batch_indices]
        state = self.state_data[batch_indices]
        state_times = self.state_times[batch_indices]
        actions = self.action_data[batch_indices]
        
        target_times = self.target_times.expand(len(batch_indices), -1)
        
        # PyTorch interpolation (slower)
        vision_aligned = self._interpolate_pytorch(vision, vision_times, target_times)
        state_aligned = self._interpolate_pytorch(state, state_times, target_times)
        
        return vision_aligned, state_aligned, actions
    
    def _interpolate_pytorch(self, data, source_times, target_times):
        """PyTorch baseline interpolation (simulates CPU dataloader)"""
        B, S, D = data.shape
        T = target_times.shape[1]
        
        output = torch.zeros(B, T, D, device=data.device, dtype=data.dtype)
        
        for b in range(B):
            for t in range(T):
                target_t = target_times[b, t]
                
                # Binary search
                left = torch.searchsorted(source_times[b], target_t, right=False)
                right = min(left + 1, S - 1)
                left = min(left, S - 1)
                
                if left == right:
                    output[b, t] = data[b, left]
                else:
                    alpha = (target_t - source_times[b, left]) / (
                        source_times[b, right] - source_times[b, left] + 1e-8
                    )
                    output[b, t] = (1 - alpha) * data[b, left] + alpha * data[b, right]
        
        return output


def measure_gpu_utilization():
    """Measure current GPU utilization"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def benchmark_training(
    use_robocache: bool,
    num_episodes: int = 1000,
    batch_size: int = 32,
    num_steps: int = 100,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark training with RoboCache or PyTorch baseline
    
    Returns:
        metrics: Dict with wall-clock time, throughput, GPU util
    """
    print("\n" + "=" * 70)
    print(f"{'RoboCache' if use_robocache else 'PyTorch Baseline'} Benchmark")
    print("=" * 70)
    
    # Setup
    dataset = RTXSyntheticDataset(num_episodes=num_episodes, device=device)
    model = TransformerPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        indices = torch.randint(0, num_episodes, (batch_size,))
        if use_robocache:
            vision, state, actions = dataset.get_batch_robocache(indices)
        else:
            vision, state, actions = dataset.get_batch_pytorch(indices)
        
        pred_actions = model(vision, state)
        loss = torch.nn.functional.mse_loss(pred_actions, actions[:, :pred_actions.shape[1], :])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    print("✓ Warmup complete")
    
    # Benchmark
    print(f"Running {num_steps} training steps...")
    
    gpu_utils = []
    step_times = []
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Sample batch
        indices = torch.randint(0, num_episodes, (batch_size,))
        
        # Preprocessing
        if use_robocache:
            vision, state, actions = dataset.get_batch_robocache(indices)
        else:
            vision, state, actions = dataset.get_batch_pytorch(indices)
        
        # Forward
        pred_actions = model(vision, state)
        
        # Loss
        target_actions = actions[:, :pred_actions.shape[1], :]
        loss = torch.nn.functional.mse_loss(pred_actions, target_actions)
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Measure GPU util
        if step % 10 == 0:
            gpu_util = measure_gpu_utilization()
            gpu_utils.append(gpu_util)
            
            if step % 50 == 0:
                avg_time = sum(step_times[-10:]) / len(step_times[-10:])
                print(f"Step {step}/{num_steps}: {avg_time*1000:.2f}ms/step, "
                      f"GPU: {gpu_util:.1f}%, Loss: {loss.item():.4f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Compute metrics
    avg_step_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_step_time  # episodes/sec
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
    
    metrics = {
        'backend': 'RoboCache' if use_robocache else 'PyTorch',
        'total_time': total_time,
        'avg_step_time_ms': avg_step_time * 1000,
        'throughput_eps_sec': throughput,
        'avg_gpu_util': avg_gpu_util,
        'batch_size': batch_size,
        'num_steps': num_steps,
    }
    
    print("\n" + "-" * 70)
    print("Results:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Avg step time:  {avg_step_time*1000:.2f}ms")
    print(f"  Throughput:     {throughput:.1f} episodes/sec")
    print(f"  Avg GPU util:   {avg_gpu_util:.1f}%")
    print("-" * 70)
    
    return metrics


def main():
    """Run full comparison benchmark"""
    print("\n" + "=" * 70)
    print("Real-World RT-X Benchmark: RoboCache vs PyTorch")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check RoboCache
    try:
        import robocache
        print(f"RoboCache: Available")
    except ImportError:
        print("ERROR: RoboCache not available")
        return
    
    # Configuration
    num_episodes = 1000
    batch_size = 32
    num_steps = 100
    
    # Run PyTorch baseline
    print("\n" + "=" * 70)
    print("BASELINE: PyTorch (CPU-style preprocessing)")
    print("=" * 70)
    pytorch_metrics = benchmark_training(
        use_robocache=False,
        num_episodes=num_episodes,
        batch_size=batch_size,
        num_steps=num_steps
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # Run RoboCache
    print("\n" + "=" * 70)
    print("OPTIMIZED: RoboCache (GPU-accelerated preprocessing)")
    print("=" * 70)
    robocache_metrics = benchmark_training(
        use_robocache=True,
        num_episodes=num_episodes,
        batch_size=batch_size,
        num_steps=num_steps
    )
    
    # Comparison
    speedup = pytorch_metrics['total_time'] / robocache_metrics['total_time']
    throughput_gain = robocache_metrics['throughput_eps_sec'] / pytorch_metrics['throughput_eps_sec']
    
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\nMetric                    PyTorch    RoboCache    Improvement")
    print("-" * 70)
    print(f"Total time:              {pytorch_metrics['total_time']:7.2f}s   {robocache_metrics['total_time']:7.2f}s      {speedup:.2f}x faster")
    print(f"Avg step time:           {pytorch_metrics['avg_step_time_ms']:7.2f}ms  {robocache_metrics['avg_step_time_ms']:7.2f}ms     {pytorch_metrics['avg_step_time_ms']/robocache_metrics['avg_step_time_ms']:.2f}x faster")
    print(f"Throughput:              {pytorch_metrics['throughput_eps_sec']:7.1f}    {robocache_metrics['throughput_eps_sec']:7.1f}      {throughput_gain:.2f}x higher")
    print(f"GPU utilization:         {pytorch_metrics['avg_gpu_util']:6.1f}%    {robocache_metrics['avg_gpu_util']:6.1f}%     {robocache_metrics['avg_gpu_util'] - pytorch_metrics['avg_gpu_util']:+.1f}%")
    print("=" * 70)
    
    # Save results
    results = {
        'pytorch': pytorch_metrics,
        'robocache': robocache_metrics,
        'speedup': speedup,
        'throughput_gain': throughput_gain,
    }
    
    output_file = Path(__file__).parent / 'rtx_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Pass/Fail criteria
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    target_speedup = 5.0
    target_gpu_util = 90.0
    
    passed = True
    
    if speedup >= target_speedup:
        print(f"✓ Speedup: {speedup:.2f}x (target: {target_speedup:.1f}x)")
    else:
        print(f"✗ Speedup: {speedup:.2f}x (target: {target_speedup:.1f}x)")
        passed = False
    
    if robocache_metrics['avg_gpu_util'] >= target_gpu_util:
        print(f"✓ GPU util: {robocache_metrics['avg_gpu_util']:.1f}% (target: {target_gpu_util:.0f}%)")
    else:
        print(f"✗ GPU util: {robocache_metrics['avg_gpu_util']:.1f}% (target: {target_gpu_util:.0f}%)")
        passed = False
    
    if passed:
        print("\n✓✓✓ BENCHMARK PASSED ✓✓✓")
    else:
        print("\n✗✗✗ BENCHMARK FAILED ✗✗✗")
    
    print("=" * 70)
    
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

