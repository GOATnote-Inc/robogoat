#!/usr/bin/env python3
"""
End-to-end training demonstration with GPU utilization tracking.

Demonstrates:
- Real training loop with RoboCache preprocessing
- GPU utilization monitoring (nvidia-smi or DCGM)
- Dataloader throughput measurement
- Before/after comparison (CPU vs GPU preprocessing)
"""

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainingMetrics:
    """Training metrics for comparison."""
    implementation: str  # "robocache" or "pytorch_cpu"
    steps_per_sec: float
    avg_step_time_ms: float
    preprocessing_time_ms: float
    model_time_ms: float
    gpu_utilization_pct: float
    dataloader_throughput_mbps: float
    total_time_sec: float


class GPUUtilizationMonitor:
    """
    Monitor GPU utilization using nvidia-smi.
    
    Runs in background thread and samples utilization every 100ms.
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.samples: List[float] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> float:
        """Stop monitoring and return average utilization."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return sum(self.samples) / len(self.samples) if self.samples else 0.0
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Query GPU utilization
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_id}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if result.returncode == 0:
                    util = float(result.stdout.strip())
                    self.samples.append(util)
            except (subprocess.TimeoutExpired, ValueError):
                pass
            time.sleep(0.1)  # Sample every 100ms


class SyntheticRobotDataset(Dataset):
    """Synthetic robot trajectory dataset."""
    
    def __init__(self, n_episodes: int = 1000, seq_len: int = 500, dim: int = 256):
        self.n_episodes = n_episodes
        self.seq_len = seq_len
        self.dim = dim
    
    def __len__(self):
        return self.n_episodes
    
    def __getitem__(self, idx):
        """Generate synthetic trajectory data."""
        # Variable-length source trajectory
        actual_len = torch.randint(self.seq_len // 2, self.seq_len, (1,)).item()
        data = torch.randn(actual_len, self.dim, dtype=torch.bfloat16)
        times = torch.linspace(0, 5, actual_len)
        
        # Target action
        action = torch.randn(self.seq_len // 2, 7, dtype=torch.bfloat16)  # 7-DOF robot
        
        return data, times, action


class SimpleTransformerPolicy(nn.Module):
    """Simple transformer policy for demonstration."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, n_actions: int = 7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, x):
        """Forward pass: x shape (B, T, D)."""
        x = self.input_proj(x.float())
        x = self.transformer(x)
        return self.action_head(x)


def resample_pytorch_cpu(data, src_times, tgt_times):
    """PyTorch CPU fallback for resampling."""
    B, S, D = data.shape
    T = tgt_times.shape[1]
    result = torch.zeros(B, T, D, dtype=data.dtype, device=data.device)
    
    for b in range(B):
        for t in range(T):
            tgt_t = tgt_times[b, t].item()
            idx = torch.searchsorted(src_times[b], tgt_t)
            
            if idx == 0:
                result[b, t] = data[b, 0]
            elif idx >= S:
                result[b, t] = data[b, -1]
            else:
                t0 = src_times[b, idx - 1].item()
                t1 = src_times[b, idx].item()
                alpha = (tgt_t - t0) / (t1 - t0 + 1e-9)
                result[b, t] = (1 - alpha) * data[b, idx - 1] + alpha * data[b, idx]
    
    return result


def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    # Pad sequences to max length in batch
    max_len = max(data.shape[0] for data, _, _ in batch)
    
    padded_data = []
    times = []
    actions = []
    
    for data, time, action in batch:
        # Pad data
        if data.shape[0] < max_len:
            pad = torch.zeros(max_len - data.shape[0], data.shape[1], dtype=data.dtype)
            data = torch.cat([data, pad], dim=0)
        padded_data.append(data)
        times.append(time)
        actions.append(action)
    
    return torch.stack(padded_data), times, torch.stack(actions)


def run_training(
    implementation: str,
    n_steps: int = 100,
    batch_size: int = 32,
    device: str = "cuda"
) -> TrainingMetrics:
    """
    Run training loop with specified preprocessing implementation.
    
    Args:
        implementation: "robocache" or "pytorch_cpu"
        n_steps: Number of training steps
        batch_size: Batch size
        device: Device for model ("cuda" or "cpu")
    
    Returns:
        TrainingMetrics with performance data
    """
    print(f"\n{'='*80}")
    print(f"Training with {implementation.upper()} preprocessing")
    print(f"{'='*80}\n")
    
    # Setup
    if implementation == "robocache":
        try:
            import robocache
        except ImportError:
            print("ERROR: robocache not installed")
            return None
    
    dataset = SyntheticRobotDataset(n_episodes=n_steps * batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single process for fair comparison
        collate_fn=collate_fn
    )
    
    model = SimpleTransformerPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Start GPU utilization monitoring
    gpu_monitor = GPUUtilizationMonitor(device_id=0)
    gpu_monitor.start()
    
    # Training loop
    step_times = []
    preprocess_times = []
    model_times = []
    
    start_time = time.time()
    
    for step, (data_batch, times_batch, actions) in enumerate(dataloader):
        if step >= n_steps:
            break
        
        step_start = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        
        target_len = actions.shape[1]
        target_times = torch.linspace(0, 5, target_len).unsqueeze(0).expand(batch_size, -1)
        
        if implementation == "robocache":
            # GPU preprocessing
            data_batch_gpu = data_batch.to(device)
            # Note: times_batch is list of tensors with different lengths
            # For demo, use max length
            src_times = torch.linspace(0, 5, data_batch.shape[1]).unsqueeze(0).expand(batch_size, -1).to(device)
            tgt_times = target_times.to(device)
            
            resampled = robocache.resample_trajectories(data_batch_gpu, src_times, tgt_times)
        else:
            # CPU preprocessing
            src_times_cpu = torch.linspace(0, 5, data_batch.shape[1]).unsqueeze(0).expand(batch_size, -1)
            resampled = resample_pytorch_cpu(data_batch, src_times_cpu, target_times)
            resampled = resampled.to(device)
        
        torch.cuda.synchronize()
        preprocess_time = (time.time() - preprocess_start) * 1000
        preprocess_times.append(preprocess_time)
        
        # Model forward + backward
        model_start = time.time()
        
        actions = actions.to(device)
        predictions = model(resampled)
        loss = criterion(predictions, actions.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        model_time = (time.time() - model_start) * 1000
        model_times.append(model_time)
        
        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{n_steps}: {step_time:.2f}ms "
                  f"(preprocess: {preprocess_time:.2f}ms, model: {model_time:.2f}ms)")
    
    total_time = time.time() - start_time
    
    # Stop monitoring
    avg_gpu_util = gpu_monitor.stop()
    
    # Compute metrics
    avg_step_time = sum(step_times) / len(step_times)
    avg_preprocess_time = sum(preprocess_times) / len(preprocess_times)
    avg_model_time = sum(model_times) / len(model_times)
    steps_per_sec = n_steps / total_time
    
    # Estimate dataloader throughput (MB/s)
    # Assume ~4 bytes/element (BF16 = 2 bytes, but accounting for overhead)
    bytes_per_batch = batch_size * 500 * 256 * 2  # source data size
    dataloader_throughput = (bytes_per_batch * n_steps / total_time) / (1024 * 1024)
    
    metrics = TrainingMetrics(
        implementation=implementation,
        steps_per_sec=steps_per_sec,
        avg_step_time_ms=avg_step_time,
        preprocessing_time_ms=avg_preprocess_time,
        model_time_ms=avg_model_time,
        gpu_utilization_pct=avg_gpu_util,
        dataloader_throughput_mbps=dataloader_throughput,
        total_time_sec=total_time
    )
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {implementation.upper()}")
    print(f"{'='*80}")
    print(f"Steps/sec: {metrics.steps_per_sec:.2f}")
    print(f"Avg step time: {metrics.avg_step_time_ms:.2f}ms")
    print(f"  - Preprocessing: {metrics.preprocessing_time_ms:.2f}ms")
    print(f"  - Model: {metrics.model_time_ms:.2f}ms")
    print(f"GPU utilization: {metrics.gpu_utilization_pct:.1f}%")
    print(f"Dataloader throughput: {metrics.dataloader_throughput_mbps:.1f} MB/s")
    print(f"Total time: {metrics.total_time_sec:.2f}s")
    print(f"{'='*80}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RoboCache Training Demo")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=Path, default=Path("bench/results/training_comparison.json"))
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    results = {}
    
    # Run with PyTorch CPU preprocessing
    print("\n🔵 Running with PyTorch CPU preprocessing...")
    cpu_metrics = run_training("pytorch_cpu", args.steps, args.batch_size, args.device)
    results["pytorch_cpu"] = asdict(cpu_metrics)
    
    # Run with RoboCache GPU preprocessing
    print("\n🟢 Running with RoboCache GPU preprocessing...")
    robocache_metrics = run_training("robocache", args.steps, args.batch_size, args.device)
    results["robocache"] = asdict(robocache_metrics)
    
    # Comparison
    speedup = cpu_metrics.steps_per_sec / robocache_metrics.steps_per_sec \
              if robocache_metrics.steps_per_sec > 0 else float('inf')
    speedup = 1.0 / speedup  # RoboCache faster than CPU
    
    gpu_util_improvement = robocache_metrics.gpu_utilization_pct - cpu_metrics.gpu_utilization_pct
    
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"RoboCache Speedup: {speedup:.2f}× faster")
    print(f"GPU Utilization: {cpu_metrics.gpu_utilization_pct:.1f}% → "
          f"{robocache_metrics.gpu_utilization_pct:.1f}% ({gpu_util_improvement:+.1f}%)")
    print(f"Preprocessing Time: {cpu_metrics.preprocessing_time_ms:.2f}ms → "
          f"{robocache_metrics.preprocessing_time_ms:.2f}ms "
          f"({(1 - robocache_metrics.preprocessing_time_ms/cpu_metrics.preprocessing_time_ms)*100:.1f}% faster)")
    print("="*80 + "\n")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "results": results,
            "comparison": {
                "speedup": speedup,
                "gpu_util_improvement_pct": gpu_util_improvement,
                "preprocessing_speedup": cpu_metrics.preprocessing_time_ms / robocache_metrics.preprocessing_time_ms
            }
        }, f, indent=2)
    
    print(f"✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()

