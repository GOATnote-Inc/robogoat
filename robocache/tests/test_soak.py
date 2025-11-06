#!/usr/bin/env python3
"""
RoboCache 8-Hour Soak Test
Validates memory stability, no leaks, sustained performance
"""
import pytest
import torch
import time
import psutil
import os
from datetime import datetime, timedelta

try:
    import robocache
    CUDA_AVAILABLE = robocache.is_cuda_available() and torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class MemoryMonitor:
    """Monitor CPU and GPU memory usage"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_cpu_mb = self.get_cpu_memory_mb()
        self.initial_gpu_mb = self.get_gpu_memory_mb() if torch.cuda.is_available() else 0
        self.samples = []
    
    def get_cpu_memory_mb(self):
        """Get CPU memory in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_mb(self):
        """Get GPU memory in MB"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / 1024 / 1024
    
    def sample(self):
        """Take a memory sample"""
        cpu_mb = self.get_cpu_memory_mb()
        gpu_mb = self.get_gpu_memory_mb()
        self.samples.append({
            'time': datetime.now(),
            'cpu_mb': cpu_mb,
            'gpu_mb': gpu_mb,
            'cpu_delta': cpu_mb - self.initial_cpu_mb,
            'gpu_delta': gpu_mb - self.initial_gpu_mb,
        })
        return self.samples[-1]
    
    def check_leak(self, max_cpu_growth_mb=100, max_gpu_growth_mb=100):
        """Check for memory leaks"""
        if len(self.samples) < 2:
            return True
        
        latest = self.samples[-1]
        
        # Check CPU leak
        if latest['cpu_delta'] > max_cpu_growth_mb:
            raise AssertionError(
                f"CPU memory leak detected: grew {latest['cpu_delta']:.1f} MB "
                f"(threshold: {max_cpu_growth_mb} MB)"
            )
        
        # Check GPU leak
        if latest['gpu_delta'] > max_gpu_growth_mb:
            raise AssertionError(
                f"GPU memory leak detected: grew {latest['gpu_delta']:.1f} MB "
                f"(threshold: {max_gpu_growth_mb} MB)"
            )
        
        return True
    
    def report(self):
        """Generate memory report"""
        if not self.samples:
            return "No samples collected"
        
        latest = self.samples[-1]
        duration = (latest['time'] - self.samples[0]['time']).total_seconds() / 3600
        
        report = f"\n{'='*60}\n"
        report += f"Memory Soak Test Report\n"
        report += f"{'='*60}\n"
        report += f"Duration: {duration:.2f} hours\n"
        report += f"Samples: {len(self.samples)}\n"
        report += f"\nInitial Memory:\n"
        report += f"  CPU: {self.initial_cpu_mb:.1f} MB\n"
        report += f"  GPU: {self.initial_gpu_mb:.1f} MB\n"
        report += f"\nFinal Memory:\n"
        report += f"  CPU: {latest['cpu_mb']:.1f} MB (Δ{latest['cpu_delta']:+.1f} MB)\n"
        report += f"  GPU: {latest['gpu_mb']:.1f} MB (Δ{latest['gpu_delta']:+.1f} MB)\n"
        report += f"\nMemory Growth Rate:\n"
        report += f"  CPU: {latest['cpu_delta'] / duration:.2f} MB/hour\n"
        report += f"  GPU: {latest['gpu_delta'] / duration:.2f} MB/hour\n"
        report += f"{'='*60}\n"
        
        return report


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
@pytest.mark.slow
def test_1hour_soak():
    """1-hour soak test (quick validation)"""
    run_soak_test(hours=1.0, sample_interval=60)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA kernels not available")
@pytest.mark.slow
def test_8hour_soak():
    """8-hour soak test (full validation)"""
    run_soak_test(hours=8.0, sample_interval=300)


def run_soak_test(hours=1.0, sample_interval=60):
    """
    Run sustained load test
    
    Args:
        hours: Duration in hours
        sample_interval: Memory sampling interval in seconds
    """
    print(f"\n{'='*60}")
    print(f"Starting {hours}-hour soak test")
    print(f"Sample interval: {sample_interval}s")
    print(f"{'='*60}\n")
    
    monitor = MemoryMonitor()
    monitor.sample()
    
    # Setup
    device = torch.device('cuda')
    batch_size = 32
    source_len = 500
    target_len = 256
    dim = 256
    dtype = torch.bfloat16
    
    # Generate data
    source_data = torch.randn(batch_size, source_len, dim, dtype=dtype, device=device)
    source_times = torch.linspace(0, 5, source_len, device=device).unsqueeze(0).expand(batch_size, -1)
    target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Run test
    start_time = time.time()
    end_time = start_time + (hours * 3600)
    last_sample = start_time
    iteration = 0
    latencies = []
    
    while time.time() < end_time:
        # Run operation
        iter_start = time.time()
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        torch.cuda.synchronize()
        iter_time = (time.time() - iter_start) * 1000
        
        latencies.append(iter_time)
        iteration += 1
        
        # Sample memory periodically
        if time.time() - last_sample >= sample_interval:
            sample = monitor.sample()
            monitor.check_leak(max_cpu_growth_mb=100, max_gpu_growth_mb=100)
            
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            
            print(f"[{elapsed/3600:.2f}h] "
                  f"Iteration {iteration:,} | "
                  f"Latency {iter_time:.2f}ms | "
                  f"CPU {sample['cpu_mb']:.0f}MB (Δ{sample['cpu_delta']:+.0f}) | "
                  f"GPU {sample['gpu_mb']:.0f}MB (Δ{sample['gpu_delta']:+.0f}) | "
                  f"Remaining {remaining/3600:.2f}h")
            
            last_sample = time.time()
    
    # Final sample
    monitor.sample()
    
    # Calculate statistics
    import numpy as np
    latencies = np.array(latencies)
    
    print(monitor.report())
    print(f"\nPerformance Statistics:")
    print(f"  Total iterations: {iteration:,}")
    print(f"  Latency mean: {np.mean(latencies):.2f} ms")
    print(f"  Latency std: {np.std(latencies):.2f} ms")
    print(f"  Latency P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"  Latency P99: {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput: {(batch_size * iteration / (hours * 3600)):.0f} samples/sec")
    
    # Validate performance stability
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    cv = std_latency / mean_latency  # Coefficient of variation
    
    print(f"\nStability Check:")
    print(f"  Coefficient of variation: {cv:.3f}")
    
    assert cv < 0.1, f"Latency too variable: CV={cv:.3f} (threshold: 0.1)"
    
    # Final memory check
    monitor.check_leak(max_cpu_growth_mb=100, max_gpu_growth_mb=100)
    
    print(f"\n✅ Soak test PASSED: {hours} hours, {iteration:,} iterations")


if __name__ == "__main__":
    # Quick test for development
    if "--quick" in os.sys.argv:
        run_soak_test(hours=0.01, sample_interval=5)  # 36 seconds
    else:
        pytest.main([__file__, "-v", "--tb=short", "-s", "-k", "1hour"])

