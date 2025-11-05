"""
compare_triton_vs_cuda.py

Objective comparison of Triton vs CUDA for multimodal sensor fusion.

This benchmark provides evidence-based analysis of:
- Performance (latency, throughput, bandwidth)
- Development time (lines of code, complexity)
- Maintainability (readability, debuggability)
- When to use each approach
"""

import torch
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels" / "triton"))

import robocache_cuda
from multimodal_fusion_triton import fused_multimodal_alignment_triton


class BenchmarkConfig:
    def __init__(self, name: str, batch: int, vision_len: int, vision_dim: int,
                 proprio_len: int, proprio_dim: int, force_len: int, force_dim: int,
                 target_len: int):
        self.name = name
        self.batch = batch
        self.vision_len = vision_len
        self.vision_dim = vision_dim
        self.proprio_len = proprio_len
        self.proprio_dim = proprio_dim
        self.force_len = force_len
        self.force_dim = force_dim
        self.target_len = target_len
        
    def data_size_mb(self) -> float:
        """Calculate total data size in MB"""
        elem_size = 2  # BF16
        vision_data = self.batch * self.vision_len * self.vision_dim * elem_size
        vision_times = self.batch * self.vision_len * 4
        proprio_data = self.batch * self.proprio_len * self.proprio_dim * elem_size
        proprio_times = self.batch * self.proprio_len * 4
        force_data = self.batch * self.force_len * self.force_dim * elem_size
        force_times = self.batch * self.force_len * 4
        target_times = self.batch * self.target_len * 4
        total_dim = self.vision_dim + self.proprio_dim + self.force_dim
        output_data = self.batch * self.target_len * total_dim * elem_size
        
        total_bytes = (vision_data + vision_times + proprio_data + proprio_times +
                      force_data + force_times + target_times + output_data)
        return total_bytes / 1024 / 1024


def create_test_data(config: BenchmarkConfig, device='cuda'):
    """Create test data for benchmarking"""
    batch = config.batch
    
    # Vision: 30 Hz
    vision_data = torch.randn(
        batch, config.vision_len, config.vision_dim,
        dtype=torch.bfloat16, device=device
    )
    vision_times = torch.stack([
        torch.arange(config.vision_len, dtype=torch.float32, device=device) / 30.0
        for _ in range(batch)
    ])
    
    # Proprioception: 100 Hz
    proprio_data = torch.randn(
        batch, config.proprio_len, config.proprio_dim,
        dtype=torch.bfloat16, device=device
    )
    proprio_times = torch.stack([
        torch.arange(config.proprio_len, dtype=torch.float32, device=device) / 100.0
        for _ in range(batch)
    ])
    
    # Force: 333 Hz
    force_data = torch.randn(
        batch, config.force_len, config.force_dim,
        dtype=torch.bfloat16, device=device
    )
    force_times = torch.stack([
        torch.arange(config.force_len, dtype=torch.float32, device=device) / 333.0
        for _ in range(batch)
    ])
    
    # Target: 50 Hz
    target_times = torch.stack([
        torch.arange(config.target_len, dtype=torch.float32, device=device) / 50.0
        for _ in range(batch)
    ])
    
    return {
        'vision_data': vision_data,
        'vision_times': vision_times,
        'proprio_data': proprio_data,
        'proprio_times': proprio_times,
        'force_data': force_data,
        'force_times': force_times,
        'target_times': target_times,
    }


def benchmark_cuda(data: Dict, config: BenchmarkConfig, num_iters=1000, warmup=10):
    """Benchmark CUDA implementation"""
    print(f"  Benchmarking CUDA...")
    
    # Warmup
    for _ in range(warmup):
        output = robocache_cuda.fused_multimodal_alignment(
            data['vision_data'], data['vision_times'],
            data['proprio_data'], data['proprio_times'],
            data['force_data'], data['force_times'],
            data['target_times']
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        output = robocache_cuda.fused_multimodal_alignment(
            data['vision_data'], data['vision_times'],
            data['proprio_data'], data['proprio_times'],
            data['force_data'], data['force_times'],
            data['target_times']
        )
    end_event.record()
    torch.cuda.synchronize()
    
    gpu_time_ms = start_event.elapsed_time(end_event) / num_iters
    
    return {
        'latency_ms': gpu_time_ms,
        'output': output,
    }


def benchmark_triton(data: Dict, config: BenchmarkConfig, num_iters=1000, warmup=10):
    """Benchmark Triton implementation"""
    print(f"  Benchmarking Triton...")
    
    # Warmup
    for _ in range(warmup):
        output = fused_multimodal_alignment_triton(
            data['vision_data'], data['vision_times'],
            data['proprio_data'], data['proprio_times'],
            data['force_data'], data['force_times'],
            data['target_times']
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        output = fused_multimodal_alignment_triton(
            data['vision_data'], data['vision_times'],
            data['proprio_data'], data['proprio_times'],
            data['force_data'], data['force_times'],
            data['target_times']
        )
    end_event.record()
    torch.cuda.synchronize()
    
    gpu_time_ms = start_event.elapsed_time(end_event) / num_iters
    
    return {
        'latency_ms': gpu_time_ms,
        'output': output,
    }


def verify_correctness(cuda_output: torch.Tensor, triton_output: torch.Tensor, tol=1e-2):
    """Verify both implementations produce same results"""
    max_diff = (cuda_output.float() - triton_output.float()).abs().max().item()
    mean_diff = (cuda_output.float() - triton_output.float()).abs().mean().item()
    
    passed = max_diff < tol
    return {
        'passed': passed,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
    }


def analyze_code_complexity():
    """Analyze code complexity metrics"""
    cuda_file = Path(__file__).parent.parent / "kernels" / "cutlass" / "multimodal_fusion.cu"
    triton_file = Path(__file__).parent.parent / "kernels" / "triton" / "multimodal_fusion_triton.py"
    
    def count_lines(filepath):
        with open(filepath) as f:
            lines = f.readlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('#')]
        return len(code_lines)
    
    cuda_loc = count_lines(cuda_file)
    triton_loc = count_lines(triton_file)
    
    return {
        'cuda_loc': cuda_loc,
        'triton_loc': triton_loc,
        'ratio': cuda_loc / triton_loc if triton_loc > 0 else 0,
    }


def run_ncu_profile(implementation: str, config: BenchmarkConfig):
    """Run NCU profiling for detailed analysis (if available)"""
    print(f"  Running NCU profiling for {implementation}...")
    
    # Create a simple profiling script
    script_path = f"/tmp/profile_{implementation}.py"
    with open(script_path, 'w') as f:
        f.write(f"""
import torch
import sys
sys.path.insert(0, '{Path(__file__).parent.parent / "python"}')
sys.path.insert(0, '{Path(__file__).parent.parent / "kernels" / "triton"}')

{'import robocache_cuda' if implementation == 'cuda' else 'from multimodal_fusion_triton import fused_multimodal_alignment_triton'}

# Create data
vision_data = torch.randn({config.batch}, {config.vision_len}, {config.vision_dim}, dtype=torch.bfloat16, device='cuda')
vision_times = torch.randn({config.batch}, {config.vision_len}, dtype=torch.float32, device='cuda')
proprio_data = torch.randn({config.batch}, {config.proprio_len}, {config.proprio_dim}, dtype=torch.bfloat16, device='cuda')
proprio_times = torch.randn({config.batch}, {config.proprio_len}, dtype=torch.float32, device='cuda')
force_data = torch.randn({config.batch}, {config.force_len}, {config.force_dim}, dtype=torch.bfloat16, device='cuda')
force_times = torch.randn({config.batch}, {config.force_len}, dtype=torch.float32, device='cuda')
target_times = torch.randn({config.batch}, {config.target_len}, dtype=torch.float32, device='cuda')

# Run once
{'robocache_cuda.fused_multimodal_alignment' if implementation == 'cuda' else 'fused_multimodal_alignment_triton'}(
    vision_data, vision_times, proprio_data, proprio_times,
    force_data, force_times, target_times
)
""")
    
    # Try to run NCU (may not be available)
    try:
        result = subprocess.run(
            ['ncu', '--metrics', 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
             '--print-summary', 'per-kernel', 'python', script_path],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "NCU not available or timed out"


def print_comparison_table(results: Dict):
    """Print comprehensive comparison table"""
    print("\n" + "="*100)
    print("TRITON VS CUDA: COMPREHENSIVE COMPARISON")
    print("="*100)
    
    for config_name, data in results.items():
        print(f"\n{config_name}:")
        print("-" * 100)
        
        cuda = data['cuda']
        triton = data['triton']
        correctness = data['correctness']
        
        # Performance metrics
        cuda_latency = cuda['latency_ms']
        triton_latency = triton['latency_ms']
        speedup = triton_latency / cuda_latency
        
        data_size = data['data_size_mb']
        cuda_bandwidth = (data_size / cuda_latency) * 1000  # GB/s
        triton_bandwidth = (data_size / triton_latency) * 1000  # GB/s
        
        print(f"{'Metric':<40} {'CUDA':<20} {'Triton':<20} {'Winner':<20}")
        print("-" * 100)
        print(f"{'Latency (ms)':<40} {cuda_latency:>18.3f}  {triton_latency:>18.3f}  {'CUDA' if cuda_latency < triton_latency else 'Triton':>18}")
        print(f"{'Bandwidth (GB/s)':<40} {cuda_bandwidth:>18.1f}  {triton_bandwidth:>18.1f}  {'CUDA' if cuda_bandwidth > triton_bandwidth else 'Triton':>18}")
        print(f"{'Speedup (CUDA/Triton)':<40} {'1.00x':>18}  {speedup:>17.2f}x {('CUDA ' + f'{speedup:.2f}x faster') if speedup > 1 else ('Triton ' + f'{1/speedup:.2f}x faster'):>18}")
        
        print(f"\n{'Correctness Check':<40} {'Status':<20}")
        print("-" * 100)
        print(f"{'Max Difference':<40} {correctness['max_diff']:>18.6f}  {'✅ PASS' if correctness['passed'] else '❌ FAIL':>38}")
        print(f"{'Mean Difference':<40} {correctness['mean_diff']:>18.6f}")


def print_development_comparison(code_stats: Dict):
    """Print development and maintainability comparison"""
    print("\n" + "="*100)
    print("DEVELOPMENT & MAINTAINABILITY COMPARISON")
    print("="*100)
    
    print(f"\n{'Metric':<50} {'CUDA':<25} {'Triton':<25}")
    print("-" * 100)
    print(f"{'Lines of Code':<50} {code_stats['cuda_loc']:>23}  {code_stats['triton_loc']:>23}")
    print(f"{'Code Ratio (CUDA/Triton)':<50} {code_stats['ratio']:>22.2f}x {'':<23}")
    print(f"{'Development Time (estimated)':<50} {'~4 hours':>23}  {'~2 hours':>23}")
    print(f"{'Debugging Ease':<50} {'Harder (cuda-gdb)':>23}  {'Easier (Python)':>23}")
    print(f"{'Maintainability':<50} {'Harder':>23}  {'Easier':>23}")
    print(f"{'Auto-tuning':<50} {'Manual':>23}  {'Built-in':>23}")
    print(f"{'Portability':<50} {'NVIDIA only':>23}  {'NVIDIA only':>23}")


def print_recommendations():
    """Print when to use each approach"""
    print("\n" + "="*100)
    print("RECOMMENDATIONS: WHEN TO USE WHAT")
    print("="*100)
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  USE TRITON WHEN:                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

✅ Rapid prototyping (2-3x faster development)
✅ You need auto-tuning (tries multiple configurations automatically)
✅ Dense operations (matmul, conv, etc.) - Triton excels here
✅ Team prefers Python (easier debugging, testing, maintenance)
✅ You're okay with 5-20% performance gap on complex kernels

╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  USE CUDA WHEN:                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

✅ Memory-latency bound workloads (binary search, sparse ops)
✅ Need warp-level primitives (__shfl, __ballot, etc.)
✅ Irregular memory access patterns (not easily expressible in Triton)
✅ Absolute maximum performance required (< 5% gap acceptable)
✅ Complex synchronization patterns (cooperative groups)
✅ You have CUDA expertise on team

╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  HYBRID APPROACH (RECOMMENDED):                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

🎯 Prototype in Triton → Profile → Optimize hotspots in CUDA if needed
🎯 Use Triton for 80% of kernels (dense ops)
🎯 Use CUDA for 20% of kernels (sparse/irregular ops)
🎯 Let benchmarks decide, not opinions

╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  THIS KERNEL (Multimodal Fusion):                                                         ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

Binary search = irregular memory access = memory-latency bound
→ **CUDA likely better** (but benchmark to confirm!)

For production: Start with Triton, profile, optimize CUDA if bottleneck.
""")


def main():
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  TRITON VS CUDA: OBJECTIVE COMPARISON                                           ║")
    print("║  Multimodal Sensor Fusion Benchmark                                             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}\n")
    
    # Test configurations
    configs = [
        BenchmarkConfig(
            "Small (1-sec, batch=32)",
            batch=32, vision_len=30, vision_dim=512,
            proprio_len=100, proprio_dim=14,
            force_len=333, force_dim=6, target_len=50
        ),
        BenchmarkConfig(
            "Medium (5-sec, batch=128)",
            batch=128, vision_len=150, vision_dim=512,
            proprio_len=500, proprio_dim=14,
            force_len=1665, force_dim=6, target_len=250
        ),
    ]
    
    results = {}
    
    # Run benchmarks
    for config in configs:
        print(f"\n{'='*100}")
        print(f"Configuration: {config.name}")
        print(f"Data size: {config.data_size_mb():.2f} MB")
        print(f"{'='*100}")
        
        data = create_test_data(config)
        
        # Benchmark CUDA
        cuda_results = benchmark_cuda(data, config)
        
        # Benchmark Triton
        triton_results = benchmark_triton(data, config)
        
        # Verify correctness
        correctness = verify_correctness(cuda_results['output'], triton_results['output'])
        
        results[config.name] = {
            'cuda': cuda_results,
            'triton': triton_results,
            'correctness': correctness,
            'data_size_mb': config.data_size_mb(),
        }
    
    # Print results
    print_comparison_table(results)
    
    # Code complexity analysis
    code_stats = analyze_code_complexity()
    print_development_comparison(code_stats)
    
    # Recommendations
    print_recommendations()
    
    print("\n✅ Comparison complete\n")


if __name__ == "__main__":
    main()

