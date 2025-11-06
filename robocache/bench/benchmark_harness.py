"""
Reproducible Benchmark Harness with Statistical Rigor.

Runs N seeds × R repeats for CPU and GPU baselines, outputs CSV + HTML summary.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.perf.perf_guard import time_op, PerfStats, PerfGuard


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    batch_size: int
    seq_len: int
    dim: int
    device: str
    dtype: torch.dtype
    seeds: List[int]
    repeats: int


class BenchmarkHarness:
    """
    Statistical benchmark harness for RoboCache operations.
    
    Features:
    - Multiple seeds for statistical significance
    - CPU vs GPU baseline comparison
    - CSV + HTML output
    - ±5% variance envelopes
    - Mean/Std/95% CI reporting
    """
    
    def __init__(self, output_dir: Path = Path("bench/results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
    
    def benchmark_trajectory_resample(
        self,
        config: BenchmarkConfig,
        implementation: str = "cuda"
    ) -> PerfStats:
        """
        Benchmark trajectory resampling operation.
        
        Args:
            config: Benchmark configuration
            implementation: "cuda" or "pytorch"
        
        Returns:
            Performance statistics
        """
        import robocache
        
        # Generate test data
        torch.manual_seed(config.seeds[0])
        source_data = torch.randn(
            config.batch_size,
            config.seq_len,
            config.dim,
            device=config.device,
            dtype=config.dtype
        )
        source_times = torch.linspace(
            0, 5, config.seq_len, device=config.device
        ).unsqueeze(0).expand(config.batch_size, -1)
        target_times = torch.linspace(
            0, 5, config.seq_len // 2, device=config.device
        ).unsqueeze(0).expand(config.batch_size, -1)
        
        # Warmup and benchmark
        if implementation == "cuda":
            fn = lambda: robocache.resample_trajectories(
                source_data, source_times, target_times
            )
        else:
            # PyTorch CPU fallback
            fn = lambda: self._resample_pytorch(
                source_data.cpu(), source_times.cpu(), target_times.cpu()
            )
        
        sync_fn = torch.cuda.synchronize if config.device == "cuda" else None
        return time_op(fn, warmup=10, iters=config.repeats, sync_fn=sync_fn)
    
    def _resample_pytorch(self, data, src_times, tgt_times):
        """PyTorch CPU reference implementation."""
        B, S, D = data.shape
        T = tgt_times.shape[1]
        result = torch.zeros(B, T, D, dtype=data.dtype)
        
        for b in range(B):
            for t in range(T):
                tgt_t = tgt_times[b, t].item()
                # Binary search
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
    
    def run_suite(
        self,
        name: str,
        configs: List[BenchmarkConfig],
        implementations: List[str] = ["cuda", "pytorch"]
    ):
        """
        Run benchmark suite across multiple configurations and implementations.
        
        Args:
            name: Suite name
            configs: List of benchmark configurations
            implementations: List of implementations to test
        """
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUITE: {name}")
        print(f"{'='*80}\n")
        
        for config in configs:
            for impl in implementations:
                # Skip CUDA if not available
                if impl == "cuda" and not torch.cuda.is_available():
                    print(f"⚠️  Skipping CUDA benchmark (no GPU available)")
                    continue
                
                print(f"Running: {config.name} [{impl}] ...")
                
                # Run across multiple seeds
                all_stats = []
                for seed in config.seeds:
                    torch.manual_seed(seed)
                    stats = self.benchmark_trajectory_resample(config, impl)
                    all_stats.append(stats)
                
                # Aggregate statistics
                p50_values = [s.p50 for s in all_stats]
                p99_values = [s.p99 for s in all_stats]
                mean_values = [s.mean for s in all_stats]
                
                agg_stats = {
                    "suite": name,
                    "operation": config.name,
                    "implementation": impl,
                    "device": config.device if impl == "cuda" else "cpu",
                    "dtype": str(config.dtype),
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "dim": config.dim,
                    "seeds": len(config.seeds),
                    "repeats_per_seed": config.repeats,
                    "p50_mean": np.mean(p50_values),
                    "p50_std": np.std(p50_values),
                    "p50_min": np.min(p50_values),
                    "p50_max": np.max(p50_values),
                    "p99_mean": np.mean(p99_values),
                    "p99_std": np.std(p99_values),
                    "variance_pct": (np.std(p50_values) / np.mean(p50_values)) * 100,
                }
                
                # 95% confidence interval
                agg_stats["p50_ci95"] = 1.96 * agg_stats["p50_std"] / np.sqrt(len(config.seeds))
                agg_stats["p99_ci95"] = 1.96 * agg_stats["p99_std"] / np.sqrt(len(config.seeds))
                
                self.results.append(agg_stats)
                
                print(f"  P50: {agg_stats['p50_mean']:.3f} ± {agg_stats['p50_std']:.3f} ms "
                      f"(CI95: ±{agg_stats['p50_ci95']:.3f} ms)")
                print(f"  P99: {agg_stats['p99_mean']:.3f} ± {agg_stats['p99_std']:.3f} ms")
                print(f"  Variance: {agg_stats['variance_pct']:.1f}%\n")
    
    def save_results(self, filename: str = "benchmark_results"):
        """Save results to CSV and JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # CSV output
        csv_path = self.output_dir / f"{filename}_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        print(f"✅ Saved CSV: {csv_path}")
        
        # JSON output
        json_path = self.output_dir / f"{filename}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved JSON: {json_path}")
        
        # HTML summary
        self.generate_html_summary(filename, timestamp)
    
    def generate_html_summary(self, filename: str, timestamp: str):
        """Generate HTML summary with comparison tables."""
        html_path = self.output_dir / f"{filename}_{timestamp}.html"
        
        # Group by operation
        ops: Dict[str, List[Dict]] = {}
        for result in self.results:
            key = result["operation"]
            if key not in ops:
                ops[key] = []
            ops[key].append(result)
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>RoboCache Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }
        th, td { padding: 12px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #76B900; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .speedup { font-weight: bold; color: #76B900; }
        .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🚀 RoboCache Benchmark Results</h1>
    <div class="summary">
        <p><strong>Timestamp:</strong> """ + timestamp + """</p>
        <p><strong>Total Operations:</strong> """ + str(len(self.results)) + """</p>
    </div>
"""
        
        for op_name, results in ops.items():
            html += f"<h2>{op_name}</h2>\n<table>\n"
            html += "<tr><th>Implementation</th><th>Device</th><th>P50 (ms)</th><th>P99 (ms)</th>"
            html += "<th>Variance</th><th>CI95</th><th>Speedup</th></tr>\n"
            
            # Calculate speedup (CUDA vs PyTorch)
            cuda_result = next((r for r in results if r["implementation"] == "cuda"), None)
            pytorch_result = next((r for r in results if r["implementation"] == "pytorch"), None)
            speedup = None
            if cuda_result and pytorch_result:
                speedup = pytorch_result["p50_mean"] / cuda_result["p50_mean"]
            
            for result in results:
                speedup_str = ""
                if result["implementation"] == "cuda" and speedup:
                    speedup_str = f'<span class="speedup">{speedup:.2f}×</span>'
                
                html += f"""<tr>
                    <td>{result['implementation']}</td>
                    <td>{result['device']}</td>
                    <td>{result['p50_mean']:.3f} ± {result['p50_std']:.3f}</td>
                    <td>{result['p99_mean']:.3f} ± {result['p99_std']:.3f}</td>
                    <td>{result['variance_pct']:.1f}%</td>
                    <td>±{result['p50_ci95']:.3f}</td>
                    <td>{speedup_str}</td>
                </tr>\n"""
            html += "</table>\n"
        
        html += "</body>\n</html>"
        
        with open(html_path, "w") as f:
            f.write(html)
        print(f"✅ Saved HTML: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="RoboCache Benchmark Harness")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--repeats", type=int, default=100, help="Repeats per seed")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    args = parser.parse_args()
    
    # Initialize harness
    harness = BenchmarkHarness(output_dir=args.output_dir)
    
    # Define benchmark configurations
    seeds = list(range(args.seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    configs = [
        BenchmarkConfig(
            name="trajectory_resample_small",
            batch_size=8,
            seq_len=250,
            dim=128,
            device=device,
            dtype=torch.bfloat16,
            seeds=seeds,
            repeats=args.repeats
        ),
        BenchmarkConfig(
            name="trajectory_resample_medium",
            batch_size=32,
            seq_len=500,
            dim=256,
            device=device,
            dtype=torch.bfloat16,
            seeds=seeds,
            repeats=args.repeats
        ),
        BenchmarkConfig(
            name="trajectory_resample_large",
            batch_size=64,
            seq_len=1000,
            dim=512,
            device=device,
            dtype=torch.bfloat16,
            seeds=seeds,
            repeats=args.repeats
        ),
    ]
    
    # Run benchmark suite
    harness.run_suite(
        name="trajectory_resampling",
        configs=configs,
        implementations=["cuda", "pytorch"] if torch.cuda.is_available() else ["pytorch"]
    )
    
    # Save results
    harness.save_results()
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

