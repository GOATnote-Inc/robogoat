"""
Performance Guard: Statistical performance validation with regression gates.

Ensures P50/P99 latencies stay within bounds to prevent performance regressions.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional
import statistics as stats


@dataclass
class PerfStats:
    """Performance statistics with percentiles and CI."""
    p50: float  # median (ms)
    p99: float  # 99th percentile (ms)
    mean: float
    std: float
    min: float
    max: float
    samples: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


def time_op(
    fn: Callable, 
    warmup: int = 5, 
    iters: int = 100,
    sync_fn: Optional[Callable] = None
) -> PerfStats:
    """
    Time an operation with warmup and multiple iterations.
    
    Args:
        fn: Function to benchmark
        warmup: Number of warmup iterations
        iters: Number of measurement iterations
        sync_fn: Optional synchronization function (e.g., torch.cuda.synchronize)
    
    Returns:
        PerfStats with percentiles and statistics
    """
    # Warmup
    for _ in range(warmup):
        fn()
        if sync_fn:
            sync_fn()
    
    # Measure
    times = []
    for _ in range(iters):
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        times.append((time.perf_counter() - t0) * 1e3)  # Convert to ms
    
    times.sort()
    return PerfStats(
        p50=times[int(0.50 * iters)],
        p99=times[min(int(0.99 * iters), iters - 1)],
        mean=stats.mean(times),
        std=stats.stdev(times) if iters > 1 else 0.0,
        min=min(times),
        max=max(times),
        samples=iters
    )


class PerfGuard:
    """
    Performance regression guard.
    
    Enforces P50/P99 thresholds and optionally compares against baseline.
    """
    
    def __init__(self, enforce: bool = None, baseline_path: Optional[Path] = None):
        """
        Args:
            enforce: Whether to raise on violations (defaults to PERF_GUARD_ENFORCE env var)
            baseline_path: Path to baseline performance JSON file
        """
        if enforce is None:
            enforce = os.environ.get("PERF_GUARD_ENFORCE", "0") == "1"
        self.enforce = enforce
        self.baseline = self._load_baseline(baseline_path) if baseline_path else {}
        self.results: Dict[str, PerfStats] = {}
    
    def _load_baseline(self, path: Path) -> Dict:
        """Load baseline performance data."""
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)
    
    def require_lt_ms(
        self,
        name: str,
        p50: float,
        p99: float,
        p50_max: float,
        p99_max: float,
        p50_tolerance: float = 0.05,  # 5% tolerance
        p99_tolerance: float = 0.10   # 10% tolerance
    ):
        """
        Require P50/P99 to be below thresholds.
        
        Args:
            name: Operation name
            p50: Measured P50 latency (ms)
            p99: Measured P99 latency (ms)
            p50_max: Maximum allowed P50 (ms)
            p99_max: Maximum allowed P99 (ms)
            p50_tolerance: Tolerance for P50 regression vs baseline
            p99_tolerance: Tolerance for P99 regression vs baseline
        
        Raises:
            AssertionError: If thresholds are violated and enforce=True
        """
        violations = []
        
        # Check absolute thresholds
        if p50 > p50_max:
            violations.append(f"P50 {p50:.3f}ms > {p50_max:.3f}ms")
        if p99 > p99_max:
            violations.append(f"P99 {p99:.3f}ms > {p99_max:.3f}ms")
        
        # Check baseline regression
        if name in self.baseline:
            baseline = self.baseline[name]
            baseline_p50 = baseline.get("p50", float("inf"))
            baseline_p99 = baseline.get("p99", float("inf"))
            
            p50_regression = (p50 - baseline_p50) / baseline_p50
            p99_regression = (p99 - baseline_p99) / baseline_p99
            
            if p50_regression > p50_tolerance:
                violations.append(
                    f"P50 regressed {p50_regression*100:.1f}% vs baseline "
                    f"({baseline_p50:.3f}ms → {p50:.3f}ms)"
                )
            if p99_regression > p99_tolerance:
                violations.append(
                    f"P99 regressed {p99_regression*100:.1f}% vs baseline "
                    f"({baseline_p99:.3f}ms → {p99:.3f}ms)"
                )
        
        if violations and self.enforce:
            raise AssertionError(f"{name} performance violations:\n  " + "\n  ".join(violations))
        elif violations:
            print(f"⚠️  {name} performance warnings:\n  " + "\n  ".join(violations))
    
    def record(self, name: str, stats: PerfStats):
        """Record performance statistics."""
        self.results[name] = stats
    
    def save_results(self, path: Path):
        """Save results as JSON baseline."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.results.items()},
                f,
                indent=2
            )
    
    def print_summary(self):
        """Print performance summary table."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Operation':<30} {'P50 (ms)':<12} {'P99 (ms)':<12} {'Mean ± Std (ms)':<20}")
        print("-"*80)
        for name, stats in self.results.items():
            print(
                f"{name:<30} {stats.p50:>10.3f}  {stats.p99:>10.3f}  "
                f"{stats.mean:>8.3f} ± {stats.std:>6.3f}"
            )
        print("="*80 + "\n")


# Global instance for convenience
perf_guard = PerfGuard()

