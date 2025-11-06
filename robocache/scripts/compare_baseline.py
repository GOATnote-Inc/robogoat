#!/usr/bin/env python3
"""
Compare current benchmark results against baseline with tolerance gates.

Used in CI to catch performance regressions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_results(path: Path) -> Dict:
    """Load benchmark results from JSON."""
    with open(path) as f:
        return json.load(f)


def compare_results(
    current: List[Dict],
    baseline: List[Dict],
    tolerance_p50: float = 0.05,
    tolerance_p99: float = 0.10
) -> bool:
    """
    Compare current results against baseline.
    
    Args:
        current: Current benchmark results
        baseline: Baseline benchmark results
        tolerance_p50: P50 regression tolerance (default 5%)
        tolerance_p99: P99 regression tolerance (default 10%)
    
    Returns:
        True if all tests pass, False if regressions detected
    """
    print("\n" + "="*80)
    print("PERFORMANCE REGRESSION ANALYSIS")
    print("="*80 + "\n")
    
    # Index baseline by operation + implementation
    baseline_map = {}
    for result in baseline:
        key = (result["operation"], result["implementation"])
        baseline_map[key] = result
    
    regressions = []
    all_pass = True
    
    for curr in current:
        key = (curr["operation"], curr["implementation"])
        
        if key not in baseline_map:
            print(f"⚠️  No baseline for {key[0]} [{key[1]}], skipping")
            continue
        
        base = baseline_map[key]
        
        # Calculate regressions
        p50_curr = curr.get("p50_mean", curr.get("p50", 0))
        p50_base = base.get("p50_mean", base.get("p50", 0))
        p99_curr = curr.get("p99_mean", curr.get("p99", 0))
        p99_base = base.get("p99_mean", base.get("p99", 0))
        
        if p50_base == 0 or p99_base == 0:
            continue
        
        p50_regression = (p50_curr - p50_base) / p50_base
        p99_regression = (p99_curr - p99_base) / p99_base
        
        # Check thresholds
        p50_pass = p50_regression <= tolerance_p50
        p99_pass = p99_regression <= tolerance_p99
        
        status = "✅ PASS" if (p50_pass and p99_pass) else "❌ FAIL"
        
        print(f"{status} {curr['operation']} [{curr['implementation']}]")
        print(f"  P50: {p50_base:.3f}ms → {p50_curr:.3f}ms "
              f"({p50_regression*100:+.1f}%, limit: {tolerance_p50*100:.0f}%)")
        print(f"  P99: {p99_base:.3f}ms → {p99_curr:.3f}ms "
              f"({p99_regression*100:+.1f}%, limit: {tolerance_p99*100:.0f}%)")
        
        if not p50_pass:
            print(f"  ⚠️  P50 regression exceeds tolerance!")
            regressions.append(f"{curr['operation']}: P50 {p50_regression*100:+.1f}%")
            all_pass = False
        
        if not p99_pass:
            print(f"  ⚠️  P99 regression exceeds tolerance!")
            regressions.append(f"{curr['operation']}: P99 {p99_regression*100:+.1f}%")
            all_pass = False
        
        print()
    
    print("="*80)
    if all_pass:
        print("✅ ALL PERFORMANCE TESTS PASSED")
    else:
        print("❌ PERFORMANCE REGRESSIONS DETECTED:")
        for reg in regressions:
            print(f"  - {reg}")
    print("="*80 + "\n")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results against baseline")
    parser.add_argument("--current", type=Path, required=True, help="Current results JSON")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline results JSON")
    parser.add_argument("--tolerance-p50", type=float, default=0.05, help="P50 tolerance (default: 5%%)")
    parser.add_argument("--tolerance-p99", type=float, default=0.10, help="P99 tolerance (default: 10%%)")
    args = parser.parse_args()
    
    if not args.current.exists():
        print(f"ERROR: Current results not found: {args.current}")
        sys.exit(1)
    
    if not args.baseline.exists():
        print(f"⚠️  Baseline not found: {args.baseline}")
        print("Skipping comparison (first run or baseline missing)")
        sys.exit(0)
    
    # Load results
    current = load_results(args.current)
    baseline = load_results(args.baseline)
    
    # Handle different JSON formats
    if isinstance(current, dict) and "results" in current:
        current = current["results"]
    if isinstance(baseline, dict) and "results" in baseline:
        baseline = baseline["results"]
    
    # Compare
    passed = compare_results(
        current,
        baseline,
        tolerance_p50=args.tolerance_p50,
        tolerance_p99=args.tolerance_p99
    )
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

