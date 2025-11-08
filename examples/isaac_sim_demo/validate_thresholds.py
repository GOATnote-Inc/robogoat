#!/usr/bin/env python3
"""
Validate Isaac Sim results against acceptance thresholds.

Used by CI to enforce latency, accuracy, and speedup gates.
"""

import argparse
import json
import yaml
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load results JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def load_thresholds(path: Path) -> dict:
    """Load acceptance thresholds YAML"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def validate_latency(results: dict, thresholds: dict) -> tuple[bool, str]:
    """Validate latency threshold"""
    actual_ms = results['latency_ms']
    max_ms = thresholds['latency']['max_ms']
    
    if actual_ms < max_ms:
        return True, f"✅ Latency: {actual_ms:.2f} ms < {max_ms} ms"
    else:
        return False, f"❌ Latency: {actual_ms:.2f} ms >= {max_ms} ms (FAILED)"


def validate_speedup(robocache_results: dict, baseline_results: dict, thresholds: dict) -> tuple[bool, str]:
    """Validate speedup vs baseline"""
    robocache_latency = robocache_results['latency_ms']
    baseline_latency = baseline_results['latency_ms']
    speedup = baseline_latency / robocache_latency
    
    min_speedup = thresholds['latency']['baseline_comparison']['min_speedup']
    
    if speedup >= min_speedup:
        return True, f"✅ Speedup: {speedup:.2f}x >= {min_speedup}x"
    else:
        return False, f"❌ Speedup: {speedup:.2f}x < {min_speedup}x (FAILED)"


def validate_accuracy(results: dict, thresholds: dict) -> tuple[bool, str]:
    """Validate numerical accuracy"""
    if 'l2_error' not in results:
        return True, "⚠️  Accuracy: Not measured (skipped)"
    
    l2_error = results['l2_error']
    max_l2_error = thresholds['accuracy']['max_l2_error']
    
    if l2_error < max_l2_error:
        return True, f"✅ Accuracy: L2 error {l2_error:.4f} < {max_l2_error}"
    else:
        return False, f"❌ Accuracy: L2 error {l2_error:.4f} >= {max_l2_error} (FAILED)"


def validate_throughput(results: dict, thresholds: dict, gpu_type: str) -> tuple[bool, str]:
    """Validate throughput"""
    if 'throughput' not in results:
        return True, "⚠️  Throughput: Not measured (skipped)"
    
    actual_eps = results['throughput']
    min_eps = thresholds['throughput']['min_episodes_per_sec'][gpu_type]
    
    if actual_eps >= min_eps:
        return True, f"✅ Throughput: {actual_eps:.0f} eps/sec >= {min_eps} eps/sec"
    else:
        return False, f"❌ Throughput: {actual_eps:.0f} eps/sec < {min_eps} eps/sec (FAILED)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robocache-results', type=Path, required=True)
    parser.add_argument('--baseline-results', type=Path, required=False)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--gpu-type', choices=['h100', 'a100', 'l4'], default='h100')
    parser.add_argument('--assert-all', action='store_true',
                        help='Fail if any threshold violated')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Isaac Sim Acceptance Threshold Validation")
    print("=" * 70)
    
    # Load data
    robocache_results = load_results(args.robocache_results)
    baseline_results = load_results(args.baseline_results) if args.baseline_results else None
    thresholds = load_thresholds(args.thresholds)
    
    print(f"\nGPU: {args.gpu_type.upper()}")
    print(f"RoboCache results: {args.robocache_results}")
    if baseline_results:
        print(f"Baseline results: {args.baseline_results}")
    print()
    
    # Run validations
    results = []
    
    # Latency
    passed, msg = validate_latency(robocache_results, thresholds)
    results.append(passed)
    print(msg)
    
    # Speedup (if baseline available)
    if baseline_results:
        passed, msg = validate_speedup(robocache_results, baseline_results, thresholds)
        results.append(passed)
        print(msg)
    
    # Accuracy
    passed, msg = validate_accuracy(robocache_results, thresholds)
    results.append(passed)
    print(msg)
    
    # Throughput
    passed, msg = validate_throughput(robocache_results, thresholds, args.gpu_type)
    results.append(passed)
    print(msg)
    
    # Summary
    print()
    print("=" * 70)
    if all(results):
        print("✅ ALL THRESHOLDS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME THRESHOLDS FAILED")
        print("=" * 70)
        
        if args.assert_all:
            sys.exit(1)
        else:
            print("(Non-blocking - continuing)")
            return 0


if __name__ == '__main__':
    sys.exit(main())

