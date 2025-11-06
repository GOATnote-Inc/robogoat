#!/usr/bin/env python3
"""
RoboCache Expert Metrics Validator
Validates Nsight profiling results against performance targets
"""
import sys
import re
import json
import glob
from pathlib import Path

# Performance targets for H100
TARGETS = {
    "sm_throughput_min": 85.0,      # SM utilization %
    "dram_throughput_min": 80.0,    # DRAM bandwidth %
    "warps_active_min": 70.0,       # Warp occupancy %
    "l1_miss_rate_max": 15.0,       # L1 cache miss rate %
}

def validate_metrics(metrics_file):
    """Validate metrics from Nsight profiling"""
    with open(metrics_file) as f:
        content = f.read()
    
    results = {}
    
    # Extract metrics using regex
    patterns = {
        "sm_throughput": r"sm__throughput.*?([0-9.]+)",
        "dram_throughput": r"dram__throughput.*?([0-9.]+)",
        "warps_active": r"warps_active.*?([0-9.]+)",
        "l1_miss_rate": r"l1tex__t_sector_miss_rate.*?([0-9.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
        else:
            results[key] = None
    
    # Validate against targets
    passed = True
    messages = []
    
    if results.get("sm_throughput") is not None:
        if results["sm_throughput"] < TARGETS["sm_throughput_min"]:
            passed = False
            messages.append(f"❌ SM Throughput: {results['sm_throughput']:.1f}% < {TARGETS['sm_throughput_min']:.1f}%")
        else:
            messages.append(f"✅ SM Throughput: {results['sm_throughput']:.1f}% >= {TARGETS['sm_throughput_min']:.1f}%")
    
    if results.get("dram_throughput") is not None:
        if results["dram_throughput"] < TARGETS["dram_throughput_min"]:
            passed = False
            messages.append(f"❌ DRAM Throughput: {results['dram_throughput']:.1f}% < {TARGETS['dram_throughput_min']:.1f}%")
        else:
            messages.append(f"✅ DRAM Throughput: {results['dram_throughput']:.1f}% >= {TARGETS['dram_throughput_min']:.1f}%")
    
    if results.get("warps_active") is not None:
        if results["warps_active"] < TARGETS["warps_active_min"]:
            messages.append(f"⚠️  Warps Active: {results['warps_active']:.1f}% < {TARGETS['warps_active_min']:.1f}%")
        else:
            messages.append(f"✅ Warps Active: {results['warps_active']:.1f}% >= {TARGETS['warps_active_min']:.1f}%")
    
    if results.get("l1_miss_rate") is not None:
        if results["l1_miss_rate"] > TARGETS["l1_miss_rate_max"]:
            messages.append(f"⚠️  L1 Miss Rate: {results['l1_miss_rate']:.1f}% > {TARGETS['l1_miss_rate_max']:.1f}%")
        else:
            messages.append(f"✅ L1 Miss Rate: {results['l1_miss_rate']:.1f}% <= {TARGETS['l1_miss_rate_max']:.1f}%")
    
    return passed, messages, results

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_metrics.py <metrics_file_glob>")
        sys.exit(1)
    
    pattern = sys.argv[1]
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No files found matching: {pattern}")
        sys.exit(1)
    
    all_passed = True
    for metrics_file in files:
        print(f"\n📊 Validating: {metrics_file}")
        print("=" * 80)
        
        passed, messages, results = validate_metrics(metrics_file)
        
        for msg in messages:
            print(msg)
        
        if not passed:
            all_passed = False
            print(f"\n❌ VALIDATION FAILED: {metrics_file}")
        else:
            print(f"\n✅ VALIDATION PASSED: {metrics_file}")
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✅ All metrics stable or improved.")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Performance regression detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()

