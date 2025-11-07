#!/usr/bin/env python3
"""
Compare RoboCache vs Baseline Training Results

Analyzes performance metrics and generates comparison report.

Usage:
    python compare_results.py --baseline baseline_results.json --robocache robocache_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns


def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(baseline: Dict, robocache: Dict):
    """Generate comprehensive comparison"""
    print(f"\n{'='*80}")
    print("ROBOCACHE VS BASELINE PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")
    
    # Overall speedup
    total_speedup = baseline['total_time_sec'] / robocache['total_time_sec']
    steps_speedup = robocache['steps_per_sec'] / baseline['steps_per_sec']
    
    print("📊 OVERALL PERFORMANCE")
    print(f"{'─'*80}")
    print(f"{'Metric':<40} {'Baseline':>15} {'RoboCache':>15} {'Speedup':>10}")
    print(f"{'─'*80}")
    print(f"{'Total Time (minutes)':<40} {baseline['total_time_min']:>15.2f} {robocache['total_time_min']:>15.2f} {total_speedup:>9.2f}x")
    print(f"{'Steps/sec':<40} {baseline['steps_per_sec']:>15.1f} {robocache['steps_per_sec']:>15.1f} {steps_speedup:>9.2f}x")
    print(f"{'Avg Step Time (ms)':<40} {baseline['avg_step_time_ms']:>15.2f} {robocache['avg_step_time_ms']:>15.2f} {baseline['avg_step_time_ms']/robocache['avg_step_time_ms']:>9.2f}x")
    print(f"{'P99 Step Time (ms)':<40} {baseline['p99_step_time_ms']:>15.2f} {robocache['p99_step_time_ms']:>15.2f} {baseline['p99_step_time_ms']/robocache['p99_step_time_ms']:>9.2f}x")
    
    # Breakdown
    print(f"\n⏱️  TIMING BREAKDOWN")
    print(f"{'─'*80}")
    print(f"{'Component':<40} {'Baseline':>15} {'RoboCache':>15} {'Speedup':>10}")
    print(f"{'─'*80}")
    
    preprocess_speedup = baseline['avg_preprocess_time_ms'] / robocache['avg_preprocess_time_ms']
    forward_speedup = baseline['avg_forward_time_ms'] / robocache['avg_forward_time_ms']
    backward_speedup = baseline['avg_backward_time_ms'] / robocache['avg_backward_time_ms']
    
    print(f"{'Preprocessing (ms)':<40} {baseline['avg_preprocess_time_ms']:>15.2f} {robocache['avg_preprocess_time_ms']:>15.2f} {preprocess_speedup:>9.2f}x")
    print(f"{'Policy Forward (ms)':<40} {baseline['avg_forward_time_ms']:>15.2f} {robocache['avg_forward_time_ms']:>15.2f} {forward_speedup:>9.2f}x")
    print(f"{'Backward Pass (ms)':<40} {baseline['avg_backward_time_ms']:>15.2f} {robocache['avg_backward_time_ms']:>15.2f} {backward_speedup:>9.2f}x")
    
    # Time percentages
    print(f"\n📈 TIME DISTRIBUTION")
    print(f"{'─'*80}")
    print(f"{'Component':<40} {'Baseline %':>15} {'RoboCache %':>15}")
    print(f"{'─'*80}")
    
    baseline_total = baseline['avg_preprocess_time_ms'] + baseline['avg_forward_time_ms'] + baseline['avg_backward_time_ms']
    robocache_total = robocache['avg_preprocess_time_ms'] + robocache['avg_forward_time_ms'] + robocache['avg_backward_time_ms']
    
    print(f"{'Preprocessing':<40} {baseline['avg_preprocess_time_ms']/baseline_total*100:>14.1f}% {robocache['avg_preprocess_time_ms']/robocache_total*100:>14.1f}%")
    print(f"{'Policy Forward':<40} {baseline['avg_forward_time_ms']/baseline_total*100:>14.1f}% {robocache['avg_forward_time_ms']/robocache_total*100:>14.1f}%")
    print(f"{'Backward Pass':<40} {baseline['avg_backward_time_ms']/baseline_total*100:>14.1f}% {robocache['avg_backward_time_ms']/robocache_total*100:>14.1f}%")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print(f"{'─'*80}")
    print(f"1. Overall Training Speedup: {total_speedup:.2f}x faster with RoboCache")
    print(f"2. Preprocessing Acceleration: {preprocess_speedup:.1f}x (sensor fusion + voxelization)")
    print(f"3. Time Saved: {baseline['total_time_min'] - robocache['total_time_min']:.1f} minutes ({(1-1/total_speedup)*100:.1f}% reduction)")
    print(f"4. Steps/sec Improvement: {robocache['steps_per_sec'] - baseline['steps_per_sec']:.1f} more steps/sec")
    
    if total_speedup >= 4.0:
        print(f"\n✅ EXCELLENT: RoboCache delivers {total_speedup:.1f}x speedup (target: 4-5x)")
    elif total_speedup >= 3.0:
        print(f"\n✅ GOOD: RoboCache delivers {total_speedup:.1f}x speedup (target: 3-4x for A100)")
    else:
        print(f"\n⚠️  BELOW TARGET: {total_speedup:.1f}x speedup (expected 3-5x)")
    
    # Cost analysis
    print(f"\n💰 COST ANALYSIS (assuming $3.50/hr GPU cost)")
    print(f"{'─'*80}")
    baseline_cost = (baseline['total_time_min'] / 60) * 3.50
    robocache_cost = (robocache['total_time_min'] / 60) * 3.50
    print(f"{'Baseline Cost':<40} ${baseline_cost:>14.2f}")
    print(f"{'RoboCache Cost':<40} ${robocache_cost:>14.2f}")
    print(f"{'Savings':<40} ${baseline_cost - robocache_cost:>14.2f} ({(1-1/total_speedup)*100:.1f}%)")
    
    print(f"\n{'='*80}\n")
    
    # Generate plots
    generate_plots(baseline, robocache)


def generate_plots(baseline: Dict, robocache: Dict):
    """Generate comparison visualizations"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall Time Comparison
    ax1 = axes[0, 0]
    categories = ['Total Time\n(min)', 'Steps/sec']
    baseline_vals = [baseline['total_time_min'], baseline['steps_per_sec']]
    robocache_vals = [robocache['total_time_min'], robocache['steps_per_sec']]
    
    x = range(len(categories))
    width = 0.35
    ax1.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', color='#e74c3c')
    ax1.bar([i + width/2 for i in x], robocache_vals, width, label='RoboCache', color='#2ecc71')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel('Value')
    ax1.set_title('Overall Performance')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Timing Breakdown
    ax2 = axes[0, 1]
    components = ['Preprocess', 'Forward', 'Backward']
    baseline_times = [
        baseline['avg_preprocess_time_ms'],
        baseline['avg_forward_time_ms'],
        baseline['avg_backward_time_ms']
    ]
    robocache_times = [
        robocache['avg_preprocess_time_ms'],
        robocache['avg_forward_time_ms'],
        robocache['avg_backward_time_ms']
    ]
    
    x = range(len(components))
    ax2.bar([i - width/2 for i in x], baseline_times, width, label='Baseline', color='#e74c3c')
    ax2.bar([i + width/2 for i in x], robocache_times, width, label='RoboCache', color='#2ecc71')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Per-Step Timing Breakdown')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Speedup by Component
    ax3 = axes[1, 0]
    speedups = [
        baseline['avg_preprocess_time_ms'] / robocache['avg_preprocess_time_ms'],
        baseline['avg_forward_time_ms'] / robocache['avg_forward_time_ms'],
        baseline['avg_backward_time_ms'] / robocache['avg_backward_time_ms'],
        baseline['avg_step_time_ms'] / robocache['avg_step_time_ms']
    ]
    components_all = ['Preprocess', 'Forward', 'Backward', 'Overall']
    colors = ['#3498db' if s > 1.5 else '#95a5a6' for s in speedups]
    
    ax3.barh(components_all, speedups, color=colors)
    ax3.axvline(x=1, color='red', linestyle='--', label='No speedup')
    ax3.set_xlabel('Speedup (x faster)')
    ax3.set_title('RoboCache Speedup by Component')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Time Distribution
    ax4 = axes[1, 1]
    baseline_pct = [
        baseline['avg_preprocess_time_ms'],
        baseline['avg_forward_time_ms'],
        baseline['avg_backward_time_ms']
    ]
    robocache_pct = [
        robocache['avg_preprocess_time_ms'],
        robocache['avg_forward_time_ms'],
        robocache['avg_backward_time_ms']
    ]
    
    labels = ['Preprocess', 'Forward', 'Backward']
    colors_pie = ['#3498db', '#e74c3c', '#f39c12']
    
    ax4_1 = plt.subplot(2, 4, 7)
    ax4_1.pie(baseline_pct, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax4_1.set_title('Baseline')
    
    ax4_2 = plt.subplot(2, 4, 8)
    ax4_2.pie(robocache_pct, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax4_2.set_title('RoboCache')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to performance_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Compare RoboCache vs Baseline results')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline results JSON')
    parser.add_argument('--robocache', type=str, required=True,
                        help='Path to RoboCache results JSON')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.baseline).exists():
        print(f"ERROR: Baseline results not found: {args.baseline}")
        return 1
    if not Path(args.robocache).exists():
        print(f"ERROR: RoboCache results not found: {args.robocache}")
        return 1
    
    # Load and compare
    baseline = load_results(args.baseline)
    robocache = load_results(args.robocache)
    
    compare_results(baseline, robocache)
    
    return 0


if __name__ == '__main__':
    exit(main())

