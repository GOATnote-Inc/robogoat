#!/usr/bin/env python3
"""
Run single reproducible benchmark from config file.

Usage:
    python run_single.py --config configs/multimodal_fusion_h100.json
    python run_single.py --config configs/multimodal_fusion_h100.json --ncu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_environment():
    """Setup Python path and imports."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'robocache' / 'python'))
    import robocache
    return robocache


def run_multimodal_fusion_benchmark(config: dict, robocache) -> dict:
    """Run multimodal fusion benchmark."""
    params = config['parameters']
    measurement = config['measurement']
    
    # Create synthetic data
    batch_size = params['batch_size']
    
    # Stream 1 (vision)
    s1_len = int(params['stream1_duration_sec'] * params['stream1_freq_hz'])
    stream1_data = torch.randn(
        batch_size, s1_len, params['stream1_shape'][1],
        dtype=getattr(torch, params['dtype']),
        device=params['device']
    )
    stream1_times = torch.linspace(
        0, params['stream1_duration_sec'], s1_len,
        device=params['device']
    ).expand(batch_size, -1)
    
    # Stream 2 (proprio)
    s2_len = int(params['stream2_duration_sec'] * params['stream2_freq_hz'])
    stream2_data = torch.randn(
        batch_size, s2_len, params['stream2_shape'][1],
        dtype=getattr(torch, params['dtype']),
        device=params['device']
    )
    stream2_times = torch.linspace(
        0, params['stream2_duration_sec'], s2_len,
        device=params['device']
    ).expand(batch_size, -1)
    
    # Stream 3 (IMU)
    s3_len = int(params['stream3_duration_sec'] * params['stream3_freq_hz'])
    stream3_data = torch.randn(
        batch_size, s3_len, params['stream3_shape'][1],
        dtype=getattr(torch, params['dtype']),
        device=params['device']
    )
    stream3_times = torch.linspace(
        0, params['stream3_duration_sec'], s3_len,
        device=params['device']
    ).expand(batch_size, -1)
    
    # Target times
    target_len = int(params['stream1_duration_sec'] * params['target_freq_hz'])
    target_times = torch.linspace(
        0, params['stream1_duration_sec'], target_len,
        device=params['device']
    ).expand(batch_size, -1)
    
    # Warmup
    for _ in range(measurement['warmup_iterations']):
        _ = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
    if measurement['synchronize']:
        torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(measurement['measurement_iterations']):
        start_event.record()
        result = robocache.fuse_multimodal(
            stream1_data, stream1_times,
            stream2_data, stream2_times,
            stream3_data, stream3_times,
            target_times
        )
        end_event.record()
        if measurement['synchronize']:
            torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    
    latencies = torch.tensor(latencies)
    return {
        'latency_ms': latencies.mean().item(),
        'latency_std_ms': latencies.std().item(),
        'latency_min_ms': latencies.min().item(),
        'latency_max_ms': latencies.max().item(),
        'latency_p50_ms': latencies.median().item(),
        'latency_p95_ms': latencies.quantile(0.95).item(),
        'latency_p99_ms': latencies.quantile(0.99).item(),
    }


def run_trajectory_resample_benchmark(config: dict, robocache) -> dict:
    """Run trajectory resampling benchmark."""
    params = config['parameters']
    measurement = config['measurement']
    
    # Create synthetic data
    source_data = torch.randn(
        params['batch_size'],
        params['source_length'],
        params['dimensions'],
        dtype=getattr(torch, params['dtype']),
        device=params['device']
    )
    source_times = torch.linspace(
        0, 1, params['source_length'],
        device=params['device']
    ).expand(params['batch_size'], -1)
    target_times = torch.linspace(
        0, 1, params['target_length'],
        device=params['device']
    ).expand(params['batch_size'], -1)
    
    # Warmup
    for _ in range(measurement['warmup_iterations']):
        _ = robocache.resample_trajectories(source_data, source_times, target_times)
    if measurement['synchronize']:
        torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(measurement['measurement_iterations']):
        start_event.record()
        result = robocache.resample_trajectories(source_data, source_times, target_times)
        end_event.record()
        if measurement['synchronize']:
            torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    
    latencies = torch.tensor(latencies)
    return {
        'latency_ms': latencies.mean().item(),
        'latency_std_ms': latencies.std().item(),
        'latency_min_ms': latencies.min().item(),
        'latency_max_ms': latencies.max().item(),
        'latency_p50_ms': latencies.median().item(),
        'latency_p95_ms': latencies.quantile(0.95).item(),
        'latency_p99_ms': latencies.quantile(0.99).item(),
    }


def run_voxelization_benchmark(config: dict, robocache) -> dict:
    """Run voxelization throughput benchmark."""
    params = config['parameters']
    measurement = config['measurement']
    
    # Create synthetic point cloud
    points = torch.rand(
        params['num_points'], 3,
        dtype=getattr(torch, params['dtype']),
        device=params['device']
    ) * 20.0 - 10.0  # Range: [-10, 10]
    
    # Warmup
    for _ in range(measurement['warmup_iterations']):
        _ = robocache.voxelize_pointcloud(
            points,
            grid_min=params['grid_min'],
            voxel_size=params['voxel_size'],
            grid_size=params['grid_size'],
            mode=params['mode']
        )
    if measurement['synchronize']:
        torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(measurement['measurement_iterations']):
        start_event.record()
        result = robocache.voxelize_pointcloud(
            points,
            grid_min=params['grid_min'],
            voxel_size=params['voxel_size'],
            grid_size=params['grid_size'],
            mode=params['mode']
        )
        end_event.record()
        if measurement['synchronize']:
            torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    
    latencies = torch.tensor(latencies)
    latency_ms = latencies.mean().item()
    latency_sec = latency_ms / 1000.0
    throughput = params['num_points'] / latency_sec
    
    return {
        'latency_ms': latency_ms,
        'latency_std_ms': latencies.std().item(),
        'throughput_points_per_sec': throughput,
        'throughput_billions_per_sec': throughput / 1e9,
    }


def evaluate_criteria(measured: dict, criteria: dict) -> dict:
    """Evaluate if measurement meets acceptance criteria."""
    metric = criteria['metric']
    measured_value = measured[metric]
    target = criteria['target']
    
    if 'tolerance_percent' in criteria:
        tolerance = criteria['tolerance_percent'] / 100.0
        deviation_percent = abs(measured_value - target) / target * 100
        passed = deviation_percent <= criteria['tolerance_percent']
    else:
        deviation_percent = None
        if 'max_acceptable' in criteria:
            passed = measured_value <= criteria['max_acceptable']
        elif 'min_acceptable' in criteria:
            passed = measured_value >= criteria['min_acceptable']
        else:
            passed = False
    
    return {
        'verdict': 'PASS' if passed else 'FAIL',
        'measured_value': measured_value,
        'target_value': target,
        'deviation_percent': deviation_percent,
        'passed': passed
    }


def main():
    parser = argparse.ArgumentParser(description='Run reproducible benchmark')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--ncu', action='store_true', help='Enable NCU profiling')
    parser.add_argument('--output', help='Output results JSON path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.verbose:
        print(f"Loaded config: {config['claim_id']}")
        print(f"README claim: {config['readme_claim']}")
    
    # Setup
    robocache = setup_environment()
    
    # Get hardware info
    hardware_info = {
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }
    
    if args.verbose:
        print(f"\nHardware: {hardware_info['gpu_name']}")
        print(f"CUDA: {hardware_info['cuda_version']}")
    
    # Run benchmark
    operation = config['operation']
    print(f"\nRunning {operation} benchmark...")
    
    if operation == 'fuse_multimodal':
        measured = run_multimodal_fusion_benchmark(config, robocache)
    elif operation == 'resample_trajectories':
        measured = run_trajectory_resample_benchmark(config, robocache)
    elif operation == 'voxelize_pointcloud':
        measured = run_voxelization_benchmark(config, robocache)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Evaluate
    evaluation = evaluate_criteria(measured, config['acceptance_criteria'])
    
    # Build results
    results = {
        'claim_id': config['claim_id'],
        'timestamp': datetime.now().isoformat(),
        'config_path': args.config,
        'hardware': hardware_info,
        'measured': measured,
        'evaluation': evaluation,
        'readme_claim': config['readme_claim'],
    }
    
    # Output
    print(f"\n{'='*60}")
    print(f"RESULTS: {config['claim_id']}")
    print(f"{'='*60}")
    print(f"Verdict: {evaluation['verdict']}")
    print(f"Measured: {evaluation['measured_value']:.4f}")
    print(f"Target: {evaluation['target_value']:.4f}")
    if evaluation['deviation_percent'] is not None:
        print(f"Deviation: {evaluation['deviation_percent']:.1f}%")
    print(f"\nFull measurements:")
    for k, v in measured.items():
        print(f"  {k}: {v:.6f}")
    print(f"{'='*60}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('results') / f"{config['claim_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Exit code based on verdict
    sys.exit(0 if evaluation['passed'] else 1)


if __name__ == '__main__':
    main()

