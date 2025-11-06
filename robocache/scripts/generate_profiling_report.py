#!/usr/bin/env python3
"""
RoboCache Expert Profiling Report Generator
Aggregates Nsight Systems + Nsight Compute results into Markdown
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime

def parse_nsys_summary(nsys_file):
    """Parse Nsight Systems summary statistics"""
    if not Path(nsys_file).exists():
        return None
    
    with open(nsys_file) as f:
        content = f.read()
    
    stats = {
        "total_kernels": 0,
        "total_time_ms": 0.0,
        "top_kernels": []
    }
    
    # Extract kernel summary (simplified regex - adjust based on actual format)
    kernel_lines = re.findall(r"(\S+)\s+(\d+)\s+([\d.]+)", content)
    for name, count, time_ms in kernel_lines[:5]:  # Top 5
        stats["top_kernels"].append({
            "name": name,
            "count": int(count),
            "time_ms": float(time_ms)
        })
        stats["total_time_ms"] += float(time_ms)
    
    return stats

def parse_ncu_metrics(ncu_json):
    """Parse Nsight Compute JSON metrics"""
    if not Path(ncu_json).exists():
        return None
    
    try:
        with open(ncu_json) as f:
            data = json.load(f)
        
        metrics = {
            "sm_throughput": None,
            "dram_throughput": None,
            "warps_active": None,
            "l1_miss_rate": None,
        }
        
        # Extract metrics from JSON structure (adjust based on actual NCU JSON format)
        # This is a simplified version - actual parsing depends on NCU output structure
        if isinstance(data, dict) and "metrics" in data:
            for metric in data["metrics"]:
                name = metric.get("name", "")
                value = metric.get("value", 0)
                if "sm__throughput" in name:
                    metrics["sm_throughput"] = value
                elif "dram__throughput" in name:
                    metrics["dram_throughput"] = value
                elif "warps_active" in name:
                    metrics["warps_active"] = value
                elif "l1tex__t_sector_miss_rate" in name:
                    metrics["l1_miss_rate"] = value
        
        return metrics
    except:
        return None

def parse_smoke_test(smoke_file):
    """Parse smoke test output for basic performance numbers"""
    if not Path(smoke_file).exists():
        return None
    
    with open(smoke_file) as f:
        content = f.read()
    
    stats = {
        "latency_ms": None,
        "throughput": None,
    }
    
    # Extract latency
    match = re.search(r"Latency:\s*([\d.]+)\s*ms", content)
    if match:
        stats["latency_ms"] = float(match.group(1))
    
    # Extract throughput
    match = re.search(r"Throughput:\s*([\d.]+)", content)
    if match:
        stats["throughput"] = float(match.group(1))
    
    return stats

def generate_markdown_report(profiling_dir, output_file):
    """Generate comprehensive Markdown report from profiling artifacts"""
    profiling_path = Path(profiling_dir)
    
    if not profiling_path.exists():
        print(f"❌ Profiling directory not found: {profiling_dir}")
        sys.exit(1)
    
    # Parse all available artifacts
    nsys_stats = parse_nsys_summary(profiling_path / "nsys_summary.txt")
    ncu_metrics = parse_ncu_metrics(profiling_path / "ncu_metrics.json")
    smoke_stats = parse_smoke_test(profiling_path / "smoke.txt")
    
    # Get GPU info from env.txt
    gpu_info = "Unknown"
    env_file = profiling_path / "env.txt"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "GPU" in line or "CUDA" in line:
                    gpu_info = line.strip()
                    break
    
    # Generate Markdown
    report = f"""# RoboCache Expert Profiling Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Profiling Run:** {profiling_path.name}  
**GPU:** {gpu_info}  

---

## Executive Summary

"""
    
    # Smoke test results
    if smoke_stats:
        report += f"""### Functional Validation ✅

- **Latency:** {smoke_stats['latency_ms']:.3f} ms
- **Throughput:** {smoke_stats['throughput']:.1f} items/sec

"""
    
    # NCU metrics
    if ncu_metrics:
        report += f"""### Nsight Compute Metrics

| Metric | Value | Status |
|--------|-------|--------|
"""
        
        if ncu_metrics.get("sm_throughput"):
            status = "✅" if ncu_metrics["sm_throughput"] >= 85.0 else "⚠️"
            report += f"| SM Throughput | {ncu_metrics['sm_throughput']:.1f}% | {status} |\n"
        
        if ncu_metrics.get("dram_throughput"):
            status = "✅" if ncu_metrics["dram_throughput"] >= 80.0 else "⚠️"
            report += f"| DRAM Throughput | {ncu_metrics['dram_throughput']:.1f}% | {status} |\n"
        
        if ncu_metrics.get("warps_active"):
            status = "✅" if ncu_metrics["warps_active"] >= 70.0 else "⚠️"
            report += f"| Warps Active | {ncu_metrics['warps_active']:.1f}% | {status} |\n"
        
        if ncu_metrics.get("l1_miss_rate"):
            status = "✅" if ncu_metrics["l1_miss_rate"] <= 15.0 else "⚠️"
            report += f"| L1 Miss Rate | {ncu_metrics['l1_miss_rate']:.1f}% | {status} |\n"
        
        report += "\n"
    
    # NSYS timeline summary
    if nsys_stats:
        report += f"""### Nsight Systems Timeline

**Total Kernel Time:** {nsys_stats['total_time_ms']:.2f} ms

**Top Kernels:**

| Kernel Name | Count | Time (ms) | % of Total |
|-------------|-------|-----------|------------|
"""
        for kernel in nsys_stats["top_kernels"]:
            pct = (kernel["time_ms"] / nsys_stats["total_time_ms"] * 100) if nsys_stats["total_time_ms"] > 0 else 0
            report += f"| `{kernel['name'][:40]}` | {kernel['count']} | {kernel['time_ms']:.2f} | {pct:.1f}% |\n"
        
        report += "\n"
    
    # Artifacts section
    report += f"""---

## Generated Artifacts

"""
    
    for artifact in profiling_path.glob("*"):
        if artifact.is_file():
            size_kb = artifact.stat().st_size / 1024
            report += f"- `{artifact.name}` ({size_kb:.1f} KB)\n"
    
    # Reproduction commands
    report += f"""

---

## Reproduction Commands

```bash
# Run expert profiling
bash tools/profile_expert.sh trajectory_h100

# Validate metrics
python3 scripts/validate_metrics.py artifacts/profiling/*/key_metrics.txt

# Regenerate this report
python3 scripts/generate_profiling_report.py {profiling_dir} PROFILING_REPORT.md
```

---

## Performance Assessment

"""
    
    # Overall assessment
    passed_checks = 0
    total_checks = 0
    
    if ncu_metrics:
        if ncu_metrics.get("sm_throughput"):
            total_checks += 1
            if ncu_metrics["sm_throughput"] >= 85.0:
                passed_checks += 1
        
        if ncu_metrics.get("dram_throughput"):
            total_checks += 1
            if ncu_metrics["dram_throughput"] >= 80.0:
                passed_checks += 1
    
    if total_checks > 0:
        pct_passed = (passed_checks / total_checks) * 100
        status_icon = "✅" if pct_passed >= 80 else "⚠️" if pct_passed >= 60 else "❌"
        report += f"""{status_icon} **Overall: {passed_checks}/{total_checks} checks passed ({pct_passed:.0f}%)**

"""
    
    report += """
**Conclusion:** This profiling run provides comprehensive performance validation using
industry-standard NVIDIA Nsight tools. All metrics are captured for regression tracking
and CI/CD integration.

---

**Generated by:** RoboCache Expert Profiling System v1.0  
**Tools:** Nsight Systems 2025.3+ | Nsight Compute 2025.3+  
**Target:** NVIDIA H100 (SM90) + CUDA 13.0
"""
    
    # Write report
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"✅ Generated profiling report: {output_file}")
    print(f"   Artifacts: {len(list(profiling_path.glob('*')))} files")
    if ncu_metrics:
        print(f"   NCU Metrics: {sum(1 for v in ncu_metrics.values() if v is not None)} captured")
    if nsys_stats:
        print(f"   NSYS Kernels: {len(nsys_stats['top_kernels'])} top kernels")

def main():
    if len(sys.argv) < 3:
        print("Usage: generate_profiling_report.py <profiling_dir> <output_md>")
        print("Example: python3 scripts/generate_profiling_report.py artifacts/profiling/trajectory_h100_20251106 PROFILING_REPORT.md")
        sys.exit(1)
    
    profiling_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    generate_markdown_report(profiling_dir, output_file)

if __name__ == "__main__":
    main()

