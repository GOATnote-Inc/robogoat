#!/bin/bash
# profile_with_ncu.sh
# Comprehensive NCU profiling for multimodal fusion kernels
#
# Copyright (c) 2025 GOATnote Inc.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "========================================================================"
echo "NCU Profiling - Multimodal Fusion Kernels"
echo "========================================================================"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu (NVIDIA Nsight Compute) not found"
    echo "Please install NVIDIA Nsight Compute from:"
    echo "  https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# Create output directory
OUTPUT_DIR="ncu_reports"
mkdir -p "$OUTPUT_DIR"

# Build path to benchmark executable
BENCHMARK_EXE="${1:-../../robocache/benchmarks/multimodal/benchmark_multimodal_fusion}"

if [ ! -f "$BENCHMARK_EXE" ]; then
    echo "Error: Benchmark executable not found at $BENCHMARK_EXE"
    echo "Usage: $0 [path_to_benchmark_exe]"
    exit 1
fi

echo "Benchmark executable: $BENCHMARK_EXE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run NCU profiling
profile_kernel() {
    local kernel_name="$1"
    local output_name="$2"
    local sections="$3"

    echo "----------------------------------------"
    echo "Profiling: $kernel_name"
    echo "Output: $OUTPUT_DIR/$output_name"
    echo "----------------------------------------"

    ncu \
        --set "$sections" \
        --kernel-name "$kernel_name" \
        --launch-skip 10 \
        --launch-count 5 \
        --cache-control=all \
        --clock-control=base \
        --export "$OUTPUT_DIR/$output_name" \
        --force-overwrite \
        "$BENCHMARK_EXE"

    echo "✓ Profile saved: $OUTPUT_DIR/$output_name.ncu-rep"
    echo ""
}

# 1. Full profiling - standard kernel
echo ""
echo "1. Full Profiling - Standard Kernel"
profile_kernel \
    "multimodal_fusion_kernel" \
    "standard_kernel_full" \
    "full"

# 2. Full profiling - optimized kernel
echo ""
echo "2. Full Profiling - Optimized Kernel"
profile_kernel \
    "multimodal_fusion_optimized_kernel" \
    "optimized_kernel_full" \
    "full"

# 3. Memory analysis
echo ""
echo "3. Memory Bandwidth Analysis"
ncu \
    --set full \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --launch-skip 10 \
    --launch-count 5 \
    --export "$OUTPUT_DIR/memory_analysis" \
    --force-overwrite \
    "$BENCHMARK_EXE"

echo "✓ Memory analysis saved: $OUTPUT_DIR/memory_analysis.ncu-rep"
echo ""

# 4. Compute utilization
echo ""
echo "4. Compute Utilization Analysis"
ncu \
    --set full \
    --section ComputeWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    --launch-skip 10 \
    --launch-count 5 \
    --export "$OUTPUT_DIR/compute_analysis" \
    --force-overwrite \
    "$BENCHMARK_EXE"

echo "✓ Compute analysis saved: $OUTPUT_DIR/compute_analysis.ncu-rep"
echo ""

# 5. Warp state analysis
echo ""
echo "5. Warp State Analysis"
ncu \
    --set full \
    --section WarpStateStats \
    --section SchedulerStats \
    --launch-skip 10 \
    --launch-count 5 \
    --export "$OUTPUT_DIR/warp_analysis" \
    --force-overwrite \
    "$BENCHMARK_EXE"

echo "✓ Warp analysis saved: $OUTPUT_DIR/warp_analysis.ncu-rep"
echo ""

# 6. Roofline analysis
echo ""
echo "6. Roofline Analysis"
ncu \
    --set roofline \
    --launch-skip 10 \
    --launch-count 5 \
    --export "$OUTPUT_DIR/roofline" \
    --force-overwrite \
    "$BENCHMARK_EXE"

echo "✓ Roofline saved: $OUTPUT_DIR/roofline.ncu-rep"
echo ""

# 7. Generate metrics summary
echo ""
echo "7. Generating Metrics Summary"

# Extract key metrics from NCU reports
ncu \
    --import "$OUTPUT_DIR/optimized_kernel_full.ncu-rep" \
    --page raw \
    --csv \
    > "$OUTPUT_DIR/metrics_summary.csv"

echo "✓ Metrics summary saved: $OUTPUT_DIR/metrics_summary.csv"
echo ""

# 8. Create comparison report
echo ""
echo "8. Creating Comparison Report"

cat > "$OUTPUT_DIR/comparison.md" << 'EOF'
# NCU Profiling Comparison Report

## Optimized Kernel Metrics

### Key Performance Indicators

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| SM Occupancy | TBD | >85% | TBD |
| Memory Bandwidth | TBD | >2.5 TB/s | TBD |
| Achieved TFLOPS | TBD | >70 TFLOPS | TBD |
| Warp Execution Efficiency | TBD | >90% | TBD |

### Memory Analysis

- **Global Load Efficiency:** TBD
- **Global Store Efficiency:** TBD
- **L2 Hit Rate:** TBD
- **DRAM Utilization:** TBD

### Compute Analysis

- **SM Active Cycles:** TBD
- **Warp Occupancy:** TBD
- **Eligible Warps Per Cycle:** TBD

## How to View Reports

1. Open in Nsight Compute GUI:
   ```bash
   ncu-ui ncu_reports/optimized_kernel_full.ncu-rep
   ```

2. Compare kernels:
   ```bash
   ncu-ui ncu_reports/standard_kernel_full.ncu-rep ncu_reports/optimized_kernel_full.ncu-rep
   ```

## Next Steps

1. Review SM occupancy - target >85%
2. Check memory bandwidth utilization - target >90% of peak
3. Analyze warp stalls and identify bottlenecks
4. Iterate on kernel optimizations based on findings
EOF

echo "✓ Comparison report template: $OUTPUT_DIR/comparison.md"
echo ""

# Summary
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_DIR/standard_kernel_full.ncu-rep"
echo "  - $OUTPUT_DIR/optimized_kernel_full.ncu-rep"
echo "  - $OUTPUT_DIR/memory_analysis.ncu-rep"
echo "  - $OUTPUT_DIR/compute_analysis.ncu-rep"
echo "  - $OUTPUT_DIR/warp_analysis.ncu-rep"
echo "  - $OUTPUT_DIR/roofline.ncu-rep"
echo "  - $OUTPUT_DIR/metrics_summary.csv"
echo "  - $OUTPUT_DIR/comparison.md"
echo ""
echo "To view reports:"
echo "  ncu-ui $OUTPUT_DIR/optimized_kernel_full.ncu-rep"
echo ""
echo "To extract specific metric:"
echo "  ncu --import $OUTPUT_DIR/optimized_kernel_full.ncu-rep --csv --page details"
echo ""
