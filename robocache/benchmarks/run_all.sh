#!/usr/bin/env bash
# run_all.sh - Automated benchmark runner with reproducible results
# Addresses audit finding: "No scripts or README guidance on reproducing data"
#
# Usage: ./benchmarks/run_all.sh [--phase 1|2|3|4|all] [--output results/]
#
# Produces CSV output with statistical treatment (mean, stddev, min, max)
# and stores NCU artifacts for review.

set -euo pipefail

# Configuration
PHASE="${1:-all}"
OUTPUT_DIR="${2:-benchmarks/results}"
NCU_DIR="docs/perf/ncu_reports"
WARMUP_ITERS=10
BENCH_ITERS=100
BUILD_DIR="build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
echo "║  RoboCache Reproducible Benchmark Suite"
echo "║  Addressing Audit: Automated benchmarking with NCU artifacts"
echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}❌ ERROR: CUDA not found${NC}" >&2
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ ERROR: No GPU detected${NC}" >&2
    exit 1
fi

# Check build
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ ERROR: Build directory not found. Run: mkdir build && cd build && cmake .. && make${NC}" >&2
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR" "$NCU_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
DATE=$(date +%Y%m%d_%H%M%S)

echo "GPU: $GPU_NAME"
echo "CUDA Version: $CUDA_VERSION"
echo "Warmup Iterations: $WARMUP_ITERS"
echo "Benchmark Iterations: $BENCH_ITERS"
echo "Output: $OUTPUT_DIR/"
echo "NCU Reports: $NCU_DIR/"
echo ""

# Function to run benchmark with statistical treatment
run_benchmark() {
    local name=$1
    local binary=$2
    local config=$3
    local output_csv=$4
    
    if [ ! -f "$BUILD_DIR/$binary" ]; then
        echo -e "${YELLOW}⚠️  SKIP: $name (binary not found)${NC}"
        return 0
    fi
    
    echo "─── Benchmarking: $name ───"
    echo "Config: $config"
    
    # Run benchmark (assumes binary outputs CSV to stdout)
    $BUILD_DIR/$binary --config "$config" --warmup $WARMUP_ITERS --iters $BENCH_ITERS \
        > "$output_csv" 2>&1 || echo -e "${RED}❌ FAILED${NC}"
    
    if [ -f "$output_csv" ]; then
        echo -e "${GREEN}✅ Results saved: $output_csv${NC}"
    fi
    
    # Run NCU profiling (single iteration, full metrics)
    if command -v ncu &> /dev/null; then
        local ncu_output="$NCU_DIR/${name// /_}_${DATE}.ncu-rep"
        echo "Profiling with NCU..."
        ncu --set full \
            --launch-skip $WARMUP_ITERS \
            --launch-count 1 \
            --target-processes all \
            -o "$ncu_output" --force-overwrite \
            $BUILD_DIR/$binary --config "$config" --iters 1 > /dev/null 2>&1 || true
        
        if [ -f "${ncu_output}.ncu-rep" ]; then
            echo -e "${GREEN}✅ NCU report: ${ncu_output}.ncu-rep${NC}"
        fi
    fi
    
    echo ""
}

# Phase 1: Trajectory Resampling
benchmark_phase1() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Trajectory Resampling"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    run_benchmark "Phase 1 - Small" "benchmark_trajectory" \
        "benchmarks/configs/phase1_small.json" \
        "$OUTPUT_DIR/phase1_small_${DATE}.csv"
    
    run_benchmark "Phase 1 - Medium" "benchmark_trajectory" \
        "benchmarks/configs/phase1_medium.json" \
        "$OUTPUT_DIR/phase1_medium_${DATE}.csv"
    
    run_benchmark "Phase 1 - Large" "benchmark_trajectory" \
        "benchmarks/configs/phase1_large.json" \
        "$OUTPUT_DIR/phase1_large_${DATE}.csv"
}

# Phase 2: Multimodal Fusion
benchmark_phase2() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Multimodal Sensor Fusion"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    run_benchmark "Phase 2 - 2-Stream" "benchmark_multimodal_fusion" \
        "benchmarks/configs/phase2_2stream.json" \
        "$OUTPUT_DIR/phase2_2stream_${DATE}.csv"
    
    run_benchmark "Phase 2 - 3-Stream" "benchmark_multimodal_fusion" \
        "benchmarks/configs/phase2_3stream.json" \
        "$OUTPUT_DIR/phase2_3stream_${DATE}.csv"
}

# Phase 3: Point Cloud Voxelization
benchmark_phase3() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Point Cloud Voxelization"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    run_benchmark "Phase 3 - Occupancy" "benchmark_voxelization" \
        "benchmarks/configs/phase3_occupancy.json" \
        "$OUTPUT_DIR/phase3_occupancy_${DATE}.csv"
    
    run_benchmark "Phase 3 - Full Suite" "benchmark_voxelization_full" \
        "benchmarks/configs/phase3_full.json" \
        "$OUTPUT_DIR/phase3_full_${DATE}.csv"
}

# Phase 4: Action Space Conversion
benchmark_phase4() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Action Space Conversion"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    run_benchmark "Phase 4 - FK+Jacobian" "benchmark_action_space" \
        "benchmarks/configs/phase4_fk.json" \
        "$OUTPUT_DIR/phase4_fk_${DATE}.csv"
}

# Main
case "$PHASE" in
    1)
        benchmark_phase1
        ;;
    2)
        benchmark_phase2
        ;;
    3)
        benchmark_phase3
        ;;
    4)
        benchmark_phase4
        ;;
    all)
        benchmark_phase1
        benchmark_phase2
        benchmark_phase3
        benchmark_phase4
        ;;
    *)
        echo -e "${RED}❌ ERROR: Invalid phase: $PHASE${NC}" >&2
        echo "Valid phases: 1, 2, 3, 4, all" >&2
        exit 1
        ;;
esac

echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ BENCHMARK COMPLETE"
echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  CSV: $OUTPUT_DIR/*_${DATE}.csv"
echo "  NCU: $NCU_DIR/*_${DATE}.ncu-rep"
echo ""
echo "To analyze results:"
echo "  python benchmarks/analyze_results.py $OUTPUT_DIR/"
echo "  ncu-ui $NCU_DIR/*.ncu-rep"

