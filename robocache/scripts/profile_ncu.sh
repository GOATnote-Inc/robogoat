#!/usr/bin/env bash
# profile_ncu.sh
# Production-grade NCU profiling script for RoboCache kernels
# Captures key performance metrics with clear interpretation
#
# Usage: ./scripts/profile_ncu.sh [phase] [mode]
#   phase: 1, 2, 3, 4, or "all" (default: all)
#   mode: "fast" or "full" (default: fast)
#
# Requirements:
# - CUDA Toolkit with NCU installed
# - RoboCache built in build/ directory
# - H100/A100 GPU

set -euo pipefail

# Configuration
BUILD_DIR="${BUILD_DIR:-build}"
PHASE="${1:-all}"
MODE="${2:-fast}"
WARMUP_LAUNCHES=100
PROFILE_LAUNCHES=1
OUTPUT_DIR="ncu_reports"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
echo "║  RoboCache NCU Profiling"
echo "║  Kernel Performance Analysis"
echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check NCU
if ! command -v ncu &> /dev/null; then
    echo "❌ ERROR: Nsight Compute (ncu) not found in PATH" >&2
    echo "Install CUDA Toolkit or load module: module load cuda" >&2
    exit 1
fi

# Check build
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ ERROR: Build directory not found: $BUILD_DIR" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "Unknown")
echo "GPU: $GPU_NAME"
echo "NCU Version: $(ncu --version | head -1)"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR/"
echo ""

# Fast metrics (for CI and quick checks)
FAST_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_average_branch_targets_threads_uniform.pct"

# Profile function
profile_kernel() {
    local name=$1
    local binary=$2
    local args=${3:-""}
    
    if [ ! -f "$BUILD_DIR/$binary" ]; then
        echo -e "${YELLOW}⚠️  SKIP: $name (binary not found)${NC}"
        return 0
    fi
    
    echo "─── Profiling: $name ───"
    
    # Warmup
    echo "Warming up..."
    $BUILD_DIR/$binary $args > /dev/null 2>&1 || true
    
    # Profile
    local output_file="$OUTPUT_DIR/${name// /_}.ncu-rep"
    
    if [ "$MODE" == "full" ]; then
        ncu --set full \
            --launch-skip $WARMUP_LAUNCHES \
            --launch-count $PROFILE_LAUNCHES \
            --target-processes all \
            -o "$output_file" --force-overwrite \
            $BUILD_DIR/$binary $args
    else
        ncu --metrics "$FAST_METRICS" \
            --launch-skip $WARMUP_LAUNCHES \
            --launch-count $PROFILE_LAUNCHES \
            --target-processes all \
            -o "$output_file" --force-overwrite \
            $BUILD_DIR/$binary $args
    fi
    
    echo -e "${GREEN}✅ Saved: $output_file${NC}"
    echo ""
}

# Profile Phase 1: Trajectory Resampling
profile_phase1() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Trajectory Resampling"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Note: Would need dedicated profiling binary
    echo -e "${YELLOW}⚠️  TODO: Create dedicated profiling binary for Phase 1${NC}"
    echo ""
}

# Profile Phase 2: Multimodal Fusion
profile_phase2() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Multimodal Sensor Fusion"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    profile_kernel "Phase 2 - Multimodal Fusion" "benchmark_multimodal_fusion" ""
}

# Profile Phase 3: Point Cloud Voxelization
profile_phase3() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Point Cloud Voxelization"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    profile_kernel "Phase 3 - Occupancy Voxelization" "benchmark_voxelization" ""
    profile_kernel "Phase 3 - Full Voxelization Suite" "benchmark_voxelization_full" ""
}

# Profile Phase 4: Action Space Conversion
profile_phase4() {
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Action Space Conversion"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    profile_kernel "Phase 4 - Action Space" "benchmark_action_space" ""
}

# Main
main() {
    case "$PHASE" in
        1)
            profile_phase1
            ;;
        2)
            profile_phase2
            ;;
        3)
            profile_phase3
            ;;
        4)
            profile_phase4
            ;;
        all)
            profile_phase1
            profile_phase2
            profile_phase3
            profile_phase4
            ;;
        *)
            echo "❌ ERROR: Invalid phase: $PHASE" >&2
            echo "Valid phases: 1, 2, 3, 4, all" >&2
            exit 1
            ;;
    esac
    
    echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ PROFILING COMPLETE"
    echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Reports saved in: $OUTPUT_DIR/"
    echo ""
    echo "View reports:"
    echo "  ncu-ui $OUTPUT_DIR/*.ncu-rep"
    echo ""
    echo "Quick analysis:"
    echo "  ncu --import $OUTPUT_DIR/Phase_3_-_Occupancy_Voxelization.ncu-rep --page details"
}

main

