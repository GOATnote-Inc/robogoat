#!/usr/bin/env bash
#
# One-click profiling script for Nsight Systems and Nsight Compute.
# Generates reproducible profiling artifacts for CI.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="$REPO_ROOT/artifacts"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] TARGET

Profile RoboCache operations with Nsight Systems and/or Nsight Compute.

TARGETS:
    trajectory      Profile trajectory resampling
    multimodal      Profile multimodal fusion
    voxelize        Profile voxelization
    train           Profile end-to-end training loop
    all             Profile all operations

OPTIONS:
    -t, --tool TOOL     Profiling tool: nsys, ncu, or both (default: both)
    -o, --output DIR    Output directory (default: artifacts/)
    -h, --help          Show this help message

EXAMPLES:
    # Profile trajectory resampling with both tools
    $0 trajectory

    # Profile training with only Nsight Systems
    $0 --tool nsys train

    # Profile all operations with Nsight Compute
    $0 --tool ncu all

ENVIRONMENT VARIABLES:
    NSYS_PATH           Path to nsys binary (default: nsys in PATH)
    NCU_PATH            Path to ncu binary (default: ncu in PATH)

EOF
    exit 1
}

# Parse arguments
TOOL="both"
OUTPUT_DIR="$ARTIFACTS_DIR"
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tool)
            TOOL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    echo -e "${RED}Error: TARGET required${NC}"
    usage
fi

# Check tools
NSYS="${NSYS_PATH:-nsys}"
NCU="${NCU_PATH:-ncu}"

if [[ "$TOOL" == "nsys" ]] || [[ "$TOOL" == "both" ]]; then
    if ! command -v "$NSYS" &> /dev/null; then
        echo -e "${RED}Error: nsys not found. Install Nsight Systems or set NSYS_PATH${NC}"
        exit 1
    fi
fi

if [[ "$TOOL" == "ncu" ]] || [[ "$TOOL" == "both" ]]; then
    if ! command -v "$NCU" &> /dev/null; then
        echo -e "${RED}Error: ncu not found. Install Nsight Compute or set NCU_PATH${NC}"
        exit 1
    fi
fi

# Create output directories
mkdir -p "$OUTPUT_DIR/nsys" "$OUTPUT_DIR/ncu"

# Timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RoboCache Profiling${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Target: ${GREEN}$TARGET${NC}"
echo -e "Tool: ${GREEN}$TOOL${NC}"
echo -e "Output: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

profile_nsys() {
    local target=$1
    local python_script="$REPO_ROOT/scripts/profile_${target}.py"
    local output_file="$OUTPUT_DIR/nsys/${target}_${TIMESTAMP}"
    
    echo -e "${BLUE}Running Nsight Systems...${NC}"
    
    "$NSYS" profile \
        -t cuda,nvtx,cublas,cudnn \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        -o "$output_file" \
        python "$python_script"
    
    echo -e "${GREEN}✅ Nsight Systems report: ${output_file}.nsys-rep${NC}"
    
    # Generate stats summary
    "$NSYS" stats --report cuda_gpu_kern_sum "$output_file.nsys-rep" > "${output_file}_kernels.txt"
    echo -e "${GREEN}✅ Kernel summary: ${output_file}_kernels.txt${NC}"
}

profile_ncu() {
    local target=$1
    local python_script="$REPO_ROOT/scripts/profile_${target}.py"
    local output_file="$OUTPUT_DIR/ncu/${target}_${TIMESTAMP}"
    
    echo -e "${BLUE}Running Nsight Compute...${NC}"
    
    "$NCU" \
        --set full \
        --target-processes all \
        --force-overwrite \
        --export "$output_file" \
        python "$python_script"
    
    echo -e "${GREEN}✅ Nsight Compute report: ${output_file}.ncu-rep${NC}"
    
    # Generate metrics CSV
    "$NCU" --import "$output_file.ncu-rep" --csv > "${output_file}_metrics.csv"
    echo -e "${GREEN}✅ Metrics CSV: ${output_file}_metrics.csv${NC}"
}

# Profile target(s)
if [[ "$TARGET" == "all" ]]; then
    TARGETS=("trajectory" "multimodal" "voxelize")
else
    TARGETS=("$TARGET")
fi

for t in "${TARGETS[@]}"; do
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Profiling: $t${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [[ "$TOOL" == "nsys" ]] || [[ "$TOOL" == "both" ]]; then
        profile_nsys "$t"
    fi
    
    if [[ "$TOOL" == "ncu" ]] || [[ "$TOOL" == "both" ]]; then
        profile_ncu "$t"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Profiling Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Artifacts saved to: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

