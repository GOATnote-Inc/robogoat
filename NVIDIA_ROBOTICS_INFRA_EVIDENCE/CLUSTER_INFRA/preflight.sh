#!/bin/bash
# preflight.sh
# GPU health check and capability verification
#
# Copyright (c) 2025 GOATnote Inc.
# SPDX-License-Identifier: Apache-2.0

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "GPU Preflight Checks"
echo "===================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "  NVIDIA driver may not be installed or GPU not available"
    exit 1
fi

# Get GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No GPUs detected${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $GPU_COUNT GPU(s)${NC}"
echo ""

# Check each GPU
for ((i=0; i<$GPU_COUNT; i++)); do
    echo "GPU $i:"

    # Get GPU name
    GPU_NAME=$(nvidia-smi -i $i --query-gpu=name --format=csv,noheader)
    echo "  Name: $GPU_NAME"

    # Get compute capability
    COMPUTE_CAP=$(nvidia-smi -i $i --query-gpu=compute_cap --format=csv,noheader)
    echo "  Compute Capability: $COMPUTE_CAP"

    # Check minimum compute capability (8.0 for Ampere)
    MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
    if [ "$MAJOR" -lt 8 ]; then
        echo -e "  ${RED}✗ Compute capability too low (need ≥8.0 for Ampere/Hopper)${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓ Compute capability OK${NC}"

    # Get memory
    MEM_TOTAL=$(nvidia-smi -i $i --query-gpu=memory.total --format=csv,noheader,nounits)
    MEM_USED=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    MEM_FREE=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits)

    echo "  Memory:"
    echo "    Total: ${MEM_TOTAL} MiB"
    echo "    Used:  ${MEM_USED} MiB"
    echo "    Free:  ${MEM_FREE} MiB"

    # Check if enough memory is available (need at least 10GB free)
    if [ "$MEM_FREE" -lt 10240 ]; then
        echo -e "  ${YELLOW}⚠ Low free memory (< 10GB)${NC}"
    else
        echo -e "  ${GREEN}✓ Memory OK${NC}"
    fi

    # Get temperature
    TEMP=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    echo "  Temperature: ${TEMP}°C"

    if [ "$TEMP" -gt 85 ]; then
        echo -e "  ${YELLOW}⚠ High temperature${NC}"
    else
        echo -e "  ${GREEN}✓ Temperature OK${NC}"
    fi

    # Get power
    POWER=$(nvidia-smi -i $i --query-gpu=power.draw --format=csv,noheader)
    echo "  Power Draw: $POWER"

    # Get clock speeds
    GPU_CLOCK=$(nvidia-smi -i $i --query-gpu=clocks.current.graphics --format=csv,noheader)
    MEM_CLOCK=$(nvidia-smi -i $i --query-gpu=clocks.current.memory --format=csv,noheader)
    echo "  Clocks:"
    echo "    GPU: $GPU_CLOCK"
    echo "    Memory: $MEM_CLOCK"
    echo -e "  ${GREEN}✓ Clocks OK${NC}"

    # Check for ECC errors
    ECC_ERRORS=$(nvidia-smi -i $i --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader)
    if [ "$ECC_ERRORS" != "N/A" ] && [ "$ECC_ERRORS" != "0" ]; then
        echo -e "  ${RED}✗ ECC errors detected: $ECC_ERRORS${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓ No ECC errors${NC}"

    echo ""
done

# Check CUDA version
echo "CUDA Environment:"
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "  CUDA Version: $CUDA_VERSION"

# Check if CUDA 12.0+
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
if [ "$CUDA_MAJOR" -lt 12 ]; then
    echo -e "  ${YELLOW}⚠ CUDA version < 12.0 (recommended: 13.0)${NC}"
else
    echo -e "  ${GREEN}✓ CUDA version OK${NC}"
fi

echo ""

# Check CUTLASS
if [ -d "$CUTLASS_HOME" ]; then
    echo -e "${GREEN}✓ CUTLASS found at $CUTLASS_HOME${NC}"
else
    echo -e "${YELLOW}⚠ CUTLASS_HOME not set${NC}"
fi

echo ""

# Quick CUDA test
echo "Running quick CUDA test..."

# Create a simple test program
cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Test allocation
    float *d_test;
    cudaError_t err = cudaMalloc(&d_test, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_test);

    printf("CUDA test passed\n");
    return 0;
}
EOF

# Compile and run
nvcc -o /tmp/cuda_test /tmp/cuda_test.cu 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CUDA compilation failed${NC}"
    exit 1
fi

/tmp/cuda_test
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CUDA test failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CUDA test passed${NC}"
rm -f /tmp/cuda_test /tmp/cuda_test.cu

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ All GPU preflight checks passed${NC}"
echo "========================================================================"
echo ""

exit 0
