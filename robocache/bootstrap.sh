#!/bin/bash
#
# RoboCache Bootstrap Script
# Unified entrypoint for environment setup, validation, and builds
#
# Usage: ./bootstrap.sh [--dev|--runtime|--test]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Banner
echo "=================================================="
echo "  RoboCache Bootstrap"
echo "  Production GPU Acceleration for Robot Learning"
echo "=================================================="
echo ""

# Parse arguments
MODE="${1:-runtime}"

case "$MODE" in
    --dev|dev)
        MODE="dev"
        log_info "Development mode selected"
        ;;
    --runtime|runtime)
        MODE="runtime"
        log_info "Runtime mode selected"
        ;;
    --test|test)
        MODE="test"
        log_info "Test mode selected"
        ;;
    *)
        log_error "Unknown mode: $MODE. Use --dev, --runtime, or --test"
        ;;
esac

# Step 1: Validate GPU environment
log_info "Step 1/6: Validating GPU environment"

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Install NVIDIA drivers first."
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    log_error "No NVIDIA GPUs detected"
fi

log_info "Found $GPU_COUNT GPU(s):"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    log_info "CUDA Toolkit: $CUDA_VERSION"
    
    # Require CUDA 12.1+
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)
    
    if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 1 ]); then
        log_warn "CUDA $CUDA_VERSION detected. Recommended: 13.0+"
    fi
else
    log_warn "nvcc not found. JIT compilation will be used."
fi

# Step 2: Initialize submodules
log_info "Step 2/6: Initializing Git submodules"

if [ -d ".git" ]; then
    git submodule update --init --recursive
    log_info "Submodules updated"
else
    log_warn "Not a Git repository. Skipping submodule init."
fi

# Step 3: Install Python dependencies
log_info "Step 3/6: Installing Python dependencies"

if ! command -v python3 &> /dev/null; then
    log_error "python3 not found. Install Python 3.10+ first."
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    log_warn "Not in a virtual environment. Consider using: python3 -m venv venv && source venv/bin/activate"
fi

# Install dependencies
if [ "$MODE" == "dev" ]; then
    pip install -e ".[dev]" || log_error "Failed to install dev dependencies"
elif [ "$MODE" == "test" ]; then
    pip install -e ".[test]" || log_error "Failed to install test dependencies"
else
    pip install -e . || log_error "Failed to install runtime dependencies"
fi

log_info "Python dependencies installed"

# Step 4: Validate PyTorch + CUDA
log_info "Step 4/6: Validating PyTorch + CUDA"

python3 << EOF || log_error "PyTorch + CUDA validation failed"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    # Test simple CUDA operation
    x = torch.rand(1000, 1000, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print("✓ CUDA operation successful")
else:
    print("WARNING: CUDA not available in PyTorch")
    exit(1)
EOF

# Step 5: Run pre-commit hooks (dev mode only)
if [ "$MODE" == "dev" ]; then
    log_info "Step 5/6: Setting up pre-commit hooks"
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        log_info "Pre-commit hooks installed"
    else
        log_warn "pre-commit not found. Install: pip install pre-commit"
    fi
else
    log_info "Step 5/6: Skipping pre-commit (not in dev mode)"
fi

# Step 6: Build/test based on mode
log_info "Step 6/6: Mode-specific setup"

case "$MODE" in
    dev)
        log_info "Development environment ready"
        log_info "Run tests: pytest tests/"
        log_info "Profile: nsys profile -o trace python benchmarks/training_loop_h100.py"
        ;;
    runtime)
        log_info "Runtime environment ready"
        log_info "Import RoboCache: python -c 'import robocache; print(robocache.__version__)'"
        ;;
    test)
        log_info "Running test suite..."
        pytest tests/ -v --tb=short || log_error "Tests failed"
        log_info "All tests passed ✓"
        ;;
esac

# Summary
echo ""
echo "=================================================="
echo "  Bootstrap Complete!"
echo "=================================================="
echo ""
echo "Environment Summary:"
echo "  Mode: $MODE"
echo "  GPUs: $GPU_COUNT"
echo "  CUDA: ${CUDA_VERSION:-N/A (JIT mode)}"
echo "  Python: $PYTHON_VERSION"
echo ""
echo "Next Steps:"
case "$MODE" in
    dev)
        echo "  - Edit code in kernels/"
        echo "  - Run tests: pytest tests/"
        echo "  - Profile: nsys profile python benchmarks/*.py"
        ;;
    runtime)
        echo "  - Use in your code: import robocache"
        echo "  - See examples/: ROS 2, cuRobo, Isaac Sim"
        ;;
    test)
        echo "  - All tests passed!"
        echo "  - Ready for production deployment"
        ;;
esac
echo ""

