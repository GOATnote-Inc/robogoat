#!/bin/bash
set -euo pipefail

echo "RoboCache Wheel Build Script"
echo "============================="

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "CUDA Version: $CUDA_VERSION"

# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: PyTorch not installed"
    exit 1
}

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "WARNING: CUDA not available in PyTorch"
}

# Build
echo ""
echo "Building wheel..."
cd "$(dirname "$0")/.."
python3 setup.py bdist_wheel

echo ""
echo "✓ Wheel built successfully"
ls -lh dist/*.whl

