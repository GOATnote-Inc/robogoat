#!/bin/bash
set -e

echo "=== Building RoboCache CUDA Extension ==="

cd "$(dirname "$0")/.."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit first."
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release)"

# Check for PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not installed. Run: pip install torch"
    exit 1
fi

echo "✓ PyTorch found: $(python3 -c 'import torch; print(torch.__version__)')"

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info csrc/**/*.o 2>/dev/null || true

# Build extension
echo ""
echo "Building CUDA extension..."
python3 setup.py build_ext --inplace

# Test import
echo ""
echo "Testing import..."
python3 -c "import robocache; print(f'✓ RoboCache {robocache.__version__} imported')"
python3 -c "import robocache; print(f'✓ CUDA kernels: {robocache.is_cuda_available()}')"

echo ""
echo "=== Build Complete ==="
echo "Run: python3 -c 'import robocache; robocache.self_test()'"

