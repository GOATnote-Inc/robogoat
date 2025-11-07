#!/bin/bash
set -e

echo "RoboCache Environment Verification"
echo "==================================="

# Python
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: PyTorch not installed"
    exit 1
}

# CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python3 -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'Compute Capability: {cc[0]}.{cc[1]}')"
    
    # Check if A100 or H100
    CC=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}')")
    if [ "$CC" = "80" ]; then
        echo "✓ A100 detected (sm_80)"
    elif [ "$CC" = "90" ]; then
        echo "✓ H100 detected (sm_90)"
    else
        echo "⚠ Non-standard GPU (expected A100 or H100)"
    fi
fi

# RoboCache
python3 -c "import robocache; print(f'RoboCache: {robocache.__version__}')" || {
    echo "RoboCache: Not installed"
}

echo ""
echo "✓ Environment OK"

