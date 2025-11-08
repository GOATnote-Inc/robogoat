#!/bin/bash
# Automated CI pipeline for RoboCache validation on H100
# Builds CUDA extension, runs pytest suite, generates NCU reports

set -e  # Exit on error

echo "=========================================="
echo "RoboCache CI: Build and Test Pipeline"
echo "=========================================="

# Configuration
WORKSPACE=${WORKSPACE:-/workspace/robocache}
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.0}
RESULTS_DIR=${RESULTS_DIR:-profiling/artifacts/ci}

export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

cd $WORKSPACE

# Check CUDA availability
echo ""
echo "📊 System Configuration:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | grep "release"
python3 --version
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -q pytest numpy torch 2>&1 | tail -5

# Create results directory
mkdir -p $RESULTS_DIR

# Run correctness tests
echo ""
echo "🧪 Running correctness tests..."
python3 -m pytest tests/test_correctness.py -v --tb=short > $RESULTS_DIR/pytest_correctness.log 2>&1 || {
    echo "❌ Correctness tests failed"
    cat $RESULTS_DIR/pytest_correctness.log
    exit 1
}
echo "✅ Correctness tests passed"

# Run benchmark harness
echo ""
echo "📈 Running benchmark harness..."
python3 benchmarks/benchmark_harness.py > $RESULTS_DIR/benchmark_stdout.log 2>&1 || {
    echo "❌ Benchmark harness failed"
    cat $RESULTS_DIR/benchmark_stdout.log
    exit 1
}
cp benchmarks/results/h100_validated.json $RESULTS_DIR/ 2>/dev/null || true
echo "✅ Benchmarks completed"

# Generate NCU profile
echo ""
echo "🔍 Generating NCU profile..."
cat > /tmp/ncu_profile_script.py << 'NCU_SCRIPT'
import torch
from torch.utils.cpp_extension import load

robocache = load(
    name='robocache_cuda_optimized',
    sources=[
        'kernels/cutlass/trajectory_resample_optimized_v2.cu',
        'kernels/cutlass/trajectory_resample_optimized_v2_torch.cu',
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False
)

batch, src_len, tgt_len, dim = 64, 4096, 1024, 32
data = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device='cuda')
src_t = torch.linspace(0, 1, src_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()
tgt_t = torch.linspace(0, 1, tgt_len, device='cuda').unsqueeze(0).expand(batch, -1).contiguous()

# Warmup
for _ in range(5):
    result = robocache.resample_trajectories(data, src_t, tgt_t)
torch.cuda.synchronize()

# Profile
result = robocache.resample_trajectories(data, src_t, tgt_t)
torch.cuda.synchronize()
NCU_SCRIPT

$CUDA_PATH/bin/ncu \
  --set full \
  --target-processes all \
  --export $RESULTS_DIR/ncu_trajectory_resample \
  python3 /tmp/ncu_profile_script.py > $RESULTS_DIR/ncu_stdout.log 2>&1 || {
    echo "⚠️  NCU profiling failed (non-fatal)"
}

# Export NCU metrics to CSV
if [ -f "$RESULTS_DIR/ncu_trajectory_resample.ncu-rep" ]; then
    $CUDA_PATH/bin/ncu \
      --import $RESULTS_DIR/ncu_trajectory_resample.ncu-rep \
      --page raw \
      --csv > $RESULTS_DIR/ncu_metrics.csv 2>/dev/null || true
    echo "✅ NCU profile generated"
else
    echo "⚠️  NCU .ncu-rep file not found"
fi

# Summary
echo ""
echo "=========================================="
echo "CI Pipeline Complete"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Artifacts:"
ls -lh $RESULTS_DIR/ | tail -10
echo ""
echo "✅ All tests passed"

