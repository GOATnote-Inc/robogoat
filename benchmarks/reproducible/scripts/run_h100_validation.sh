#!/bin/bash
# H100 Performance Validation Suite
# Executes all reproducible benchmarks and generates evidence matrix

set -e

echo "=== RoboCache H100 Performance Validation ==="
echo "Hardware: NVIDIA H100 PCIe 80GB"
echo "Start: $(date)"
echo ""

# Setup
cd /workspace/robocache/robocache
export PYTHONPATH=/workspace/robocache/robocache/python:$PYTHONPATH
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo ""

# Install dependencies if needed
if ! python3 -c "import robocache" 2>/dev/null; then
    echo "Installing robocache..."
    cd python && pip3 install -e . --user && cd ..
fi

# Verify installation
echo "Verifying installation..."
python3 -c "import robocache; print('RoboCache version:', robocache.__version__); print('CUDA available:', robocache.is_cuda_available())"
echo ""

# Create results directory
mkdir -p benchmarks/reproducible/results/h100_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmarks/reproducible/results/h100_$(date +%Y%m%d_%H%M%S)"

# Run benchmarks
echo "=== Running Benchmarks ==="

# 1. Multimodal Fusion
echo ""
echo "[1/3] Multimodal Fusion Latency..."
python3 benchmarks/reproducible/scripts/run_single.py \
    --config benchmarks/reproducible/configs/multimodal_fusion_h100.json \
    --output "$RESULTS_DIR/multimodal_fusion_h100.json" \
    --verbose || echo "FAILED: multimodal_fusion"

# 2. Trajectory Resampling  
echo ""
echo "[2/3] Trajectory Resampling Latency..."
python3 benchmarks/reproducible/scripts/run_single.py \
    --config benchmarks/reproducible/configs/trajectory_resample_h100.json \
    --output "$RESULTS_DIR/trajectory_resample_h100.json" \
    --verbose || echo "FAILED: trajectory_resample"

# 3. Voxelization Throughput
echo ""
echo "[3/3] Voxelization Throughput..."
python3 benchmarks/reproducible/scripts/run_single.py \
    --config benchmarks/reproducible/configs/voxelization_throughput_h100.json \
    --output "$RESULTS_DIR/voxelization_throughput_h100.json" \
    --verbose || echo "FAILED: voxelization"

echo ""
echo "=== Benchmark Suite Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo "End: $(date)"

# Generate summary
echo ""
echo "=== Results Summary ==="
for result_file in "$RESULTS_DIR"/*.json; do
    if [ -f "$result_file" ]; then
        claim_id=$(python3 -c "import json; print(json.load(open('$result_file'))['claim_id'])")
        verdict=$(python3 -c "import json; print(json.load(open('$result_file'))['evaluation']['verdict'])")
        measured=$(python3 -c "import json; print(json.load(open('$result_file'))['evaluation']['measured_value'])")
        target=$(python3 -c "import json; print(json.load(open('$result_file'))['evaluation']['target_value'])")
        echo "$claim_id: $verdict (measured: $measured, target: $target)"
    fi
done

