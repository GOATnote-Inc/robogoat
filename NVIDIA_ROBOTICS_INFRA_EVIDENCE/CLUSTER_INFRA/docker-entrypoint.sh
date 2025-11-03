#!/bin/bash
# docker-entrypoint.sh
# Container entrypoint for RoboCache validation
#
# Copyright (c) 2025 GOATnote Inc.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "========================================================================"
echo "RoboCache - Multimodal Fusion Kernel"
echo "CUDA 13.0 + CUTLASS 4.3.0"
echo "========================================================================"
echo ""

# Run GPU preflight checks
echo "Running GPU preflight checks..."
/usr/local/bin/preflight.sh

if [ $? -ne 0 ]; then
    echo "ERROR: GPU preflight checks failed"
    exit 1
fi

echo ""
echo "========================================  "
echo "Running Validation Suite"
echo "========================================================================"
echo ""

cd /workspace

# 1. Unit tests
echo "1. Running Unit Tests..."
echo "----------------------------------------"
make test
if [ $? -ne 0 ]; then
    echo "ERROR: Unit tests failed"
    exit 1
fi

echo ""

# 2. Benchmarks
echo "2. Running Performance Benchmarks..."
echo "----------------------------------------"
make benchmark
if [ $? -ne 0 ]; then
    echo "ERROR: Benchmarks failed"
    exit 1
fi

echo ""

# 3. Reproducibility
echo "3. Testing Reproducibility..."
echo "----------------------------------------"
make reproducibility
if [ $? -ne 0 ]; then
    echo "ERROR: Reproducibility test failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ ALL VALIDATION PASSED"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ GPU preflight: PASS"
echo "  ✓ Unit tests: PASS"
echo "  ✓ Benchmarks: PASS"
echo "  ✓ Reproducibility: PASS"
echo ""
echo "For detailed profiling, run:"
echo "  docker run --rm --gpus all goatnote/robocache:cuda13-cutlass43 make ncu-profile"
echo ""

exit 0
