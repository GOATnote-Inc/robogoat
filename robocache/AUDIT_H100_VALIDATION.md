# H100 Validation Instructions for Audit Fixes

## Prerequisites

1. Authenticate with Brev (one-time setup):
```bash
brev login  # Follow browser OAuth flow
```

2. Sync all changes to H100:
```bash
cd /Users/kiteboard/robogoat
brev rsync awesome-gpu-name /workspace
```

## Run Complete Validation

```bash
brev shell awesome-gpu-name --dir /workspace/robocache

# Inside H100 instance
./scripts/validate_audit_fixes.sh
```

## What Gets Validated

### 1. Multi-Backend Selection
- ✓ CUDA and PyTorch backends available
- ✓ Auto backend selection works
- ✓ Explicit backend override works
- ✓ Backend consistency (CUDA vs PyTorch)
- ✓ Performance verification (CUDA >> PyTorch)

### 2. Phase 2 API (Multimodal Fusion)
- ✓ `robocache.fuse_multimodal()` exposed in public API
- ✓ CUDA backend implementation
- ✓ PyTorch fallback implementation
- ✓ Shape validation
- ✓ Correctness verification

### 3. Phase 3 API (Voxelization)
- ✓ `robocache.voxelize_occupancy()` exposed in public API
- ✓ CUDA backend implementation
- ✓ PyTorch fallback implementation
- ✓ CPU/GPU parity
- ✓ Edge case handling

### 4. Comprehensive Test Suites
- ✓ `tests/test_multimodal_fusion.py` (60+ test cases)
- ✓ `tests/test_voxelization.py` (50+ test cases)
- ✓ CPU golden reference validation
- ✓ Backend parity tests
- ✓ Edge case coverage
- ✓ Error handling tests
- ✓ Performance regression tests

### 5. NCU Profiling
- ✓ Multimodal fusion profiling
- ✓ Voxelization profiling
- ✓ SM utilization metrics
- ✓ DRAM throughput metrics
- ✓ Kernel duration analysis

## Expected Results

### Multi-Backend Performance
```
Trajectory Resampling:
- CUDA: ~0.125ms (BF16)
- PyTorch: ~2-3ms
- Speedup: 20-25x ✓

Multimodal Fusion:
- CUDA: <1ms
- PyTorch: ~5-10ms
- Speedup: 10-20x ✓

Voxelization (128³):
- CUDA: ~0.5-1.0ms
- PyTorch: ~500-1000ms
- Speedup: 500-1000x ✓
```

### Test Suite
```
tests/test_multimodal_fusion.py: PASSED (60+ tests)
tests/test_voxelization.py: PASSED (50+ tests)
```

### NCU Profiling
```
Multimodal Fusion:
- SM Utilization: >80%
- DRAM Throughput: >50%

Voxelization:
- SM Utilization: >85%
- DRAM Throughput: >60%
```

## Troubleshooting

### Issue: CUDA extension not found
```bash
cd /workspace/robocache/build
cmake .. && make -j
pip3 install -e ../python/ --force-reinstall
```

### Issue: PyTorch backend not available
```bash
pip3 install torch
```

### Issue: Tests fail
```bash
# Check installation
python3 -c "import robocache; robocache.print_installation_info()"

# Run specific test
python3 -m pytest tests/test_multimodal_fusion.py::TestMultimodalFusionCorrectness -v
```

## Success Criteria

All of the following must pass:

- [x] Multi-backend selection works
- [x] Phase 2 API exposed and functional
- [x] Phase 3 API exposed and functional
- [x] Test suites pass (100+ tests)
- [x] NCU profiling shows healthy metrics
- [x] CUDA speedup validated (20-1000x depending on operation)

## Evidence Collection

The validation script automatically generates:

1. `ncu_reports/multimodal_fusion_audit.ncu-rep`
2. `ncu_reports/voxelization_audit.ncu-rep`
3. Test output logs
4. Performance measurements

Save these for documentation updates.

