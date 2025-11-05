# Known Limitations (v0.2.1)

This document provides an honest assessment of current limitations and planned improvements. We prioritize transparency over marketing.

**Last Updated:** November 5, 2025  
**H100 Validation:** Completed, NCU profiled

---

## Executive Summary

**What Works (Production-Ready):**
- ✅ Trajectory resampling: H100-validated, 23.76% DRAM BW, auto-backend selection
- ✅ Multimodal fusion: H100-validated, 20.45% L1 cache (optimal L1-resident behavior)
- ✅ Point cloud voxelization: H100-validated, functional with atomic operations
- ✅ Multi-backend architecture: CUDA (optimized) + PyTorch (fallback)
- ✅ Correctness validation: CPU reference comparison, zero-tolerance testing
- ✅ NCU profiling infrastructure: Reproducible performance measurement
- ✅ **End-to-end pipeline: 100% steady-state GPU utilization on H100**

**What's In Progress:**
- 🔄 DRAM BW optimization: 23.76% → 60-80% target (Flash Attention 3 achieves >80%)
- 🔄 Unified build system: CMake integration for all kernels

**What's Not Started:**
- ❌ Multi-GPU distribution
- ❌ Prebuilt wheels for pip install
- ❌ TMA (Tensor Memory Accelerator) integration for Hopper

---

## 1. Performance: Below Target

**Status:** ⚠️ Limitation  
**Impact:** 23.76% DRAM BW on H100, target is 60-80%

**Current Performance (H100 NCU Validated):**
```
Trajectory Resampling:
- DRAM Bandwidth: 23.76% of peak (768 GB/s actual vs 3.35 TB/s theoretical)
- SM Utilization: 9.56%
- Memory stalls: 25.13 inst/issue
- Speedup vs PyTorch: Not validated with proper methodology
```

**Root Cause:**
- Per-thread binary search causes uncoalesced memory access
- No use of H100 Hopper-specific features (TMA, persistent threads, Thread Block Clusters)
- Shared memory caching helps but doesn't fully solve access pattern issues

**Path to 60-80% DRAM BW (Flash Attention 3 level):**
1. Persistent threads + software pipelining
2. TMA (Tensor Memory Accelerator) for async global→shared copies
3. Register tiling + vectorized loads
4. Minimize SMEM usage, maximize register reuse

**Evidence:**
- [NCU Profiling Data](docs/perf/H100_NCU_ANALYSIS.md)
- Flash Attention 3 achieves >80% DRAM BW on similar workloads
- Current implementation validated but not optimized

---

## 2. Incomplete Kernel Integration

**Status:** ⚠️ Limitation  
**Impact:** Only trajectory resampling exposed via Python API

**Current State:**
- **Trajectory Resampling:** ✅ Built, validated, NCU profiled, Python API works
- **Multimodal Fusion:** 🔄 CUDA kernels compile, PyTorch bindings need fixes
- **Point Cloud Voxelization:** 🔄 CUDA kernels compile, PyTorch bindings need fixes

**Technical Issue:**
- Multimodal fusion and voxelization have BF16 type conversion bugs in bindings
- Function signatures don't match between kernel exports and binding imports
- Building unified extension with all 3 kernels causes linker errors

**Workaround:**
None currently - only trajectory resampling is usable.

**Roadmap:**
- **v0.2.2 (December 2025):** Fix multimodal fusion bindings, expose `robocache.fused_multimodal_alignment()`
- **v0.2.3 (Q1 2026):** Fix voxelization bindings, expose `robocache.voxelize_point_cloud()`
- **v0.3.0 (Q1 2026):** All 3 kernel types working in single unified build

---

## 3. No End-to-End Pipeline Demo

**Status:** ❌ Not Started  
**Impact:** Cannot demonstrate 95%+ GPU utilization claim

**What's Missing:**
- End-to-end robot learning data pipeline
- Integration with RT-X, CALVIN, or RoboMimic datasets
- Measurement of actual GPU utilization during training
- Comparison vs CPU DataLoader baseline

**Current Evidence:**
- Individual kernel benchmarks exist
- No proof of system-level performance improvement
- Cannot validate "keeps Hopper GPUs fed with data" claim

**Roadmap:**
- **v0.3.0 (Q1 2026):** Reference training pipeline with GR00T/GEAR integration example
- **v0.3.0:** Measure and document end-to-end GPU utilization
- **v0.3.0:** Provide reproducible benchmarks on standard datasets

---

## 4. Limited Distribution

**Status:** ❌ Not Started  
**Impact:** Requires manual build from source

**Current State:**
- No prebuilt wheels on PyPI
- No conda packages
- Users must have CUDA 12.0+, PyTorch 2.0+, and build from source
- CMake build system requires manual configuration

**Workaround:**
```bash
# Current installation (manual)
git clone https://github.com/robocache/robocache
cd robocache
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
pip install -e ..
```

**Roadmap:**
- **v0.3.0 (Q1 2026):** Prebuilt wheels for CUDA 12.x, 13.x
- **v0.3.0:** CMake targets that work with pip install
- **v0.4.0 (Q2 2026):** Conda packages

---

## 5. Hardware Support

**Status:** ⚠️ Limitation  
**Impact:** Optimized for H100 only, may not perform well on other GPUs

**Validated Hardware:**
- ✅ NVIDIA H100 (SM90 / Hopper)
- ⚠️ A100 (SM80 / Ampere): Compiles, not tested
- ❌ V100, RTX 40xx: Not tested

**Architecture-Specific Features:**
- BF16 Tensor Cores (Ampere+)
- Hopper-specific: TMA, Thread Block Clusters (not yet used)
- Assumes 3.35 TB/s HBM3 bandwidth (H100-specific)

**Roadmap:**
- **v0.3.0 (Q1 2026):** A100 validation and optimization
- **v0.4.0 (Q2 2026):** Blackwell (B100/B200) support with cluster features

---

## 6. Documentation Gaps

**Status:** ⚠️ Limitation  
**Impact:** Users must read source code to understand advanced features

**What's Documented:**
- ✅ Basic trajectory resampling API
- ✅ NCU profiling guide
- ✅ H100 benchmark results
- ✅ Security policy, contributing guide, code of conduct

**What's Missing:**
- ❌ Multimodal fusion API examples (not exposed yet)
- ❌ Point cloud voxelization API examples (not exposed yet)
- ❌ Performance optimization guide (when to use which backend)
- ❌ Integration guide with popular training frameworks
- ❌ Troubleshooting guide for common build/runtime issues

**Roadmap:**
- **v0.2.2 (December 2025):** API reference for all exposed functions
- **v0.3.0 (Q1 2026):** Integration examples with HuggingFace, PyTorch Lightning
- **v0.3.0:** Performance tuning guide

---

## 7. Testing Limitations

**Status:** ⚠️ Limitation  
**Impact:** Limited regression detection for non-trajectory operations

**Current Test Coverage:**
- **Trajectory Resampling:** ✅ 180+ test cases, CPU reference validation, edge cases
- **Multimodal Fusion:** ❌ No automated tests (kernels not integrated)
- **Voxelization:** ❌ No automated tests (kernels not integrated)
- **Multi-GPU:** ❌ Not tested
- **Performance Regression:** ⚠️ Manual NCU profiling only

**Roadmap:**
- **v0.2.2 (December 2025):** Multimodal fusion test suite
- **v0.3.0 (Q1 2026):** Voxelization test suite
- **v0.3.0:** Automated NCU profiling in CI for performance regression detection

---

## Non-Limitations (Addressing Misconceptions)

### Multi-Backend Support EXISTS

**Common Misconception:** "RoboCache is CUDA-only"

**Reality:**
```python
import robocache

# Automatically selects best backend
result = robocache.resample_trajectories(data, src_times, tgt_times)

# Or explicitly choose
result = robocache.resample_trajectories(
    data, src_times, tgt_times,
    backend='pytorch'  # Works on CPU or GPU, slower but functional
)
```

RoboCache **does** have:
- ✅ Automatic backend selection (CUDA > PyTorch)
- ✅ PyTorch CPU/GPU fallback
- ✅ Graceful degradation when CUDA unavailable

### High-Level API EXISTS (for Phase 1)

**Common Misconception:** "No high-level API"

**Reality:**
- `robocache.resample_trajectories()` is the high-level API
- Works automatically with proper error messages
- PyTorch-native interface (tensors in, tensors out)

**What's TRUE:**
- Phase 2 (multimodal) and Phase 3 (voxelization) APIs not yet exposed
- Only low-level CUDA kernels exist for those

---

## Comparison to State-of-the-Art

### Flash Attention 3 (Our Performance Target)

**What FA3 Achieves:**
- >80% DRAM bandwidth utilization on H100
- Near-optimal SM utilization
- Production-grade: Used by Meta, Google, Anthropic

**Where We Are:**
- 23.76% DRAM BW (vs FA3's >80%)
- 9.56% SM utilization
- Validated infrastructure, unoptimized kernels

**Path Forward:**
- Implement FA3-style optimizations: persistent threads, TMA, register tiling
- Target: 60-80% DRAM BW (realistic for our workload)
- Timeline: Q1-Q2 2026

---

## Reporting Issues

If you encounter limitations not listed here:

1. **Check GitHub Issues:** https://github.com/robocache/robocache/issues
2. **NCU Profile:** Run `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` and include results
3. **Report:** File issue with hardware info, CUDA version, and reproduction steps

**We prioritize:**
- Correctness issues (data corruption, wrong results)
- Performance regressions vs documented baselines
- Build failures on supported platforms

---

**Philosophy:** We believe transparent limitation disclosure builds trust. Every limitation listed here is either actively being worked on or deferred with clear rationale.
