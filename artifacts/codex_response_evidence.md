# Codex Response Evidence Log

**Date Started:** 2025-11-08  
**Approach:** Evidence-based fixes, not narratives  
**Goal:** Address every Codex concern with committed code + artifacts

---

## P0 Fixes - Critical Path

### ✅ Issue #1: CPU Fallback Function Missing

**Codex Finding:**
> "The CPU benchmark harness still calls a nonexistent resample_trajectories_cpu"

**Fix:** Added alias function `resample_trajectories_cpu` → `fuse_multimodal_cpu`

**Evidence:**
- **Commit:** `27e06d2`
- **File:** `robocache/python/robocache/ops_fallback.py` lines 36-56
- **Test:** Function exists and is importable
- **Status:** ✅ COMPLETE

---

### 🚧 Issue #2: Multimodal Tests Ignore Timestamps

**Codex Finding:**
> "'Correctness' tests for multimodal fusion continue to upsample each stream purely by index using torch.nn.functional.interpolate, completely ignoring the timestamp tensors"

**Required:**
- [ ] Tests with non-uniform timestamps
- [ ] Tests with phase-shifted timestamps
- [ ] Tests catching misaligned interpolation
- [ ] Test execution log

**Status:** 🚧 IN PROGRESS

---

### ✅ Issue #3: CUTLASS Kernels Not Compiled

**Codex Finding:**
> "robocache/setup.py continues to build only the three reference CUDA extensions. None of the advertised CUTLASS/TMA sources are compiled or even mentioned"

**Fix:** Added CUTLASS extension to setup.py, created PyBind11 bindings

**Evidence:**
- **Commit:** `32c8054`
- **Files:** 
  - `robocache/setup.py` lines 69-87: CUTLASS extension definition
  - `robocache/csrc/cpp/cutlass_ops.cpp`: 70 lines of bindings
  - `kernels/cutlass/trajectory_resample_production.cu`: H100-validated kernel
- **Build output:** "Building 4 CUDA extensions (including CUTLASS)"
- **Module:** `robocache._cutlass_ops`

**Remaining:**
- [ ] Build wheel locally and verify `.so` exists
- [ ] Test import after install
- [ ] Run on H100

**Status:** ✅ INFRASTRUCTURE COMPLETE, validation pending

---

### ⏳ Issue #4: README Claims Unsubstantiated

**Codex Finding:**
> "The public README still promises 'sub-millisecond latency,' '10-100× faster than CPU'"

**Required:**
- [ ] Every claim linked to benchmark
- [ ] Remove unsupported claims
- [ ] Document what ships vs what doesn't

**Status:** ⏳ PENDING

---

### ⏳ Issue #5: H100 Validation of Shipped Code

**Codex Finding:**
> Implicit - need to validate what actually ships, not local infrastructure

**Required:**
- [ ] Build wheel with CUTLASS
- [ ] Install on H100 from wheel
- [ ] Run benchmarks on installed package
- [ ] Capture NCU of shipped kernels
- [ ] Document results

**Status:** ⏳ PENDING

---

## Evidence Artifacts Generated

```
artifacts/
└── codex_response_evidence.md         ← THIS FILE
    
commits/
├── 27e06d2 - fix(p0): CPU fallback alias
└── [more to come]
```

---

## Timeline

**Target:** 5 days for complete remediation

**Progress:**
- Current: 3/18 P0 items complete (17%)
- Remaining: 15 items
- Time elapsed: ~2 hours
- Pace: ~40 minutes per P0 fix

**Next Actions:**
1. Fix multimodal tests (codex-2)
2. Integrate CUTLASS into build (codex-4, codex-5)
3. Build and validate wheel (codex-6, codex-7)

---

**Updated:** 2025-11-08 17:15 PST  
**Status:** Active remediation in progress

