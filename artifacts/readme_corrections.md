# README Performance Claims - Required Corrections

**Date:** 2025-11-09  
**Based on:** H100 validation (artifacts/h100_validation_final_results.md)  
**Purpose:** Document exact corrections needed for README.md

---

## Critical Corrections (P0 - Apply Immediately)

### 1. Fix Latency Range (Line 23)

**BEFORE:**
```markdown
- 🚀 **Sub-millisecond latency** - 0.018-2.6ms on H100
```

**AFTER:**
```markdown
- 🚀 **Sub-millisecond latency** - 0.021-0.035ms on H100 (measured)
```

**Evidence:**
- Voxelization: 0.0205ms
- Multimodal fusion: 0.0339ms
- Trajectory resample: 0.0353ms
- Source: `artifacts/h100_validation_final_results.md`

**Rationale:** The "2.6ms" claim is **73x too high** compared to measured data. Replace with actual measured range.

---

### 2. Remove Unsubstantiated CPU Speedup Claim (Line 24)

**BEFORE:**
```markdown
- ⚡ **10-100× faster than CPU** - Validated with Nsight profiling
```

**AFTER:**
```markdown
- ⚡ **GPU-accelerated with BF16** - CUDA kernels with vectorized loads
```

**Evidence:** ❌ NONE - No CPU baseline benchmarks exist

**Rationale:** Without CPU comparison data, the "10-100×" claim is unsubstantiated. Replace with factual statement about GPU optimization.

---

### 3. Add Measurement Uncertainty to Quick Start (Line 56)

**BEFORE:**
```python
# H100: 0.034ms | A100: 0.057ms
```

**AFTER:**
```python
# H100: 0.034ms ± 0.002ms (n=100) | A100: 0.057ms (P50)
```

**Evidence:**
- H100: Mean 0.0339ms, Std 0.0022ms from artifacts/h100_validation_final_results.md
- A100: No detailed validation found, but P50 metric noted

**Rationale:** Add measurement rigor (sample size, uncertainty)

---

### 4. Update Voxelization Comment (Line 71)

**BEFORE:**
```python
# H100: 24 billion points/sec | A100: 21 billion points/sec
```

**AFTER:**
```python
# H100: 24.3 billion points/sec (500K pts @ 0.0205ms)
# A100: [Evidence pending - see artifacts/a100_validation.md]
```

**Evidence:**
- H100: 24.34 B pts/sec measured
- A100: ❌ NO VALIDATION FOUND in artifacts/

**Rationale:** H100 claim verified, A100 claim unsubstantiated

---

### 5. Remove Memory Efficiency Claim (Line 162)

**BEFORE:**
```markdown
- Coalesced memory access (>95% efficiency)
```

**AFTER:**
```markdown
- L1-resident workloads (99%+ cache hit rate for fusion/resample)
```

**Evidence:**
- NCU reports show 99%+ L1 hit rate
- NO "memory access efficiency" metric found
- NCU data: 0.05% DRAM (trajectory), 0.03% DRAM (multimodal) → L1-resident

**Rationale:** Replace unsubstantiated ">95%" with measured L1 hit rate

---

### 6. Clarify Vectorization Claim (Line 161)

**BEFORE:**
```markdown
- Vectorized BF16 loads (4× throughput vs scalar)
```

**AFTER:**
```markdown
- Vectorized BF16 loads (4-element vectors, 4× bandwidth vs scalar)
```

**Evidence:** Architectural fact (BF16x4 = 8 bytes vs BF16x1 = 2 bytes)

**Rationale:** Change "throughput" (requires measurement) to "bandwidth" (architectural fact)

---

## NCU Metrics Table Verification (Lines 173-177)

**Current README Claims:**
| Kernel | DRAM BW | SM Throughput | Warps Active | L1 Hit Rate |
|--------|---------|---------------|--------------|-------------|
| Trajectory Resample | 0.05% | 1.27% | 12.48% | 99%+ |
| Multimodal Fusion | 0.03% | 2.15% | 12.49% | 99%+ |
| Voxelization | 54.17% | 14.06% | 64.83% | N/A |

**Verification:**
Source: `robocache/profiling/NCU_COMPLETE_ANALYSIS.md`

✅ **ALL NUMBERS VERIFIED:**
- Trajectory: DRAM 0.05%, SM 1.27%, Warps 12.48% ✅
- Multimodal: DRAM 0.03-0.04%, SM 2.14-2.15%, Warps 12.49% ✅
- Voxelization: Lines 96-100 confirm these metrics ✅

**Recommendation:** Table is CORRECT. Keep as-is.

---

## Nsight Systems Claims Verification (Lines 196-202)

**Current README Claims:**
```markdown
- **End-to-end latency:** 1.56ms/step (12.84× faster than 20ms target)
- **RoboCache preprocessing:** 19.3% of GPU time (83.4μs per call)
- **Throughput:** 20,548 episodes/sec
- **Memory overhead:** 0.15% (negligible)
```

**Verification:**
Source: `robocache/profiling/NSIGHT_SYSTEMS_H100.md`

✅ **ALL NUMBERS VERIFIED:**
- End-to-end: 1.56ms ✅ (line 57)
- 12.84× faster: ✅ (1.56ms vs 20ms target)
- RoboCache: 19.3% of GPU time ✅ (line 71)
- 83.4μs per call ✅ (line 71)
- Throughput: 20,548 episodes/sec ✅ (line 59)

**Memory overhead:** Need to check line 89-100 for "0.15%" claim...

From NSIGHT_SYSTEMS_H100.md line 89:
```
| memset | 72.3% | 177,666 | 220 | 808 |
```

Total memory time: ~245μs (177,666 + 55,586 + 12,577 nanoseconds = 245,829 ns)
Total runtime: 160ms (for 100 steps)
Memory overhead: 245μs / 1600μs per step = 0.15% ✅

**Recommendation:** All Nsight Systems claims CORRECT. Keep as-is.

---

## Summary of Required Changes

### MUST FIX (P0)
1. ✅ Line 23: Change "0.018-2.6ms" → "0.021-0.035ms"
2. ✅ Line 24: Remove "10-100× faster than CPU" → replace with factual GPU claim
3. ✅ Line 56: Add "± 0.002ms (n=100)"
4. ✅ Line 71: Remove A100 claim or add "[pending validation]"
5. ✅ Line 161: Change "throughput" → "bandwidth"
6. ✅ Line 162: Replace ">95% efficiency" with "99%+ L1 cache hit rate"

### VERIFIED CORRECT (No Changes)
- ✅ Lines 173-177: NCU metrics table - ALL NUMBERS CORRECT
- ✅ Lines 196-202: Nsight Systems claims - ALL NUMBERS CORRECT
- ✅ Lines 115-117: H100 benchmarks table - CORRECT (from h100_validation_final_results.md)

---

## Additional Recommendations

### Add Configuration Details
For each performance claim in Quick Start examples, add comment with:
- Exact tensor shapes
- Data type (bf16, fp32)
- Hardware tested
- Sample size
- Measurement method

**Example:**
```python
fused = robocache.fuse_multimodal(...)
# H100: 0.034ms ± 0.002ms (n=100, torch.cuda.Event timing)
# Config: batch=4, vision=(30,512), proprio=(100,64), imu=(200,12), target=50
# Measured: NVIDIA H100 PCIe 80GB, CUDA 13.0, Driver 580.95
```

### Link Claims to Evidence
Add footnotes linking to evidence files:
```markdown
**Statistical Rigor:** 5 seeds × 50 repeats = 250 measurements per config  
**Hardware:** NVIDIA H100 PCIe 81GB, CUDA 13.0, Driver 580.95  
**Methodology:** `torch.cuda.Event` timing with warmup, CSV export  
**Full Report:** [H100 Validation](artifacts/h100_validation_final_results.md)  
**NCU Profiling:** [Expert Analysis](robocache/profiling/NCU_COMPLETE_ANALYSIS.md)  
**Nsight Systems:** [End-to-End](robocache/profiling/NSIGHT_SYSTEMS_H100.md)
```

---

## Verification Status

| Claim | Location | Status | Evidence File |
|-------|----------|--------|---------------|
| Latency range | Line 23 | ❌ WRONG | h100_validation_final_results.md |
| CPU speedup | Line 24 | ❌ UNSUBSTANTIATED | None |
| Multimodal latency | Line 56 | ✅ CORRECT | h100_validation_final_results.md |
| Voxelization H100 | Line 71 | ✅ CORRECT | h100_validation_final_results.md |
| Voxelization A100 | Line 71 | ❌ UNSUBSTANTIATED | None |
| Vectorization | Line 161 | ⚠️ WORDING | Architectural (no file) |
| Memory efficiency | Line 162 | ❌ UNSUBSTANTIATED | NCU shows L1 hit rate only |
| NCU table | Lines 173-177 | ✅ CORRECT | NCU_COMPLETE_ANALYSIS.md |
| Nsight Systems | Lines 196-202 | ✅ CORRECT | NSIGHT_SYSTEMS_H100.md |

**Overall:** 4 verified, 3 need fixes, 2 need wording changes

---

**Next Action:** Apply these corrections to README.md with detailed commit message documenting each change and its rationale.

