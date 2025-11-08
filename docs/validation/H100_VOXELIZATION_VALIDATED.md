# H100 Voxelization Kernel - Hardware Validated

**Date:** November 8, 2025  
**Hardware:** NVIDIA H100 (SM90)  
**Instance:** awesome-gpu-name (Shadeform/Brev)  
**Validation Type:** Functional Benchmark (NCU unavailable - not installed)

---

## Results

**Configuration:** 500,000 points → 128³ grid

| Metric | Value | Status |
|--------|-------|--------|
| **P50 Latency** | 0.024 ms | ✅ |
| **P99 Latency** | 0.034 ms | ✅ |
| **Throughput** | 21.06 B pts/sec | ✅ |
| **Regression Gate** | >5B pts/sec | ✅ PASSED (4.2× above threshold) |

---

## Validation Status

✅ **Functional Test:** Passed  
❌ **NCU Profile:** Not available (NCU not installed on instance)  
❌ **Occupancy Proof:** Cannot measure without NCU  
✅ **Regression Gate:** Passed (21.06B > 5B threshold)

---

## Comparison

| GPU | Throughput | Latency (P50) |
|-----|------------|---------------|
| **H100** | 21.06 B pts/sec | 0.024 ms |
| **A100** | 11.76 B pts/sec | 0.043 ms |

**H100 Speedup:** 1.79× faster than A100

---

## Assessment

**Functional Performance:** ✅ VALIDATED  
**NCU Occupancy Proof:** ❌ BLOCKED (NCU not installed)  
**Recommendation:** Install NCU on GPU instance for occupancy validation

**Current Status:** Production kernel validated functionally, NCU metrics pending infrastructure setup.

---

*Validated: 2025-11-08*  
*Method: 200 iterations, P50/P99 latency*  
*Infrastructure Blocker: NCU tools not available on brev/shadeform instances*

