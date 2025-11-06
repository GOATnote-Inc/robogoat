# Blackwell (B100/B200) CI Runner Acquisition Plan

**Status:** Planned Q1 2026  
**Owner:** Brandon Dent <b@thegoatnote.com>  
**Budget:** TBD  
**Priority:** P1 (High)

---

## Objective

Acquire NVIDIA Blackwell (B100/B200) GPU hardware for CI/CD integration, enabling:
1. SM100 kernel compilation and validation
2. 5th-generation Tensor Core optimization
3. Performance benchmarking vs H100/A100
4. Future-proofing for next-gen robot foundation models

---

## Hardware Options

### Option 1: NVIDIA DGX B200 (Recommended)
**Specs:**
- 8× NVIDIA B200 GPUs (SM100 architecture)
- 1.4 TB HBM3e memory (175 GB/GPU)
- NVLink Switch for multi-GPU scaling
- 2× Intel Xeon Platinum CPUs

**Pros:**
- Official NVIDIA hardware
- Full NVLink/NVSwitch support
- Enterprise support and warranty
- Known good configuration

**Cons:**
- High cost (~$500K-$1M estimated)
- Long lead time (Q1-Q2 2026)
- Requires data center infrastructure

**Vendor:** NVIDIA Direct or authorized resellers

---

### Option 2: Self-Hosted B100 Workstation
**Specs:**
- 1× NVIDIA B100 GPU (PCIe Gen5)
- 192 GB HBM3e memory
- AMD Threadripper Pro or Intel Xeon W
- PCIe Gen5 support required

**Pros:**
- Lower cost (~$30K-$50K estimated)
- Easier to deploy (desktop form factor)
- Good for single-GPU testing

**Cons:**
- No NVLink for multi-GPU
- Limited availability (enterprise prioritization)
- Less representative of production DGX systems

**Vendor:** PNY, ASUS, Supermicro (enterprise channels)

---

### Option 3: Cloud Provider (Lambda Labs, AWS, GCP, Azure)
**Specs:**
- On-demand Blackwell instances
- Hourly/monthly rental
- Variable availability

**Pros:**
- No upfront capital
- Flexible scaling
- Rapid deployment when available

**Cons:**
- High operational cost ($10-20/hr estimated)
- Availability uncertain (Q2 2026+)
- Network latency for CI/CD
- Limited persistent storage

**Vendors:** 
- Lambda Labs (likely first to market)
- AWS EC2 P6 instances (future)
- GCP A3 Ultra (future)
- Azure ND H100 successor (future)

---

### Option 4: NVIDIA Developer Access Program
**Specs:**
- Early access to Blackwell hardware
- Remote development environment
- Limited availability

**Pros:**
- Free or low-cost access
- Official NVIDIA support
- Early feedback opportunity

**Cons:**
- Competitive application
- Shared resources
- Limited CI/CD integration
- Time-limited access

**Application:** https://developer.nvidia.com/blackwell-access

---

## Recommended Strategy

### Phase 1: Cloud Access (Q1 2026)
1. **Timeline:** January-March 2026
2. **Action:** Monitor Lambda Labs, AWS, GCP for Blackwell availability
3. **Goal:** Initial SM100 kernel compilation and smoke tests
4. **Budget:** $5K-10K for spot testing

### Phase 2: Developer Program (Q1 2026)
1. **Timeline:** Concurrent with Phase 1
2. **Action:** Apply to NVIDIA Blackwell Developer Access Program
3. **Goal:** Extended access for performance tuning
4. **Budget:** $0 (if accepted)

### Phase 3: Self-Hosted CI Runner (Q2 2026)
1. **Timeline:** April-June 2026
2. **Action:** Purchase 1× B100 workstation or 2× B100 PCIe cards
3. **Goal:** Dedicated CI/CD integration
4. **Budget:** $50K-100K
5. **Justification:** Amortized over 3 years, ~$1.4K/month vs $7K+/month cloud

### Phase 4: DGX B200 (Q3-Q4 2026)
1. **Timeline:** Conditional on business needs
2. **Action:** Evaluate need for multi-GPU / NVLink validation
3. **Goal:** Production-scale testing (8-GPU configurations)
4. **Budget:** $500K+ (requires Series A+ funding)

---

## Technical Requirements

### For CI/CD Integration

**Minimum:**
- 1× Blackwell GPU (B100 or B200)
- CUDA 14.0+ support (estimated)
- Ubuntu 22.04/24.04 LTS
- 512 GB system RAM
- 4 TB NVMe SSD (fast build cache)
- 10 GbE network (for artifact uploads)

**Recommended:**
- 2× Blackwell GPUs (redundancy + multi-GPU testing)
- CUDA 14.0+
- Ubuntu 24.04 LTS
- 1 TB system RAM
- 8 TB NVMe SSD RAID 0
- 25 GbE network

**Software Stack:**
- CUDA Toolkit 14.0+ (when released)
- cuDNN 9.5+ (when released)
- TensorRT 11.0+ (when released)
- CUTLASS 4.5+ (main branch support)
- PyTorch 2.7+ (with Blackwell support)

---

## GitHub Actions Integration

### Self-Hosted Runner Setup

```yaml
# .github/workflows/blackwell-validation.yml
name: Blackwell Validation

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Nightly

jobs:
  blackwell-test:
    runs-on: [self-hosted, gpu, blackwell, sm100]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Verify Blackwell GPU
        run: |
          nvidia-smi --query-gpu=name,compute_cap --format=csv
          # Expect: B100/B200, 10.0 (SM100)
      
      - name: Build for SM100
        run: |
          pip install -e .
          # Kernels compile with -gencode arch=compute_100,code=sm_100
      
      - name: Run benchmarks
        run: |
          python bench/benchmark_harness.py --device cuda --gpu blackwell
      
      - name: Compare vs H100/A100
        run: |
          python scripts/compare_baseline.py \
            --current artifacts/blackwell.csv \
            --baseline artifacts/h100_baseline.csv
```

---

## Performance Targets (Estimated)

Based on NVIDIA's published Blackwell specs (2× H100 performance):

| Operation | H100 Baseline | B100 Target | B200 Target |
|-----------|---------------|-------------|-------------|
| Trajectory (32×500×256) | 2.605ms | <1.3ms | <1.0ms |
| Multimodal (3-stream) | <1ms | <0.5ms | <0.4ms |
| Voxelization (1M pts) | <0.5ms | <0.25ms | <0.2ms |
| DRAM Bandwidth | 3.0 TB/s | 4+ TB/s | 4+ TB/s |
| Tensor Core TFLOPS | 990 (FP16) | 2000+ (FP8) | 2000+ (FP8) |

**Key Optimizations for Blackwell:**
1. Enhanced FP8 precision (Transformer Engine v2)
2. Larger WGMMA tiles (256×256 vs 128×128)
3. Improved TMA (Tensor Memory Accelerator)
4. Larger L2 cache (expected 60+ MB)

---

## Risk Mitigation

### Risk 1: Hardware Unavailable in Q1 2026
**Mitigation:**
- Continue H100/A100 validation
- Simulate SM100 with SM90 + newer CUTLASS APIs
- Defer Blackwell-specific optimizations to Q2 2026

### Risk 2: CUDA 14.0 API Changes
**Mitigation:**
- Abstract kernel APIs via CUTLASS templates
- Maintain backward compatibility with SM80/SM90
- Beta test with CUDA 14.0 RC releases

### Risk 3: Budget Constraints
**Mitigation:**
- Prioritize cloud access over self-hosted
- Leverage NVIDIA developer program
- Defer DGX B200 until Series A funding

### Risk 4: CI/CD Integration Complexity
**Mitigation:**
- Test GitHub Actions self-hosted runner setup with H100 first
- Document runner provisioning and security
- Use containerized builds (Docker) for reproducibility

---

## Success Criteria

1. ✅ **Q1 2026:** SM100 kernel compilation successful (cloud or dev program)
2. ✅ **Q2 2026:** Performance benchmarks vs H100 (<2× speedup validated)
3. ✅ **Q2 2026:** Self-hosted CI runner operational (nightly builds)
4. ✅ **Q3 2026:** Blackwell optimizations merged to main (WGMMA, TMA)
5. ✅ **Q4 2026:** Multi-GPU scaling validated (if DGX B200 acquired)

---

## Action Items

### Immediate (Q4 2025)
- [x] Document Blackwell acquisition plan
- [ ] Monitor NVIDIA announcements for availability timeline
- [ ] Apply to NVIDIA Blackwell Developer Access Program
- [ ] Budget approval for $50K-100K hardware purchase

### Q1 2026
- [ ] Test on Lambda Labs Blackwell instances (when available)
- [ ] Port kernels to SM100 (-gencode arch=compute_100,code=sm_100)
- [ ] Validate CUTLASS 4.5+ compatibility
- [ ] Initial performance benchmarks

### Q2 2026
- [ ] Purchase 1-2× B100 GPUs for self-hosted CI
- [ ] Set up GitHub Actions self-hosted runner
- [ ] Implement Blackwell-specific optimizations
- [ ] Update documentation with Blackwell results

---

## Budget Summary

| Item | Cost | Timeline | Priority |
|------|------|----------|----------|
| Cloud testing (spot) | $5K-10K | Q1 2026 | P0 |
| B100 workstation | $50K-100K | Q2 2026 | P1 |
| DGX B200 | $500K-1M | Q4 2026 | P2 |
| **Total (Phase 1-2)** | **$55K-110K** | **Q1-Q2 2026** | - |

**ROI Analysis (Self-Hosted B100 vs Cloud):**
- Self-hosted: $50K upfront + $500/month power/cooling = $56K/year
- Cloud: $15/hr × 8 hrs/day × 365 days = $43K/year
- **Breakeven:** ~14 months
- **Advantage:** Control, availability, CI/CD integration

---

**Maintained By:** Brandon Dent <b@thegoatnote.com>  
**Last Updated:** 2025-11-06  
**Next Review:** 2026-01-01 (quarterly)

