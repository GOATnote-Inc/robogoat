# GPU CI/CD Status & Roadmap

**Last Updated:** November 8, 2025  
**Status:** ⚠️ CPU-Only CI + Manual GPU Validation

---

## Current State

### ✅ What We Have (Automated)

**CPU-Only Continuous Integration**
- **Workflow:** `.github/workflows/ci.yml`
- **Triggers:** Every PR and push to `main`
- **Runs on:** `ubuntu-latest` (GitHub-hosted CPU runners)

**Tests:**
- ✅ **Linting:** flake8, mypy (Python type checking)
- ✅ **CPU Fallbacks:** All ops work without CUDA
- ✅ **Unit Tests:** Functional correctness (no performance)
- ✅ **Import Tests:** Graceful degradation when CUDA unavailable

**What This Validates:**
- Python API correctness
- CPU fallback functionality
- Code quality and style
- Cross-platform compatibility (Linux/macOS)

**What This CANNOT Validate:**
- ❌ CUDA kernel compilation
- ❌ GPU runtime errors (OOM, illegal memory, race conditions)
- ❌ Performance regressions on real hardware
- ❌ Architecture-specific correctness (SM80 vs SM90)
- ❌ Multi-GPU functionality (NVLink, DDP)

---

### ✅ What We Have (Manual)

**Weekly GPU Validation**
- **Hardware:** Brev/Lambda Labs cloud instances
  - H100 PCIe (SM90): Primary validation platform
  - A100 SXM4 (SM80): Cross-architecture validation
  
- **Frequency:** Weekly (Sundays 6am UTC)
- **Process:** Manual execution of validation scripts
- **Results:** Published to `docs/validation/`

**Validation Tasks:**
1. **Build Test:** Compile CUDA extensions for sm_80 and sm_90
2. **Smoke Test:** Run `benchmarks/smoke.py` with performance thresholds
3. **Full Benchmarks:** Generate CSVs with 5 seeds × 50 repeats
4. **Nsight Profiling:** Capture NCU/NSys traces for performance analysis
5. **Long-Running:** 1-hour burn-in test for memory leaks

**Artifacts Generated:**
- Benchmark CSVs: `bench/results/benchmark_h100_YYYYMMDD_HHMMSS.csv`
- Validation reports: `docs/validation/*_COMPLETE.md`
- NCU reports: `artifacts/ncu_reports/*.ncu-rep`
- System info: GPU clocks, driver version, CUDA version

---

## Why No Automated GPU CI?

### Cost Analysis

**Option A: GitHub-Hosted GPU Runners**
- **Pricing:** $4.00/minute for GPU runners (2024 pricing)
- **Typical run:** 15 minutes (build + test + benchmark)
- **Cost per run:** $60
- **Monthly cost (daily runs):** ~$1,800
- **Verdict:** Prohibitively expensive for small team

**Option B: Self-Hosted GPU Runner**
- **Hardware:** Dedicated A100/H100 instance
- **Cloud cost:** $1.10-$3.00/hour (A100/H100)
- **24/7 availability:** ~$800-$2,200/month
- **Alternative:** On-prem hardware (~$50K capex + maintenance)
- **Verdict:** Significant ongoing cost or upfront investment

**Option C: Spot/On-Demand Instances**
- **Strategy:** Spin up GPU instance only for CI runs
- **Challenges:**
  - Instance startup: 2-5 minutes (cold start penalty)
  - Spot availability: Not guaranteed
  - Setup complexity: Docker, CUDA, dependencies
- **Cost:** $0.50-$1.50 per run (optimistic)
- **Monthly (daily runs):** ~$15-$45
- **Verdict:** Most cost-effective but complex to implement

**Our Choice:** Manual validation until project funding supports Option C or B.

---

### Technical Challenges

1. **GitHub Actions Limitations:**
   - No native GPU runner support (requires self-hosted)
   - No CUDA toolkit pre-installed
   - Self-hosted runner setup is complex (security, networking, Docker)

2. **Build Environment Reproducibility:**
   - CUDA version must match PyTorch build
   - Driver version must support CUDA version
   - Kernel compilation takes 5-10 minutes

3. **Performance Testing Requirements:**
   - GPU must be idle (no other workloads)
   - Consistent thermal state (warmup required)
   - Multiple runs for statistical significance (adds time)

4. **Artifact Storage:**
   - .ncu-rep files: 2-20 MB each
   - Benchmark CSVs: 1-5 MB each
   - GitHub Actions artifact limit: 500 MB per run
   - Long-term storage costs add up

---

## Current Workarounds

### 1. Robust CPU Fallbacks

**All operations work without CUDA:**
```python
# Graceful degradation
if robocache._cuda_available and tensor.is_cuda:
    return _cuda_ops.resample(...)  # Fast GPU path
else:
    return _pytorch_cpu(...)  # Slow but correct CPU path
```

**Benefit:** CI can validate API correctness and functional behavior.

**Limitation:** Cannot catch GPU-specific bugs or performance regressions.

---

### 2. Deterministic Builds

**Reproducible compilation:**
- Fixed CUDA architectures: `-gencode arch=compute_80,code=sm_80` (A100)
- Fixed compiler flags: `-O3 --use_fast_math -std=c++17`
- Pinned dependencies: `requirements.txt` with exact versions

**Benefit:** Manual builds are reproducible across machines.

**Limitation:** Still manual - no automated verification.

---

### 3. Comprehensive Validation Reports

**Detailed documentation of manual runs:**
- System info: GPU model, driver, CUDA version, clocks
- Full benchmark results with statistical analysis
- Nsight profiling data (NCU/NSys)
- Acceptance criteria pass/fail

**Benefit:** External reviewers can assess validation quality.

**Limitation:** Labor-intensive, not real-time feedback.

---

### 4. Performance Regression Gates

**Smoke test with thresholds:**
```bash
python benchmarks/smoke.py --assert-min-throughput

# Exits with code 1 if performance regresses
# Thresholds: H100 > 5000 ops/s, A100 > 3000 ops/s
```

**Benefit:** Catch major regressions in manual validation.

**Limitation:** Runs manually, not in PR workflow.

---

## Roadmap

### Q4 2025 (Next 2 Months)

**Goal:** Improve manual validation process

- [x] Document GPU CI status honestly (this file)
- [ ] Automate manual validation with scripts
  - `scripts/weekly_validation.sh` - One-click validation
  - Outputs: CSV, Nsight traces, attestation report
- [ ] Publish attestation with signed hashes
  - Hardware config + benchmark results
  - GPG-signed or Sigstore attestation
- [ ] Add performance dashboard (static site)
  - Jekyll/Hugo site showing historical benchmarks
  - Hosted on GitHub Pages (free)

**Budget:** $0 (no cloud costs)

---

### Q1 2026

**Goal:** Semi-automated GPU CI

- [ ] Implement spot instance CI (Option C)
  - Trigger: Manual `workflow_dispatch` or nightly
  - Spin up Lambda Labs A100 spot instance
  - Run full validation suite
  - Publish results, tear down instance
- [ ] Cost target: < $100/month
- [ ] Frequency: Nightly (if spot available) or 3×/week

**Budget:** ~$50-100/month

---

### Q2 2026 (If Funded)

**Goal:** Continuous GPU CI

- [ ] Self-hosted GPU runner (Option B)
  - Dedicated A100 instance (24/7 availability)
  - Docker-based runners with CUDA pre-installed
  - Full integration with GitHub Actions
- [ ] Run on every PR (with caching)
- [ ] Real-time performance feedback

**Budget:** ~$800-2,000/month (A100 cloud) or $50K capex (on-prem)

---

## How External Contributors Can Help

### For PRs from External Contributors

**Current process:**
1. CPU CI runs automatically (linting, CPU tests)
2. Maintainers manually validate GPU changes on H100/A100
3. Results published before merge

**Estimated turnaround:** 2-7 days for GPU validation

**How to help:**
- Test on your own GPU hardware
- Include benchmark results in PR description
- Run `benchmarks/smoke.py` and include output
- Generate Nsight traces if possible

---

### For Organizations with GPU Resources

**Sponsorship opportunities:**
1. **Compute Credits:**
   - Lambda Labs, Paperspace, Runpod credits
   - Use for automated nightly validation
   
2. **Hardware Donation:**
   - Spare A100/H100 for self-hosted runner
   - Tax-deductible (we're a 501(c)(3) - pending)
   
3. **CI/CD Integration:**
   - Buildkite/CircleCI credits with GPU runners
   - We'll acknowledge in README

**Contact:** b@thegoatnote.com

---

## FAQ

### Q: Why not use free GPU CI services?

**A:** Most free tiers (Colab, Kaggle) prohibit automated CI usage and have strict time limits (30-60 min sessions). Enterprise plans are $200+/month.

### Q: Can I trust manual validation?

**A:** Yes, with caveats:
- ✅ We publish full system info, raw CSVs, Nsight traces
- ✅ Validation scripts are in repo (reproducible)
- ✅ Multiple architectures tested (H100, A100)
- ⚠️ Self-reported (no external verification yet)
- ⚠️ Weekly frequency (not real-time)

**Invitation:** Replicate our benchmarks and file an issue if results differ!

### Q: What happens if I submit a PR?

**A:** 
1. CPU CI runs automatically (~5 min)
2. If CPU tests pass, we manually test on GPU (~2-7 days)
3. Results posted in PR comments
4. Merge if all checks pass

### Q: How do I run GPU tests locally?

**A:**
```bash
# Prerequisites: NVIDIA GPU, CUDA 12.1+, PyTorch 2.5+

# Build
cd robocache
python setup.py develop

# Smoke test
python benchmarks/smoke.py

# Full benchmark
python bench/benchmark_harness.py --output my_run.csv

# Nsight profiling
ncu --set full python scripts/profile_trajectory.py
```

---

## Transparency Commitment

**We will always:**
- ✅ Clearly label automated vs manual testing
- ✅ Publish raw benchmark data (CSVs, JSONs)
- ✅ Document hardware configs and system info
- ✅ Admit when we cannot afford automated GPU CI

**We will never:**
- ❌ Claim automated GPU CI when it's manual
- ❌ Hide performance regressions
- ❌ Cherry-pick favorable benchmark runs
- ❌ Misrepresent validation status

---

**This document is a living record. Updates published with each quarterly review.**

**Last Updated:** November 8, 2025  
**Next Review:** February 1, 2026

