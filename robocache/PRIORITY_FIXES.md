# Development Roadmap: 4-Week Plan

**Status:** Active Development  
**Timeline:** November 5 - December 3, 2025

**Owner:** Expert CUDA/NVIDIA Engineer (15+ years)  
**Start:** November 5, 2025  
**Target:** December 3, 2025 (4 weeks)

---

## Week 1: Complete the Python API (Nov 5-12)

### Day 1-2: Multimodal Fusion API
**Goal:** `robocache.fuse_multimodal()` works

**Tasks:**
- [ ] Create PyBind11 bindings for `multimodal_fusion.cu`
- [ ] Add to `_cuda_ext.py` load sources
- [ ] Expose in `__init__.py` public API
- [ ] Write PyTorch CPU fallback
- [ ] Add basic unit test
- [ ] Document usage with example

**Deliverable:** Working Python function, tested on H100

---

### Day 3-4: Voxelization API
**Goal:** `robocache.voxelize_point_cloud()` works

**Tasks:**
- [ ] Create PyBind11 bindings for `point_cloud_voxelization.cu`
- [ ] Add to `_cuda_ext.py` load sources
- [ ] Expose in `__init__.py` public API
- [ ] Write PyTorch CPU fallback
- [ ] Add basic unit test
- [ ] Document usage with example

**Deliverable:** Working Python function, tested on H100

---

### Day 5: Integration Testing
**Goal:** All 3 operations work together

**Tasks:**
- [ ] Test all 3 APIs in sequence
- [ ] Fix any JIT compilation issues
- [ ] Add end-to-end smoke test
- [ ] Update README with all 3 examples
- [ ] Run linter, fix all warnings

**Deliverable:** Complete Python package with all operations

---

## Week 2: End-to-End Dataset Integration (Nov 12-19)

### Day 6-7: RT-X DataLoader
**Goal:** Load and preprocess RT-X data with RoboCache

**Tasks:**
- [ ] Create `robocache/datasets/rtx.py`
- [ ] Implement DataLoader with trajectory resampling
- [ ] Add multimodal fusion (vision + proprio)
- [ ] Benchmark vs PyTorch baseline
- [ ] Measure GPU utilization (nvidia-smi)

**Code Structure:**
```python
class RT_X_DataLoader:
    def __init__(self, dataset_path, batch_size, use_robocache=True):
        self.use_robocache = use_robocache
        # Load RT-X episodes
        
    def __iter__(self):
        for episode in self.episodes:
            # Preprocess with RoboCache or PyTorch
            if self.use_robocache:
                yield self._robocache_preprocess(episode)
            else:
                yield self._pytorch_preprocess(episode)
```

**Deliverable:** Working dataloader with benchmark results

---

### Day 8-9: End-to-End Pipeline Test
**Goal:** Prove 95%+ GPU utilization claim

**Tasks:**
- [ ] Build simple training loop (dummy model)
- [ ] Measure GPU utilization over 100 batches
- [ ] Profile with nvidia-smi, nvprof
- [ ] Compare RoboCache vs PyTorch dataloader
- [ ] Document results in `benchmarks/e2e_results.md`

**Metrics to capture:**
- GPU utilization % (target: 95%+)
- Throughput (samples/sec)
- Latency per batch
- Memory usage

**Deliverable:** Proof of end-to-end GPU saturation

---

### Day 10: CALVIN Integration (Stretch)
**Goal:** Second dataset to prove generality

**Tasks:**
- [ ] Create `robocache/datasets/calvin.py`
- [ ] Similar structure to RT-X loader
- [ ] Benchmark and document

**Deliverable:** Second dataset integration (if time permits)

---

## Week 3: Distribution & Multi-Hardware (Nov 19-26)

### Day 11-12: Wheel Building
**Goal:** `pip install robocache` just works

**Tasks:**
- [ ] Setup `cibuildwheel` in CI
- [ ] Build wheels for:
  - cu118 (CUDA 11.8)
  - cu121 (CUDA 12.1)
  - cu124 (CUDA 12.4)
- [ ] Test installation on clean system
- [ ] Upload to TestPyPI
- [ ] Document installation

**Deliverable:** Prebuilt wheels, <5 min install

---

### Day 13: A100 Validation
**Goal:** Prove it works beyond H100

**Tasks:**
- [ ] Rent A100 instance (Lambda Labs/Vast.ai)
- [ ] Run full test suite
- [ ] Benchmark all 3 operations
- [ ] Fix any SM80 issues
- [ ] Document A100 results

**Deliverable:** Multi-GPU validation proof

---

### Day 14: RTX 4090 Validation (Consumer GPU)
**Goal:** Works on prosumer hardware

**Tasks:**
- [ ] Test on RTX 4090 (SM89)
- [ ] Run benchmarks
- [ ] Fix any compatibility issues
- [ ] Document results

**Deliverable:** Consumer GPU support

---

### Day 15: CI/CD Setup
**Goal:** Automated testing for all configs

**Tasks:**
- [ ] GitHub Actions for:
  - Unit tests (CPU)
  - CUDA tests (if GPU available)
  - Wheel building
  - Documentation build
- [ ] Add badges to README
- [ ] Setup automated releases

**Deliverable:** Production CI/CD pipeline

---

## Week 4: Performance Tuning & Polish (Nov 26-Dec 3)

### Day 16-17: Large-Batch Optimization
**Goal:** Hit 60-80% DRAM BW for large problems

**Tasks:**
- [ ] Benchmark trajectory at B=256, T=2048
- [ ] Profile with NCU
- [ ] If <60% DRAM, investigate TMA
- [ ] Implement and validate
- [ ] Document results

**Deliverable:** Large-scale performance data

---

### Day 18: Kernel Fusion
**Goal:** Fuse voxelization count+occupancy passes

**Tasks:**
- [ ] Implement fused kernel
- [ ] Benchmark vs 2-pass
- [ ] If >2x speedup, make default
- [ ] Document

**Deliverable:** Optimized voxelization (if beneficial)

---

### Day 19: Documentation Polish
**Goal:** Production-quality docs

**Tasks:**
- [ ] Update README with honest status
- [ ] Add installation guide
- [ ] Add usage examples for all 3 ops
- [ ] Add troubleshooting section
- [ ] Add contributing guide
- [ ] Fix all doc inconsistencies

**Deliverable:** Complete, honest documentation

---

### Day 20: Final Validation
**Goal:** Everything works, everything's documented

**Tasks:**
- [ ] Run full test suite on H100, A100, RTX 4090
- [ ] Verify all examples work
- [ ] Check all documentation links
- [ ] Final performance benchmarks
- [ ] Tag v0.2.0 release

**Deliverable:** Production-ready v0.2.0

---

## Success Criteria (4 Weeks)

### Must Have ✅
- [ ] All 3 operations in Python API (trajectory, fusion, voxelization)
- [ ] PyTorch CPU fallbacks for all operations
- [ ] RT-X dataloader with end-to-end benchmark
- [ ] GPU utilization proof (with actual %)
- [ ] Prebuilt wheels (cu118, cu121, cu124)
- [ ] Validation on H100 + A100
- [ ] Honest documentation (no false claims)
- [ ] CI/CD for automated testing

### Nice to Have 🎯
- [ ] CALVIN dataloader
- [ ] RTX 4090 validation
- [ ] TMA optimization for large batches
- [ ] Fused voxelization kernel
- [ ] Multi-GPU tests
- [ ] Docker images

### Success Metrics
- **API completeness:** 3/3 operations exposed
- **Hardware validation:** 2+ GPUs (H100, A100)
- **End-to-end proof:** GPU utilization % documented
- **Install time:** <5 minutes with wheels
- **Test coverage:** >80% for exposed APIs

---

## Resource Requirements

### Hardware Access
- ✅ H100 (have via Shadeform)
- ⏳ A100 (rent for 1 day: ~$50)
- ⏳ RTX 4090 (local or rent: ~$20)

### Time Commitment
- **Full-time (40 hrs/week):** 4 weeks  
- **Part-time (20 hrs/week):** 8 weeks  
- **Focused sprints (10 hrs/week):** 16 weeks

### Budget
- GPU rental: ~$100
- PyPI/TestPyPI: Free
- CI/CD (GitHub Actions): Free for public repos
- **Total:** ~$100

---

## Risk Mitigation

### Risk 1: JIT Compilation Issues
**Mitigation:** We've already proven JIT works on H100. Use same pattern for other ops.

### Risk 2: Dataset Access
**Mitigation:** RT-X is public. CALVIN is public. No access issues.

### Risk 3: Hardware Availability
**Mitigation:** Can rent A100/4090 on-demand. Validated process.

### Risk 4: Time Overruns
**Mitigation:** MVP is Week 1-2. Everything else is additive.

---

## Weekly Check-ins

### End of Week 1
**Question:** Do all 3 operations work in Python?  
**Evidence:** `import robocache; robocache.fuse_multimodal(...)` works

### End of Week 2
**Question:** Do we have end-to-end GPU utilization proof?  
**Evidence:** Benchmark results showing % utilization

### End of Week 3
**Question:** Can anyone `pip install robocache` and run it?  
**Evidence:** Wheels on TestPyPI, installation takes <5 min

### End of Week 4
**Question:** Is this ready for NVIDIA to evaluate?  
**Evidence:** Complete API, real dataset, honest docs

---

## Post-MVP Roadmap (After 4 Weeks)

### Month 2
- Multi-GPU support (DataParallel, DDP)
- Triton kernels for auto-tuning
- More datasets (RoboMimic, Bridge)
- Advanced optimizations (TMA, persistent threads)

### Month 3
- Independent validation (NVIDIA partner lab)
- Research paper submission
- Conference demos
- Production adoption by 1+ robotics teams

---

**Next Step:** Start Week 1, Day 1 - Multimodal Fusion API  
**Owner:** Expert CUDA/NVIDIA Engineer  
**Status:** Ready to execute

**Let's fucking build this properly.** ✅

