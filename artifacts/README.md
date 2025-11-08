# RoboCache GPU Profiling Artifacts

This directory contains comprehensive GPU profiling data for RoboCache, validated using NVIDIA's professional profiling tools (NCU and Nsight Systems).

## 📊 Validated Hardware

| GPU | Architecture | CUDA | Status | Report |
|-----|--------------|------|--------|--------|
| **NVIDIA H100 PCIe** | SM90 (Hopper) | 13.0 | ✅ Validated | [H100 Report](h100/H100_NCU_NSIGHT_REPORT.md) |
| **NVIDIA A100** | SM80 (Ampere) | 12.0 | ✅ Validated | [A100 Validation](../docs/validation/A100_SM80_VALIDATION.md) |

---

## 🔬 Profiling Methodology

All profiling follows **expert standards** matching PyTorch, Triton, and FlashAttention-3:

### Tools Used
- **NCU (Nsight Compute):** Kernel-level performance analysis with `--set full` metrics
- **Nsys (Nsight Systems):** End-to-end timeline and API trace analysis
- **Configuration:** DCGM disabled, `perf_event_paranoid=0`, 100+ iterations per benchmark

### Metrics Collected
- Warp occupancy & active warps
- Memory bandwidth utilization (DRAM + L1/L2)
- SM throughput & instruction mix
- Kernel launch overhead
- CPU-GPU synchronization latency

---

## 📁 Artifact Structure

```
artifacts/
├── README.md                                 # This file
├── h100/                                     # H100 (SM90) profiling data
│   ├── H100_NCU_NSIGHT_REPORT.md            # Comprehensive expert report
│   ├── gpu_info.txt                         # GPU specs and driver version
│   ├── ncu_reports/
│   │   ├── robocache_h100_full.ncu-rep      # Full NCU binary report (22MB)
│   │   ├── metrics.csv                      # Extracted metrics (CSV)
│   │   └── ncu_full_output.txt              # Console output
│   └── nsys_reports/
│       ├── robocache_h100_e2e.nsys-rep      # Nsight Systems timeline (4MB)
│       ├── robocache_h100_e2e.sqlite        # SQLite database for analysis
│       └── nsys_output.txt                  # Stats summary
└── a100/                                     # A100 (SM80) profiling data (future)
```

---

## 🚀 Key Performance Results

### NVIDIA H100 PCIe (SM90)

| Metric | Voxelization | Trajectory Resampling |
|--------|--------------|------------------------|
| **Kernel Latency** | 9.86 μs | 29.03 μs |
| **Throughput** | 16.86 B points/sec | 261.65 M samples/sec |
| **Warp Occupancy** | 78.62% | N/A |
| **Memory BW** | 51.82% | 14.85% |
| **Status** | ✅ Production-ready | ✅ Production-ready |

### NVIDIA A100 (SM80)

| Metric | Voxelization | Trajectory Resampling |
|--------|--------------|------------------------|
| **P50 Latency** | 0.036 ms | 0.063 ms |
| **Throughput** | 11.76 B points/sec | 126.27 M samples/sec |
| **Memory Stability** | ✅ 0 MB growth (10K iter) | ✅ 0 MB growth (10K iter) |
| **Status** | ✅ Production-ready | ✅ Production-ready |

---

## 🔍 Viewing the Reports

### NCU Reports (.ncu-rep)
```bash
# Open in Nsight Compute GUI
nsight-compute --import artifacts/h100/ncu_reports/robocache_h100_full.ncu-rep

# Or analyze via CLI
ncu --import robocache_h100_full.ncu-rep --page details
```

### Nsight Systems Reports (.nsys-rep)
```bash
# Open in Nsight Systems GUI
nsys-ui artifacts/h100/nsys_reports/robocache_h100_e2e.nsys-rep

# Or export stats via CLI
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum artifacts/h100/nsys_reports/robocache_h100_e2e.nsys-rep
```

### CSV Metrics
```bash
# Quick analysis with pandas
python3 -c "
import pandas as pd
df = pd.read_csv('artifacts/h100/ncu_reports/metrics.csv')
print(df[df['Kernel Name'].str.contains('voxelize|resample')][['Kernel Name', 'sm__warps_active.avg.pct_of_peak_sustained_active', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed']])
"
```

---

## 📝 Reproducibility

All results are **100% reproducible** using the provided scripts:

1. **NCU Profiling:**
   ```bash
   cd robocache
   sudo systemctl stop dcgm  # Disable DCGM interference
   ncu --set full --export ncu_report python3 benchmark.py
   ```

2. **Nsight Systems Profiling:**
   ```bash
   nsys profile --trace=cuda,nvtx,osrt --stats=true python3 benchmark.py
   ```

3. **Automated CI Validation:**
   ```bash
   # Runs automatically on H100/A100 GPU runners
   .github/workflows/gpu_ci_h100.yml
   .github/workflows/gpu_ci_a100.yml
   ```

---

## ✅ Validation Status

- ✅ **H100 (SM90):** Full NCU + Nsys profiling complete
- ✅ **A100 (SM80):** Functional benchmarks + memory leak tests complete
- 🔄 **L4 (SM89):** Pending (inference-optimized workload)

---

## 📖 References

- [H100 Expert Report](h100/H100_NCU_NSIGHT_REPORT.md) - Comprehensive NCU/Nsys analysis
- [A100 Validation](../docs/validation/A100_SM80_VALIDATION.md) - Performance benchmarks
- [GPU CI Documentation](../docs/GPU_CI_STATUS.md) - Automated testing infrastructure

---

**Last Updated:** 2025-11-08  
**Methodology:** Matches NVIDIA expert standards (PyTorch, Triton, FlashAttention-3)  
**Contact:** For questions about profiling methodology, see [CONTRIBUTING.md](../CONTRIBUTING.md)

