# Reproducible Performance Benchmarks

**Purpose:** Verify every performance claim in main README with exact configurations  
**Hardware:** H100 PCIe 80GB, A100 PCIe 80GB  
**Validation:** Each claim maps to specific config + measurement

---

## Quick Start

```bash
# On H100 instance
cd /workspace/robocache
python benchmarks/reproducible/scripts/run_all.py --hardware h100

# On A100 instance
python benchmarks/reproducible/scripts/run_all.py --hardware a100
```

---

## Benchmark Configurations

Each JSON config specifies exact test parameters for one README claim.

### Format

```json
{
  "claim_id": "multimodal_fusion_latency",
  "readme_claim": "H100: 0.018ms for 3 streams @ 100Hz -> 50Hz",
  "operation": "fuse_multimodal",
  "hardware": "h100",
  "parameters": {
    "batch_size": 4,
    "stream1_shape": [30, 512],
    "stream1_freq_hz": 30,
    "stream2_shape": [100, 64],
    "stream2_freq_hz": 100,
    "stream3_shape": [200, 12],
    "stream3_freq_hz": 200,
    "target_freq_hz": 50,
    "dtype": "bfloat16"
  },
  "acceptance_criteria": {
    "metric": "latency_ms",
    "target": 0.018,
    "tolerance": 0.01,
    "max_acceptable": 0.028
  },
  "ncu_metrics": [
    "Duration",
    "dram__bytes_read.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
  ]
}
```

---

## Claims Registry

| Claim ID | README Statement | Config File | Status |
|----------|------------------|-------------|--------|
| `multimodal_h100_latency` | "H100: 0.018ms multimodal fusion" | `multimodal_fusion_h100.json` | ⏳ Pending |
| `trajectory_h100_latency` | "H100: 2.6ms trajectory resample" | `trajectory_resample_h100.json` | ⏳ Pending |
| `voxel_h100_throughput` | "H100: >2.5B points/sec" | `voxelization_throughput_h100.json` | ⏳ Pending |
| `multimodal_a100_latency` | "A100: 0.057ms multimodal fusion" | `multimodal_fusion_a100.json` | ⏳ Pending |
| `trajectory_a100_latency` | "A100: 3.1ms trajectory resample" | `trajectory_resample_a100.json` | ⏳ Pending |

---

## Output Format

Each benchmark run produces:
- `results/<claim_id>_<timestamp>.json` - Structured results
- `results/<claim_id>_<timestamp>.txt` - Human-readable summary
- `results/<claim_id>_<timestamp>.ncu-rep` - NCU binary report (if enabled)

### Results JSON Schema

```json
{
  "claim_id": "multimodal_h100_latency",
  "timestamp": "2025-11-08T10:30:00Z",
  "hardware": {
    "gpu_name": "NVIDIA H100 PCIe",
    "cuda_version": "12.1",
    "driver_version": "535.104.05"
  },
  "measured": {
    "latency_ms": 0.019,
    "latency_std_ms": 0.002,
    "throughput_items_per_sec": 52631,
    "memory_mb": 145
  },
  "verdict": "PASS",
  "deviation_percent": 5.6,
  "notes": "Within 10% tolerance"
}
```

---

## Running Individual Benchmarks

```bash
# Run single benchmark
python scripts/run_single.py --config configs/multimodal_fusion_h100.json

# With NCU profiling
python scripts/run_single.py --config configs/multimodal_fusion_h100.json --ncu

# With verbose output
python scripts/run_single.py --config configs/multimodal_fusion_h100.json --verbose
```

---

## Validation Checklist

For each claim:
- [ ] Config file created with exact parameters
- [ ] Benchmark script executes successfully
- [ ] Measurement within tolerance
- [ ] NCU metrics captured
- [ ] Results committed to `results/`
- [ ] Verdict documented (PASS/FAIL)

---

## Maintenance

**Adding New Claim:**
1. Create config JSON in `configs/`
2. Add entry to Claims Registry table
3. Run benchmark and commit results
4. Update main README if claim changes

**Updating Existing Claim:**
1. Modify config JSON
2. Re-run benchmark
3. Update README to match measured values
4. Commit updated results

---

**Status:** Framework complete, benchmarks pending H100/A100 execution

