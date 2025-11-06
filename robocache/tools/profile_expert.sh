#!/usr/bin/env bash
set -euo pipefail

# =========================================================
#  RoboCache Expert Nsight Profiling Script (v1.0)
#  Targets: H100 (sm_90a), CUDA 13.0.2, PyTorch ≥2.5
# =========================================================
#  Generates:
#   - NSYS timeline traces  (.nsys-rep + .qdrep)
#   - NCU metric reports    (.ncu-rep + .json)
#   - Auto-diff regression  (.compare)
#   - Text summaries        (.txt)
# =========================================================

DATESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${1:-trajectory_h100}
OUTDIR="artifacts/profiling/${RUN_NAME}_${DATESTAMP}"
mkdir -p "${OUTDIR}"

# Set writable temp directory
export TMPDIR="${OUTDIR}/tmp"
mkdir -p "${TMPDIR}"

echo "=== [1/6] Environment Check ==="
nvidia-smi --query-gpu=name,driver_version,pstate,utilization.gpu,temperature.gpu --format=csv
which nvcc && nvcc --version | grep "release" || echo "WARNING: nvcc not in PATH"
which nsys || { echo "WARNING: nsys not found, skipping NSYS"; SKIP_NSYS=1; }
which ncu || { echo "WARNING: ncu not found, skipping NCU"; SKIP_NCU=1; }

echo ""
echo "=== [2/6] Baseline Run (functional smoke) ==="
python3 -m torch.utils.collect_env > "${OUTDIR}/env.txt"
python3 scripts/profile_trajectory.py 2>&1 | tee "${OUTDIR}/smoke.txt"

if [ "${SKIP_NSYS:-0}" == "0" ]; then
  echo ""
  echo "=== [3/6] Nsight Systems Timeline ==="
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o "${OUTDIR}/timeline" \
    python3 scripts/profile_trajectory.py

  echo ""
  echo "=== [3.1/6] Extracting NSYS Stats ==="
  nsys stats --report cuda_gpu_kernel_sum,cuda_gpu_mem_time_sum \
    "${OUTDIR}/timeline.nsys-rep" > "${OUTDIR}/nsys_summary.txt" || true
else
  echo ""
  echo "=== [3/6] NSYS SKIPPED ==="
fi

if [ "${SKIP_NCU:-0}" == "0" ]; then
  echo ""
  echo "=== [4/6] Nsight Compute Deep Metrics ==="
  ncu --set full --target-processes all \
    --metrics \
      sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,lts__t_sectors_srcunit_tex_op_read.sum.per_second,l1tex__t_sector_miss_rate.pct,smsp__sass_average_branch_targets_threads_uniform.pct \
    --force-overwrite \
    -o "${OUTDIR}/ncu_full" \
    python3 scripts/profile_trajectory.py || true

  echo ""
  echo "=== [5/6] Export & Compare Metrics ==="
  if [ -f "${OUTDIR}/ncu_full.ncu-rep" ]; then
    ncu --export json --export-file "${OUTDIR}/ncu_metrics.json" "${OUTDIR}/ncu_full.ncu-rep" || true
    if [ -f artifacts/baseline/ncu_baseline.ncu-rep ]; then
      echo "Comparing against baseline..."
      ncu --compare artifacts/baseline/ncu_baseline.ncu-rep "${OUTDIR}/ncu_full.ncu-rep" \
          > "${OUTDIR}/ncu_compare.txt" || true
    fi
  fi
else
  echo ""
  echo "=== [4-5/6] NCU SKIPPED ==="
fi

echo ""
echo "=== [6/6] Summary ==="
ls -lh "${OUTDIR}"
echo ""
echo "Reports written to: ${OUTDIR}"
echo "✅ PROFILING COMPLETE"

