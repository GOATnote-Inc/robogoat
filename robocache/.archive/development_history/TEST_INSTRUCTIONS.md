# H100 Testing Instructions

## Upload Package

```bash
# On local machine
cd /Users/kiteboard/robogoat/robocache
scp optimization_package.tar.gz <h100-host>:/workspace/
scp RUN_ON_H100.sh <h100-host>:/workspace/
```

## Run on H100

```bash
# On H100
cd /workspace
chmod +x RUN_ON_H100.sh
./RUN_ON_H100.sh
```

## Expected Output

### C++ Benchmark
```
Batch |  Src |  Tgt |  Dim |       Kernel | Time(ms)   | BW(GB/s)   | Eff(%)     | Speedup
   256 |  100 |   50 |   32 |     Baseline |      ~2.5  |     ~360   |    ~12     | -
   256 |  100 |   50 |   32 |    Optimized |      ~1.3  |     ~700   |    ~23     |  ~1.9x
```

### Python Tests
```
TEST 1: Correctness - Max diff < 1e-4
TEST 2: Performance - Speedup 1.5-2.0x
TEST 3: Mixed Precision - All dtypes pass
TEST 4: Scaling - Best for src_len <= 512
```

### NCU Metrics
```
Baseline:  dram__throughput ~12%, sm__throughput ~15%
Optimized: dram__throughput ~20%, sm__throughput ~25%
```

## Retrieve Results

```bash
scp <h100-host>:/workspace/robocache/*_results.txt ./
```

