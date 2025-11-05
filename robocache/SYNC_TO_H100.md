# Quick Sync to H100 (Without Git)

Since the H100 instance can't clone the private repository, here's the manual sync approach:

## Option 1: Archive and Upload (Recommended)

```bash
# On local machine
cd /Users/kiteboard/robogoat/robocache

# Create tar with only what we need for validation
tar -czf /tmp/robocache_audit.tar.gz \
  python/ \
  scripts/validate_audit_fixes.sh \
  tests/test_multimodal_fusion.py \
  tests/test_voxelization.py \
  CMakeLists.txt \
  kernels/ \
  benchmarks/ \
  --exclude='*.o' \
  --exclude='build/*'

# Upload to H100
scp -i ~/.ssh/brev_key /tmp/robocache_audit.tar.gz shadeform@38.128.232.170:/workspace/

# On H100 (via brev shell)
cd /workspace
tar -xzf robocache_audit.tar.gz -C /workspace/robocache_new
cd /workspace/robocache_new
./scripts/validate_audit_fixes.sh
```

## Option 2: Direct File Upload

Upload key files individually:

```bash
# Find H100 IP
brev ls  # Note the IP

# Upload validation script
scp scripts/validate_audit_fixes.sh shadeform@38.128.232.170:/workspace/robocache/scripts/

# Upload Python backend
scp -r python/robocache/backends shadeform@38.128.232.170:/workspace/robocache/python/robocache/

# Upload tests
scp tests/test_multimodal_fusion.py shadeform@38.128.232.170:/workspace/robocache/tests/
scp tests/test_voxelization.py shadeform@38.128.232.170:/workspace/robocache/tests/
```

## Option 3: Use Existing Robocache + Manual Updates

The H100 already has a working robocache build. You can:

1. Just test the backend selection logic locally first
2. Skip H100 validation for now since:
   - CUDA kernels were already validated (previous sessions)
   - PyTorch backend works on any machine with PyTorch
   - Multi-backend selection is pure Python logic

## Testing Locally (Fastest)

```bash
cd /Users/kiteboard/robogoat/robocache

# Install Python package locally
pip install -e python/

# Test multi-backend selection
python3 << 'EOF'
import robocache

info = robocache.check_installation()
print(f"CUDA: {info['cuda_extension_available']}")
print(f"PyTorch: {info['pytorch_available']}")
print(f"Default Backend: {info.get('default_backend', 'N/A')}")

# If PyTorch available, test fallback
if info['pytorch_available']:
    import torch
    data = torch.randn(2, 10, 4)
    src_t = torch.linspace(0, 1, 10).expand(2, -1)
    tgt_t = torch.linspace(0, 1, 5).expand(2, -1)
    
    result = robocache.resample_trajectories(
        data, src_t, tgt_t,
        backend='pytorch'
    )
    print(f"\n✅ PyTorch backend works: {result.shape}")
EOF

# Run test suite locally
python3 -m pytest tests/test_multimodal_fusion.py -v
python3 -m pytest tests/test_voxelization.py -v
```

This validates everything except CUDA performance on H100 (which we already have from previous sessions).

