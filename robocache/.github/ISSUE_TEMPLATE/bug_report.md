---
name: Bug Report
about: Report a bug to help us improve RoboCache
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

Please provide detailed steps to reproduce the behavior:

1. Create a dataset/tensor with '...'
2. Call function '...'
3. Run command '...'
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened instead. Include full error messages and stack traces.

```
Paste error messages here
```

## System Information

Please provide the following information:

**GPU Information:**
- GPU Model: [e.g., NVIDIA H100, A100, RTX 4090]
- CUDA Version: [run `nvcc --version`]
- GPU Memory: [e.g., 80 GB]
- Driver Version: [run `nvidia-smi`]

**Software Versions:**
- RoboCache Version: [e.g., 0.1.0]
- PyTorch Version: [run `python -c "import torch; print(torch.__version__)"`]
- Python Version: [run `python --version`]
- Operating System: [e.g., Ubuntu 22.04]
- CMake Version: [run `cmake --version`]
- CUTLASS Version: [e.g., 4.3.0]

**Installation Method:**
- [ ] Built from source
- [ ] Pip install
- [ ] Docker container

## Minimal Reproducible Example

Please provide a minimal code example that reproduces the issue:

```python
import torch
import robocache

# Your minimal reproducible example here
```

## Additional Context

Add any other context about the problem here:
- Does the issue occur consistently or intermittently?
- Does it work with different batch sizes or data dimensions?
- Have you tried with different dtypes (FP32, FP16, BF16)?
- Any relevant configuration settings?

## Possible Solution (Optional)

If you have suggestions on how to fix the bug, please share them here.

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided all required system information
- [ ] I have included a minimal reproducible example
- [ ] I have included the full error message and stack trace
- [ ] I have tested with the latest version of RoboCache
