"""
RoboCache Backend Implementations

Provides multiple backend implementations for each operation:
- CUDA: Hand-optimized CUTLASS kernels (H100, highest performance)
- PyTorch: Native PyTorch operations (CPU/GPU fallback, compatibility)
- Triton: Auto-tuned kernels (future, experimental)

Backend selection can be automatic (based on availability) or manual.
"""

from .pytorch_backend import PyTorchBackend
from .backend_selector import select_backend, BackendType

__all__ = [
    "PyTorchBackend",
    "select_backend",
    "BackendType",
]

