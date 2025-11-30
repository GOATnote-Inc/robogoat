"""
RoboCache Backend Implementations

Provides multiple backend implementations for each operation:
- CUDA: Hand-optimized CUTLASS kernels (H100, highest performance)
- PyTorch: Native PyTorch operations (CPU/GPU fallback, compatibility)
- Triton: Auto-tuned kernels (future, experimental)

Backend selection can be automatic (based on availability) or manual.
"""

from .pytorch_backend import PyTorchBackend
from .backend_selector import (
    BackendStatus,
    BackendType,
    get_backend_status,
    select_backend,
)

__all__ = [
    "PyTorchBackend",
    "BackendStatus",
    "select_backend",
    "get_backend_status",
    "BackendType",
]

