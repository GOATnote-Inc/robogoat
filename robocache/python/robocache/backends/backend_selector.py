"""
Backend Selection Logic

Automatically selects the best available backend or allows manual override.
Priority: CUDA > PyTorch (GPU) > PyTorch (CPU)
"""

from enum import Enum
from typing import Optional
import warnings


class BackendType(Enum):
    """Available backend types"""
    CUDA = "cuda"
    PYTORCH = "pytorch"
    AUTO = "auto"


class BackendSelector:
    """Manages backend availability and selection"""
    
    def __init__(self):
        self._cuda_available = False
        self._torch_available = False
        self._cuda_import_error = None
        
        # Check CUDA extension availability
        try:
            from .. import robocache_cuda
            self._cuda_available = True
        except ImportError as e:
            self._cuda_import_error = str(e)
        
        # Check PyTorch availability
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
    
    @property
    def cuda_available(self) -> bool:
        return self._cuda_available
    
    @property
    def pytorch_available(self) -> bool:
        return self._torch_available
    
    def select_backend(self, backend: Optional[str] = None) -> BackendType:
        """
        Select the best available backend.
        
        Args:
            backend: Optional manual override ('cuda', 'pytorch', or 'auto')
        
        Returns:
            BackendType: Selected backend
        
        Raises:
            RuntimeError: If requested backend is unavailable
        """
        if backend is None or backend == "auto":
            return self._select_auto()
        
        backend_lower = backend.lower()
        
        if backend_lower == "cuda":
            if not self._cuda_available:
                raise RuntimeError(
                    f"CUDA backend requested but not available.\n"
                    f"Import error: {self._cuda_import_error}\n\n"
                    f"To build the CUDA extension:\n"
                    f"  cd robocache && mkdir build && cd build\n"
                    f"  cmake .. && make -j\n\n"
                    f"Or use backend='pytorch' for CPU/PyTorch GPU fallback."
                )
            return BackendType.CUDA
        
        elif backend_lower == "pytorch":
            if not self._torch_available:
                raise RuntimeError(
                    "PyTorch backend requested but PyTorch is not installed.\n"
                    "Install with: pip install torch"
                )
            return BackendType.PYTORCH
        
        else:
            raise ValueError(
                f"Unknown backend: {backend}\n"
                f"Supported backends: 'cuda', 'pytorch', 'auto'"
            )
    
    def _select_auto(self) -> BackendType:
        """Automatically select best available backend"""
        if self._cuda_available:
            return BackendType.CUDA
        
        if self._torch_available:
            warnings.warn(
                "CUDA extension not available, falling back to PyTorch.\n"
                "Performance will be significantly slower (20-70x).\n"
                "For production use, build the CUDA extension:\n"
                "  cd robocache && mkdir build && cd build && cmake .. && make -j",
                UserWarning,
                stacklevel=3
            )
            return BackendType.PYTORCH
        
        raise RuntimeError(
            "No backend available. Install PyTorch or build CUDA extension.\n"
            "Install PyTorch: pip install torch\n"
            "Build CUDA: cd robocache && mkdir build && cd build && cmake .. && make -j"
        )


# Global singleton instance
_selector = BackendSelector()


def select_backend(backend: Optional[str] = None) -> BackendType:
    """
    Select backend for operations.
    
    Args:
        backend: 'cuda', 'pytorch', 'auto', or None (auto)
    
    Returns:
        BackendType: Selected backend
    """
    return _selector.select_backend(backend)


def is_cuda_available() -> bool:
    """Check if CUDA backend is available"""
    return _selector.cuda_available


def is_pytorch_available() -> bool:
    """Check if PyTorch backend is available"""
    return _selector.pytorch_available

