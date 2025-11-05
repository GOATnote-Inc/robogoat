"""
Observability, logging, and telemetry for RoboCache.

Provides:
- Health checks
- Performance metrics collection
- Telemetry hooks (opt-in)
- Debugging utilities
"""

import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Collect and track performance metrics."""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.Lock()
        self._enabled = False
    
    def enable(self):
        """Enable metrics collection."""
        self._enabled = True
        logger.info("Performance metrics collection enabled")
    
    def disable(self):
        """Disable metrics collection."""
        self._enabled = False
    
    def record(self, operation: str, duration_ms: float, **kwargs):
        """Record a performance metric."""
        if not self._enabled:
            return
        
        with self._lock:
            self._metrics[operation].append({
                "duration_ms": duration_ms,
                "timestamp": time.time(),
                **kwargs
            })
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self._metrics:
                return {}
            
            durations = [m["duration_ms"] for m in self._metrics[operation]]
            
            if not durations:
                return {}
            
            return {
                "count": len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "mean_ms": sum(durations) / len(durations),
                "total_ms": sum(durations),
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        with self._lock:
            return {
                op: self.get_stats(op)
                for op in self._metrics.keys()
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
    
    def print_stats(self):
        """Print all statistics."""
        stats = self.get_all_stats()
        
        if not stats:
            print("No performance metrics collected.")
            print("Enable with: robocache.enable_metrics()")
            return
        
        print("=" * 80)
        print("RoboCache Performance Metrics")
        print("=" * 80)
        
        for operation, data in stats.items():
            print(f"\n{operation}:")
            print(f"  Count:    {data['count']}")
            print(f"  Mean:     {data['mean_ms']:.3f} ms")
            print(f"  Min:      {data['min_ms']:.3f} ms")
            print(f"  Max:      {data['max_ms']:.3f} ms")
            print(f"  Total:    {data['total_ms']:.1f} ms")
        
        print("=" * 80)


# Global metrics instance
_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Get the global metrics instance."""
    return _metrics


@contextmanager
def profile_operation(operation: str, **kwargs):
    """
    Context manager for profiling an operation.
    
    Example:
        with profile_operation("resample_trajectories", batch_size=64):
            result = resample_trajectories(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _metrics.record(operation, duration_ms, **kwargs)


def health_check() -> Dict[str, Any]:
    """
    Perform a comprehensive health check.
    
    Returns:
        Dict with health status of all components
    """
    from .backends import get_backend_status, get_backend_info
    from ._version import __version__
    from .config import get_config
    
    health = {
        "version": __version__,
        "status": "healthy",
        "checks": {}
    }
    
    # Check PyTorch
    try:
        import torch
        health["checks"]["pytorch"] = {
            "status": "ok",
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            health["checks"]["pytorch"]["cuda_version"] = torch.version.cuda
            health["checks"]["pytorch"]["gpu_count"] = torch.cuda.device_count()
            health["checks"]["pytorch"]["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        health["checks"]["pytorch"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check backends
    try:
        backend_status = get_backend_status()
        backend_info = get_backend_info()
        
        health["checks"]["backends"] = {
            "status": "ok",
            "cuda": backend_status.cuda_available,
            "pytorch": backend_status.pytorch_available,
            "triton": backend_status.triton_available,
            "default": backend_info["default_backend"],
        }
        
        if not (backend_status.cuda_available or backend_status.pytorch_available):
            health["status"] = "critical"
            health["checks"]["backends"]["status"] = "error"
            health["checks"]["backends"]["error"] = "No backends available"
    
    except Exception as e:
        health["checks"]["backends"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "critical"
    
    # Check configuration
    try:
        config = get_config()
        health["checks"]["config"] = {
            "status": "ok",
            "backend": config.backend,
            "cache_dir": str(config.cache_dir) if config.cache_dir else None,
        }
    except Exception as e:
        health["checks"]["config"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check metrics collection
    health["checks"]["metrics"] = {
        "status": "ok",
        "enabled": _metrics._enabled,
        "operations_tracked": len(_metrics._metrics),
    }
    
    return health


def print_health_check():
    """Print formatted health check results."""
    health = health_check()
    
    status_icon = {
        "healthy": "✓",
        "degraded": "⚠",
        "critical": "✗"
    }
    
    print("=" * 80)
    print(f"RoboCache Health Check {status_icon.get(health['status'], '?')} {health['status'].upper()}")
    print("=" * 80)
    print(f"Version: {health['version']}")
    print()
    
    for check_name, check_data in health["checks"].items():
        status = check_data.get("status", "unknown")
        icon = {"ok": "✓", "error": "✗", "warning": "⚠"}.get(status, "?")
        
        print(f"{icon} {check_name.upper()}: {status}")
        
        for key, value in check_data.items():
            if key != "status":
                print(f"    {key}: {value}")
        print()
    
    print("=" * 80)
    
    return health


def log_operation(operation: str, level: str = "INFO", **kwargs):
    """
    Log an operation with structured data.
    
    Args:
        operation: Operation name
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        **kwargs: Additional structured data to log
    """
    log_func = getattr(logger, level.lower())
    
    msg = f"Operation: {operation}"
    if kwargs:
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        msg += f" | {details}"
    
    log_func(msg)


# Telemetry placeholder (opt-in only, currently disabled)
class Telemetry:
    """
    Telemetry collection (opt-in only).
    
    Currently disabled by default. Future: anonymous usage statistics
    to help prioritize features and identify common issues.
    """
    
    def __init__(self):
        self._enabled = False
    
    def enable(self):
        """Enable telemetry (opt-in)."""
        self._enabled = True
        logger.info("Telemetry enabled (opt-in)")
    
    def disable(self):
        """Disable telemetry."""
        self._enabled = False
    
    def record_event(self, event_type: str, **data):
        """Record a telemetry event."""
        if not self._enabled:
            return
        
        # Future: Send to telemetry backend
        # For now: just log locally
        logger.debug(f"Telemetry event: {event_type} | {data}")


_telemetry = Telemetry()


def get_telemetry() -> Telemetry:
    """Get the global telemetry instance."""
    return _telemetry


__all__ = [
    "PerformanceMetrics",
    "get_metrics",
    "profile_operation",
    "health_check",
    "print_health_check",
    "log_operation",
    "Telemetry",
    "get_telemetry",
]

