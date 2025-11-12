"""Convenience wrapper that exposes the Python implementation of RoboCache."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import ModuleType
import sys as _sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_IMPL_ROOT = _PACKAGE_ROOT / "python" / "robocache"

if not _IMPL_ROOT.exists():  # pragma: no cover - defensive
    raise ImportError("Unable to locate RoboCache Python sources")


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    """Load a module from the given file path under ``module_name``."""
    loader = SourceFileLoader(module_name, str(file_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to create spec for {module_name}")
    module = module_from_spec(spec)
    _sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


# Preload modules that the implementation imports directly from ``robocache``
# during initialisation (e.g. ``from robocache import ops_fallback``).  Loading
# them first prevents circular import issues.
_ops_fallback_module = _load_module(
    "robocache.ops_fallback", _IMPL_ROOT / "ops_fallback.py"
)
globals()["ops_fallback"] = _ops_fallback_module
_sys.modules.setdefault("robocache.ops_fallback", _ops_fallback_module)

# Load the main implementation module.
_impl = _load_module("robocache._impl", _IMPL_ROOT / "__init__.py")

# Re-export the public API defined by the implementation.
__all__ = list(getattr(_impl, "__all__", []))
if not __all__:
    __all__ = [name for name in dir(_impl) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_impl, name)

# Backward compatibility: some tests expect the snake_case variant with an
# extra underscore between "point" and "cloud".
if "voxelize_point_cloud" not in globals() and "voxelize_pointcloud" in globals():
    def voxelize_point_cloud(points, grid_size, voxel_size, origin, backend=None):
        if "voxelize_occupancy" in globals():
            return globals()["voxelize_occupancy"](
                points, grid_size, voxel_size, origin, backend=backend
            )
        return globals()["voxelize_pointcloud"](
            points=points,
            features=None,
            grid_min=origin,
            voxel_size=voxel_size,
            grid_size=grid_size,
            mode="occupancy",
            backend=backend,
        )

    globals()["voxelize_point_cloud"] = voxelize_point_cloud
    __all__.append("voxelize_point_cloud")

# Propagate metadata for introspection tools.
__doc__ = getattr(_impl, "__doc__", None)
__version__ = getattr(_impl, "__version__", None)

# Ensure common submodules remain importable via ``robocache.<module>``.
_OPTIONAL_SUBMODULES: tuple[tuple[str, str], ...] = (
    ("backends", "backends.py"),
    ("config", "config.py"),
    ("datasets", "datasets/__init__.py"),
    ("metrics", "metrics.py"),
    ("temporal_fusion", "temporal_fusion.py"),
    ("logging", "logging.py"),
    ("observability", "observability.py"),
    ("sim_to_real", "sim_to_real.py"),
    ("cuda_graph_cache", "cuda_graph_cache.py"),
)

for module_name, relative_path in _OPTIONAL_SUBMODULES:
    module_path = _IMPL_ROOT / relative_path
    if not module_path.exists():
        continue
    full_name = f"robocache.{module_name}"
    module = _load_module(full_name, module_path)
    globals()[module_name] = module
    _sys.modules.setdefault(full_name, module)

# Register the implementation module itself for completeness.
_sys.modules.setdefault("robocache._impl", _impl)
