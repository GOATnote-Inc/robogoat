"""Version information for RoboCache."""

__version__ = "0.2.1"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Build metadata
__build_date__ = "2025-11-05"
__git_commit__ = "3708bef"  # Updated by CI
__cuda_version__ = "13.0"
__cutlass_version__ = "4.3.0"

# API version for backward compatibility tracking
__api_version__ = "0.2"  # Major.minor only, patch changes don't break API

