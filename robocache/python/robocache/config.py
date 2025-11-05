"""
Configuration management for RoboCache.

Supports configuration via:
1. Environment variables (highest priority)
2. Config file (JSON/YAML)
3. Programmatic API
4. Defaults (lowest priority)
"""

import os
import json
import logging
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """
    Global configuration for RoboCache.
    
    Environment variables (highest priority):
    - ROBOCACHE_BACKEND: Force backend ('cuda', 'pytorch', 'triton', 'auto')
    - ROBOCACHE_LOG_LEVEL: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    - ROBOCACHE_CUDA_VERBOSE: Verbose CUDA compilation (0 or 1)
    - ROBOCACHE_CACHE_DIR: Directory for compiled kernels
    - ROBOCACHE_DISABLE_TELEMETRY: Disable usage telemetry (0 or 1)
    """
    
    def __init__(self):
        # Backend settings
        self.backend: Optional[str] = os.getenv("ROBOCACHE_BACKEND", "auto")
        self.cuda_verbose: bool = bool(int(os.getenv("ROBOCACHE_CUDA_VERBOSE", "0")))
        
        # Logging settings
        self.log_level: str = os.getenv("ROBOCACHE_LOG_LEVEL", "INFO")
        
        # Cache settings
        self.cache_dir: Optional[Path] = self._get_cache_dir()
        
        # Telemetry settings
        self.disable_telemetry: bool = bool(int(os.getenv("ROBOCACHE_DISABLE_TELEMETRY", "1")))
        
        # Performance settings
        self.enable_profiling: bool = bool(int(os.getenv("ROBOCACHE_ENABLE_PROFILING", "0")))
        self.performance_warnings: bool = bool(int(os.getenv("ROBOCACHE_PERF_WARNINGS", "1")))
        
        # Numerical settings
        self.numerical_checks: bool = bool(int(os.getenv("ROBOCACHE_NUMERICAL_CHECKS", "0")))
        self.tolerance: float = float(os.getenv("ROBOCACHE_TOLERANCE", "1e-5"))
    
    def _get_cache_dir(self) -> Optional[Path]:
        """Get the cache directory for compiled kernels."""
        if "ROBOCACHE_CACHE_DIR" in os.environ:
            return Path(os.environ["ROBOCACHE_CACHE_DIR"])
        
        # Default: use torch's cache directory
        try:
            import torch
            torch_cache = Path(torch.utils.cpp_extension._get_build_directory("robocache", False))
            return torch_cache
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "backend": self.backend,
            "cuda_verbose": self.cuda_verbose,
            "log_level": self.log_level,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "disable_telemetry": self.disable_telemetry,
            "enable_profiling": self.enable_profiling,
            "performance_warnings": self.performance_warnings,
            "numerical_checks": self.numerical_checks,
            "tolerance": self.tolerance,
        }
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 60)
        print("RoboCache Configuration")
        print("=" * 60)
        for key, value in self.to_dict().items():
            print(f"{key:25s} = {value}")
        print("=" * 60)
    
    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        """Load configuration from JSON file."""
        config = cls()
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return config
        
        try:
            with open(config_file) as f:
                data = json.load(f)
            
            # Update config from file
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
        
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
        
        return config
    
    def save_to_file(self, config_file: Path):
        """Save configuration to JSON file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Config saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        
        # Setup logging based on config
        logging.basicConfig(
            level=getattr(logging, _config.log_level),
            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
    
    return _config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = None


__all__ = [
    "Config",
    "get_config",
    "set_config",
    "reset_config",
]

