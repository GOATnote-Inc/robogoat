"""Structured logging for RoboCache operations"""

import logging
import time
from contextlib import contextmanager
from typing import Optional


class RoboCacheLogger:
    """Structured logger with performance tracking"""
    
    def __init__(self, name: str = "robocache", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @contextmanager
    def log_operation(self, operation: str, **kwargs):
        """Context manager for logging operations with timing"""
        start = time.perf_counter()
        self.logger.info(f"Starting {operation}", extra=kwargs)
        
        try:
            yield
            elapsed = time.perf_counter() - start
            self.logger.info(
                f"Completed {operation}",
                extra={"elapsed_ms": elapsed * 1000, **kwargs}
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.logger.error(
                f"Failed {operation}: {e}",
                extra={"elapsed_ms": elapsed * 1000, **kwargs}
            )
            raise


# Global logger instance
_logger = RoboCacheLogger()


def get_logger() -> RoboCacheLogger:
    """Get global logger instance"""
    return _logger


def set_log_level(level: int):
    """Set logging level"""
    _logger.logger.setLevel(level)

