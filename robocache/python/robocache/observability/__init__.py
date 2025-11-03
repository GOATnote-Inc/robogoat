"""
RoboCache Observability Module

Provides production-grade monitoring and telemetry for GPU-accelerated
trajectory preprocessing pipelines.

Components:
- RoboCacheMetrics: Prometheus metrics collector
- MetricsContext: Context manager for automatic metric recording

Example:
    from robocache.observability import RoboCacheMetrics

    metrics = RoboCacheMetrics(
        job_name='preprocessing',
        pushgateway='pushgateway:9091'
    )

    metrics.record_batch(
        batch_size=256,
        source_length=100,
        target_length=50,
        action_dim=32,
        duration=0.0007,
        dtype='bfloat16'
    )

    metrics.push()  # Push to Prometheus Pushgateway
"""

from .metrics import RoboCacheMetrics, MetricsContext

__all__ = ['RoboCacheMetrics', 'MetricsContext']
