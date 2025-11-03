"""
Prometheus Metrics Integration for RoboCache

This module provides production-grade observability for RoboCache operations,
enabling monitoring of throughput, latency, bandwidth utilization, and errors
in distributed GPU preprocessing pipelines.

Features:
- Prometheus metrics (Counter, Histogram, Gauge, Summary)
- Pushgateway support for batch jobs
- Context managers for automatic timing
- GPU utilization tracking
- Bandwidth calculation

Usage:
    from robocache.observability import RoboCacheMetrics

    metrics = RoboCacheMetrics(
        job_name='trajectory_preprocessing',
        pushgateway='prometheus-pushgateway:9091'
    )

    with metrics.timer('resample_duration_seconds'):
        result = robocache.resample_trajectories(...)
"""

import time
import torch
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, CollectorRegistry,
        push_to_gateway, delete_from_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class RoboCacheMetrics:
    """
    Prometheus metrics collector for RoboCache operations.

    Tracks performance metrics for GPU-accelerated trajectory preprocessing:
    - Throughput (trajectories/second)
    - Latency (kernel execution time)
    - Bandwidth (GPU memory bandwidth utilization)
    - GPU utilization (compute and memory)
    - Error rates

    Args:
        job_name: Identifier for this preprocessing job
        pushgateway: URL of Prometheus Pushgateway (e.g., 'localhost:9091')
        namespace: Prometheus namespace (default: 'robocache')
        labels: Additional labels to attach to all metrics
    """

    def __init__(
        self,
        job_name: str = 'robocache',
        pushgateway: Optional[str] = None,
        namespace: str = 'robocache',
        labels: Optional[Dict[str, str]] = None
    ):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required for metrics. "
                "Install with: pip install prometheus-client"
            )

        self.job_name = job_name
        self.pushgateway = pushgateway
        self.namespace = namespace
        self.labels = labels or {}

        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics collectors."""

        # Counter: Total trajectories processed
        self.trajectories_processed = Counter(
            f'{self.namespace}_trajectories_processed_total',
            'Total number of trajectories resampled',
            labelnames=['gpu_id', 'dtype', 'job'],
            registry=self.registry
        )

        # Counter: Total bytes processed
        self.bytes_processed = Counter(
            f'{self.namespace}_bytes_processed_total',
            'Total bytes read and written',
            labelnames=['gpu_id', 'operation', 'job'],  # operation: read/write
            registry=self.registry
        )

        # Counter: Error counter
        self.errors = Counter(
            f'{self.namespace}_errors_total',
            'Total number of errors',
            labelnames=['gpu_id', 'error_type', 'job'],
            registry=self.registry
        )

        # Histogram: Resampling duration
        self.resample_duration = Histogram(
            f'{self.namespace}_resample_duration_seconds',
            'Time spent in resampling kernel',
            labelnames=['batch_size', 'dtype', 'gpu_id', 'job'],
            buckets=[
                0.0001, 0.0002, 0.0005,  # 0.1-0.5 ms
                0.001, 0.002, 0.005,      # 1-5 ms
                0.01, 0.02, 0.05,         # 10-50 ms
                0.1, 0.2, 0.5,            # 100-500 ms
                1.0, 2.0, 5.0             # 1-5 seconds
            ],
            registry=self.registry
        )

        # Gauge: Current GPU memory usage
        self.gpu_memory_allocated = Gauge(
            f'{self.namespace}_gpu_memory_allocated_bytes',
            'Current GPU memory allocated',
            labelnames=['gpu_id', 'job'],
            registry=self.registry
        )

        self.gpu_memory_reserved = Gauge(
            f'{self.namespace}_gpu_memory_reserved_bytes',
            'Current GPU memory reserved',
            labelnames=['gpu_id', 'job'],
            registry=self.registry
        )

        # Gauge: GPU memory bandwidth
        self.bandwidth_utilization = Gauge(
            f'{self.namespace}_bandwidth_gbps',
            'Achieved memory bandwidth in GB/s',
            labelnames=['gpu_id', 'dtype', 'job'],
            registry=self.registry
        )

        # Gauge: GPU compute utilization
        self.gpu_compute_utilization = Gauge(
            f'{self.namespace}_gpu_compute_utilization_percent',
            'GPU compute utilization percentage',
            labelnames=['gpu_id', 'job'],
            registry=self.registry
        )

        # Histogram: Batch size distribution
        self.batch_size_histogram = Histogram(
            f'{self.namespace}_batch_size',
            'Distribution of batch sizes processed',
            labelnames=['job'],
            buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            registry=self.registry
        )

        # Summary: Throughput statistics
        self.throughput = Summary(
            f'{self.namespace}_throughput_trajectories_per_second',
            'Throughput in trajectories per second',
            labelnames=['gpu_id', 'dtype', 'job'],
            registry=self.registry
        )

    @contextmanager
    def timer(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for timing operations.

        Usage:
            with metrics.timer('resample_duration_seconds', {'gpu_id': '0'}):
                result = robocache.resample_trajectories(...)

        Args:
            metric_name: Name of the metric to time
            labels: Labels to attach to this timing measurement
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start

            if metric_name == 'resample_duration_seconds':
                _labels = labels or {}
                self.resample_duration.labels(
                    batch_size=_labels.get('batch_size', 'unknown'),
                    dtype=_labels.get('dtype', 'unknown'),
                    gpu_id=_labels.get('gpu_id', '0'),
                    job=self.job_name
                ).observe(duration)

    def record_batch(
        self,
        batch_size: int,
        source_length: int,
        target_length: int,
        action_dim: int,
        duration: float,
        dtype: str,
        gpu_id: Optional[int] = None
    ):
        """
        Record metrics for a batch of trajectories processed.

        Args:
            batch_size: Number of trajectories in batch
            source_length: Length of source trajectories
            target_length: Length of resampled trajectories
            action_dim: Dimensionality of action space
            duration: Time taken to process batch (seconds)
            dtype: Data type used ('bfloat16', 'float16', 'float32')
            gpu_id: GPU device ID (auto-detected if None)
        """
        if gpu_id is None:
            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        gpu_id_str = str(gpu_id)

        # Record trajectories processed
        self.trajectories_processed.labels(
            gpu_id=gpu_id_str,
            dtype=dtype,
            job=self.job_name
        ).inc(batch_size)

        # Calculate bytes processed
        dtype_bytes = {'bfloat16': 2, 'float16': 2, 'float32': 4}.get(dtype, 4)
        bytes_read = batch_size * source_length * action_dim * dtype_bytes
        bytes_written = batch_size * target_length * action_dim * dtype_bytes

        self.bytes_processed.labels(
            gpu_id=gpu_id_str,
            operation='read',
            job=self.job_name
        ).inc(bytes_read)

        self.bytes_processed.labels(
            gpu_id=gpu_id_str,
            operation='write',
            job=self.job_name
        ).inc(bytes_written)

        # Record duration
        self.resample_duration.labels(
            batch_size=str(batch_size),
            dtype=dtype,
            gpu_id=gpu_id_str,
            job=self.job_name
        ).observe(duration)

        # Calculate and record bandwidth
        total_bytes = bytes_read + bytes_written
        bandwidth_gbps = (total_bytes / duration) / 1e9
        self.bandwidth_utilization.labels(
            gpu_id=gpu_id_str,
            dtype=dtype,
            job=self.job_name
        ).set(bandwidth_gbps)

        # Record throughput
        throughput_traj_per_sec = batch_size / duration
        self.throughput.labels(
            gpu_id=gpu_id_str,
            dtype=dtype,
            job=self.job_name
        ).observe(throughput_traj_per_sec)

        # Record batch size distribution
        self.batch_size_histogram.labels(
            job=self.job_name
        ).observe(batch_size)

        # Update GPU memory metrics
        if torch.cuda.is_available():
            self.update_gpu_memory_metrics(gpu_id)

    def record_error(
        self,
        error_type: str,
        gpu_id: Optional[int] = None
    ):
        """
        Record an error event.

        Args:
            error_type: Type of error (e.g., 'oom', 'invalid_shape', 'cuda_error')
            gpu_id: GPU device ID
        """
        if gpu_id is None:
            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        self.errors.labels(
            gpu_id=str(gpu_id),
            error_type=error_type,
            job=self.job_name
        ).inc()

    def update_gpu_memory_metrics(self, gpu_id: int = 0):
        """
        Update GPU memory usage metrics.

        Args:
            gpu_id: GPU device ID to query
        """
        if not torch.cuda.is_available():
            return

        # Get memory stats
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)

        self.gpu_memory_allocated.labels(
            gpu_id=str(gpu_id),
            job=self.job_name
        ).set(allocated)

        self.gpu_memory_reserved.labels(
            gpu_id=str(gpu_id),
            job=self.job_name
        ).set(reserved)

    def push(self):
        """
        Push metrics to Prometheus Pushgateway.

        Use this in batch jobs where metrics need to be pushed
        rather than scraped by Prometheus.
        """
        if not self.pushgateway:
            raise ValueError(
                "Pushgateway URL not configured. "
                "Set pushgateway parameter in constructor."
            )

        try:
            push_to_gateway(
                self.pushgateway,
                job=self.job_name,
                registry=self.registry,
                grouping_key=self.labels
            )
        except Exception as e:
            print(f"Warning: Failed to push metrics to {self.pushgateway}: {e}")

    def delete(self):
        """
        Delete metrics from Prometheus Pushgateway.

        Use this to clean up metrics after job completion.
        """
        if not self.pushgateway:
            return

        try:
            delete_from_gateway(
                self.pushgateway,
                job=self.job_name,
                grouping_key=self.labels
            )
        except Exception as e:
            print(f"Warning: Failed to delete metrics from {self.pushgateway}: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected metrics.

        Returns:
            Dictionary with metric summaries
        """
        # This is a simplified summary - actual implementation would
        # query the metric collectors directly
        return {
            'job_name': self.job_name,
            'pushgateway': self.pushgateway,
            'namespace': self.namespace,
            'labels': self.labels,
        }


class MetricsContext:
    """
    Context manager for automatic metrics recording.

    Usage:
        metrics = RoboCacheMetrics(...)

        with MetricsContext(metrics, batch_size=256, dtype='bfloat16'):
            result = robocache.resample_trajectories(...)
        # Metrics are automatically recorded on exit
    """

    def __init__(
        self,
        metrics: RoboCacheMetrics,
        batch_size: int,
        source_length: int,
        target_length: int,
        action_dim: int,
        dtype: str,
        gpu_id: Optional[int] = None,
        push_on_exit: bool = False
    ):
        self.metrics = metrics
        self.batch_size = batch_size
        self.source_length = source_length
        self.target_length = target_length
        self.action_dim = action_dim
        self.dtype = dtype
        self.gpu_id = gpu_id
        self.push_on_exit = push_on_exit
        self.start_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - self.start_time

        if exc_type is not None:
            # Record error
            error_type = exc_type.__name__
            self.metrics.record_error(error_type, self.gpu_id)
        else:
            # Record successful batch
            self.metrics.record_batch(
                batch_size=self.batch_size,
                source_length=self.source_length,
                target_length=self.target_length,
                action_dim=self.action_dim,
                duration=duration,
                dtype=self.dtype,
                gpu_id=self.gpu_id
            )

            if self.push_on_exit:
                self.metrics.push()
