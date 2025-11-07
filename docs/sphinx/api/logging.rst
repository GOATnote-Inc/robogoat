Logging & Metrics
=================

Structured logging and metrics collection for production deployments.

Logging
-------

.. automodule:: robocache.logging
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from robocache.logging import get_logger, set_log_level
   import logging

   # Set log level
   set_log_level(logging.DEBUG)

   # Get logger
   logger = get_logger()

   # Log operation with timing
   with logger.log_operation("multimodal_fusion", batch_size=4):
       fused = robocache.fuse_multimodal(...)
   
   # Output:
   # 2025-11-07 12:34:56 - robocache - INFO - Starting multimodal_fusion
   # 2025-11-07 12:34:56 - robocache - INFO - Completed multimodal_fusion (elapsed_ms: 0.018)

Metrics
-------

.. automodule:: robocache.metrics
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from robocache.metrics import get_metrics_collector, export_prometheus

   # Get collector
   metrics = get_metrics_collector()

   # Record metrics
   metrics.increment('multimodal_fusions_total')
   metrics.record_time('multimodal_fusion_latency_ms', 0.018)
   metrics.set_gauge('gpu_utilization_pct', 87.3)

   # Export to Prometheus
   export_prometheus('metrics.prom')

   # Output (metrics.prom):
   # robocache_multimodal_fusions_total 1
   # robocache_multimodal_fusion_latency_ms_count 1
   # robocache_multimodal_fusion_latency_ms_mean_ms 0.018
   # robocache_gpu_utilization_pct 87.3

Prometheus Integration
----------------------

**Grafana Dashboard:**

.. code-block:: yaml

   # prometheus.yml
   scrape_configs:
     - job_name: 'robocache'
       static_configs:
         - targets: ['localhost:8000']
       metrics_path: '/metrics'

**Example Queries:**

.. code-block:: promql

   # P50 latency
   histogram_quantile(0.5, robocache_multimodal_fusion_latency_ms)

   # Throughput (batches/sec)
   rate(robocache_multimodal_fusions_total[1m])

   # GPU utilization
   robocache_gpu_utilization_pct

