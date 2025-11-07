"""Metrics collection for RoboCache operations"""

import time
from collections import defaultdict
from typing import Dict, Optional
import threading


class MetricsCollector:
    """Thread-safe metrics collector"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
    
    def increment(self, name: str, value: int = 1):
        """Increment counter"""
        with self._lock:
            self._counters[name] += value
    
    def record_time(self, name: str, ms: float):
        """Record timing"""
        with self._lock:
            self._timers[name].append(ms)
    
    def set_gauge(self, name: str, value: float):
        """Set gauge value"""
        with self._lock:
            self._gauges[name] = value
    
    def get_metrics(self) -> Dict:
        """Get all metrics"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'timers': {k: {
                    'count': len(v),
                    'mean': sum(v) / len(v) if v else 0,
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0,
                } for k, v in self._timers.items()},
                'gauges': dict(self._gauges),
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics


def export_prometheus(output_file: str = "metrics.prom"):
    """Export metrics in Prometheus format"""
    metrics = _metrics.get_metrics()
    
    with open(output_file, 'w') as f:
        # Counters
        for name, value in metrics['counters'].items():
            f.write(f"robocache_{name}_total {value}\n")
        
        # Timers
        for name, stats in metrics['timers'].items():
            f.write(f"robocache_{name}_count {stats['count']}\n")
            f.write(f"robocache_{name}_mean_ms {stats['mean']}\n")
            f.write(f"robocache_{name}_min_ms {stats['min']}\n")
            f.write(f"robocache_{name}_max_ms {stats['max']}\n")
        
        # Gauges
        for name, value in metrics['gauges'].items():
            f.write(f"robocache_{name} {value}\n")

