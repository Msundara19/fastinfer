from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import contextmanager

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
error_counter = Counter('prediction_errors_total', 'Total prediction errors')
batch_size_gauge = Gauge('current_batch_size', 'Current batch size')

@contextmanager
def track_time():
    """Context manager to track execution time"""
    start = time.time()
    yield lambda: time.time() - start