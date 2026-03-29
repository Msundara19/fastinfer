"""
Compare PyTorch vs ONNX performance
"""

import requests
import time
import statistics

def benchmark_endpoint(endpoint, n_requests=50):
    """Benchmark a single endpoint"""
    print(f"\nBenchmarking {endpoint}...")
    latencies = []

    for i in range(n_requests):
        with open('test_dog.jpg', 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"http://localhost:8000{endpoint}", files=files)

            if response.status_code == 503:
                print(f"  Endpoint unavailable: {response.json().get('detail', '')}")
                return None

            if response.status_code == 200:
                latencies.append(response.json()['latency_ms'])

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_requests} complete")

    if not latencies:
        return None

    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p95': sorted(latencies)[int(len(latencies) * 0.95)]
    }

def print_result(name, results, baseline_mean):
    speedup = baseline_mean / results['mean']
    print(f"\n  {name}:")
    print(f"    Mean:    {results['mean']:.2f}ms  ({speedup:.2f}x vs baseline)")
    print(f"    Median:  {results['median']:.2f}ms")
    print(f"    P95:     {results['p95']:.2f}ms")
    print(f"    Range:   {results['min']:.2f}ms - {results['max']:.2f}ms")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GPU-to-GPU Latency Comparison (sequential, single requests)")
    print("=" * 60)

    endpoints = [
        ("/predict",           "PyTorch MPS (GPU baseline)"),
        ("/predict/onnx",      "ONNX CoreML FP32"),
        ("/predict/coreml",    "Direct CoreML FP16"),
    ]

    results = {}
    for endpoint, name in endpoints:
        r = benchmark_endpoint(endpoint, n_requests=50)
        if r:
            results[name] = r
        else:
            print(f"  Skipping {name} — endpoint unavailable")

    if not results:
        print("No results collected.")
        sys.exit(1)

    baseline_mean = list(results.values())[0]['mean']

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, r in results.items():
        print_result(name, r, baseline_mean)

    print("\n" + "=" * 60)