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
            
            start = time.time()
            response = requests.post(f"http://localhost:8000{endpoint}", files=files)
            end = time.time()
            
            if response.status_code == 200:
                data = response.json()
                server_latency = data['latency_ms']
                latencies.append(server_latency)
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_requests} complete")
    
    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p95': sorted(latencies)[int(len(latencies) * 0.95)]
    }

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch vs ONNX Performance Comparison")
    print("=" * 60)
    
    pytorch_results = benchmark_endpoint("/predict", n_requests=50)
    onnx_results = benchmark_endpoint("/predict/onnx", n_requests=50)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 PyTorch Baseline:")
    print(f"  Mean:   {pytorch_results['mean']:.2f}ms")
    print(f"  Median: {pytorch_results['median']:.2f}ms")
    print(f"  P95:    {pytorch_results['p95']:.2f}ms")
    print(f"  Range:  {pytorch_results['min']:.2f}ms - {pytorch_results['max']:.2f}ms")
    
    print(f"\n🚀 ONNX Optimized:")
    print(f"  Mean:   {onnx_results['mean']:.2f}ms")
    print(f"  Median: {onnx_results['median']:.2f}ms")
    print(f"  P95:    {onnx_results['p95']:.2f}ms")
    print(f"  Range:  {onnx_results['min']:.2f}ms - {onnx_results['max']:.2f}ms")
    
    speedup_mean = pytorch_results['mean'] / onnx_results['mean']
    speedup_p95 = pytorch_results['p95'] / onnx_results['p95']
    
    print(f"\n✨ IMPROVEMENT:")
    print(f"  Mean latency: {speedup_mean:.2f}× faster")
    print(f"  P95 latency:  {speedup_p95:.2f}× faster")
    print(f"  Latency reduction: {pytorch_results['mean'] - onnx_results['mean']:.2f}ms")
    
    print("\n" + "=" * 60)