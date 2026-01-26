"""
Comprehensive throughput benchmark comparing all endpoints
"""

import asyncio
import aiohttp
import time
import statistics
from pathlib import Path

async def make_request(session, url, image_path):
    """Make async prediction request"""
    with open(image_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename='test.jpg', content_type='image/jpeg')
        
        start = time.time()
        async with session.post(url, data=data) as response:
            result = await response.json()
            end = time.time()
            
            return {
                'latency': (end - start) * 1000,
                'server_latency': result.get('latency_ms', 0),
                'status': response.status
            }

async def benchmark_endpoint(endpoint, n_concurrent=10, duration_seconds=30):
    """
    Benchmark endpoint with concurrent requests
    
    Args:
        endpoint: API endpoint to test
        n_concurrent: Number of concurrent requests
        duration_seconds: Test duration
    """
    url = f"http://localhost:8000{endpoint}"
    image_path = "test_dog.jpg"
    
    print(f"\nBenchmarking {endpoint}")
    print(f"  Concurrency: {n_concurrent}")
    print(f"  Duration: {duration_seconds}s")
    
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration_seconds:
            # Create batch of concurrent requests
            tasks = [
                make_request(session, url, image_path)
                for _ in range(n_concurrent)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    latencies = [r['latency'] for r in results]
    server_latencies = [r['server_latency'] for r in results]
    
    return {
        'endpoint': endpoint,
        'duration': elapsed,
        'total_requests': len(results),
        'requests_per_second': len(results) / elapsed,
        'latency': {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p95': sorted(latencies)[int(len(latencies) * 0.95)],
            'min': min(latencies),
            'max': max(latencies)
        },
        'server_latency': {
            'mean': statistics.mean(server_latencies),
            'p95': sorted(server_latencies)[int(len(server_latencies) * 0.95)]
        }
    }

async def run_benchmarks():
    """Run comprehensive throughput benchmarks"""
    print("=" * 70)
    print("THROUGHPUT BENCHMARK - Comparing All Endpoints")
    print("=" * 70)
    
    endpoints = [
        ("/predict", "PyTorch Baseline"),
        ("/predict/onnx", "ONNX Optimized"),
        ("/predict/batched/onnx", "ONNX + Batching")
    ]
    
    results = []
    
    for endpoint, name in endpoints:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        result = await benchmark_endpoint(
            endpoint,
            n_concurrent=10,
            duration_seconds=30
        )
        result['name'] = name
        results.append(result)
        
        print(f"\n✓ Completed: {result['total_requests']} requests in {result['duration']:.1f}s")
        print(f"  Throughput: {result['requests_per_second']:.2f} req/s")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_throughput = results[0]['requests_per_second']
    
    for result in results:
        speedup = result['requests_per_second'] / baseline_throughput
        print(f"\n📊 {result['name']}:")
        print(f"  Throughput: {result['requests_per_second']:.2f} req/s")
        print(f"  Speedup: {speedup:.2f}×")
        print(f"  Mean latency: {result['latency']['mean']:.2f}ms")
        print(f"  P95 latency: {result['latency']['p95']:.2f}ms")
        print(f"  Total requests: {result['total_requests']}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmarks())