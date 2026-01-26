"""
Aggressive throughput benchmark with higher concurrency
"""

import asyncio
import aiohttp
import time
import statistics

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

async def benchmark_endpoint(endpoint, n_concurrent=50, duration_seconds=30):
    """
    Benchmark with HIGH concurrency to trigger batching
    """
    url = f"http://localhost:8000{endpoint}"
    image_path = "test_dog.jpg"
    
    print(f"\n{'='*70}")
    print(f"Benchmarking {endpoint}")
    print(f"  Concurrency: {n_concurrent}")
    print(f"  Duration: {duration_seconds}s")
    print(f"{'='*70}")
    
    results = []
    start_time = time.time()
    request_count = 0
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration_seconds:
            # Create large batch of concurrent requests
            tasks = [
                make_request(session, url, image_path)
                for _ in range(n_concurrent)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                    request_count += 1
            
            # Progress update
            if request_count % 100 == 0:
                elapsed = time.time() - start_time
                current_rps = request_count / elapsed
                print(f"  Progress: {request_count} requests, {current_rps:.2f} req/s")
    
    elapsed = time.time() - start_time
    
    if not results:
        return None
    
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
    """Run aggressive throughput benchmarks"""
    print("=" * 70)
    print("AGGRESSIVE THROUGHPUT BENCHMARK")
    print("High Concurrency (50 parallel requests)")
    print("=" * 70)
    
    endpoints = [
        ("/predict", "PyTorch Baseline (no batching)"),
        ("/predict/batched", "PyTorch + Dynamic Batching")
    ]
    
    results = []
    
    for endpoint, name in endpoints:
        print(f"\n\n🚀 Testing: {name}")
        
        result = await benchmark_endpoint(
            endpoint,
            n_concurrent=50,  # HIGH concurrency
            duration_seconds=30
        )
        
        if result:
            result['name'] = name
            results.append(result)
            
            print(f"\n✅ Completed:")
            print(f"   Total requests: {result['total_requests']}")
            print(f"   Throughput: {result['requests_per_second']:.2f} req/s")
            print(f"   Mean latency: {result['latency']['mean']:.2f}ms")
    
    # Print comparison
    print("\n\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    if len(results) >= 2:
        baseline_throughput = results[0]['requests_per_second']
        
        for i, result in enumerate(results):
            speedup = result['requests_per_second'] / baseline_throughput
            
            print(f"\n{i+1}. {result['name']}:")
            print(f"   Throughput: {result['requests_per_second']:.2f} req/s")
            print(f"   Speedup: {speedup:.2f}×")
            print(f"   Mean latency: {result['latency']['mean']:.2f}ms")
            print(f"   P95 latency: {result['latency']['p95']:.2f}ms")
            print(f"   Total requests: {result['total_requests']}")
        
        improvement = results[1]['requests_per_second'] / results[0]['requests_per_second']
        print(f"\n🎯 Dynamic Batching Improvement: {improvement:.2f}× throughput increase")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmarks())