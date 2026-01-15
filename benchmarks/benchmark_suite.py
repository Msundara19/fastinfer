"""
Comprehensive benchmarking suite for FastInfer
Measures latency, throughput, and resource usage
"""

import requests
import time
import statistics
import json
from datetime import datetime
from pathlib import Path
import psutil
import os

class BenchmarkSuite:
    def __init__(self, base_url="http://localhost:8000", test_image="test_dog.jpg"):
        self.base_url = base_url
        self.test_image = test_image
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "server_url": base_url,
            "test_image": test_image
        }
    
    def warm_up(self, n=5):
        """Warm up the model with a few requests"""
        print(f"Warming up with {n} requests...")
        for i in range(n):
            self._make_prediction()
        print("✓ Warm-up complete\n")
    
    def _make_prediction(self):
        """Make a single prediction request"""
        with open(self.test_image, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/predict", files=files)
        return response.json()
    
    def benchmark_latency(self, n_requests=100):
        """Benchmark single-request latency"""
        print(f"Running latency benchmark ({n_requests} requests)...")
        
        latencies = []
        server_latencies = []
        
        for i in range(n_requests):
            start = time.time()
            result = self._make_prediction()
            end = time.time()
            
            total_latency = (end - start) * 1000  # Convert to ms
            server_latency = result.get('latency_ms', 0)
            
            latencies.append(total_latency)
            server_latencies.append(server_latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_requests} requests")
        
        self.results['latency'] = {
            'n_requests': n_requests,
            'total_latency': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'p95': self._percentile(latencies, 95),
                'p99': self._percentile(latencies, 99)
            },
            'server_latency': {
                'mean': statistics.mean(server_latencies),
                'median': statistics.median(server_latencies),
                'min': min(server_latencies),
                'max': max(server_latencies),
                'p95': self._percentile(server_latencies, 95),
                'p99': self._percentile(server_latencies, 99)
            }
        }
        
        print("✓ Latency benchmark complete\n")
    
    def benchmark_throughput(self, duration_seconds=30):
        """Benchmark requests per second"""
        print(f"Running throughput benchmark ({duration_seconds}s)...")
        
        start_time = time.time()
        request_count = 0
        latencies = []
        
        while time.time() - start_time < duration_seconds:
            req_start = time.time()
            self._make_prediction()
            req_end = time.time()
            
            request_count += 1
            latencies.append((req_end - req_start) * 1000)
        
        elapsed = time.time() - start_time
        requests_per_second = request_count / elapsed
        
        self.results['throughput'] = {
            'duration_seconds': elapsed,
            'total_requests': request_count,
            'requests_per_second': requests_per_second,
            'mean_latency_ms': statistics.mean(latencies)
        }
        
        print(f"✓ Throughput: {requests_per_second:.2f} req/s\n")
    
    def benchmark_resource_usage(self, n_requests=50):
        """Monitor CPU and memory during inference"""
        print(f"Running resource usage benchmark ({n_requests} requests)...")
        
        process = psutil.Process(os.getpid())
        cpu_percentages = []
        memory_mb = []
        
        for i in range(n_requests):
            cpu_before = process.cpu_percent(interval=0.1)
            mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
            
            self._make_prediction()
            
            cpu_after = process.cpu_percent(interval=0.1)
            mem_after = process.memory_info().rss / 1024 / 1024
            
            cpu_percentages.append(cpu_after)
            memory_mb.append(mem_after)
        
        self.results['resources'] = {
            'cpu_percent': {
                'mean': statistics.mean(cpu_percentages),
                'max': max(cpu_percentages)
            },
            'memory_mb': {
                'mean': statistics.mean(memory_mb),
                'max': max(memory_mb)
            }
        }
        
        print("✓ Resource usage benchmark complete\n")
    
    def _percentile(self, data, percentile):
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[index]
    
    def print_summary(self):
        """Print benchmark results summary"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if 'latency' in self.results:
            lat = self.results['latency']['server_latency']
            print(f"\n📊 LATENCY (Server-side)")
            print(f"  Mean:     {lat['mean']:.2f}ms")
            print(f"  Median:   {lat['median']:.2f}ms")
            print(f"  P95:      {lat['p95']:.2f}ms")
            print(f"  P99:      {lat['p99']:.2f}ms")
            print(f"  Min/Max:  {lat['min']:.2f}ms / {lat['max']:.2f}ms")
        
        if 'throughput' in self.results:
            tp = self.results['throughput']
            print(f"\n🚀 THROUGHPUT")
            print(f"  Requests/sec: {tp['requests_per_second']:.2f}")
            print(f"  Total requests: {tp['total_requests']}")
            print(f"  Duration: {tp['duration_seconds']:.1f}s")
        
        if 'resources' in self.results:
            res = self.results['resources']
            print(f"\n💻 RESOURCE USAGE")
            print(f"  CPU (mean): {res['cpu_percent']['mean']:.1f}%")
            print(f"  CPU (max):  {res['cpu_percent']['max']:.1f}%")
            print(f"  Memory (mean): {res['memory_mb']['mean']:.1f}MB")
            print(f"  Memory (max):  {res['memory_mb']['max']:.1f}MB")
        
        print("\n" + "="*60)
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmarks/results/baseline_{timestamp}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def run_all(self):
        """Run complete benchmark suite"""
        print("\n🔥 Starting FastInfer Benchmark Suite\n")
        
        # Check server health
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                print("❌ Server health check failed!")
                return
            print("✓ Server is healthy\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return
        
        # Run benchmarks
        self.warm_up(n=5)
        self.benchmark_latency(n_requests=100)
        self.benchmark_throughput(duration_seconds=30)
        self.benchmark_resource_usage(n_requests=50)
        
        # Display and save results
        self.print_summary()
        self.save_results()
        
        print("\n✅ Benchmark suite complete!")


if __name__ == "__main__":
    # Run the benchmark suite
    suite = BenchmarkSuite()
    suite.run_all()