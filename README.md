
# FastInfer

A machine learning inference API that demonstrates systematic performance optimization on Apple Silicon — taking a standard PyTorch model and making it significantly faster through hardware-specific acceleration techniques.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Project Does

FastInfer serves a ResNet-50 image classifier through multiple optimized backends and measures the real-world impact of each optimization. The goal was to understand — and demonstrate — the actual performance gains available on Apple Silicon hardware, and where the genuine bottlenecks lie.

**Hardware:** Apple M5 (Neural Engine + GPU via Metal Performance Shaders)

---

## Results

All measurements are taken against a PyTorch MPS (GPU) baseline to ensure a fair comparison. Each backend runs on accelerated hardware.

### Latency — How fast is a single prediction?

| Backend | Median Latency | vs Baseline |
|---------|---------------|-------------|
| PyTorch MPS (baseline) | 16.2ms | — |
| ONNX + CoreML FP32 | 10.1ms | **1.6× faster** |
| Direct CoreML FP16 | 10.3ms | **1.6× faster** |

Pure inference time (model only, no image processing): CoreML FP16 = **1.17ms**.
The end-to-end latency is dominated by image preprocessing (~8.5ms), not the model itself — a finding that shaped the architecture decisions in this project.

### Throughput — How many requests per second under load?

Tested with 10 concurrent clients over 30 seconds, server running 4 worker processes.

| Backend | Requests/sec | vs Single-Worker Baseline |
|---------|-------------|--------------------------|
| PyTorch MPS, 1 worker (original baseline) | 45.8 | — |
| PyTorch MPS, 4 workers | 82.5 | **1.8×** |
| CoreML FP16 + Static Batching, 4 workers | 104.4 | **2.3×** |
| ONNX + CoreML FP32, 4 workers | 166.2 | *(see note below)* |

> **On the ONNX number:** The 166 req/s figure is partially inflated by Redis cache hits when the benchmark repeatedly sends the same image. The CoreML Static Batching result (104 req/s) is the most representative real-world number as it bypasses the cache.

---

## What Was Built

Each optimization layer was implemented, measured, and compared:

- **ONNX Export + CoreML Execution Provider** — routes inference to the Apple Neural Engine (121 of 124 ops accelerated)
- **Direct CoreML FP16 Conversion** — bypasses ONNX entirely for native half-precision inference
- **Static Batch Models** — pre-compiled CoreML models at batch sizes 1/2/4/8, avoiding dynamic shape fallback to CPU
- **Dynamic Request Batching** — asyncio queue that groups concurrent requests before inference
- **Multi-Worker Serving** — multiple uvicorn processes, each with its own model instance
- **Redis Response Caching** — identical requests served from cache
- **INT8 Quantization** — ONNX INT8 for CPU-only deployments (CoreML does not support INT8 ops)
- **Prometheus Metrics** — request counts, latency histograms, cache hit rates

---

## Key Findings

**The Neural Engine is fast, but preprocessing is the real bottleneck.**
After profiling, inference accounts for only ~1ms of a ~10ms request. PIL image decode and transforms take ~8.5ms. Optimizing the model backend has a ceiling unless preprocessing is also addressed.

**Dynamic batching helps only with a single worker.**
With multiple workers, requests are distributed across processes. Each worker's batcher receives too few concurrent requests to form meaningful batches, so the queue delay hurts more than batching helps. Static batch CoreML models outperform dynamic batching at scale.

**INT8 quantization doesn't work on CoreML.**
`ConvInteger` and `DynamicQuantizeLinear` ops are unsupported by the CoreML execution provider, routing the model to CPU. The INT8 path exists for non-Apple deployments only.

---

## Getting Started

### Option 1 — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/Msundara19/fastinfer.git
cd fastinfer
docker compose up
```

That's it. On first run, Docker will install dependencies, convert the PyTorch model to ONNX, and start the server alongside Redis. The API is available at `http://localhost:8000` and interactive docs at `http://localhost:8000/docs`.

> **Note:** Docker runs on Linux, so the CoreML and Apple Neural Engine backends are not available in this mode. The `/predict` (PyTorch CPU) and `/predict/onnx` (ONNX CPU) endpoints are fully functional. The CoreML performance results in this README were measured natively on Apple Silicon.

---

### Option 2 — Native (Apple Silicon, full performance)

Requires Python 3.11+, Apple Silicon Mac (M1 or later), and Redis if caching is enabled.

```bash
git clone https://github.com/Msundara19/fastinfer.git
cd fastinfer

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install coremltools  # macOS only
```

Convert models (one-time):

```bash
python scripts/convert_to_onnx.py       # PyTorch → ONNX
python scripts/convert_to_coreml.py     # PyTorch → CoreML FP16 (b1/b2/b4/b8)
python scripts/quantize_model.py        # ONNX → INT8 (optional)
```

Start the server:

```bash
python run.py --workers 4
```

The API will be available at `http://localhost:8000`.

---

## API

| Endpoint | Backend | Best For |
|----------|---------|----------|
| `POST /predict` | PyTorch MPS | Baseline / debugging |
| `POST /predict/onnx` | ONNX + CoreML FP32 | Low latency, cache-friendly |
| `POST /predict/coreml` | Direct CoreML FP16 | Lowest single-request latency |
| `POST /predict/batched` | PyTorch MPS + Dynamic Batching | Single-worker high concurrency |
| `POST /predict/batched/coreml` | CoreML FP16 + Static Batch | Best multi-worker throughput |
| `POST /predict/quantized` | INT8 ONNX (CPU) | Non-Apple hardware |
| `GET /cache/stats` | — | Redis hit/miss rates |
| `GET /metrics` | — | Prometheus metrics |

**Example:**

```bash
curl -X POST "http://localhost:8000/predict/coreml" -F "file=@image.jpg"
```

```json
{
  "class": "golden_retriever",
  "confidence": 0.89,
  "latency_ms": 10.3,
  "model": "coreml_fp16"
}
```

---

## Running Benchmarks

```bash
# Latency benchmark (50 sequential requests per backend)
python benchmarks/compare_onnx.py

# Throughput benchmark (10 concurrent clients, 30 seconds per backend)
python benchmarks/benchmark_throughput.py
```

Make sure the server is running before executing benchmarks.

---

## Project Structure

```
fastinfer/
├── src/
│   ├── server.py                  # FastAPI app and all endpoints
│   ├── config.py                  # Configuration and environment variables
│   ├── models/
│   │   ├── loader.py              # PyTorch model loading (MPS/CPU)
│   │   ├── optimized.py           # ONNX and quantized model classes
│   │   └── coreml_model.py        # CoreML model classes (FP16, static batch)
│   ├── optimization/
│   │   └── batching.py            # Dynamic request batcher
│   └── utils/
│       ├── preprocessing.py       # Image preprocessing pipeline
│       ├── cache.py               # Redis async cache
│       └── metrics.py             # Prometheus instrumentation
├── scripts/
│   ├── convert_to_onnx.py         # Export PyTorch model to ONNX
│   ├── convert_to_coreml.py       # Convert to CoreML FP16
│   └── quantize_model.py          # ONNX INT8 quantization
├── benchmarks/
│   ├── compare_onnx.py            # Sequential latency comparison
│   └── benchmark_throughput.py    # Concurrent throughput benchmark
├── run.py                         # Server launcher (single / multi-worker)
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + uvicorn |
| ML Runtime | PyTorch 2.6.0, ONNX Runtime, coremltools |
| Hardware Acceleration | Apple Neural Engine (CoreML), Apple GPU (MPS) |
| Caching | Redis + aioredis |
| Monitoring | Prometheus |
| Benchmarking | aiohttp async load testing |

---

## License

MIT — see [LICENSE](LICENSE) for details.
