# FastInfer

**ML inference optimization API — ResNet-50 on Apple Silicon, deployed to production.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo →](https://fastinfer-production.up.railway.app)**

---

## The Problem

Running machine learning models in production is slow by default. A standard PyTorch model makes no use of the specialized hardware sitting right on the chip — the Apple Neural Engine, half-precision compute, or hardware-level batching. Most inference services just ship the model as-is and call it done.

The question this project asks: **how much faster can a real model go, and where does the ceiling actually come from?**

---

## What Was Built

FastInfer is a production-ready image classification API built around ResNet-50. Rather than applying a single optimization and stopping, this project layers every meaningful acceleration technique available on Apple Silicon, benchmarks each one honestly, and exposes them all as separate API endpoints for direct comparison.

**Hardware:** Apple M5 (Neural Engine + GPU via Metal Performance Shaders)

Each optimization was implemented, measured, and compared:

- **ONNX Export + CoreML Execution Provider** — routes inference through the Apple Neural Engine (121 of 124 ops accelerated)
- **Direct CoreML FP16 Conversion** — bypasses ONNX entirely for native half-precision inference
- **Static Batch Models** — pre-compiled CoreML models at batch sizes 1/2/4/8, eliminating dynamic shape fallback to CPU
- **Dynamic Request Batching** — asyncio queue that groups concurrent requests before a single inference call
- **Multi-Worker Serving** — multiple uvicorn processes, each with its own model instance, for horizontal scaling
- **Redis Response Caching** — repeated identical requests served from memory, skipping preprocessing and inference entirely
- **INT8 Quantization** — ONNX INT8 model for CPU-only deployments (cloud or non-Apple hardware)
- **Prometheus Metrics** — request counts, latency histograms, and cache hit rates
- **Interactive Dashboard** — live inference UI with backend toggle and AI-powered result analysis via Groq LLM

---

## Results

All measurements are taken against a PyTorch MPS (GPU) baseline on the same hardware.

### Latency — Single request, how fast?

| Backend | Median Latency | vs Baseline |
|---------|---------------|-------------|
| PyTorch MPS (baseline) | 16.2ms | — |
| ONNX + CoreML FP32 | 10.1ms | **1.6× faster** |
| Direct CoreML FP16 | 10.3ms | **1.6× faster** |

Pure model inference time (excluding preprocessing): CoreML FP16 = **1.17ms**.

### Throughput — Under load, how many requests per second?

Tested with 10 concurrent clients over 30 seconds, 4 worker processes.

| Backend | Requests/sec | vs Baseline |
|---------|-------------|-------------|
| PyTorch MPS, 1 worker | 45.8 | — |
| PyTorch MPS, 4 workers | 82.5 | **1.8×** |
| CoreML FP16 + Static Batching, 4 workers | 104.4 | **2.3×** |
| ONNX + CoreML FP32, 4 workers | 166.2 | *(see note)* |

> The 166 req/s ONNX figure is partially inflated by Redis cache hits during repeated-image benchmarking. The CoreML static batch result (104 req/s) is the more representative real-world number.

---

## What the Data Revealed

**Preprocessing is the real bottleneck, not the model.**
After profiling end-to-end request time, inference accounts for only ~1ms of a ~10ms request. PIL image decoding and normalization take ~8.5ms. Accelerating the model backend has a hard ceiling until the preprocessing pipeline is also optimized — a finding that isn't obvious without actually measuring it.

**Dynamic batching backfires with multiple workers.**
With a single worker, dynamic batching groups concurrent requests and amortizes inference cost. With multiple workers, requests spread across processes — each worker's batcher sees too few concurrent requests to fill a batch, and the added queue wait hurts more than batching helps. Static CoreML batch models outperform dynamic batching at scale.

**INT8 quantization is incompatible with CoreML.**
`ConvInteger` and `DynamicQuantizeLinear` ops are unsupported by the CoreML execution provider, silently routing the quantized model to CPU. The INT8 path exists for non-Apple hardware deployments only.

---

## Live Demo

**[fastinfer-production.up.railway.app](https://fastinfer-production.up.railway.app)**

Upload any image and compare PyTorch vs ONNX inference side by side. The dashboard runs live on Railway with Redis caching enabled.

---

## Getting Started

### Option 1 — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/Msundara19/fastinfer.git
cd fastinfer
docker compose up
```

On first run, Docker installs dependencies, converts the PyTorch model to ONNX, and starts the server alongside Redis. The API is available at `http://localhost:8000`.

> **Note:** Docker runs on Linux — CoreML and Apple Neural Engine backends are unavailable in this mode. The PyTorch CPU and ONNX CPU endpoints are fully functional. CoreML benchmark numbers were measured natively on Apple Silicon.

### Option 2 — Native (Apple Silicon, full performance)

Requires Python 3.11+, Apple Silicon Mac, and Redis if caching is enabled.

```bash
git clone https://github.com/Msundara19/fastinfer.git
cd fastinfer

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install coremltools  # macOS only
```

Convert models (one-time setup):

```bash
python scripts/convert_to_onnx.py       # PyTorch → ONNX
python scripts/convert_to_coreml.py     # PyTorch → CoreML FP16 (b1/b2/b4/b8)
python scripts/quantize_model.py        # ONNX → INT8 (optional)
```

Start the server:

```bash
python run.py --workers 4
```

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

Server must be running before executing benchmarks.

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
| AI Analysis | Groq LLM (llama-3.1-8b-instant) |
| Deployment | Railway (Docker-based) |
| Benchmarking | aiohttp async load testing |

---

## License

MIT — see [LICENSE](LICENSE) for details.
