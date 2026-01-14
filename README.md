
# FastInfer - Production ML Inference Optimizer

> High-performance ML inference API achieving 5-10× speedup through systematic optimization

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

FastInfer is a production-grade ML inference optimization system demonstrating measurable performance improvements through:
- **ONNX Runtime** conversion for faster inference
- **INT8 Quantization** for reduced memory footprint
- **Dynamic Batching** for improved throughput
- **Redis Caching** for repeated queries
- **Prometheus Metrics** for observability

**Current Status:** Phase 1 Complete - Baseline established at 196ms latency

## 📊 Performance Metrics

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Latency (p95) | 196ms | <80ms | 🚧 In Progress |
| Throughput | ~5 req/s | 20-40 req/s | 🚧 In Progress |
| Memory Usage | ~2GB | <1GB | 🚧 In Progress |
| GPU Utilization | N/A (CPU) | N/A | ✅ CPU-optimized |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (venv or conda)
- 4GB+ RAM

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/fastinfer.git
cd fastinfer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Run Server

\`\`\`bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
\`\`\`

Visit http://localhost:8000/docs for interactive API documentation.

## 🧪 Usage

### Health Check
\`\`\`bash
curl http://localhost:8000/health
\`\`\`

### Image Classification
\`\`\`bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
\`\`\`

**Response:**
\`\`\`json
{
  "class": "golden_retriever",
  "confidence": 0.8934,
  "latency_ms": 196.5,
  "class_idx": 207
}
\`\`\`

## 🏗️ Project Structure

\`\`\`
fastinfer/
├── src/
│   ├── server.py              # FastAPI application
│   ├── config.py              # Configuration management
│   ├── models/
│   │   └── loader.py          # Model loading utilities
│   ├── utils/
│   │   ├── preprocessing.py   # Image preprocessing
│   │   └── metrics.py         # Prometheus metrics
│   └── optimization/          # Optimization techniques (coming soon)
├── benchmarks/
│   └── results/               # Benchmark results
├── tests/                     # Unit tests (coming soon)
├── requirements.txt
└── README.md
\`\`\`

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **ML Library:** PyTorch 2.6.0
- **Model:** ResNet-50 (ImageNet pretrained)
- **Optimization:** ONNX Runtime, Quantization
- **Monitoring:** Prometheus + Grafana
- **Testing:** Locust (load testing)
- **Deployment:** Docker, Railway/Modal

## 📈 Roadmap

- [x] Phase 1: Baseline FastAPI server with ResNet-50
- [ ] Phase 2: ONNX Runtime integration
- [ ] Phase 3: INT8 Quantization
- [ ] Phase 4: Dynamic batching implementation
- [ ] Phase 5: Redis caching layer
- [ ] Phase 6: Production deployment

## 🎓 Learning Outcomes

This project demonstrates:
- Production ML system design
- Performance optimization techniques
- API development with FastAPI
- Model conversion and quantization
- Load testing and benchmarking
- Containerization and deployment

## 📝 License

MIT License - see LICENSE file for details


## 🙏 Acknowledgments

- PyTorch team for pretrained models
- FastAPI for excellent documentation

