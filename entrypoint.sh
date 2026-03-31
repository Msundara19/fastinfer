#!/bin/bash
set -e

# Convert PyTorch model to ONNX on first run (cached in Docker volume after that)
if [ ! -f "models/resnet50.onnx" ]; then
    echo "First run: converting PyTorch model to ONNX (this takes ~30 seconds)..."
    python scripts/convert_to_onnx.py
    echo "Model conversion complete."
fi

echo "Starting FastInfer server with ${WORKERS:-1} workers on port ${PORT:-8000}..."
exec python run.py --workers "${WORKERS:-1}" --port "${PORT:-8000}"
