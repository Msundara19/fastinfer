from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.models.optimized import ONNXModel, QuantizedONNXModel
try:
    from src.models.coreml_model import CoreMLModel, StaticBatchCoreMLModel
    _COREML_AVAILABLE = True
except (ImportError, RuntimeError):
    _COREML_AVAILABLE = False
from src.optimization.batching import DynamicBatcher
import numpy as np
import torch
import json
import time

from src.config import get_settings
from src.models.loader import ModelLoader
from src.utils.preprocessing import ImagePreprocessor
from src.utils.cache import PredictionCache
from src.utils.metrics import (
    prediction_counter, prediction_latency, error_counter, track_time
)

# Initialize
app = FastAPI(title="FastInfer", version="0.1.0")
settings = get_settings()

# Load ImageNet labels
with open("imagenet_classes.json", "r") as f:
    IMAGENET_CLASSES = json.load(f)

# Global instances
model_loader = ModelLoader(settings.MODEL_NAME)
model = None
preprocessor = ImagePreprocessor()
onnx_model = None
coreml_model = None
coreml_batcher = None
pytorch_batcher = None
quantized_model = None
cache: PredictionCache = None


@app.on_event("startup")
async def startup_event():
    """Load models and start batchers on startup"""
    global model, onnx_model, coreml_model, coreml_batcher, pytorch_batcher, quantized_model, cache

    # Load PyTorch model on GPU (MPS/CUDA)
    model = model_loader.load_model()
    print(f"PyTorch model ready: {model_loader.get_model_info()}")

    # Initialize PyTorch MPS batcher
    pytorch_batcher = DynamicBatcher(
        model=model,
        max_batch_size=settings.BATCH_SIZE,
        max_wait_ms=settings.MAX_BATCH_WAIT_MS,
        use_onnx=False
    )
    await pytorch_batcher.start()

    # Load ONNX model (CoreML on Apple Silicon)
    try:
        onnx_model = ONNXModel()
        print(f"ONNX model ready: {onnx_model.get_model_info()}")
    except FileNotFoundError:
        print("⚠️  ONNX model not found. Run: python scripts/convert_to_onnx.py")
        onnx_model = None

    # Load direct CoreML models (FP16, static batch) — macOS/Apple Silicon only
    if not _COREML_AVAILABLE:
        print("ℹ️  CoreML not available on this platform — skipping CoreML endpoints")
    else:
        try:
            coreml_model = CoreMLModel()
            print(f"CoreML single model ready: {coreml_model.get_model_info()}")
            coreml_batch_model = StaticBatchCoreMLModel()
            coreml_batcher = DynamicBatcher(
                model=coreml_batch_model,
                max_batch_size=settings.BATCH_SIZE,
                max_wait_ms=settings.MAX_BATCH_WAIT_MS,
                use_onnx=True,  # uses model.predict(numpy_array) interface
            )
            await coreml_batcher.start()
            print(f"CoreML batch model ready: {coreml_batch_model.get_model_info()}")
        except FileNotFoundError:
            print("⚠️  CoreML models not found. Run: python scripts/convert_to_coreml.py")
        coreml_model = None
        coreml_batcher = None

    # Connect to Redis cache if enabled
    if settings.ENABLE_CACHE:
        cache = PredictionCache(settings.REDIS_HOST, settings.REDIS_PORT, settings.CACHE_TTL)
        try:
            await cache.connect()
        except Exception as e:
            print(f"⚠️  Redis unavailable, caching disabled: {e}")
            cache = None

    # Load INT8 quantized model if enabled (CPU-only, for non-Apple deployments)
    if settings.ENABLE_QUANTIZATION:
        try:
            quantized_model = QuantizedONNXModel(settings.QUANTIZED_MODEL_PATH)
            print(f"INT8 quantized model ready: {quantized_model.get_model_info()}")
        except FileNotFoundError:
            print("⚠️  INT8 model not found. Run: python scripts/quantize_model.py")
            quantized_model = None


@app.on_event("shutdown")
async def shutdown_event():
    if pytorch_batcher:
        await pytorch_batcher.stop()
    if coreml_batcher:
        await coreml_batcher.stop()
    if cache:
        await cache.close()


@app.get("/")
async def root():
    return {
        "service": "FastInfer",
        "version": "0.1.0",
        "model": settings.MODEL_NAME,
        "status": "healthy"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """PyTorch MPS inference — GPU baseline"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_bytes = await file.read()

        with track_time() as get_elapsed:
            input_tensor = preprocessor.preprocess(image_bytes)
            input_tensor = input_tensor.to(model_loader.device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            confidence, class_idx = torch.max(probabilities, dim=0)
            predicted_class = IMAGENET_CLASSES[class_idx.item()]
            latency_ms = get_elapsed() * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        return {
            "class": predicted_class,
            "confidence": float(confidence),
            "latency_ms": round(latency_ms, 2),
            "class_idx": int(class_idx)
        }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/onnx")
async def predict_onnx(file: UploadFile = File(...)):
    """ONNX + CoreML inference — GPU optimized, Redis cached"""
    if onnx_model is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX model not loaded. Run: python scripts/convert_to_onnx.py"
        )

    try:
        image_bytes = await file.read()

        # Cache lookup — skips all preprocessing and inference on hit
        if cache:
            cached = await cache.get(image_bytes)
            if cached:
                prediction_counter.inc()
                return cached

        with track_time() as get_elapsed:
            input_tensor = preprocessor.preprocess(image_bytes)
            input_numpy = input_tensor.numpy()

            outputs = onnx_model.predict(input_numpy)

            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
            probabilities = exp_outputs / exp_outputs.sum()
            class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[class_idx])
            predicted_class = IMAGENET_CLASSES[class_idx]
            latency_ms = get_elapsed() * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        result = {
            "class": predicted_class,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "class_idx": class_idx,
            "model": "onnx_coreml"
        }

        if cache:
            await cache.set(image_bytes, result)

        return result

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/coreml")
async def predict_coreml(file: UploadFile = File(...)):
    """Direct CoreML FP16 — single image, fastest single-request path"""
    if coreml_model is None:
        raise HTTPException(
            status_code=503,
            detail="CoreML model not found. Run: python scripts/convert_to_coreml.py"
        )

    try:
        image_bytes = await file.read()

        if cache:
            cached = await cache.get(image_bytes)
            if cached:
                prediction_counter.inc()
                return cached

        with track_time() as get_elapsed:
            input_numpy = preprocessor.preprocess_for_coreml(image_bytes)
            outputs = coreml_model.predict(input_numpy)

            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
            probabilities = exp_outputs / exp_outputs.sum()
            class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[class_idx])
            predicted_class = IMAGENET_CLASSES[class_idx]
            latency_ms = get_elapsed() * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        result = {
            "class": predicted_class,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "class_idx": class_idx,
            "model": "coreml_fp16"
        }
        if cache:
            await cache.set(image_bytes, result)
        return result

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batched/coreml")
async def predict_batched_coreml(file: UploadFile = File(...)):
    """Direct CoreML FP16 with static-batch routing — high concurrency path"""
    if coreml_batcher is None:
        raise HTTPException(
            status_code=503,
            detail="CoreML batch models not found. Run: python scripts/convert_to_coreml.py"
        )

    try:
        image_bytes = await file.read()
        start = time.time()

        input_numpy = preprocessor.preprocess_for_coreml(image_bytes)

        outputs = await coreml_batcher.predict(input_numpy)

        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        probabilities = exp_outputs / exp_outputs.sum()
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        predicted_class = IMAGENET_CLASSES[class_idx]
        latency_ms = (time.time() - start) * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        return {
            "class": predicted_class,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "class_idx": class_idx,
            "model": "coreml_fp16_batched"
        }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batched")
async def predict_batched(file: UploadFile = File(...)):
    """PyTorch MPS with dynamic batching — high concurrency path"""
    if pytorch_batcher is None:
        raise HTTPException(status_code=503, detail="Batcher not initialized")

    try:
        image_bytes = await file.read()
        start = time.time()

        input_tensor = preprocessor.preprocess(image_bytes)
        input_numpy = input_tensor.numpy()

        outputs = await pytorch_batcher.predict(input_numpy)

        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        probabilities = exp_outputs / exp_outputs.sum()
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        predicted_class = IMAGENET_CLASSES[class_idx]
        latency_ms = (time.time() - start) * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        return {
            "class": predicted_class,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "class_idx": class_idx,
            "model": "pytorch_mps_batched"
        }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/quantized")
async def predict_quantized(file: UploadFile = File(...)):
    """INT8 quantized ONNX — CPU-only, for non-Apple/cloud deployments"""
    if quantized_model is None:
        raise HTTPException(
            status_code=503,
            detail="Quantized model not loaded. Set ENABLE_QUANTIZATION=True and run scripts/quantize_model.py"
        )

    try:
        image_bytes = await file.read()

        with track_time() as get_elapsed:
            input_tensor = preprocessor.preprocess(image_bytes)
            input_numpy = input_tensor.numpy()

            outputs = quantized_model.predict(input_numpy)

            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
            probabilities = exp_outputs / exp_outputs.sum()
            class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[class_idx])
            predicted_class = IMAGENET_CLASSES[class_idx]
            latency_ms = get_elapsed() * 1000

        prediction_counter.inc()
        prediction_latency.observe(latency_ms / 1000)

        return {
            "class": predicted_class,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "class_idx": class_idx,
            "model": "onnx_int8_dynamic",
            "backend": "cpu"
        }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def cache_stats():
    """Redis cache hit/miss statistics"""
    if not cache:
        return {"enabled": False, "message": "Set ENABLE_CACHE=True to enable caching"}
    return {"enabled": True, **cache.stats()}


@app.get("/batch/stats")
async def batch_stats():
    """Get batching statistics"""
    if pytorch_batcher:
        return {"pytorch": pytorch_batcher.get_stats()}
    return {}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if settings.ENABLE_METRICS:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return {"error": "Metrics disabled"}


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_loader.get_model_info()
