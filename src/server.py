from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch
import json
import time

from src.config import get_settings
from src.models.loader import ModelLoader
from src.utils.preprocessing import ImagePreprocessor
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

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    model = model_loader.load_model()
    print(f"Server ready! Model info: {model_loader.get_model_info()}")

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
    """Single image prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Track inference time
        with track_time() as get_elapsed:
            # Preprocess
            input_tensor = preprocessor.preprocess(image_bytes)
            input_tensor = input_tensor.to(model_loader.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            confidence, class_idx = torch.max(probabilities, dim=0)
            predicted_class = IMAGENET_CLASSES[class_idx.item()]
            
            latency_ms = get_elapsed() * 1000
        
        # Update metrics
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