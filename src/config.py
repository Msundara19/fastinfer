from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Model
    MODEL_NAME: str = "resnet50"
    BATCH_SIZE: int = 8
    MAX_BATCH_WAIT_MS: int = 10
    
    # Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_TTL: int = 3600
    ENABLE_CACHE: bool = False
    
    # Optimization flags
    ENABLE_ONNX: bool = False
    ENABLE_QUANTIZATION: bool = False
    ENABLE_BATCHING: bool = False

    # Model paths
    QUANTIZED_MODEL_PATH: str = "models/resnet50_int8.onnx"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()