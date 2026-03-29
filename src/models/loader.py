import torch
import torchvision.models as models
from typing import Dict
import time

class ModelLoader:
    def __init__(self, model_name: str = "resnet50", force_cpu: bool = False):
        self.model_name = model_name
        self.model = None
        if force_cpu:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
    def load_model(self) -> torch.nn.Module:
        """Load pretrained ResNet-50 model"""
        print(f"Loading {self.model_name} on {self.device}...")
        start = time.time()
        
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # torch.compile reduces dispatch overhead on MPS; falls back silently if unsupported
        if self.device.type == "mps":
            try:
                self.model = torch.compile(self.model, backend="aot_eager")
                print("torch.compile enabled (aot_eager backend)")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s")
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": param_count,
            "parameters_millions": f"{param_count / 1e6:.1f}M"
        }