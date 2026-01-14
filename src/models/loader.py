import torch
import torchvision.models as models
from typing import Dict
import time

class ModelLoader:
    def __init__(self, model_name: str = "resnet50"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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