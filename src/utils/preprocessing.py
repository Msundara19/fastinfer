import torch
from torchvision import transforms
from PIL import Image
import io

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to preprocessed tensor"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_batch(self, image_bytes_list: list) -> torch.Tensor:
        """Preprocess multiple images"""
        tensors = [self.preprocess(img_bytes).squeeze(0) for img_bytes in image_bytes_list]
        return torch.stack(tensors)