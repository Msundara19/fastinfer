import torch
import numpy as np
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
        # Resize + crop only — no ToTensor/Normalize.
        # Used for CoreML endpoints whose models have normalization baked in.
        self._resize_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to float32 normalized tensor [1, 3, 224, 224]."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def preprocess_for_coreml(self, image_bytes: bytes) -> np.ndarray:
        """
        Resize + crop + /255 → float32 numpy [1, 3, 224, 224] in [0, 1].

        Skips Normalize — baked into the CoreML model (NormalizedResNet wrapper).
        Saves ~0.5ms vs full preprocess() pipeline.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self._resize_crop(image)
        # HWC uint8 → CHW float32 [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0  # [224, 224, 3]
        arr = arr.transpose(2, 0, 1)[np.newaxis]          # [1, 3, 224, 224]
        return arr

    def preprocess_batch(self, image_bytes_list: list) -> torch.Tensor:
        """Preprocess multiple images into a float32 normalized batch tensor."""
        tensors = [self.preprocess(img_bytes).squeeze(0) for img_bytes in image_bytes_list]
        return torch.stack(tensors)