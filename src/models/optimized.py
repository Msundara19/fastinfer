"""
ONNX optimized model inference
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

class ONNXModel:
    def __init__(self, model_path="models/resnet50.onnx"):
        """Initialize ONNX Runtime session"""
        self.model_path = model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Run: python scripts/convert_to_onnx.py"
            )
        
        print(f"Loading ONNX model from {model_path}...")
        start = time.time()
        
        # Configure ONNX Runtime session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4  # Use 4 threads for CPU
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']  # Use CPU provider
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        load_time = time.time() - start
        print(f"ONNX model loaded in {load_time:.2f}s")
    
    def predict(self, input_tensor):
        """
        Run inference on input tensor
        
        Args:
            input_tensor: numpy array of shape [batch_size, 3, 224, 224]
        
        Returns:
            numpy array of shape [batch_size, 1000] (logits)
        """
        # Ensure input is numpy array with correct dtype
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = input_tensor.numpy()
        
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        
        return outputs[0]
    
    def get_model_info(self):
        """Get model metadata"""
        return {
            "model_path": str(self.model_path),
            "model_type": "ONNX",
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers()
        }